// CUDA Aperiodic Monotile Candidate Hunter v2
// ------------------------------------------------------------
// This is a finite-search candidate hunter, NOT a mathematical proof engine.
// It searches connected polyiamond monotiles on the triangular lattice.
// It uses the GPU to generate and heuristically score huge numbers of candidate shapes,
// then the CPU runs exact finite torus tests and greedy patch-growth tests.
//
// Compile:
//   nvcc -O3 -std=c++17 -arch=sm_80 einstein_hunter.cu -o einstein_hunter
//
// Example run on an A100:
//   ./einstein_hunter --min 8 --max 14 --batch 131072 --period 10 --nodes 200000 --patch-radius 42 --patch-trials 80 --out records_einstein_v1
//
// Important interpretation:
//   periodic        = exact periodic tiling found on a finite torus, so discard.
//   non_tiler_like  = did not grow much in greedy patch tests.
//   candidate       = grew patches and no small periodic tiling found.
//   strong_candidate= grew large patches and no small periodic tiling found.
//
// A true "einstein" claim still requires a formal proof/substitution/SAT proof.
// ------------------------------------------------------------

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cstdio>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define MAX_CELLS 32

#define CUDA_CHECK(call)                                                                  \
    do {                                                                                  \
        cudaError_t err__ = (call);                                                       \
        if (err__ != cudaSuccess) {                                                       \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " -> "       \
                      << cudaGetErrorString(err__) << std::endl;                         \
            std::exit(1);                                                                 \
        }                                                                                 \
    } while (0)

namespace fs = std::filesystem;

static std::atomic<bool> g_stop(false);

void signal_handler(int) {
    g_stop.store(true);
}

struct Cell {
    int x;
    int y;
    int o; // 0 = upward triangle in rhombus (x,y), 1 = downward triangle in rhombus (x,y)
};

struct GShape {
    int n;
    int x[MAX_CELLS];
    int y[MAX_CELLS];
    int o[MAX_CELLS];
    uint64_t seed;
};

static inline bool operator<(const Cell& a, const Cell& b) {
    if (a.x != b.x) return a.x < b.x;
    if (a.y != b.y) return a.y < b.y;
    return a.o < b.o;
}

static inline bool operator==(const Cell& a, const Cell& b) {
    return a.x == b.x && a.y == b.y && a.o == b.o;
}

static inline uint64_t pack_cell(int x, int y, int o) {
    // Pack signed 32-bit x/y safely into a 64-bit key.
    return (static_cast<uint64_t>(static_cast<uint32_t>(x)) << 33) ^
           (static_cast<uint64_t>(static_cast<uint32_t>(y)) << 1) ^
           static_cast<uint64_t>(o & 1);
}

// Triangular lattice adjacency.
// Up(x,y) has down-neighbours D(x,y), D(x-1,y), D(x,y-1).
// Down(x,y) has up-neighbours U(x,y), U(x+1,y), U(x,y+1).
static inline Cell neighbour(const Cell& c, int dir) {
    dir %= 3;
    if (c.o == 0) {
        if (dir == 0) return {c.x, c.y, 1};
        if (dir == 1) return {c.x - 1, c.y, 1};
        return {c.x, c.y - 1, 1};
    } else {
        if (dir == 0) return {c.x, c.y, 0};
        if (dir == 1) return {c.x + 1, c.y, 0};
        return {c.x, c.y + 1, 0};
    }
}

__device__ static inline uint64_t splitmix64_dev(uint64_t& x) {
    uint64_t z = (x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

__device__ static inline uint32_t rng32(uint64_t& s) {
    return static_cast<uint32_t>(splitmix64_dev(s) >> 32);
}

__device__ static inline void d_neighbour(int x, int y, int o, int dir, int* nx, int* ny, int* no) {
    dir %= 3;
    if (o == 0) {
        if (dir == 0) { *nx = x;     *ny = y;     *no = 1; return; }
        if (dir == 1) { *nx = x - 1; *ny = y;     *no = 1; return; }
                    { *nx = x;     *ny = y - 1; *no = 1; return; }
    } else {
        if (dir == 0) { *nx = x;     *ny = y;     *no = 0; return; }
        if (dir == 1) { *nx = x + 1; *ny = y;     *no = 0; return; }
                    { *nx = x;     *ny = y + 1; *no = 0; return; }
    }
}

__device__ static inline bool d_exists(const GShape& s, int x, int y, int o) {
    for (int i = 0; i < s.n; ++i) {
        if (s.x[i] == x && s.y[i] == y && s.o[i] == o) return true;
    }
    return false;
}

__device__ static inline GShape d_make_random_shape(uint64_t& s, int min_cells, int max_cells) {
    int span = max_cells - min_cells + 1;
    int target = min_cells + static_cast<int>(rng32(s) % static_cast<uint32_t>(span));
    if (target > MAX_CELLS) target = MAX_CELLS;

    GShape sh;
    sh.n = 1;
    sh.x[0] = 0;
    sh.y[0] = 0;
    sh.o[0] = 0;
    sh.seed = s;

    int attempts = 0;
    while (sh.n < target && attempts < 4096) {
        ++attempts;
        int base = static_cast<int>(rng32(s) % static_cast<uint32_t>(sh.n));
        int dir = static_cast<int>(rng32(s) % 3U);
        int nx, ny, no;
        d_neighbour(sh.x[base], sh.y[base], sh.o[base], dir, &nx, &ny, &no);
        if (!d_exists(sh, nx, ny, no)) {
            int j = sh.n++;
            sh.x[j] = nx;
            sh.y[j] = ny;
            sh.o[j] = no;
        }
    }
    return sh;
}

__device__ static inline int d_shape_heuristic_score(const GShape& sh, uint64_t& s) {
    int minx = sh.x[0], maxx = sh.x[0], miny = sh.y[0], maxy = sh.y[0];
    int up = 0, down = 0;
    int perimeter = 0;
    int contacts = 0;

    for (int i = 0; i < sh.n; ++i) {
        minx = min(minx, sh.x[i]); maxx = max(maxx, sh.x[i]);
        miny = min(miny, sh.y[i]); maxy = max(maxy, sh.y[i]);
        if (sh.o[i] == 0) ++up; else ++down;
        int local_contacts = 0;
        for (int d = 0; d < 3; ++d) {
            int nx, ny, no;
            d_neighbour(sh.x[i], sh.y[i], sh.o[i], d, &nx, &ny, &no);
            if (d_exists(sh, nx, ny, no)) ++local_contacts;
            else ++perimeter;
        }
        contacts += local_contacts;
    }

    int width = maxx - minx + 1;
    int height = maxy - miny + 1;
    int bbox_area = width * height * 2;
    int holes_or_sprawl_penalty = max(0, bbox_area - sh.n * 3);
    int balance_penalty = abs(up - down);

    // This is only a prefilter. It tries to favour non-trivial, jagged, asymmetric-enough
    // connected polyiamonds that are neither tiny blobs nor ridiculous spaghetti.
    int score = 0;
    score += sh.n * 200;
    score += perimeter * 65;
    score += contacts * 15;
    score -= holes_or_sprawl_penalty * 18;
    score -= balance_penalty * 9;
    score -= abs(width - height) * 12;
    score += static_cast<int>(rng32(s) & 127U); // tie-breaking diversity
    return score;
}

__global__ void generate_shapes_kernel(GShape* out,
                                       int* scores,
                                       int batch,
                                       int min_cells,
                                       int max_cells,
                                       int gpu_iters,
                                       uint64_t global_seed,
                                       uint64_t epoch) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= batch) return;

    uint64_t s = global_seed ^ (0xD1B54A32D192ED03ULL * static_cast<uint64_t>(tid + 1)) ^
                 (0xABC98388FB8FAC03ULL * static_cast<uint64_t>(epoch + 1));

    int best_score = -2147483647;
    GShape best = d_make_random_shape(s, min_cells, max_cells);

    gpu_iters = max(1, gpu_iters);
    for (int it = 0; it < gpu_iters; ++it) {
        GShape sh = d_make_random_shape(s, min_cells, max_cells);
        int sc = d_shape_heuristic_score(sh, s);
        if (sc > best_score) {
            best_score = sc;
            best = sh;
        }
    }

    out[tid] = best;
    scores[tid] = best_score;
}

static std::vector<Cell> to_host_shape(const GShape& g) {
    std::vector<Cell> v;
    v.reserve(g.n);
    for (int i = 0; i < g.n; ++i) v.push_back({g.x[i], g.y[i], g.o[i]});
    return v;
}

static std::vector<Cell> normalize_cells(std::vector<Cell> v) {
    if (v.empty()) return v;
    int minx = v[0].x, miny = v[0].y;
    for (const auto& c : v) {
        minx = std::min(minx, c.x);
        miny = std::min(miny, c.y);
    }
    for (auto& c : v) {
        c.x -= minx;
        c.y -= miny;
    }
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
    return v;
}

static std::string key_of_cells(const std::vector<Cell>& v) {
    std::ostringstream oss;
    oss << v.size() << ":";
    for (const auto& c : v) oss << c.x << "," << c.y << "," << c.o << ";";
    return oss.str();
}

using V2 = std::pair<int,int>;

static std::array<V2,3> vertices_of_cell(const Cell& c) {
    if (c.o == 0) {
        return { V2{c.x, c.y}, V2{c.x + 1, c.y}, V2{c.x, c.y + 1} };
    }
    return { V2{c.x + 1, c.y + 1}, V2{c.x + 1, c.y}, V2{c.x, c.y + 1} };
}

static V2 rotate60_once(V2 p) {
    // Axial triangular-lattice coordinate rotation by +60 degrees.
    // e1 -> e2, e2 -> e2 - e1.
    int a = p.first;
    int b = p.second;
    return {-b, a + b};
}

static V2 transform_vertex(V2 p, int rot, bool reflect) {
    // Reflection across line x=y followed by rot*60 degrees.
    if (reflect) std::swap(p.first, p.second);
    for (int i = 0; i < rot; ++i) p = rotate60_once(p);
    return p;
}

static bool same_vertex_set(std::array<V2,3> a, std::array<V2,3> b) {
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    return a == b;
}

static Cell cell_from_vertices(std::array<V2,3> vs) {
    int minx = vs[0].first, miny = vs[0].second;
    for (auto p : vs) {
        minx = std::min(minx, p.first);
        miny = std::min(miny, p.second);
    }
    Cell u{minx, miny, 0};
    if (same_vertex_set(vs, vertices_of_cell(u))) return u;
    Cell d{minx, miny, 1};
    if (same_vertex_set(vs, vertices_of_cell(d))) return d;

    // Should not happen for valid triangular-lattice symmetries.
    std::cerr << "Internal transform error: triangle did not map to a cell.\n";
    std::exit(2);
}

static std::vector<Cell> transform_shape(const std::vector<Cell>& shape, int rot, bool reflect) {
    std::vector<Cell> out;
    out.reserve(shape.size());
    for (const auto& c : shape) {
        auto vs = vertices_of_cell(c);
        for (auto& p : vs) p = transform_vertex(p, rot, reflect);
        out.push_back(cell_from_vertices(vs));
    }
    return normalize_cells(out);
}

static std::vector<std::vector<Cell>> all_orientations(const std::vector<Cell>& shape, bool allow_reflections) {
    std::vector<std::vector<Cell>> result;
    std::unordered_set<std::string> seen;
    for (int ref = 0; ref <= (allow_reflections ? 1 : 0); ++ref) {
        for (int rot = 0; rot < 6; ++rot) {
            auto t = transform_shape(shape, rot, ref != 0);
            std::string k = key_of_cells(t);
            if (seen.insert(k).second) result.push_back(std::move(t));
        }
    }
    return result;
}

static std::string canonical_key(const std::vector<Cell>& shape, bool allow_reflections) {
    std::string best;
    bool first = true;
    for (const auto& ori : all_orientations(shape, allow_reflections)) {
        std::string k = key_of_cells(ori);
        if (first || k < best) {
            best = k;
            first = false;
        }
    }
    return best;
}

static bool connected_shape(const std::vector<Cell>& shape) {
    if (shape.empty()) return false;
    std::unordered_set<uint64_t> S;
    for (const auto& c : shape) S.insert(pack_cell(c.x, c.y, c.o));
    std::vector<Cell> stack{shape[0]};
    std::unordered_set<uint64_t> seen;
    seen.insert(pack_cell(shape[0].x, shape[0].y, shape[0].o));
    while (!stack.empty()) {
        Cell c = stack.back();
        stack.pop_back();
        for (int d = 0; d < 3; ++d) {
            Cell n = neighbour(c, d);
            uint64_t k = pack_cell(n.x, n.y, n.o);
            if (S.count(k) && !seen.count(k)) {
                seen.insert(k);
                stack.push_back(n);
            }
        }
    }
    return seen.size() == shape.size();
}

static int mod_pos(int a, int m) {
    int r = a % m;
    return r < 0 ? r + m : r;
}

struct Placement {
    std::vector<uint64_t> bits;
    std::vector<int> cells;
};

static inline int torus_index(int x, int y, int o, int W, int H) {
    x = mod_pos(x, W);
    y = mod_pos(y, H);
    return ((y * W + x) << 1) | (o & 1);
}

static inline bool bit_is_set(const std::vector<uint64_t>& b, int idx) {
    return (b[static_cast<size_t>(idx) >> 6] >> (idx & 63)) & 1ULL;
}

static inline void set_bit(std::vector<uint64_t>& b, int idx) {
    b[static_cast<size_t>(idx) >> 6] |= (1ULL << (idx & 63));
}

static inline bool no_overlap(const std::vector<uint64_t>& a, const std::vector<uint64_t>& p) {
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] & p[i]) return false;
    }
    return true;
}

static int first_uncovered(const std::vector<uint64_t>& occ, int nbits) {
    int words = static_cast<int>(occ.size());
    for (int w = 0; w < words; ++w) {
        uint64_t full = ~occ[w];
        if (w == words - 1 && (nbits & 63)) {
            uint64_t mask = (1ULL << (nbits & 63)) - 1ULL;
            full &= mask;
        }
        if (full) {
            return w * 64 + __builtin_ctzll(full);
        }
    }
    return -1;
}

struct ExactStats {
    bool found = false;
    bool limit_hit = false;
    uint64_t nodes = 0;
};

static bool exact_cover_dfs(const std::vector<std::vector<int>>& cover,
                            const std::vector<Placement>& placements,
                            std::vector<uint64_t>& occ,
                            int nbits,
                            uint64_t max_nodes,
                            ExactStats& stats) {
    if (++stats.nodes > max_nodes) {
        stats.limit_hit = true;
        return false;
    }

    int u = first_uncovered(occ, nbits);
    if (u < 0) {
        stats.found = true;
        return true;
    }

    const auto& choices = cover[u];
    for (int pi : choices) {
        const Placement& p = placements[pi];
        if (!no_overlap(occ, p.bits)) continue;
        for (size_t w = 0; w < occ.size(); ++w) occ[w] |= p.bits[w];
        if (exact_cover_dfs(cover, placements, occ, nbits, max_nodes, stats)) return true;
        for (size_t w = 0; w < occ.size(); ++w) occ[w] &= ~p.bits[w];
        if (stats.limit_hit) return false;
    }
    return false;
}

static ExactStats torus_tiles_exact(const std::vector<std::vector<Cell>>& orientations,
                                    int W,
                                    int H,
                                    uint64_t max_nodes) {
    ExactStats stats;
    if (orientations.empty()) return stats;
    int n = static_cast<int>(orientations[0].size());
    int board_cells = 2 * W * H;
    if (board_cells % n != 0) return stats;

    int nwords = (board_cells + 63) / 64;
    std::vector<Placement> placements;
    placements.reserve(orientations.size() * W * H);
    std::vector<std::vector<int>> cover(static_cast<size_t>(board_cells));

    for (const auto& ori : orientations) {
        for (int tx = 0; tx < W; ++tx) {
            for (int ty = 0; ty < H; ++ty) {
                Placement p;
                p.bits.assign(static_cast<size_t>(nwords), 0ULL);
                p.cells.reserve(ori.size());
                bool ok = true;
                for (const auto& c : ori) {
                    int idx = torus_index(c.x + tx, c.y + ty, c.o, W, H);
                    if (bit_is_set(p.bits, idx)) { ok = false; break; } // self-overlap after wrap
                    set_bit(p.bits, idx);
                    p.cells.push_back(idx);
                }
                if (!ok) continue;
                int pi = static_cast<int>(placements.size());
                for (int idx : p.cells) cover[idx].push_back(pi);
                placements.push_back(std::move(p));
            }
        }
    }

    std::vector<uint64_t> occ(static_cast<size_t>(nwords), 0ULL);
    exact_cover_dfs(cover, placements, occ, board_cells, max_nodes, stats);
    return stats;
}

struct PeriodicResult {
    bool found = false;
    bool any_limit = false;
    int W = 0;
    int H = 0;
    uint64_t total_nodes = 0;
};

static PeriodicResult search_periodic_tori(const std::vector<std::vector<Cell>>& orientations,
                                           int max_period,
                                           uint64_t max_nodes_per_torus) {
    PeriodicResult r;
    for (int W = 2; W <= max_period; ++W) {
        for (int H = 2; H <= max_period; ++H) {
            int area = 2 * W * H;
            int n = static_cast<int>(orientations[0].size());
            if (area % n != 0) continue;
            ExactStats s = torus_tiles_exact(orientations, W, H, max_nodes_per_torus);
            r.total_nodes += s.nodes;
            if (s.limit_hit) r.any_limit = true;
            if (s.found) {
                r.found = true;
                r.W = W;
                r.H = H;
                return r;
            }
        }
    }
    return r;
}

struct PatchResult {
    int max_tiles = 1;
    int max_cells = 0;
    std::vector<Cell> best_occupied;
    std::vector<std::vector<Cell>> best_tiles;
};

static bool placement_fits(const std::vector<Cell>& ori,
                           int tx,
                           int ty,
                           int radius,
                           const std::unordered_set<uint64_t>& occupied) {
    for (const auto& c : ori) {
        int x = c.x + tx;
        int y = c.y + ty;
        if (std::abs(x) > radius || std::abs(y) > radius || std::abs(x + y) > radius * 2) return false;
        if (occupied.count(pack_cell(x, y, c.o))) return false;
    }
    return true;
}

static std::vector<Cell> translate_placement(const std::vector<Cell>& ori,
                                           int tx,
                                           int ty) {
    std::vector<Cell> placed;
    placed.reserve(ori.size());
    for (const auto& c : ori) placed.push_back(Cell{c.x + tx, c.y + ty, c.o});
    return placed;
}

static void add_placement(const std::vector<Cell>& ori,
                          int tx,
                          int ty,
                          std::unordered_set<uint64_t>& occupied,
                          std::vector<Cell>& occ_list) {
    for (const auto& c : ori) {
        Cell z{c.x + tx, c.y + ty, c.o};
        uint64_t k = pack_cell(z.x, z.y, z.o);
        if (occupied.insert(k).second) occ_list.push_back(z);
    }
}

static PatchResult greedy_patch_growth(const std::vector<std::vector<Cell>>& orientations,
                                        int radius,
                                        int trials,
                                        uint64_t seed) {
    PatchResult best;
    if (orientations.empty()) return best;
    std::mt19937_64 rng(seed);
    int n = static_cast<int>(orientations[0].size());
    best.max_cells = n;

    for (int t = 0; t < trials; ++t) {
        std::unordered_set<uint64_t> occupied;
        occupied.reserve(8192);
        std::vector<Cell> occ_list;
        occ_list.reserve(8192);
        std::vector<std::vector<Cell>> placed_tile_cells;
        placed_tile_cells.reserve(2048);

        int start_ori = static_cast<int>(rng() % orientations.size());
        add_placement(orientations[start_ori], 0, 0, occupied, occ_list);
        placed_tile_cells.push_back(translate_placement(orientations[start_ori], 0, 0));
        int placed_tiles = 1;

        int failed_streak = 0;
        int max_steps = radius * radius * 20;
        for (int step = 0; step < max_steps && failed_streak < 4000; ++step) {
            if (occ_list.empty()) break;
            const Cell base = occ_list[static_cast<size_t>(rng() % occ_list.size())];
            Cell target = neighbour(base, static_cast<int>(rng() % 3));
            if (occupied.count(pack_cell(target.x, target.y, target.o))) {
                ++failed_streak;
                continue;
            }

            bool placed = false;
            int ori_start = static_cast<int>(rng() % orientations.size());
            for (size_t oi0 = 0; oi0 < orientations.size() && !placed; ++oi0) {
                const auto& ori = orientations[(ori_start + oi0) % orientations.size()];
                int cell_start = static_cast<int>(rng() % ori.size());
                for (size_t ci0 = 0; ci0 < ori.size(); ++ci0) {
                    const Cell& anchor = ori[(cell_start + ci0) % ori.size()];
                    if (anchor.o != target.o) continue;
                    int tx = target.x - anchor.x;
                    int ty = target.y - anchor.y;
                    if (placement_fits(ori, tx, ty, radius, occupied)) {
                        add_placement(ori, tx, ty, occupied, occ_list);
                        placed_tile_cells.push_back(translate_placement(ori, tx, ty));
                        ++placed_tiles;
                        failed_streak = 0;
                        placed = true;
                        break;
                    }
                }
            }
            if (!placed) ++failed_streak;
        }

        if (placed_tiles > best.max_tiles) {
            best.max_tiles = placed_tiles;
            best.max_cells = static_cast<int>(occ_list.size());
            best.best_occupied = occ_list;
            best.best_tiles = placed_tile_cells;
        }
    }
    return best;
}

static std::pair<double,double> lattice_to_xy(int a, int b) {
    static const double SQ3 = std::sqrt(3.0);
    return {static_cast<double>(a) + 0.5 * static_cast<double>(b),
            (SQ3 * 0.5) * static_cast<double>(b)};
}

static void write_svg_cells(const std::vector<Cell>& cells, const fs::path& file, const std::string& title) {
    if (cells.empty()) return;
    double minx = 1e100, miny = 1e100, maxx = -1e100, maxy = -1e100;
    std::vector<std::array<std::pair<double,double>,3>> polys;
    polys.reserve(cells.size());
    for (const auto& c : cells) {
        auto vs = vertices_of_cell(c);
        std::array<std::pair<double,double>,3> poly;
        for (int i = 0; i < 3; ++i) {
            poly[i] = lattice_to_xy(vs[i].first, vs[i].second);
            minx = std::min(minx, poly[i].first);
            miny = std::min(miny, poly[i].second);
            maxx = std::max(maxx, poly[i].first);
            maxy = std::max(maxy, poly[i].second);
        }
        polys.push_back(poly);
    }

    double scale = 24.0;
    double pad = 20.0;
    double W = (maxx - minx) * scale + 2 * pad;
    double H = (maxy - miny) * scale + 2 * pad + 20;

    std::ofstream out(file);
    out << "<svg xmlns='http://www.w3.org/2000/svg' width='" << W << "' height='" << H
        << "' viewBox='0 0 " << W << " " << H << "'>\n";
    out << "<rect width='100%' height='100%' fill='white'/>\n";
    out << "<text x='10' y='16' font-size='12' font-family='monospace'>" << title << "</text>\n";
    for (const auto& p : polys) {
        out << "<polygon points='";
        for (int i = 0; i < 3; ++i) {
            double x = (p[i].first - minx) * scale + pad;
            double y = (maxy - p[i].second) * scale + pad + 20;
            out << x << "," << y << " ";
        }
        out << "' fill='none' stroke='black' stroke-width='1.2'/>\n";
    }
    out << "</svg>\n";
}


static std::string palette_color(size_t idx) {
    static const std::array<std::string, 20> palette = {
        "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
        "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC",
        "#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD",
        "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF"
    };
    return palette[idx % palette.size()];
}

static void write_svg_patch_tiles(const std::vector<std::vector<Cell>>& tiles,
                                  const fs::path& file,
                                  const std::string& title) {
    if (tiles.empty()) return;

    double minx = 1e100, miny = 1e100, maxx = -1e100, maxy = -1e100;
    std::vector<std::vector<std::array<std::pair<double,double>,3>>> tile_polys;
    tile_polys.reserve(tiles.size());

    for (const auto& tile : tiles) {
        std::vector<std::array<std::pair<double,double>,3>> polys;
        polys.reserve(tile.size());

        for (const auto& c : tile) {
            auto vs = vertices_of_cell(c);
            std::array<std::pair<double,double>,3> poly;

            for (int i = 0; i < 3; ++i) {
                poly[i] = lattice_to_xy(vs[i].first, vs[i].second);
                minx = std::min(minx, poly[i].first);
                miny = std::min(miny, poly[i].second);
                maxx = std::max(maxx, poly[i].first);
                maxy = std::max(maxy, poly[i].second);
            }

            polys.push_back(poly);
        }

        tile_polys.push_back(std::move(polys));
    }

    double scale = 24.0;
    double pad = 20.0;
    double W = (maxx - minx) * scale + 2 * pad;
    double H = (maxy - miny) * scale + 2 * pad + 20;

    std::ofstream out(file);
    out << "<svg xmlns='http://www.w3.org/2000/svg' width='" << W << "' height='" << H
        << "' viewBox='0 0 " << W << " " << H << "'>\n";
    out << "<rect width='100%' height='100%' fill='white'/>\n";
    out << "<text x='10' y='16' font-size='12' font-family='monospace'>" << title << "</text>\n";

    for (size_t ti = 0; ti < tile_polys.size(); ++ti) {
        const std::string fill = palette_color(ti);

        for (const auto& p : tile_polys[ti]) {
            out << "<polygon points='";

            for (int i = 0; i < 3; ++i) {
                double x = (p[i].first - minx) * scale + pad;
                double y = (maxy - p[i].second) * scale + pad + 20;
                out << x << "," << y << " ";
            }

            out << "' fill='" << fill
                << "' fill-opacity='0.72' stroke='black' stroke-width='1.0'/>\n";
        }
    }

    out << "</svg>\n";
}


struct Config {
    int min_cells = 8;
    int max_cells = 14;
    int batch = 65536;
    int gpu_iters = 1;
    int cpu_checks = 0; // 0 = check all returned GPU candidates
    int period = 10;
    uint64_t nodes = 200000;
    int patch_radius = 36;
    int patch_trials = 64;
    int strong_tiles = 120;
    int candidate_tiles = 40;
    bool allow_reflections = true;
    uint64_t seed = 0x123456789abcdef0ULL;
    std::string out_dir = "records_einstein_v1";
};

static bool parse_int(const char* s, int& v) {
    char* end = nullptr;
    long x = std::strtol(s, &end, 10);
    if (!end || *end != '\0') return false;
    v = static_cast<int>(x);
    return true;
}

static bool parse_u64(const char* s, uint64_t& v) {
    char* end = nullptr;
    unsigned long long x = std::strtoull(s, &end, 10);
    if (!end || *end != '\0') return false;
    v = static_cast<uint64_t>(x);
    return true;
}

static Config parse_args(int argc, char** argv) {
    Config c;
    c.seed = static_cast<uint64_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    for (int i = 1; i < argc; ++i) {
        auto need = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::cerr << "Missing value after " << name << "\n";
                std::exit(1);
            }
            return argv[++i];
        };
        std::string a = argv[i];
        if (a == "--min") parse_int(need("--min"), c.min_cells);
        else if (a == "--max") parse_int(need("--max"), c.max_cells);
        else if (a == "--batch") parse_int(need("--batch"), c.batch);
        else if (a == "--gpu-iters") parse_int(need("--gpu-iters"), c.gpu_iters);
        else if (a == "--cpu-checks") parse_int(need("--cpu-checks"), c.cpu_checks);
        else if (a == "--period") parse_int(need("--period"), c.period);
        else if (a == "--nodes") parse_u64(need("--nodes"), c.nodes);
        else if (a == "--patch-radius") parse_int(need("--patch-radius"), c.patch_radius);
        else if (a == "--patch-trials") parse_int(need("--patch-trials"), c.patch_trials);
        else if (a == "--strong-tiles") parse_int(need("--strong-tiles"), c.strong_tiles);
        else if (a == "--candidate-tiles") parse_int(need("--candidate-tiles"), c.candidate_tiles);
        else if (a == "--allow-reflections") {
            int x = 1; parse_int(need("--allow-reflections"), x); c.allow_reflections = (x != 0);
        }
        else if (a == "--seed") parse_u64(need("--seed"), c.seed);
        else if (a == "--out") c.out_dir = need("--out");
        else if (a == "--help" || a == "-h") {
            std::cout << "CUDA Aperiodic Monotile Candidate Hunter v2\n"
                      << "Options:\n"
                      << "  --min N                  minimum triangular cells [8]\n"
                      << "  --max N                  maximum triangular cells [14], max 32\n"
                      << "  --batch N                GPU winners returned per epoch [65536]\n"
                      << "  --gpu-iters N            random shapes searched per GPU thread [1]\n"
                      << "  --cpu-checks N           CPU validations per epoch, 0=all [0]\n"
                      << "  --period N               max finite torus W/H to test [10]\n"
                      << "  --nodes N                DFS node limit per torus [200000]\n"
                      << "  --patch-radius N         greedy patch radius [36]\n"
                      << "  --patch-trials N         greedy patch trials [64]\n"
                      << "  --candidate-tiles N      weak candidate patch threshold [40]\n"
                      << "  --strong-tiles N         strong candidate patch threshold [120]\n"
                      << "  --allow-reflections 0/1  include reflected copies in tests [1]\n"
                      << "  --seed N                 random seed [time]\n"
                      << "  --out DIR                output directory [records_einstein_v1]\n";
            std::exit(0);
        } else {
            std::cerr << "Unknown option: " << a << "\n";
            std::exit(1);
        }
    }
    if (c.min_cells < 1) c.min_cells = 1;
    if (c.max_cells > MAX_CELLS) c.max_cells = MAX_CELLS;
    if (c.max_cells < c.min_cells) std::swap(c.max_cells, c.min_cells);
    if (c.batch < 1) c.batch = 1;
    if (c.gpu_iters < 1) c.gpu_iters = 1;
    if (c.cpu_checks < 0) c.cpu_checks = 0;
    return c;
}

struct ValidationReport {
    std::string status;
    PeriodicResult periodic;
    PatchResult patch;
    int orientation_count = 0;
};

static ValidationReport validate_candidate(const std::vector<Cell>& shape,
                                           bool allow_reflections,
                                           int max_period,
                                           uint64_t nodes,
                                           int patch_radius,
                                           int patch_trials,
                                           int candidate_tiles,
                                           int strong_tiles,
                                           uint64_t seed) {
    ValidationReport rep;
    auto orientations = all_orientations(shape, allow_reflections);
    rep.orientation_count = static_cast<int>(orientations.size());

    rep.periodic = search_periodic_tori(orientations, max_period, nodes);
    if (rep.periodic.found) {
        rep.status = "periodic";
        return rep;
    }

    rep.patch = greedy_patch_growth(orientations, patch_radius, patch_trials, seed);
    if (rep.patch.max_tiles >= strong_tiles) rep.status = "strong_candidate";
    else if (rep.patch.max_tiles >= candidate_tiles) rep.status = "candidate";
    else rep.status = "non_tiler_like";
    return rep;
}

static std::string now_string() {
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S");
    return oss.str();
}

static void save_candidate(const fs::path& dir,
                           int id,
                           const std::vector<Cell>& shape,
                           const std::string& key,
                           const ValidationReport& rep,
                           uint64_t seed) {
    fs::create_directories(dir);
    std::ostringstream base;
    base << "candidate_" << std::setw(7) << std::setfill('0') << id;
    fs::path txt = dir / (base.str() + ".txt");
    fs::path tile_svg = dir / (base.str() + "_tile.svg");
    fs::path patch_svg = dir / (base.str() + "_patch.svg");

    std::ofstream out(txt);
    out << "status: " << rep.status << "\n";
    out << "note: finite tests only; this is not a proof of being an einstein tile.\n";
    out << "seed: " << seed << "\n";
    out << "cells: " << shape.size() << "\n";
    out << "orientations_tested: " << rep.orientation_count << "\n";
    out << "canonical_key: " << key << "\n";
    out << "periodic_found: " << (rep.periodic.found ? 1 : 0) << "\n";
    out << "periodic_W: " << rep.periodic.W << "\n";
    out << "periodic_H: " << rep.periodic.H << "\n";
    out << "periodic_limit_hit: " << (rep.periodic.any_limit ? 1 : 0) << "\n";
    out << "torus_nodes: " << rep.periodic.total_nodes << "\n";
    out << "patch_max_tiles: " << rep.patch.max_tiles << "\n";
    out << "patch_max_cells: " << rep.patch.max_cells << "\n";
    out << "\nnormalized_cells x y orientation:\n";
    for (const auto& c : shape) out << c.x << " " << c.y << " " << c.o << "\n";

    write_svg_cells(shape, tile_svg, rep.status + " tile cells=" + std::to_string(shape.size()));
    if (!rep.patch.best_tiles.empty()) {
        write_svg_patch_tiles(rep.patch.best_tiles, patch_svg,
                              rep.status + " greedy patch tiles=" + std::to_string(rep.patch.max_tiles));
    } else if (!rep.patch.best_occupied.empty()) {
        write_svg_cells(rep.patch.best_occupied, patch_svg,
                        rep.status + " greedy patch tiles=" + std::to_string(rep.patch.max_tiles));
    }
}

// ------------------------------------------------------------
// Second-pass validator
// ------------------------------------------------------------

struct CandidateFile {
    fs::path path;
    std::string filename;
    std::string old_status = "unknown";
    std::string key = "";
    int id = -1;
    int old_patch_tiles = 0;
    std::vector<Cell> cells;
};

struct RegionExactResult {
    bool found = false;
    bool limit_hit = false;
    int radius = 0;
    int cells = 0;
    uint64_t nodes = 0;
};

struct DeepReport {
    std::string classification;
    PeriodicResult periodic;
    PeriodicResult periodic_no_reflect;
    PatchResult random_patch;
    PatchResult boundary_patch;
    PatchResult forced_patch;
    RegionExactResult exact_region;
    int orientation_count = 0;
    int orientation_count_no_reflect = 0;
    int best_patch_tiles = 0;
    int best_patch_cells = 0;
    std::string best_patch_method = "none";
    double score = 0.0;
};

struct ValidatorConfig {
    fs::path input_dir = "records_einstein_v3_very_hard";
    fs::path out_dir = "validation_second_pass";
    int period = 18;
    uint64_t nodes = 20000000ULL;
    int patch_radius = 100;
    int random_trials = 800;
    int boundary_trials = 500;
    int forced_trials = 500;
    int region_min = 3;
    int region_max = 8;
    uint64_t region_nodes = 5000000ULL;
    int weak_tiles = 180;
    int strong_tiles = 400;
    int very_strong_tiles = 700;
    int allow_reflections = 1;
    int test_no_reflection = 1;
    int max_candidates = 0; // 0 = all
    int skip_periodic_old = 1;
    int only_strong_old = 0;
    uint64_t seed = 0;
};

static bool starts_with(const std::string& s, const std::string& p) {
    return s.rfind(p, 0) == 0;
}

static bool ends_with(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

static std::string trim_copy(std::string s) {
    auto not_space = [](unsigned char ch){ return !std::isspace(ch); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
    s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
    return s;
}

static int parse_candidate_id(const std::string& name) {
    // candidate_0001234.txt -> 1234
    const std::string pre = "candidate_";
    const std::string suf = ".txt";
    if (!starts_with(name, pre) || !ends_with(name, suf)) return -1;
    std::string mid = name.substr(pre.size(), name.size() - pre.size() - suf.size());
    try { return std::stoi(mid); } catch (...) { return -1; }
}

static bool read_candidate_file(const fs::path& p, CandidateFile& cand) {
    std::ifstream in(p);
    if (!in) return false;
    cand.path = p;
    cand.filename = p.filename().string();
    cand.id = parse_candidate_id(cand.filename);
    cand.cells.clear();

    bool in_cells = false;
    std::string line;
    while (std::getline(in, line)) {
        std::string t = trim_copy(line);
        if (t.empty()) continue;

        if (!in_cells) {
            if (starts_with(t, "status:")) cand.old_status = trim_copy(t.substr(7));
            else if (starts_with(t, "canonical_key:")) cand.key = trim_copy(t.substr(14));
            else if (starts_with(t, "patch_max_tiles:")) {
                try { cand.old_patch_tiles = std::stoi(trim_copy(t.substr(16))); } catch (...) {}
            }
            else if (t.find("normalized_cells") != std::string::npos) in_cells = true;
            continue;
        }

        std::istringstream iss(t);
        int x, y, o;
        if (iss >> x >> y >> o) cand.cells.push_back(Cell{x, y, o});
    }

    if (cand.cells.empty()) return false;
    cand.cells = normalize_cells(cand.cells);
    return true;
}

static std::vector<CandidateFile> load_candidates(const fs::path& dir,
                                                  int skip_periodic_old,
                                                  int only_strong_old) {
    std::vector<CandidateFile> out;
    if (!fs::exists(dir)) {
        std::cerr << "Input directory does not exist: " << dir << "\n";
        return out;
    }

    for (const auto& e : fs::directory_iterator(dir)) {
        if (!e.is_regular_file()) continue;
        std::string name = e.path().filename().string();
        if (!starts_with(name, "candidate_") || !ends_with(name, ".txt")) continue;

        CandidateFile c;
        if (!read_candidate_file(e.path(), c)) continue;
        if (skip_periodic_old && c.old_status == "periodic") continue;
        if (only_strong_old && c.old_status != "strong_candidate") continue;
        out.push_back(std::move(c));
    }

    std::sort(out.begin(), out.end(), [](const CandidateFile& a, const CandidateFile& b) {
        if (a.old_patch_tiles != b.old_patch_tiles) return a.old_patch_tiles > b.old_patch_tiles;
        return a.id < b.id;
    });
    return out;
}

static bool within_radius_cell(const Cell& c, int radius) {
    return std::abs(c.x) <= radius && std::abs(c.y) <= radius && std::abs(c.x + c.y) <= radius * 2;
}

static std::vector<Cell> collect_boundary_targets(const std::vector<Cell>& occ_list,
                                                  const std::unordered_set<uint64_t>& occupied,
                                                  int radius) {
    std::vector<Cell> targets;
    std::unordered_set<uint64_t> seen;
    seen.reserve(occ_list.size() * 2 + 64);
    for (const auto& c : occ_list) {
        for (int d = 0; d < 3; ++d) {
            Cell n = neighbour(c, d);
            if (!within_radius_cell(n, radius)) continue;
            uint64_t k = pack_cell(n.x, n.y, n.o);
            if (occupied.count(k)) continue;
            if (seen.insert(k).second) targets.push_back(n);
        }
    }
    return targets;
}

static int contact_score(const std::vector<Cell>& placed,
                         const std::unordered_set<uint64_t>& occupied) {
    int score = 0;
    for (const auto& c : placed) {
        for (int d = 0; d < 3; ++d) {
            Cell n = neighbour(c, d);
            if (occupied.count(pack_cell(n.x, n.y, n.o))) ++score;
        }
    }
    return score;
}

static bool find_best_boundary_placement(const std::vector<std::vector<Cell>>& orientations,
                                         const Cell& target,
                                         int radius,
                                         const std::unordered_set<uint64_t>& occupied,
                                         std::mt19937_64& rng,
                                         std::vector<Cell>& best_tile) {
    int best_score = -1;
    best_tile.clear();
    for (const auto& ori : orientations) {
        for (const auto& anchor : ori) {
            if (anchor.o != target.o) continue;
            int tx = target.x - anchor.x;
            int ty = target.y - anchor.y;
            if (!placement_fits(ori, tx, ty, radius, occupied)) continue;
            std::vector<Cell> placed = translate_placement(ori, tx, ty);
            int sc = contact_score(placed, occupied);
            // tiny random tie-break so repeated trials explore alternatives
            sc = sc * 1000 + static_cast<int>(rng() % 997);
            if (sc > best_score) {
                best_score = sc;
                best_tile = std::move(placed);
            }
        }
    }
    return best_score >= 0;
}

static void add_translated_tile(const std::vector<Cell>& placed,
                                std::unordered_set<uint64_t>& occupied,
                                std::vector<Cell>& occ_list) {
    for (const auto& z : placed) {
        uint64_t k = pack_cell(z.x, z.y, z.o);
        if (occupied.insert(k).second) occ_list.push_back(z);
    }
}

static PatchResult boundary_first_patch_growth(const std::vector<std::vector<Cell>>& orientations,
                                               int radius,
                                               int trials,
                                               uint64_t seed) {
    PatchResult best;
    if (orientations.empty()) return best;
    std::mt19937_64 rng(seed ^ 0xB0A11DAB1EFULL);
    best.max_cells = static_cast<int>(orientations[0].size());

    for (int t = 0; t < trials; ++t) {
        std::unordered_set<uint64_t> occupied;
        occupied.reserve(65536);
        std::vector<Cell> occ_list;
        occ_list.reserve(65536);
        std::vector<std::vector<Cell>> placed_tiles;
        placed_tiles.reserve(4096);

        int start_ori = static_cast<int>(rng() % orientations.size());
        std::vector<Cell> first = translate_placement(orientations[start_ori], 0, 0);
        add_translated_tile(first, occupied, occ_list);
        placed_tiles.push_back(first);

        int max_steps = radius * radius * 16;
        int stuck = 0;
        for (int step = 0; step < max_steps && stuck < 250; ++step) {
            std::vector<Cell> targets = collect_boundary_targets(occ_list, occupied, radius);
            if (targets.empty()) break;
            std::shuffle(targets.begin(), targets.end(), rng);

            bool placed = false;
            std::vector<Cell> best_tile;
            int scan_limit = std::min<int>(static_cast<int>(targets.size()), 128);
            for (int i = 0; i < scan_limit; ++i) {
                if (find_best_boundary_placement(orientations, targets[i], radius, occupied, rng, best_tile)) {
                    add_translated_tile(best_tile, occupied, occ_list);
                    placed_tiles.push_back(best_tile);
                    placed = true;
                    stuck = 0;
                    break;
                }
            }
            if (!placed) ++stuck;
        }

        int placed_count = static_cast<int>(placed_tiles.size());
        if (placed_count > best.max_tiles) {
            best.max_tiles = placed_count;
            best.max_cells = static_cast<int>(occ_list.size());
            best.best_occupied = occ_list;
            best.best_tiles = placed_tiles;
        }
    }
    return best;
}

static int enumerate_valid_placements_for_target(const std::vector<std::vector<Cell>>& orientations,
                                                 const Cell& target,
                                                 int radius,
                                                 const std::unordered_set<uint64_t>& occupied,
                                                 std::vector<Cell>& only_tile,
                                                 int max_count = 2) {
    int count = 0;
    only_tile.clear();
    for (const auto& ori : orientations) {
        for (const auto& anchor : ori) {
            if (anchor.o != target.o) continue;
            int tx = target.x - anchor.x;
            int ty = target.y - anchor.y;
            if (!placement_fits(ori, tx, ty, radius, occupied)) continue;
            ++count;
            if (count == 1) only_tile = translate_placement(ori, tx, ty);
            if (count >= max_count) return count;
        }
    }
    return count;
}

static PatchResult forced_patch_growth(const std::vector<std::vector<Cell>>& orientations,
                                       int radius,
                                       int trials,
                                       uint64_t seed) {
    PatchResult best;
    if (orientations.empty()) return best;
    std::mt19937_64 rng(seed ^ 0xF04CED1234ULL);
    best.max_cells = static_cast<int>(orientations[0].size());

    for (int t = 0; t < trials; ++t) {
        std::unordered_set<uint64_t> occupied;
        occupied.reserve(65536);
        std::vector<Cell> occ_list;
        occ_list.reserve(65536);
        std::vector<std::vector<Cell>> placed_tiles;
        placed_tiles.reserve(4096);

        int start_ori = static_cast<int>(rng() % orientations.size());
        std::vector<Cell> first = translate_placement(orientations[start_ori], 0, 0);
        add_translated_tile(first, occupied, occ_list);
        placed_tiles.push_back(first);

        int max_steps = radius * radius * 20;
        int stuck = 0;
        for (int step = 0; step < max_steps && stuck < 400; ++step) {
            std::vector<Cell> targets = collect_boundary_targets(occ_list, occupied, radius);
            if (targets.empty()) break;
            std::shuffle(targets.begin(), targets.end(), rng);

            bool placed = false;
            std::vector<Cell> tile;
            // First pass: look for genuinely forced cells: exactly one placement can cover target.
            int scan_limit = std::min<int>(static_cast<int>(targets.size()), 160);
            for (int i = 0; i < scan_limit; ++i) {
                int cnt = enumerate_valid_placements_for_target(orientations, targets[i], radius, occupied, tile, 2);
                if (cnt == 1) {
                    add_translated_tile(tile, occupied, occ_list);
                    placed_tiles.push_back(tile);
                    placed = true;
                    stuck = 0;
                    break;
                }
            }

            // Fallback: if nothing is forced, use boundary-first. This keeps the patch alive.
            if (!placed) {
                std::vector<Cell> best_tile;
                for (int i = 0; i < scan_limit; ++i) {
                    if (find_best_boundary_placement(orientations, targets[i], radius, occupied, rng, best_tile)) {
                        add_translated_tile(best_tile, occupied, occ_list);
                        placed_tiles.push_back(best_tile);
                        placed = true;
                        stuck = 0;
                        break;
                    }
                }
            }

            if (!placed) ++stuck;
        }

        int placed_count = static_cast<int>(placed_tiles.size());
        if (placed_count > best.max_tiles) {
            best.max_tiles = placed_count;
            best.max_cells = static_cast<int>(occ_list.size());
            best.best_occupied = occ_list;
            best.best_tiles = placed_tiles;
        }
    }
    return best;
}

static std::vector<Cell> hex_region_cells(int radius) {
    std::vector<Cell> cells;
    for (int x = -radius; x <= radius; ++x) {
        for (int y = -radius; y <= radius; ++y) {
            if (std::abs(x + y) > radius) continue;
            cells.push_back(Cell{x, y, 0});
            cells.push_back(Cell{x, y, 1});
        }
    }
    return normalize_cells(cells);
}

static RegionExactResult exact_fill_hex_region(const std::vector<std::vector<Cell>>& orientations,
                                               int radius,
                                               uint64_t max_nodes) {
    RegionExactResult rr;
    rr.radius = radius;
    if (orientations.empty()) return rr;
    std::vector<Cell> board = hex_region_cells(radius);
    rr.cells = static_cast<int>(board.size());
    int n = static_cast<int>(orientations[0].size());
    if (rr.cells == 0 || rr.cells % n != 0) return rr;

    std::unordered_map<uint64_t, int> cell_to_idx;
    cell_to_idx.reserve(board.size() * 2);
    for (int i = 0; i < static_cast<int>(board.size()); ++i) {
        cell_to_idx[pack_cell(board[i].x, board[i].y, board[i].o)] = i;
    }

    int nwords = (rr.cells + 63) / 64;
    std::vector<Placement> placements;
    std::vector<std::vector<int>> cover(static_cast<size_t>(rr.cells));

    // Translations are searched over the board coordinate range, padded by tile size.
    int pad = n + 2;
    for (const auto& ori : orientations) {
        for (int tx = -radius - pad; tx <= radius + pad; ++tx) {
            for (int ty = -radius - pad; ty <= radius + pad; ++ty) {
                Placement p;
                p.bits.assign(static_cast<size_t>(nwords), 0ULL);
                p.cells.reserve(ori.size());
                bool ok = true;
                for (const auto& c : ori) {
                    auto it = cell_to_idx.find(pack_cell(c.x + tx, c.y + ty, c.o));
                    if (it == cell_to_idx.end()) { ok = false; break; }
                    int idx = it->second;
                    if (bit_is_set(p.bits, idx)) { ok = false; break; }
                    set_bit(p.bits, idx);
                    p.cells.push_back(idx);
                }
                if (!ok) continue;
                int pi = static_cast<int>(placements.size());
                for (int idx : p.cells) cover[idx].push_back(pi);
                placements.push_back(std::move(p));
            }
        }
    }

    if (placements.empty()) return rr;
    std::vector<uint64_t> occ(static_cast<size_t>(nwords), 0ULL);
    ExactStats stats;
    exact_cover_dfs(cover, placements, occ, rr.cells, max_nodes, stats);
    rr.nodes = stats.nodes;
    rr.limit_hit = stats.limit_hit;
    rr.found = stats.found;
    return rr;
}

static RegionExactResult best_exact_region(const std::vector<std::vector<Cell>>& orientations,
                                           int rmin,
                                           int rmax,
                                           uint64_t nodes_per_region) {
    RegionExactResult best;
    for (int r = rmin; r <= rmax; ++r) {
        RegionExactResult cur = exact_fill_hex_region(orientations, r, nodes_per_region);
        if (cur.found && cur.radius > best.radius) best = cur;
        else if (!best.found && cur.limit_hit && cur.radius > best.radius) best = cur;
    }
    return best;
}

static PatchResult choose_best_patch(const DeepReport& r) {
    PatchResult best = r.random_patch;
    if (r.boundary_patch.max_tiles > best.max_tiles) best = r.boundary_patch;
    if (r.forced_patch.max_tiles > best.max_tiles) best = r.forced_patch;
    return best;
}

static DeepReport validate_deep(const CandidateFile& cand, const ValidatorConfig& cfg, uint64_t seed) {
    DeepReport r;
    auto orientations = all_orientations(cand.cells, cfg.allow_reflections != 0);
    r.orientation_count = static_cast<int>(orientations.size());

    r.periodic = search_periodic_tori(orientations, cfg.period, cfg.nodes);

    if (cfg.test_no_reflection) {
        auto no_ref = all_orientations(cand.cells, false);
        r.orientation_count_no_reflect = static_cast<int>(no_ref.size());
        r.periodic_no_reflect = search_periodic_tori(no_ref, cfg.period, cfg.nodes);
    }

    // Still run patching even if periodic is found. This gives useful diagnostics and images.
    r.random_patch = greedy_patch_growth(orientations, cfg.patch_radius, cfg.random_trials, seed ^ 0x1111ULL);
    r.boundary_patch = boundary_first_patch_growth(orientations, cfg.patch_radius, cfg.boundary_trials, seed ^ 0x2222ULL);
    r.forced_patch = forced_patch_growth(orientations, cfg.patch_radius, cfg.forced_trials, seed ^ 0x3333ULL);

    r.best_patch_tiles = r.random_patch.max_tiles;
    r.best_patch_cells = r.random_patch.max_cells;
    r.best_patch_method = "random";
    if (r.boundary_patch.max_tiles > r.best_patch_tiles) {
        r.best_patch_tiles = r.boundary_patch.max_tiles;
        r.best_patch_cells = r.boundary_patch.max_cells;
        r.best_patch_method = "boundary";
    }
    if (r.forced_patch.max_tiles > r.best_patch_tiles) {
        r.best_patch_tiles = r.forced_patch.max_tiles;
        r.best_patch_cells = r.forced_patch.max_cells;
        r.best_patch_method = "forced";
    }

    r.exact_region = best_exact_region(orientations, cfg.region_min, cfg.region_max, cfg.region_nodes);

    r.score = static_cast<double>(r.best_patch_tiles)
            + 40.0 * static_cast<double>(r.exact_region.radius)
            + (r.exact_region.found ? 120.0 : 0.0)
            - (r.periodic.found ? 100000.0 : 0.0)
            - (r.periodic.any_limit ? 25.0 : 0.0);

    if (r.periodic.found) r.classification = "periodic_reject";
    else if (r.best_patch_tiles < cfg.weak_tiles) r.classification = "weak_patch_reject";
    else if (r.best_patch_tiles >= cfg.very_strong_tiles && r.exact_region.found) r.classification = "very_strong_candidate";
    else if (r.best_patch_tiles >= cfg.strong_tiles) r.classification = "strong_patch_candidate";
    else r.classification = "unstable_candidate";

    return r;
}

static std::string csv_escape(const std::string& s) {
    std::string out = "\"";
    for (char c : s) {
        if (c == '"') out += "\"\"";
        else out += c;
    }
    out += "\"";
    return out;
}

static void save_deep_artifacts(const fs::path& out_dir,
                                const CandidateFile& cand,
                                const DeepReport& rep) {
    fs::path bucket = out_dir / rep.classification;
    fs::create_directories(bucket);
    std::ostringstream base;
    base << "candidate_" << std::setw(7) << std::setfill('0') << cand.id;

    fs::path txt = bucket / (base.str() + "_validation.txt");
    fs::path tile_svg = bucket / (base.str() + "_tile.svg");
    fs::path patch_svg = bucket / (base.str() + "_best_patch.svg");

    std::ofstream out(txt);
    out << "classification: " << rep.classification << "\n";
    out << "note: second-pass finite validation only; not a proof of being an einstein tile.\n";
    out << "source_file: " << cand.filename << "\n";
    out << "old_status: " << cand.old_status << "\n";
    out << "old_patch_tiles: " << cand.old_patch_tiles << "\n";
    out << "cells: " << cand.cells.size() << "\n";
    out << "canonical_key: " << cand.key << "\n";
    out << "orientations_reflection_mode: " << rep.orientation_count << "\n";
    out << "periodic_found: " << (rep.periodic.found ? 1 : 0) << "\n";
    out << "periodic_W: " << rep.periodic.W << "\n";
    out << "periodic_H: " << rep.periodic.H << "\n";
    out << "periodic_limit_hit: " << (rep.periodic.any_limit ? 1 : 0) << "\n";
    out << "torus_nodes: " << rep.periodic.total_nodes << "\n";
    out << "no_reflection_periodic_found: " << (rep.periodic_no_reflect.found ? 1 : 0) << "\n";
    out << "no_reflection_periodic_W: " << rep.periodic_no_reflect.W << "\n";
    out << "no_reflection_periodic_H: " << rep.periodic_no_reflect.H << "\n";
    out << "random_patch_tiles: " << rep.random_patch.max_tiles << "\n";
    out << "boundary_patch_tiles: " << rep.boundary_patch.max_tiles << "\n";
    out << "forced_patch_tiles: " << rep.forced_patch.max_tiles << "\n";
    out << "best_patch_method: " << rep.best_patch_method << "\n";
    out << "best_patch_tiles: " << rep.best_patch_tiles << "\n";
    out << "exact_region_found: " << (rep.exact_region.found ? 1 : 0) << "\n";
    out << "exact_region_radius: " << rep.exact_region.radius << "\n";
    out << "exact_region_cells: " << rep.exact_region.cells << "\n";
    out << "exact_region_limit_hit: " << (rep.exact_region.limit_hit ? 1 : 0) << "\n";
    out << "exact_region_nodes: " << rep.exact_region.nodes << "\n";
    out << "score: " << rep.score << "\n";
    out << "\nnormalized_cells x y orientation:\n";
    for (const auto& c : cand.cells) out << c.x << " " << c.y << " " << c.o << "\n";

    write_svg_cells(cand.cells, tile_svg, rep.classification + " tile cells=" + std::to_string(cand.cells.size()));
    PatchResult best = choose_best_patch(rep);
    if (!best.best_tiles.empty()) {
        write_svg_patch_tiles(best.best_tiles, patch_svg,
                              rep.classification + " " + rep.best_patch_method +
                              " patch tiles=" + std::to_string(best.max_tiles));
    } else if (!best.best_occupied.empty()) {
        write_svg_cells(best.best_occupied, patch_svg,
                        rep.classification + " patch tiles=" + std::to_string(best.max_tiles));
    }
}

static void usage_validator() {
    std::cout << "CUDA Einstein Candidate Second-Pass Validator\n"
              << "Options:\n"
              << "  --input DIR              candidate directory from hunter\n"
              << "  --out DIR                validation output directory\n"
              << "  --period N               max torus W/H to search [18]\n"
              << "  --nodes N                DFS node limit per torus [20000000]\n"
              << "  --patch-radius N         patch radius [100]\n"
              << "  --random-trials N        random greedy trials [800]\n"
              << "  --boundary-trials N      boundary-first trials [500]\n"
              << "  --forced-trials N        forced-placement trials [500]\n"
              << "  --region-min N           smallest exact hex region [3]\n"
              << "  --region-max N           largest exact hex region [8]\n"
              << "  --region-nodes N         DFS node limit per exact region [5000000]\n"
              << "  --weak-tiles N           weak reject threshold [180]\n"
              << "  --strong-tiles N         strong threshold [400]\n"
              << "  --very-strong-tiles N    very strong threshold [700]\n"
              << "  --allow-reflections 0/1  include reflected copies [1]\n"
              << "  --test-no-reflection 0/1 also test chiral/no-reflection mode [1]\n"
              << "  --max-candidates N       validate only top N by old patch size [0=all]\n"
              << "  --skip-periodic-old 0/1  skip old periodic rejects [1]\n"
              << "  --only-strong-old 0/1    only validate old strong_candidate files [0]\n"
              << "  --seed N                 random seed [time]\n";
}

static ValidatorConfig parse_validator_config(int argc, char** argv) {
    ValidatorConfig c;
    c.seed = static_cast<uint64_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    for (int i = 1; i < argc; ++i) {
        auto need = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::cerr << "Missing value after " << name << "\n";
                std::exit(1);
            }
            return argv[++i];
        };
        std::string a = argv[i];
        if (a == "--input") c.input_dir = need("--input");
        else if (a == "--out") c.out_dir = need("--out");
        else if (a == "--period") parse_int(need("--period"), c.period);
        else if (a == "--nodes") parse_u64(need("--nodes"), c.nodes);
        else if (a == "--patch-radius") parse_int(need("--patch-radius"), c.patch_radius);
        else if (a == "--random-trials") parse_int(need("--random-trials"), c.random_trials);
        else if (a == "--boundary-trials") parse_int(need("--boundary-trials"), c.boundary_trials);
        else if (a == "--forced-trials") parse_int(need("--forced-trials"), c.forced_trials);
        else if (a == "--region-min") parse_int(need("--region-min"), c.region_min);
        else if (a == "--region-max") parse_int(need("--region-max"), c.region_max);
        else if (a == "--region-nodes") parse_u64(need("--region-nodes"), c.region_nodes);
        else if (a == "--weak-tiles") parse_int(need("--weak-tiles"), c.weak_tiles);
        else if (a == "--strong-tiles") parse_int(need("--strong-tiles"), c.strong_tiles);
        else if (a == "--very-strong-tiles") parse_int(need("--very-strong-tiles"), c.very_strong_tiles);
        else if (a == "--allow-reflections") parse_int(need("--allow-reflections"), c.allow_reflections);
        else if (a == "--test-no-reflection") parse_int(need("--test-no-reflection"), c.test_no_reflection);
        else if (a == "--max-candidates") parse_int(need("--max-candidates"), c.max_candidates);
        else if (a == "--skip-periodic-old") parse_int(need("--skip-periodic-old"), c.skip_periodic_old);
        else if (a == "--only-strong-old") parse_int(need("--only-strong-old"), c.only_strong_old);
        else if (a == "--seed") parse_u64(need("--seed"), c.seed);
        else if (a == "--help" || a == "-h") { usage_validator(); std::exit(0); }
        else { std::cerr << "Unknown option: " << a << "\n"; usage_validator(); std::exit(1); }
    }
    if (c.region_max < c.region_min) std::swap(c.region_max, c.region_min);
    if (c.patch_radius < 4) c.patch_radius = 4;
    if (c.period < 2) c.period = 2;
    return c;
}

int main(int argc, char** argv) {
    ValidatorConfig cfg = parse_validator_config(argc, argv);
    fs::create_directories(cfg.out_dir);

    std::cout << "\n╔══════════════════════════════════════════════════════════════╗\n"
              << "║      CUDA EINSTEIN SECOND-PASS VALIDATOR v1                 ║\n"
              << "╚══════════════════════════════════════════════════════════════╝\n";
    std::cout << "Input              : " << cfg.input_dir << "\n";
    std::cout << "Output             : " << cfg.out_dir << "\n";
    std::cout << "Torus period       : <= " << cfg.period << "\n";
    std::cout << "Torus nodes        : " << cfg.nodes << " per torus\n";
    std::cout << "Patch radius       : " << cfg.patch_radius << "\n";
    std::cout << "Trials             : random=" << cfg.random_trials
              << " boundary=" << cfg.boundary_trials
              << " forced=" << cfg.forced_trials << "\n";
    std::cout << "Exact hex regions  : " << cfg.region_min << ".." << cfg.region_max
              << " radius, nodes=" << cfg.region_nodes << "\n";
    std::cout << "Reflections        : " << (cfg.allow_reflections ? "allowed" : "not allowed") << "\n";
    std::cout << "Seed               : " << cfg.seed << "\n\n";

    // Initialise CUDA context so nvidia-smi shows the validator. The deep checks are CPU-heavy
    // exact searches plus stochastic patch builders; this file is compiled with nvcc for convenience.
    cudaFree(0);

    auto candidates = load_candidates(cfg.input_dir, cfg.skip_periodic_old, cfg.only_strong_old);
    if (cfg.max_candidates > 0 && static_cast<int>(candidates.size()) > cfg.max_candidates) {
        candidates.resize(static_cast<size_t>(cfg.max_candidates));
    }
    std::cout << "Loaded candidates  : " << candidates.size() << "\n";
    if (candidates.empty()) return 1;

    fs::path csv_path = cfg.out_dir / "validation.csv";
    std::ofstream csv(csv_path);
    csv << "time,index,total,candidate_id,filename,old_status,old_patch_tiles,classification,n,orientations,"
           "periodic_found,periodic_W,periodic_H,periodic_limit_hit,torus_nodes,"
           "no_reflect_periodic_found,no_reflect_W,no_reflect_H,"
           "random_patch_tiles,boundary_patch_tiles,forced_patch_tiles,best_patch_method,best_patch_tiles,best_patch_cells,"
           "exact_region_found,exact_region_radius,exact_region_cells,exact_region_limit_hit,exact_region_nodes,score,key\n";

    int periodic_rejects = 0, weak_rejects = 0, unstable = 0, strong = 0, very_strong = 0;
    int idx = 0;
    for (const auto& cand : candidates) {
        ++idx;
        uint64_t seed = cfg.seed ^ (static_cast<uint64_t>(cand.id) * 0x9E3779B97F4A7C15ULL) ^ static_cast<uint64_t>(idx);
        std::cout << "[" << idx << "/" << candidates.size() << "] " << cand.filename
                  << " old_patch=" << cand.old_patch_tiles << " ..." << std::flush;

        DeepReport rep = validate_deep(cand, cfg, seed);
        save_deep_artifacts(cfg.out_dir, cand, rep);

        if (rep.classification == "periodic_reject") ++periodic_rejects;
        else if (rep.classification == "weak_patch_reject") ++weak_rejects;
        else if (rep.classification == "unstable_candidate") ++unstable;
        else if (rep.classification == "strong_patch_candidate") ++strong;
        else if (rep.classification == "very_strong_candidate") ++very_strong;

        csv << now_string() << "," << idx << "," << candidates.size() << ","
            << cand.id << "," << csv_escape(cand.filename) << "," << cand.old_status << "," << cand.old_patch_tiles << ","
            << rep.classification << "," << cand.cells.size() << "," << rep.orientation_count << ","
            << (rep.periodic.found ? 1 : 0) << "," << rep.periodic.W << "," << rep.periodic.H << ","
            << (rep.periodic.any_limit ? 1 : 0) << "," << rep.periodic.total_nodes << ","
            << (rep.periodic_no_reflect.found ? 1 : 0) << "," << rep.periodic_no_reflect.W << "," << rep.periodic_no_reflect.H << ","
            << rep.random_patch.max_tiles << "," << rep.boundary_patch.max_tiles << "," << rep.forced_patch.max_tiles << ","
            << rep.best_patch_method << "," << rep.best_patch_tiles << "," << rep.best_patch_cells << ","
            << (rep.exact_region.found ? 1 : 0) << "," << rep.exact_region.radius << "," << rep.exact_region.cells << ","
            << (rep.exact_region.limit_hit ? 1 : 0) << "," << rep.exact_region.nodes << ","
            << std::fixed << std::setprecision(2) << rep.score << "," << csv_escape(cand.key) << "\n";
        csv.flush();

        std::cout << " " << rep.classification
                  << " best_patch=" << rep.best_patch_tiles
                  << " method=" << rep.best_patch_method
                  << " periodic=" << (rep.periodic.found ? "yes" : "no")
                  << " exactR=" << (rep.exact_region.found ? rep.exact_region.radius : 0)
                  << "\n";
    }

    std::cout << "\nDone. Results: " << csv_path << "\n";
    std::cout << "periodic_reject=" << periodic_rejects
              << " weak_patch_reject=" << weak_rejects
              << " unstable_candidate=" << unstable
              << " strong_patch_candidate=" << strong
              << " very_strong_candidate=" << very_strong << "\n";
    return 0;
}
