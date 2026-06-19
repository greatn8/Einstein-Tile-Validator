// -----------------------------------------------------------------------------
// Einstein Torus Audit
// -----------------------------------------------------------------------------
// A standalone, per-(W,H) finite-torus exact-cover audit for one triangular-
// lattice monotile candidate. It is intended to complement, not replace, the
// existing second-pass validator.
//
// What it does:
//   * Reads one candidate_XXXXXXX.txt file.
//   * Generates allowed rotations/reflections of the tile.
//   * Tests every finite W x H wrapped triangular-lattice torus up to a chosen
//     maximum period, where the board area is divisible by the tile area.
//   * Logs one durable CSV row after EVERY individual torus case.
//   * Prints heartbeat progress while a difficult exact-cover case is running.
//   * Supports --resume, so an interrupted sweep continues from the first
//     missing case rather than losing prior work.
//
// Interpretation:
//   FOUND       = an exact periodic torus tiling was constructed. The candidate
//                 is not an einstein tile under the tested orientation mode.
//   NO_TILING   = this particular finite torus was exhaustively searched within
//                 the configured solver and node budget; no tiling was found.
//   LIMIT       = the case hit its node cap. It is unresolved, not a pass.
//   INTERRUPTED = the program received Ctrl+C / SIGTERM while on this case.
//
// This is finite evidence only. It cannot prove a candidate is aperiodic.
// -----------------------------------------------------------------------------

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

static std::atomic<bool> g_stop(false);

static void signal_handler(int) {
    g_stop.store(true);
}

struct Cell {
    int x = 0;
    int y = 0;
    int o = 0; // 0=up triangle, 1=down triangle
};

static inline bool operator<(const Cell& a, const Cell& b) {
    if (a.x != b.x) return a.x < b.x;
    if (a.y != b.y) return a.y < b.y;
    return a.o < b.o;
}

static inline bool operator==(const Cell& a, const Cell& b) {
    return a.x == b.x && a.y == b.y && a.o == b.o;
}

static std::vector<Cell> normalize_cells(std::vector<Cell> cells) {
    if (cells.empty()) return cells;
    int min_x = cells.front().x;
    int min_y = cells.front().y;
    for (const auto& c : cells) {
        min_x = std::min(min_x, c.x);
        min_y = std::min(min_y, c.y);
    }
    for (auto& c : cells) {
        c.x -= min_x;
        c.y -= min_y;
    }
    std::sort(cells.begin(), cells.end());
    cells.erase(std::unique(cells.begin(), cells.end()), cells.end());
    return cells;
}

static bool parse_candidate(const fs::path& path, std::vector<Cell>& out_cells) {
    std::ifstream in(path);
    if (!in) return false;

    bool in_cells = false;
    std::string line;
    std::vector<Cell> cells;

    while (std::getline(in, line)) {
        if (line.find("normalized_cells") != std::string::npos) {
            in_cells = true;
            continue;
        }
        if (!in_cells) continue;

        std::istringstream iss(line);
        Cell c;
        if (iss >> c.x >> c.y >> c.o) {
            if (c.o == 0 || c.o == 1) cells.push_back(c);
        }
    }

    out_cells = normalize_cells(std::move(cells));
    return !out_cells.empty();
}

using V2 = std::pair<int, int>;

static std::array<V2, 3> vertices_of_cell(const Cell& c) {
    if (c.o == 0) {
        return {V2{c.x, c.y}, V2{c.x + 1, c.y}, V2{c.x, c.y + 1}};
    }
    return {V2{c.x + 1, c.y + 1}, V2{c.x + 1, c.y}, V2{c.x, c.y + 1}};
}

static V2 rotate60_once(V2 p) {
    const int a = p.first;
    const int b = p.second;
    return {-b, a + b};
}

static V2 transform_vertex(V2 p, int rot, bool reflect) {
    if (reflect) std::swap(p.first, p.second);
    for (int i = 0; i < rot; ++i) p = rotate60_once(p);
    return p;
}

static bool same_vertices(std::array<V2, 3> a, std::array<V2, 3> b) {
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    return a == b;
}

static Cell cell_from_vertices(const std::array<V2, 3>& vertices) {
    int min_x = vertices[0].first;
    int min_y = vertices[0].second;
    for (const auto& p : vertices) {
        min_x = std::min(min_x, p.first);
        min_y = std::min(min_y, p.second);
    }

    Cell up{min_x, min_y, 0};
    if (same_vertices(vertices, vertices_of_cell(up))) return up;

    Cell down{min_x, min_y, 1};
    if (same_vertices(vertices, vertices_of_cell(down))) return down;

    throw std::runtime_error("Could not map transformed triangle back to lattice cell");
}

static std::vector<Cell> transform_shape(const std::vector<Cell>& shape, int rot, bool reflect) {
    std::vector<Cell> result;
    result.reserve(shape.size());
    for (const auto& c : shape) {
        const auto vertices = vertices_of_cell(c);
        std::array<V2, 3> transformed{};
        for (int i = 0; i < 3; ++i) transformed[i] = transform_vertex(vertices[i], rot, reflect);
        result.push_back(cell_from_vertices(transformed));
    }
    return normalize_cells(std::move(result));
}

static std::string cells_key(const std::vector<Cell>& cells) {
    std::ostringstream oss;
    for (const auto& c : cells) oss << c.x << ',' << c.y << ',' << c.o << ';';
    return oss.str();
}

static std::vector<std::vector<Cell>> all_orientations(const std::vector<Cell>& shape,
                                                         bool allow_reflections) {
    std::vector<std::vector<Cell>> result;
    std::unordered_set<std::string> seen;
    const int reflection_count = allow_reflections ? 2 : 1;

    for (int reflection = 0; reflection < reflection_count; ++reflection) {
        for (int rotation = 0; rotation < 6; ++rotation) {
            auto transformed = transform_shape(shape, rotation, reflection != 0);
            const std::string key = cells_key(transformed);
            if (seen.insert(key).second) result.push_back(std::move(transformed));
        }
    }
    return result;
}

static int torus_index(int x, int y, int o, int W, int H) {
    x %= W;
    y %= H;
    if (x < 0) x += W;
    if (y < 0) y += H;
    return ((y * W + x) << 1) | (o & 1);
}

static inline void set_bit(std::vector<uint64_t>& words, int bit) {
    words[static_cast<size_t>(bit >> 6)] |= (1ULL << (bit & 63));
}

static inline bool bit_is_set(const std::vector<uint64_t>& words, int bit) {
    return (words[static_cast<size_t>(bit >> 6)] >> (bit & 63)) & 1ULL;
}

static inline bool no_overlap(const std::vector<uint64_t>& occupied,
                              const std::vector<uint64_t>& candidate) {
    for (size_t i = 0; i < occupied.size(); ++i) {
        if (occupied[i] & candidate[i]) return false;
    }
    return true;
}

struct Placement {
    std::vector<uint64_t> bits;
    std::vector<int> cells;
    int orientation_index = -1;
    int tx = 0;
    int ty = 0;
};

static int first_uncovered(const std::vector<uint64_t>& occupied, int bit_count) {
    const int word_count = static_cast<int>(occupied.size());
    for (int w = 0; w < word_count; ++w) {
        uint64_t available = ~occupied[static_cast<size_t>(w)];
        if (w == word_count - 1 && (bit_count & 63)) {
            const uint64_t mask = (1ULL << (bit_count & 63)) - 1ULL;
            available &= mask;
        }
        if (available) return w * 64 + __builtin_ctzll(available);
    }
    return -1;
}

struct CaseProgress {
    std::string mode;
    int W = 0;
    int H = 0;
    uint64_t node_limit = 0;
    int interval_seconds = 10;
    std::chrono::steady_clock::time_point started{};
    std::chrono::steady_clock::time_point last_print{};
};

struct ExactStats {
    bool found = false;
    bool limit_hit = false;
    bool interrupted = false;
    uint64_t nodes = 0;
    uint64_t placements = 0;
};

static std::string elapsed_string(double seconds) {
    const uint64_t total = static_cast<uint64_t>(seconds);
    const uint64_t hours = total / 3600;
    const uint64_t minutes = (total % 3600) / 60;
    const uint64_t secs = total % 60;
    std::ostringstream oss;
    if (hours) oss << hours << 'h';
    if (hours || minutes) oss << std::setw(2) << std::setfill('0') << minutes << 'm';
    oss << std::setw(2) << std::setfill('0') << secs << 's';
    return oss.str();
}

static void maybe_print_heartbeat(const CaseProgress& progress, const ExactStats& stats) {
    if ((stats.nodes & 0xFFFFULL) != 0ULL) return;
    const auto now = std::chrono::steady_clock::now();
    const double since_last = std::chrono::duration<double>(now - progress.last_print).count();
    if (since_last < progress.interval_seconds) return;

    const double elapsed = std::chrono::duration<double>(now - progress.started).count();
    const double rate = elapsed > 0.0 ? static_cast<double>(stats.nodes) / elapsed : 0.0;

    std::cout << "[heartbeat] mode=" << progress.mode
              << " W=" << progress.W
              << " H=" << progress.H
              << " elapsed=" << elapsed_string(elapsed)
              << " nodes=" << stats.nodes
              << "/" << progress.node_limit
              << " rate=" << std::fixed << std::setprecision(0) << rate << " nodes/s"
              << std::defaultfloat
              << std::endl;

    const_cast<CaseProgress&>(progress).last_print = now;
}

// Exact cover DFS with an MRV (minimum remaining values) pivot. This is a
// separate implementation from the original validator's first-uncovered DFS,
// deliberately making it a useful independent finite-torus cross-check.
static bool exact_cover_dfs_mrv(const std::vector<std::vector<int>>& cover,
                                const std::vector<Placement>& placements,
                                std::vector<uint64_t>& occupied,
                                int bit_count,
                                uint64_t max_nodes,
                                ExactStats& stats,
                                const CaseProgress& progress,
                                std::vector<int>* chosen_placements) {
    ++stats.nodes;

    if (g_stop.load()) {
        stats.interrupted = true;
        return false;
    }

    maybe_print_heartbeat(progress, stats);

    if (stats.nodes > max_nodes) {
        stats.limit_hit = true;
        return false;
    }

    if (first_uncovered(occupied, bit_count) < 0) {
        stats.found = true;
        return true;
    }

    int selected_cell = -1;
    std::vector<int> selected_choices;
    size_t fewest_choices = std::numeric_limits<size_t>::max();

    for (int cell = 0; cell < bit_count; ++cell) {
        if (bit_is_set(occupied, cell)) continue;

        std::vector<int> choices;
        choices.reserve(cover[static_cast<size_t>(cell)].size());
        for (int placement_index : cover[static_cast<size_t>(cell)]) {
            if (no_overlap(occupied, placements[static_cast<size_t>(placement_index)].bits)) {
                choices.push_back(placement_index);
            }
        }

        if (choices.empty()) return false;

        if (choices.size() < fewest_choices) {
            selected_cell = cell;
            selected_choices = std::move(choices);
            fewest_choices = selected_choices.size();
            if (fewest_choices == 1) break;
        }
    }

    if (selected_cell < 0) return false;

    for (int placement_index : selected_choices) {
        const auto& placement = placements[static_cast<size_t>(placement_index)];
        if (!no_overlap(occupied, placement.bits)) continue;

        for (size_t word = 0; word < occupied.size(); ++word) {
            occupied[word] |= placement.bits[word];
        }

        if (chosen_placements) chosen_placements->push_back(placement_index);

        if (exact_cover_dfs_mrv(cover, placements, occupied, bit_count, max_nodes, stats,
                                progress, chosen_placements)) {
            return true;
        }

        if (chosen_placements) chosen_placements->pop_back();

        for (size_t word = 0; word < occupied.size(); ++word) {
            occupied[word] &= ~placement.bits[word];
        }

        if (stats.limit_hit || stats.interrupted) return false;
    }

    return false;
}

static ExactStats torus_tiles_exact(const std::vector<std::vector<Cell>>& orientations,
                                    int W,
                                    int H,
                                    uint64_t max_nodes,
                                    const CaseProgress& progress,
                                    std::vector<Placement>* witness_placements = nullptr,
                                    std::vector<int>* witness_solution = nullptr) {
    ExactStats stats;
    if (orientations.empty()) return stats;

    const int tile_cells = static_cast<int>(orientations.front().size());
    const int board_cells = 2 * W * H;
    if (board_cells % tile_cells != 0) return stats;

    const int word_count = (board_cells + 63) / 64;
    std::vector<Placement> placements;
    placements.reserve(orientations.size() * static_cast<size_t>(W) * static_cast<size_t>(H));
    std::vector<std::vector<int>> cover(static_cast<size_t>(board_cells));
    std::unordered_set<std::string> placement_seen;

    for (size_t orientation_index = 0; orientation_index < orientations.size(); ++orientation_index) {
        const auto& orientation = orientations[orientation_index];
        for (int tx = 0; tx < W; ++tx) {
            for (int ty = 0; ty < H; ++ty) {
                Placement placement;
                placement.bits.assign(static_cast<size_t>(word_count), 0ULL);
                placement.cells.reserve(orientation.size());
                bool valid = true;

                for (const auto& cell : orientation) {
                    const int index = torus_index(cell.x + tx, cell.y + ty, cell.o, W, H);
                    if (bit_is_set(placement.bits, index)) {
                        valid = false; // self-overlap after wrapping
                        break;
                    }
                    set_bit(placement.bits, index);
                    placement.cells.push_back(index);
                }

                if (!valid) continue;

                // Deduplicate placements that arise from symmetries of this tile.
                std::ostringstream key_stream;
                for (uint64_t word : placement.bits) key_stream << word << ':';
                if (!placement_seen.insert(key_stream.str()).second) continue;

                placement.orientation_index = static_cast<int>(orientation_index);
                placement.tx = tx;
                placement.ty = ty;

                const int placement_index = static_cast<int>(placements.size());
                for (int cell_index : placement.cells) {
                    cover[static_cast<size_t>(cell_index)].push_back(placement_index);
                }
                placements.push_back(std::move(placement));
            }
        }
    }

    stats.placements = static_cast<uint64_t>(placements.size());
    std::vector<uint64_t> occupied(static_cast<size_t>(word_count), 0ULL);
    std::vector<int> solution;
    exact_cover_dfs_mrv(cover, placements, occupied, board_cells, max_nodes, stats, progress,
                        witness_solution ? &solution : nullptr);

    if (stats.found && witness_placements && witness_solution) {
        *witness_placements = std::move(placements);
        *witness_solution = std::move(solution);
    }
    return stats;
}

struct Config {
    fs::path input_file;
    fs::path out_dir = "torus_audit";
    int max_period = 22;
    uint64_t nodes_per_case = 50000000ULL;
    std::string mode = "both"; // reflect, no-reflect, both
    int progress_seconds = 10;
    bool resume = false;
    bool single_case = false;
    bool save_witness = false;
    int W = 0;
    int H = 0;
};

static bool parse_int(const std::string& value, int& out) {
    try {
        size_t consumed = 0;
        const long long parsed = std::stoll(value, &consumed);
        if (consumed != value.size() || parsed < std::numeric_limits<int>::min() ||
            parsed > std::numeric_limits<int>::max()) return false;
        out = static_cast<int>(parsed);
        return true;
    } catch (...) {
        return false;
    }
}

static bool parse_u64(const std::string& value, uint64_t& out) {
    try {
        size_t consumed = 0;
        const unsigned long long parsed = std::stoull(value, &consumed);
        if (consumed != value.size()) return false;
        out = static_cast<uint64_t>(parsed);
        return true;
    } catch (...) {
        return false;
    }
}

static void usage() {
    std::cout << "Einstein Torus Audit\n\n"
              << "Usage:\n"
              << "  ./einstein_torus_audit --input candidate_0000858.txt [options]\n\n"
              << "Options:\n"
              << "  --input FILE                 original candidate .txt file (required)\n"
              << "  --out DIR                    audit output directory [torus_audit]\n"
              << "  --max-period N               test W,H from 2..N [22]\n"
              << "  --nodes N                    exact-cover node cap per W,H case [50000000]\n"
              << "  --mode reflect|no-reflect|both  orientation mode [both]\n"
              << "  --case W H                   audit only one W,H torus\n"
              << "  --progress-sec N             heartbeat interval for hard cases [10]\n"
              << "  --resume                     skip cases already recorded in audit.csv\n"
              << "  --save-witness               write placement witness for a FOUND single case\n"
              << "  --help                       show this help\n";
}

static Config parse_config(int argc, char** argv) {
    Config config;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need = [&](const char* name) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(std::string("Missing value after ") + name);
            return argv[++i];
        };

        if (arg == "--input") config.input_file = need("--input");
        else if (arg == "--out") config.out_dir = need("--out");
        else if (arg == "--max-period") {
            if (!parse_int(need("--max-period"), config.max_period)) throw std::runtime_error("Bad --max-period");
        }
        else if (arg == "--nodes") {
            if (!parse_u64(need("--nodes"), config.nodes_per_case)) throw std::runtime_error("Bad --nodes");
        }
        else if (arg == "--mode") config.mode = need("--mode");
        else if (arg == "--case") {
            if (!parse_int(need("--case W"), config.W) || !parse_int(need("--case H"), config.H)) {
                throw std::runtime_error("Bad --case W H");
            }
            config.single_case = true;
        }
        else if (arg == "--progress-sec") {
            if (!parse_int(need("--progress-sec"), config.progress_seconds)) throw std::runtime_error("Bad --progress-sec");
        }
        else if (arg == "--resume") config.resume = true;
        else if (arg == "--save-witness") config.save_witness = true;
        else if (arg == "--help" || arg == "-h") {
            usage();
            std::exit(0);
        }
        else throw std::runtime_error("Unknown option: " + arg);
    }

    if (config.input_file.empty()) throw std::runtime_error("--input is required");
    if (config.max_period < 2) throw std::runtime_error("--max-period must be at least 2");
    if (config.nodes_per_case < 1) throw std::runtime_error("--nodes must be at least 1");
    if (config.progress_seconds < 1) config.progress_seconds = 1;
    if (config.mode != "reflect" && config.mode != "no-reflect" && config.mode != "both") {
        throw std::runtime_error("--mode must be reflect, no-reflect, or both");
    }
    if (config.single_case && (config.W < 2 || config.H < 2)) {
        throw std::runtime_error("--case W H values must both be at least 2");
    }
    if (config.save_witness && !config.single_case) {
        throw std::runtime_error("--save-witness requires --case W H so the witness is unambiguous");
    }

    return config;
}

static std::string now_string() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

static std::set<std::string> load_completed_case_keys(const fs::path& csv_path) {
    std::set<std::string> completed;
    std::ifstream in(csv_path);
    if (!in) return completed;

    std::string line;
    std::getline(in, line); // header
    while (std::getline(in, line)) {
        std::vector<std::string> fields;
        std::stringstream stream(line);
        std::string field;
        while (std::getline(stream, field, ',')) fields.push_back(field);
        if (fields.size() < 12) continue;

        const std::string& mode = fields[1];
        const std::string& W = fields[2];
        const std::string& H = fields[3];
        const std::string& status = fields[10];
        if (status == "FOUND" || status == "NO_TILING" || status == "LIMIT") {
            completed.insert(mode + ":" + W + ":" + H);
        }
    }
    return completed;
}

static std::string case_key(const std::string& mode, int W, int H) {
    return mode + ":" + std::to_string(W) + ":" + std::to_string(H);
}

static void index_to_torus_cell(int index, int W, int& x, int& y, int& o) {
    o = index & 1;
    const int cell_pair = index >> 1;
    x = cell_pair % W;
    y = cell_pair / W;
}

static std::string witness_stem(const std::string& mode, int W, int H) {
    return "periodic_witness_" + mode + "_W" + std::to_string(W) + "_H" + std::to_string(H);
}

static void write_periodic_witness(const fs::path& out_dir,
                                   const Config& config,
                                   const std::string& mode,
                                   int W,
                                   int H,
                                   int tile_cells,
                                   int orientation_count,
                                   const std::vector<Placement>& placements,
                                   const std::vector<int>& solution) {
    const int board_cells = 2 * W * H;
    const std::string stem = witness_stem(mode, W, H);
    const fs::path summary_path = out_dir / (stem + ".txt");
    const fs::path placement_path = out_dir / (stem + "_placements.csv");
    const fs::path cells_path = out_dir / (stem + "_cells.csv");

    std::vector<int> multiplicity(static_cast<size_t>(board_cells), 0);
    for (int placement_index : solution) {
        if (placement_index < 0 || placement_index >= static_cast<int>(placements.size())) {
            throw std::runtime_error("Internal error: witness contains invalid placement index");
        }
        for (int cell : placements[static_cast<size_t>(placement_index)].cells) {
            if (cell < 0 || cell >= board_cells) {
                throw std::runtime_error("Internal error: witness contains out-of-range cell");
            }
            ++multiplicity[static_cast<size_t>(cell)];
        }
    }

    int uncovered = 0;
    int multiply_covered = 0;
    for (int count : multiplicity) {
        if (count == 0) ++uncovered;
        if (count > 1) ++multiply_covered;
    }

    std::ofstream summary(summary_path);
    if (!summary) throw std::runtime_error("Could not write witness summary: " + summary_path.string());
    summary << "Periodic torus witness\n"
            << "======================\n"
            << "input_file: " << config.input_file.string() << "\n"
            << "mode: " << mode << "\n"
            << "W: " << W << "\n"
            << "H: " << H << "\n"
            << "board_cells: " << board_cells << "\n"
            << "tile_cells: " << tile_cells << "\n"
            << "tiles_required: " << (board_cells / tile_cells) << "\n"
            << "orientation_count: " << orientation_count << "\n"
            << "chosen_placements: " << solution.size() << "\n"
            << "witness_coverage_uncovered_cells: " << uncovered << "\n"
            << "witness_coverage_multiply_covered_cells: " << multiply_covered << "\n"
            << "coverage_check: " << ((uncovered == 0 && multiply_covered == 0 &&
                                         static_cast<int>(solution.size()) * tile_cells == board_cells)
                                             ? "PASS" : "FAIL") << "\n"
            << "\n"
            << "The placement CSV lists one translated orientation per tile.\n"
            << "The cell CSV lists each wrapped triangular cell covered by each tile.\n";

    std::ofstream placement_csv(placement_path);
    if (!placement_csv) throw std::runtime_error("Could not write witness placements: " + placement_path.string());
    placement_csv << "tile_index,placement_index,orientation_index,tx,ty,cell_count\n";

    std::ofstream cell_csv(cells_path);
    if (!cell_csv) throw std::runtime_error("Could not write witness cells: " + cells_path.string());
    cell_csv << "tile_index,placement_index,orientation_index,tx,ty,cell_index,x,y,triangle_orientation\n";

    for (size_t tile_index = 0; tile_index < solution.size(); ++tile_index) {
        const int placement_index = solution[tile_index];
        const auto& placement = placements[static_cast<size_t>(placement_index)];
        placement_csv << tile_index << ',' << placement_index << ',' << placement.orientation_index << ','
                      << placement.tx << ',' << placement.ty << ',' << placement.cells.size() << '\n';

        for (int cell_index : placement.cells) {
            int x = 0, y = 0, o = 0;
            index_to_torus_cell(cell_index, W, x, y, o);
            cell_csv << tile_index << ',' << placement_index << ',' << placement.orientation_index << ','
                     << placement.tx << ',' << placement.ty << ',' << cell_index << ','
                     << x << ',' << y << ',' << o << '\n';
        }
    }

    summary.flush();
    placement_csv.flush();
    cell_csv.flush();

    std::cout << "\nWitness exported:\n"
              << "  " << summary_path << "\n"
              << "  " << placement_path << "\n"
              << "  " << cells_path << "\n";
}

static void append_summary(const fs::path& out_dir,
                           const Config& config,
                           int tile_cells,
                           int reflect_orientations,
                           int no_reflect_orientations) {
    std::ofstream summary(out_dir / "summary.txt", std::ios::app);
    summary << "Einstein torus audit\n"
            << "====================\n"
            << "input_file: " << config.input_file.string() << "\n"
            << "tile_cells: " << tile_cells << "\n"
            << "orientations_reflect: " << reflect_orientations << "\n"
            << "orientations_no_reflect: " << no_reflect_orientations << "\n"
            << "max_period: " << config.max_period << "\n"
            << "nodes_per_case: " << config.nodes_per_case << "\n"
            << "mode: " << config.mode << "\n"
            << "resume: " << (config.resume ? 1 : 0) << "\n"
            << "\n";
}

int main(int argc, char** argv) {
    try {
        const Config config = parse_config(argc, argv);
        std::signal(SIGINT, signal_handler);
        std::signal(SIGTERM, signal_handler);

        std::vector<Cell> shape;
        if (!parse_candidate(config.input_file, shape)) {
            std::cerr << "Could not read normalized_cells from: " << config.input_file << "\n";
            return 1;
        }

        const auto reflected_orientations = all_orientations(shape, true);
        const auto no_reflect_orientations = all_orientations(shape, false);
        fs::create_directories(config.out_dir);

        const fs::path csv_path = config.out_dir / "audit.csv";
        const bool write_header = !fs::exists(csv_path) || fs::file_size(csv_path) == 0;
        const std::set<std::string> completed = config.resume ? load_completed_case_keys(csv_path)
                                                               : std::set<std::string>{};

        std::ofstream csv(csv_path, std::ios::app);
        if (!csv) {
            std::cerr << "Could not open output CSV: " << csv_path << "\n";
            return 1;
        }
        if (write_header) {
            csv << "time,mode,W,H,board_cells,tile_cells,orientations,placements,nodes,node_limit,status,elapsed_seconds\n";
            csv.flush();
        }

        append_summary(config.out_dir, config, static_cast<int>(shape.size()),
                       static_cast<int>(reflected_orientations.size()),
                       static_cast<int>(no_reflect_orientations.size()));

        std::cout << "\n╔══════════════════════════════════════════════════════════════╗\n"
                  << "║             EINSTEIN TORUS AUDIT — PER CASE                  ║\n"
                  << "╚══════════════════════════════════════════════════════════════╝\n"
                  << "Input              : " << config.input_file << "\n"
                  << "Output              : " << config.out_dir << "\n"
                  << "Tile cells          : " << shape.size() << "\n"
                  << "Orientations        : reflect=" << reflected_orientations.size()
                  << " no-reflect=" << no_reflect_orientations.size() << "\n"
                  << "Max period          : " << config.max_period << "\n"
                  << "Node cap per case   : " << config.nodes_per_case << "\n"
                  << "Mode                : " << config.mode << "\n"
                  << "Resume              : " << (config.resume ? "yes" : "no") << "\n"
                  << "Save witness        : " << (config.save_witness ? "yes" : "no") << "\n\n";

        struct ModeRun {
            std::string name;
            const std::vector<std::vector<Cell>>* orientations = nullptr;
        };

        std::vector<ModeRun> modes;
        if (config.mode == "reflect" || config.mode == "both") {
            modes.push_back({"reflect", &reflected_orientations});
        }
        if (config.mode == "no-reflect" || config.mode == "both") {
            modes.push_back({"no_reflect", &no_reflect_orientations});
        }

        int total_cases = 0;
        for (size_t mode_index = 0; mode_index < modes.size(); ++mode_index) {
            if (config.single_case) {
                ++total_cases;
            } else {
                for (int W = 2; W <= config.max_period; ++W) {
                    for (int H = 2; H <= config.max_period; ++H) {
                        const int board_cells = 2 * W * H;
                        if (board_cells % static_cast<int>(shape.size()) == 0) ++total_cases;
                    }
                }
            }
        }

        int case_index = 0;
        int no_tiling = 0;
        int limits = 0;
        int found = 0;
        int skipped = 0;

        for (const auto& mode : modes) {
            const auto& orientations = *mode.orientations;

            const int start_W = config.single_case ? config.W : 2;
            const int end_W = config.single_case ? config.W : config.max_period;
            const int start_H = config.single_case ? config.H : 2;
            const int end_H = config.single_case ? config.H : config.max_period;

            for (int W = start_W; W <= end_W; ++W) {
                for (int H = start_H; H <= end_H; ++H) {
                    if (g_stop.load()) break;

                    const int board_cells = 2 * W * H;
                    if (board_cells % static_cast<int>(shape.size()) != 0) continue;

                    ++case_index;
                    const std::string key = case_key(mode.name, W, H);
                    if (config.resume && completed.count(key)) {
                        ++skipped;
                        std::cout << "[" << case_index << "/" << total_cases << "] "
                                  << mode.name << " W=" << W << " H=" << H
                                  << " already recorded; skipping\n";
                        continue;
                    }

                    std::cout << "[" << case_index << "/" << total_cases << "] "
                              << mode.name << " W=" << W << " H=" << H
                              << " board_cells=" << board_cells
                              << " ..." << std::flush;

                    CaseProgress progress;
                    progress.mode = mode.name;
                    progress.W = W;
                    progress.H = H;
                    progress.node_limit = config.nodes_per_case;
                    progress.interval_seconds = config.progress_seconds;
                    progress.started = std::chrono::steady_clock::now();
                    progress.last_print = progress.started;

                    const auto case_start = progress.started;
                    std::vector<Placement> witness_placements;
                    std::vector<int> witness_solution;
                    ExactStats stats = torus_tiles_exact(
                        orientations, W, H, config.nodes_per_case, progress,
                        config.save_witness ? &witness_placements : nullptr,
                        config.save_witness ? &witness_solution : nullptr);
                    const auto case_end = std::chrono::steady_clock::now();
                    const double elapsed = std::chrono::duration<double>(case_end - case_start).count();

                    std::string status;
                    if (stats.interrupted) status = "INTERRUPTED";
                    else if (stats.found) status = "FOUND";
                    else if (stats.limit_hit) status = "LIMIT";
                    else status = "NO_TILING";

                    if (status == "FOUND") ++found;
                    else if (status == "LIMIT") ++limits;
                    else if (status == "NO_TILING") ++no_tiling;

                    csv << now_string() << ','
                        << mode.name << ','
                        << W << ','
                        << H << ','
                        << board_cells << ','
                        << shape.size() << ','
                        << orientations.size() << ','
                        << stats.placements << ','
                        << stats.nodes << ','
                        << config.nodes_per_case << ','
                        << status << ','
                        << std::fixed << std::setprecision(3) << elapsed << '\n';
                    csv.flush();

                    std::cout << " " << status
                              << " placements=" << stats.placements
                              << " nodes=" << stats.nodes
                              << " elapsed=" << elapsed_string(elapsed)
                              << "\n";

                    if (stats.found && config.save_witness) {
                        write_periodic_witness(config.out_dir, config, mode.name, W, H,
                                               static_cast<int>(shape.size()),
                                               static_cast<int>(orientations.size()),
                                               witness_placements, witness_solution);
                    }

                    if (stats.interrupted) break;
                }
                if (g_stop.load()) break;
            }
            if (g_stop.load()) break;
        }

        std::ofstream summary(config.out_dir / "summary.txt", std::ios::app);
        summary << "Completed run summary\n"
                << "---------------------\n"
                << "no_tiling_cases: " << no_tiling << "\n"
                << "limit_cases: " << limits << "\n"
                << "periodic_found_cases: " << found << "\n"
                << "skipped_resumed_cases: " << skipped << "\n"
                << "interrupted: " << (g_stop.load() ? 1 : 0) << "\n\n";

        std::cout << "\nAudit complete.\n"
                  << "NO_TILING=" << no_tiling
                  << " LIMIT=" << limits
                  << " FOUND=" << found
                  << " skipped=" << skipped
                  << "\nCSV: " << csv_path << "\n";

        return g_stop.load() ? 130 : 0;

    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << "\n";
        usage();
        return 1;
    }
}
