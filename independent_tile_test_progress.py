#!/usr/bin/env python3
import argparse
import random
import sys
import threading
import time

sys.setrecursionlimit(1000000)


# ------------------------------------------------------------
# Progress heartbeat
# ------------------------------------------------------------

class Progress:
    def __init__(self):
        self.lock = threading.Lock()
        self.phase = "starting"
        self.detail = ""
        self.nodes = 0
        self.best_tiles = 0
        self.trial = 0
        self.total_trials = 0
        self.start_time = time.time()
        self.phase_start = time.time()
        self.done = False

    def update(self, phase=None, detail=None, nodes=None, best_tiles=None,
               trial=None, total_trials=None):
        with self.lock:
            if phase is not None and phase != self.phase:
                self.phase = phase
                self.phase_start = time.time()

            if detail is not None:
                self.detail = detail
            if nodes is not None:
                self.nodes = nodes
            if best_tiles is not None:
                self.best_tiles = best_tiles
            if trial is not None:
                self.trial = trial
            if total_trials is not None:
                self.total_trials = total_trials

    def stop(self):
        with self.lock:
            self.done = True

    def heartbeat_loop(self, seconds):
        while True:
            time.sleep(seconds)

            with self.lock:
                if self.done:
                    return

                elapsed = time.time() - self.start_time
                phase_elapsed = time.time() - self.phase_start

                msg = (
                    f"[heartbeat] total={fmt_time(elapsed)} "
                    f"phase={self.phase} "
                    f"phase_time={fmt_time(phase_elapsed)} "
                    f"detail={self.detail} "
                    f"nodes={self.nodes:,} "
                    f"best_tiles={self.best_tiles}"
                )

                if self.total_trials:
                    msg += f" trial={self.trial}/{self.total_trials}"

            print(msg, flush=True)


PROGRESS = Progress()


def fmt_time(seconds):
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


# ------------------------------------------------------------
# Triangular grid helpers
# ------------------------------------------------------------

def neighbour(c, d):
    x, y, o = c
    d %= 3

    if o == 0:
        if d == 0:
            return (x, y, 1)
        if d == 1:
            return (x - 1, y, 1)
        return (x, y - 1, 1)

    if d == 0:
        return (x, y, 0)
    if d == 1:
        return (x + 1, y, 0)
    return (x, y + 1, 0)


def normalize(cells):
    cells = list(set(cells))
    minx = min(c[0] for c in cells)
    miny = min(c[1] for c in cells)
    return sorted((x - minx, y - miny, o) for x, y, o in cells)


def key_of(cells):
    cells = normalize(cells)
    return f"{len(cells)}:" + "".join(f"{x},{y},{o};" for x, y, o in cells)


def vertices_of_cell(c):
    x, y, o = c

    if o == 0:
        return [(x, y), (x + 1, y), (x, y + 1)]

    return [(x + 1, y + 1), (x + 1, y), (x, y + 1)]


def rotate60(p):
    a, b = p
    return (-b, a + b)


def transform_vertex(p, rot, reflect):
    a, b = p

    if reflect:
        a, b = b, a

    for _ in range(rot):
        a, b = rotate60((a, b))

    return (a, b)


def same_vertices(a, b):
    return sorted(a) == sorted(b)


def cell_from_vertices(vs):
    minx = min(p[0] for p in vs)
    miny = min(p[1] for p in vs)

    up = (minx, miny, 0)
    if same_vertices(vs, vertices_of_cell(up)):
        return up

    down = (minx, miny, 1)
    if same_vertices(vs, vertices_of_cell(down)):
        return down

    raise RuntimeError(f"Could not map transformed triangle to cell: {vs}")


def transform_shape(shape, rot, reflect):
    out = []

    for c in shape:
        vs = vertices_of_cell(c)
        tvs = [transform_vertex(p, rot, reflect) for p in vs]
        out.append(cell_from_vertices(tvs))

    return normalize(out)


def all_orientations(shape, allow_reflections=True):
    result = []
    seen = set()

    reflection_count = 2 if allow_reflections else 1

    for ref in range(reflection_count):
        for rot in range(6):
            t = transform_shape(shape, rot, bool(ref))
            k = key_of(t)

            if k not in seen:
                seen.add(k)
                result.append(t)

    return result


def parse_candidate(path):
    cells = []
    in_cells = False

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()

            if not t:
                continue

            if "normalized_cells" in t:
                in_cells = True
                continue

            if in_cells:
                parts = t.split()

                if len(parts) >= 3:
                    try:
                        x, y, o = map(int, parts[:3])
                        cells.append((x, y, o))
                    except ValueError:
                        pass

    if not cells:
        raise ValueError("No normalized_cells section found in candidate file.")

    return normalize(cells)


def connected(shape):
    S = set(shape)
    stack = [shape[0]]
    seen = {shape[0]}

    while stack:
        c = stack.pop()

        for d in range(3):
            n = neighbour(c, d)

            if n in S and n not in seen:
                seen.add(n)
                stack.append(n)

    return len(seen) == len(S)


# ------------------------------------------------------------
# Periodic torus exact-cover search
# ------------------------------------------------------------

def torus_index(x, y, o, W, H):
    x %= W
    y %= H
    return ((y * W + x) << 1) | (o & 1)


def exact_torus(orientations, W, H, max_nodes):
    tile_cells = len(orientations[0])
    board_cells = 2 * W * H

    if board_cells % tile_cells != 0:
        return {
            "found": False,
            "limit_hit": False,
            "nodes": 0,
            "skipped": True,
            "placements": 0,
        }

    all_mask = (1 << board_cells) - 1
    placements = []
    cover = [[] for _ in range(board_cells)]
    seen_masks = set()

    PROGRESS.update(
        phase="building placements",
        detail=f"W={W} H={H}",
        nodes=0,
    )

    for ori in orientations:
        for tx in range(W):
            for ty in range(H):
                mask = 0
                ok = True
                idxs = []

                for x, y, o in ori:
                    idx = torus_index(x + tx, y + ty, o, W, H)
                    bit = 1 << idx

                    if mask & bit:
                        ok = False
                        break

                    mask |= bit
                    idxs.append(idx)

                if not ok:
                    continue

                if mask in seen_masks:
                    continue

                seen_masks.add(mask)
                pi = len(placements)
                placements.append(mask)

                for idx in idxs:
                    cover[idx].append(pi)

    nodes = 0
    limit_hit = False

    PROGRESS.update(
        phase="torus search",
        detail=f"W={W} H={H} placements={len(placements):,}",
        nodes=0,
    )

    def choose_uncovered_cell(occ):
        best_choices = None

        for idx in range(board_cells):
            if (occ >> idx) & 1:
                continue

            choices = []

            for pi in cover[idx]:
                pmask = placements[pi]

                if pmask & occ == 0:
                    choices.append(pmask)

            if not choices:
                return []

            if best_choices is None or len(choices) < len(best_choices):
                best_choices = choices

                if len(best_choices) == 1:
                    break

        return best_choices

    def dfs(occ):
        nonlocal nodes, limit_hit

        nodes += 1

        if nodes % 4096 == 0:
            PROGRESS.update(nodes=nodes)

        if nodes > max_nodes:
            limit_hit = True
            PROGRESS.update(nodes=nodes)
            return False

        if occ == all_mask:
            PROGRESS.update(nodes=nodes)
            return True

        choices = choose_uncovered_cell(occ)

        if not choices:
            return False

        for pmask in choices:
            if pmask & occ:
                continue

            if dfs(occ | pmask):
                return True

            if limit_hit:
                return False

        return False

    found = dfs(0)

    PROGRESS.update(nodes=nodes)

    return {
        "found": found,
        "limit_hit": limit_hit,
        "nodes": nodes,
        "skipped": False,
        "placements": len(placements),
    }


def search_periodic(orientations, max_period, max_nodes):
    total_nodes = 0
    any_limit = False

    for W in range(2, max_period + 1):
        for H in range(2, max_period + 1):
            result = exact_torus(orientations, W, H, max_nodes)

            if result["skipped"]:
                continue

            total_nodes += result["nodes"]

            if result["limit_hit"]:
                any_limit = True

            print(
                f"[torus done] W={W:2d} H={H:2d} "
                f"placements={result['placements']:7,} "
                f"nodes={result['nodes']:10,} "
                f"found={int(result['found'])} "
                f"limit={int(result['limit_hit'])}",
                flush=True,
            )

            if result["found"]:
                return {
                    "found": True,
                    "W": W,
                    "H": H,
                    "limit_hit": any_limit,
                    "nodes": total_nodes,
                }

    return {
        "found": False,
        "W": 0,
        "H": 0,
        "limit_hit": any_limit,
        "nodes": total_nodes,
    }


# ------------------------------------------------------------
# Independent patch growth
# ------------------------------------------------------------

def within_radius(c, radius):
    x, y, _ = c
    return abs(x) <= radius and abs(y) <= radius and abs(x + y) <= radius * 2


def translate(ori, tx, ty):
    return [(x + tx, y + ty, o) for x, y, o in ori]


def placement_fits(ori, tx, ty, radius, occupied):
    for x, y, o in ori:
        z = (x + tx, y + ty, o)

        if not within_radius(z, radius):
            return False

        if z in occupied:
            return False

    return True


def contact_score(tile, occupied):
    score = 0

    for c in tile:
        for d in range(3):
            if neighbour(c, d) in occupied:
                score += 1

    return score


def collect_boundary_targets(occ_list, occupied, radius):
    targets = []
    seen = set()

    for c in occ_list:
        for d in range(3):
            n = neighbour(c, d)

            if not within_radius(n, radius):
                continue

            if n in occupied:
                continue

            if n not in seen:
                seen.add(n)
                targets.append(n)

    return targets


def add_tile(tile, occupied, occ_list):
    for c in tile:
        if c not in occupied:
            occupied.add(c)
            occ_list.append(c)


def enumerate_valid_placements_for_target(orientations, target, radius, occupied, max_count=2):
    count = 0
    only_tile = None

    tx0, ty0, to = target

    for ori in orientations:
        for ax, ay, ao in ori:
            if ao != to:
                continue

            tx = tx0 - ax
            ty = ty0 - ay

            if placement_fits(ori, tx, ty, radius, occupied):
                count += 1

                if count == 1:
                    only_tile = translate(ori, tx, ty)

                if count >= max_count:
                    return count, only_tile

    return count, only_tile


def best_boundary_placement(orientations, target, radius, occupied, rng):
    best_tile = None
    best_score = -1

    tx0, ty0, to = target

    for ori in orientations:
        for ax, ay, ao in ori:
            if ao != to:
                continue

            tx = tx0 - ax
            ty = ty0 - ay

            if not placement_fits(ori, tx, ty, radius, occupied):
                continue

            tile = translate(ori, tx, ty)
            score = contact_score(tile, occupied) * 1000 + rng.randrange(997)

            if score > best_score:
                best_score = score
                best_tile = tile

    return best_tile


def forced_patch_growth(orientations, radius, trials, seed, progress_every):
    rng = random.Random(seed)
    best_tiles = 0
    best_cells = 0

    PROGRESS.update(
        phase="patch growth",
        detail=f"radius={radius} trials={trials}",
        nodes=0,
        best_tiles=0,
        trial=0,
        total_trials=trials,
    )

    for trial in range(1, trials + 1):
        occupied = set()
        occ_list = []

        first = translate(rng.choice(orientations), 0, 0)
        add_tile(first, occupied, occ_list)

        placed_count = 1
        max_steps = radius * radius * 20
        stuck = 0

        for step in range(1, max_steps + 1):
            if stuck >= 400:
                break

            if step % 200 == 0:
                PROGRESS.update(
                    detail=f"radius={radius} trial={trial}/{trials} step={step} placed={placed_count}",
                    best_tiles=best_tiles,
                    trial=trial,
                    total_trials=trials,
                )

            targets = collect_boundary_targets(occ_list, occupied, radius)

            if not targets:
                break

            rng.shuffle(targets)
            scan_limit = min(len(targets), 160)
            placed = False

            # First try genuinely forced placements.
            for target in targets[:scan_limit]:
                count, tile = enumerate_valid_placements_for_target(
                    orientations, target, radius, occupied, max_count=2
                )

                if count == 1 and tile is not None:
                    add_tile(tile, occupied, occ_list)
                    placed_count += 1
                    placed = True
                    stuck = 0
                    break

            # Fallback to strongest boundary placement.
            if not placed:
                for target in targets[:scan_limit]:
                    tile = best_boundary_placement(
                        orientations, target, radius, occupied, rng
                    )

                    if tile is not None:
                        add_tile(tile, occupied, occ_list)
                        placed_count += 1
                        placed = True
                        stuck = 0
                        break

            if not placed:
                stuck += 1

        if placed_count > best_tiles:
            best_tiles = placed_count
            best_cells = len(occ_list)

            print(
                f"[new best patch] tiles={best_tiles:,} "
                f"cells={best_cells:,} "
                f"trial={trial}/{trials}",
                flush=True,
            )

        if trial % progress_every == 0 or trial == trials:
            print(
                f"[patch progress] trial={trial}/{trials} "
                f"latest_tiles={placed_count:,} "
                f"best_tiles={best_tiles:,} "
                f"best_cells={best_cells:,}",
                flush=True,
            )

        PROGRESS.update(
            detail=f"radius={radius} trial={trial}/{trials}",
            best_tiles=best_tiles,
            trial=trial,
            total_trials=trials,
        )

    return best_tiles, best_cells


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Independent Python finite test for one triangular-lattice monotile candidate."
    )

    ap.add_argument("candidate_file")
    ap.add_argument("--max-period", type=int, default=14)
    ap.add_argument("--nodes", type=int, default=500000)
    ap.add_argument("--radius", type=int, default=70)
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--skip-patch", action="store_true")
    ap.add_argument("--skip-periodic", action="store_true")
    ap.add_argument("--progress-every", type=int, default=10)
    ap.add_argument("--heartbeat-sec", type=int, default=10)

    args = ap.parse_args()

    seed = args.seed if args.seed is not None else int(time.time_ns() & 0xFFFFFFFFFFFF)

    heartbeat = threading.Thread(
        target=PROGRESS.heartbeat_loop,
        args=(args.heartbeat_sec,),
        daemon=True,
    )
    heartbeat.start()

    try:
        PROGRESS.update(phase="loading candidate", detail=args.candidate_file)

        shape = parse_candidate(args.candidate_file)
        ori_ref = all_orientations(shape, allow_reflections=True)
        ori_no_ref = all_orientations(shape, allow_reflections=False)

        print("=== Independent Python Tile Test With Progress ===")
        print(f"candidate_file: {args.candidate_file}")
        print(f"cells: {len(shape)}")
        print(f"connected: {connected(shape)}")
        print(f"orientations_with_reflections: {len(ori_ref)}")
        print(f"orientations_no_reflection: {len(ori_no_ref)}")
        print(f"seed: {seed}")
        print(f"heartbeat_sec: {args.heartbeat_sec}")
        print()

        if not args.skip_periodic:
            print("=== Periodic torus search: reflections allowed ===", flush=True)
            r1 = search_periodic(ori_ref, args.max_period, args.nodes)
            print()
            print("reflection_periodic_found:", int(r1["found"]))
            print("reflection_periodic_W:", r1["W"])
            print("reflection_periodic_H:", r1["H"])
            print("reflection_periodic_limit_hit:", int(r1["limit_hit"]))
            print("reflection_torus_nodes:", r1["nodes"])
            print()

            print("=== Periodic torus search: no reflections ===", flush=True)
            r2 = search_periodic(ori_no_ref, args.max_period, args.nodes)
            print()
            print("no_reflection_periodic_found:", int(r2["found"]))
            print("no_reflection_periodic_W:", r2["W"])
            print("no_reflection_periodic_H:", r2["H"])
            print("no_reflection_periodic_limit_hit:", int(r2["limit_hit"]))
            print("no_reflection_torus_nodes:", r2["nodes"])
            print()

        if not args.skip_patch:
            print("=== Independent forced/boundary patch growth ===", flush=True)
            best_tiles, best_cells = forced_patch_growth(
                ori_ref,
                args.radius,
                args.trials,
                seed,
                max(1, args.progress_every),
            )
            print()
            print("python_forced_patch_tiles:", best_tiles)
            print("python_forced_patch_cells:", best_cells)

        print()
        print("Done.")

    finally:
        PROGRESS.stop()


if __name__ == "__main__":
    main()