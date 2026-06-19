#!/usr/bin/env python3
"""
Independent forced-structure probe for a triangular-lattice monotile candidate.

This is a research aid, not a proof engine. It grows finite patches using local
placement rules, records placements that were locally unique relative to the
current patch boundary, samples boundary option counts, and looks for repeated
local placement neighbourhoods.

Outputs:
  summary.txt
  tile.svg
  best_patch.svg
  placements.csv
  forced_events.csv
  boundary_option_counts.csv
  repeated_cluster_report.txt
  snapshots/*.svg

The phrase "forced" here means: exactly one legal placement covering a chosen
boundary cell, given the current finite patch and the probe's finite radius.
It does NOT prove that the placement is globally forced in every infinite tiling.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

sys.setrecursionlimit(1_000_000)

Cell = Tuple[int, int, int]  # x, y, orientation; orientation 0 = up, 1 = down


# -----------------------------------------------------------------------------
# Progress helpers
# -----------------------------------------------------------------------------

class Reporter:
    def __init__(self, heartbeat_seconds: int) -> None:
        self.start = time.time()
        self.last_heartbeat = self.start
        self.heartbeat_seconds = max(1, heartbeat_seconds)

    @staticmethod
    def elapsed_text(seconds: float) -> str:
        seconds = int(seconds)
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        if hours:
            return f"{hours}h{minutes:02d}m{secs:02d}s"
        if minutes:
            return f"{minutes}m{secs:02d}s"
        return f"{secs}s"

    def log(self, text: str) -> None:
        print(text, flush=True)

    def heartbeat(self, phase: str, detail: str) -> None:
        now = time.time()
        if now - self.last_heartbeat >= self.heartbeat_seconds:
            self.last_heartbeat = now
            self.log(
                f"[heartbeat] elapsed={self.elapsed_text(now - self.start)} "
                f"phase={phase} {detail}"
            )


# -----------------------------------------------------------------------------
# Triangular lattice geometry and tile orientations
# -----------------------------------------------------------------------------

def neighbour(cell: Cell, side: int) -> Cell:
    """Return the triangle sharing the requested edge of a triangular cell."""
    x, y, orient = cell
    side %= 3

    if orient == 0:  # upward-pointing triangle
        if side == 0:
            return (x, y, 1)
        if side == 1:
            return (x - 1, y, 1)
        return (x, y - 1, 1)

    # downward-pointing triangle
    if side == 0:
        return (x, y, 0)
    if side == 1:
        return (x + 1, y, 0)
    return (x, y + 1, 0)


def normalize(cells: Iterable[Cell]) -> List[Cell]:
    cells = list(set(cells))
    if not cells:
        return []
    min_x = min(x for x, _, _ in cells)
    min_y = min(y for _, y, _ in cells)
    return sorted((x - min_x, y - min_y, orient) for x, y, orient in cells)


def key_of(cells: Sequence[Cell]) -> str:
    return f"{len(cells)}:" + "".join(f"{x},{y},{orient};" for x, y, orient in normalize(cells))


def vertices_of_cell(cell: Cell) -> List[Tuple[int, int]]:
    x, y, orient = cell
    if orient == 0:
        return [(x, y), (x + 1, y), (x, y + 1)]
    return [(x + 1, y + 1), (x + 1, y), (x, y + 1)]


def rotate60(vertex: Tuple[int, int]) -> Tuple[int, int]:
    a, b = vertex
    return (-b, a + b)


def transform_vertex(vertex: Tuple[int, int], rotation: int, reflect: bool) -> Tuple[int, int]:
    a, b = vertex
    if reflect:
        a, b = b, a
    for _ in range(rotation % 6):
        a, b = rotate60((a, b))
    return a, b


def cell_from_vertices(vertices: Sequence[Tuple[int, int]]) -> Cell:
    min_x = min(x for x, _ in vertices)
    min_y = min(y for _, y in vertices)

    up = (min_x, min_y, 0)
    if sorted(vertices) == sorted(vertices_of_cell(up)):
        return up

    down = (min_x, min_y, 1)
    if sorted(vertices) == sorted(vertices_of_cell(down)):
        return down

    raise ValueError(f"Could not map transformed vertices to a grid cell: {vertices}")


def transform_shape(shape: Sequence[Cell], rotation: int, reflect: bool) -> List[Cell]:
    transformed: List[Cell] = []
    for cell in shape:
        transformed_vertices = [
            transform_vertex(vertex, rotation, reflect)
            for vertex in vertices_of_cell(cell)
        ]
        transformed.append(cell_from_vertices(transformed_vertices))
    return normalize(transformed)


def all_orientations(shape: Sequence[Cell], allow_reflections: bool) -> List[List[Cell]]:
    orientations: List[List[Cell]] = []
    seen: Set[str] = set()

    for reflected in range(2 if allow_reflections else 1):
        for rotation in range(6):
            transformed = transform_shape(shape, rotation, bool(reflected))
            signature = key_of(transformed)
            if signature not in seen:
                seen.add(signature)
                orientations.append(transformed)

    return orientations


def parse_candidate(path: Path) -> List[Cell]:
    """Read the normalized_cells section from either a candidate or validation report."""
    cells: List[Cell] = []
    in_cells = False

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if "normalized_cells" in line:
            in_cells = True
            continue

        if in_cells:
            parts = line.split()
            if len(parts) >= 3:
                try:
                    x, y, orient = map(int, parts[:3])
                except ValueError:
                    continue
                cells.append((x, y, orient))

    if not cells:
        raise ValueError(
            "No normalized_cells section was found. Provide the original candidate_*.txt "
            "file or a compatible validation report."
        )

    return normalize(cells)


def is_connected(shape: Sequence[Cell]) -> bool:
    if not shape:
        return False
    shape_set = set(shape)
    seen = {shape[0]}
    stack = [shape[0]]

    while stack:
        current = stack.pop()
        for side in range(3):
            nxt = neighbour(current, side)
            if nxt in shape_set and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)

    return len(seen) == len(shape_set)


# -----------------------------------------------------------------------------
# Placement and patch growth
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Placement:
    orientation: int
    tx: int
    ty: int
    method: str
    step: int
    trial: int


def translated_cells(orientations: Sequence[Sequence[Cell]], orientation: int, tx: int, ty: int) -> List[Cell]:
    return [(x + tx, y + ty, orient) for x, y, orient in orientations[orientation]]


def within_radius(cell: Cell, radius: int) -> bool:
    x, y, _ = cell
    # Hex-like axial distance in this coordinate system, allowing a slightly
    # wider diagonal so triangular-cell patches are not overly clipped.
    return max(abs(x), abs(y), abs(x + y)) <= radius * 2


def placement_fits(
    orientations: Sequence[Sequence[Cell]],
    orientation: int,
    tx: int,
    ty: int,
    radius: int,
    occupied: Set[Cell],
) -> bool:
    for x, y, orient in orientations[orientation]:
        cell = (x + tx, y + ty, orient)
        if not within_radius(cell, radius) or cell in occupied:
            return False
    return True


def placement_contact_score(tile_cells: Sequence[Cell], occupied: Set[Cell]) -> int:
    score = 0
    for cell in tile_cells:
        for side in range(3):
            if neighbour(cell, side) in occupied:
                score += 1
    return score


def collect_boundary_targets(occupied_cells: Sequence[Cell], occupied: Set[Cell], radius: int) -> List[Cell]:
    targets: List[Cell] = []
    seen: Set[Cell] = set()

    for cell in occupied_cells:
        for side in range(3):
            target = neighbour(cell, side)
            if target in occupied or target in seen or not within_radius(target, radius):
                continue
            seen.add(target)
            targets.append(target)

    return targets


def placements_covering_target(
    orientations: Sequence[Sequence[Cell]],
    target: Cell,
    radius: int,
    occupied: Set[Cell],
    cap: int = 3,
) -> List[Tuple[int, int, int]]:
    """Find distinct legal placements which cover one boundary target cell."""
    target_x, target_y, target_orient = target
    answers: List[Tuple[int, int, int]] = []
    seen: Set[Tuple[int, int, int]] = set()

    for orientation, shape in enumerate(orientations):
        for anchor_x, anchor_y, anchor_orient in shape:
            if anchor_orient != target_orient:
                continue
            tx = target_x - anchor_x
            ty = target_y - anchor_y
            descriptor = (orientation, tx, ty)
            if descriptor in seen:
                continue
            seen.add(descriptor)

            if placement_fits(orientations, orientation, tx, ty, radius, occupied):
                answers.append(descriptor)
                if len(answers) >= cap:
                    return answers

    return answers


def add_placement(
    placement: Placement,
    orientations: Sequence[Sequence[Cell]],
    occupied: Set[Cell],
    occupied_cells: List[Cell],
) -> None:
    for cell in translated_cells(orientations, placement.orientation, placement.tx, placement.ty):
        if cell not in occupied:
            occupied.add(cell)
            occupied_cells.append(cell)


def choose_fallback_placement(
    orientations: Sequence[Sequence[Cell]],
    targets: Sequence[Cell],
    radius: int,
    occupied: Set[Cell],
    rng: random.Random,
    scan_limit: int,
    reporter: Reporter | None = None,
    trial: int = 0,
    step: int = 0,
) -> Tuple[int, int, int] | None:
    """Choose a legal boundary placement with high shared-edge contact."""
    best: Tuple[int, int, int] | None = None
    best_score = -1

    for index, target in enumerate(targets[:scan_limit], start=1):
        if reporter is not None and index % 10 == 0:
            reporter.heartbeat(
                "fallback scan",
                f"trial={trial} step={step} scanned={index}/{min(scan_limit, len(targets))} boundary={len(targets)}",
            )
        candidates = placements_covering_target(
            orientations, target, radius, occupied, cap=24
        )
        for orientation, tx, ty in candidates:
            cells = translated_cells(orientations, orientation, tx, ty)
            # Random tie-break avoids a completely deterministic spiral.
            score = placement_contact_score(cells, occupied) * 10_000 + rng.randrange(10_000)
            if score > best_score:
                best_score = score
                best = (orientation, tx, ty)

    return best


def sampled_boundary_options(
    orientations: Sequence[Sequence[Cell]],
    targets: Sequence[Cell],
    radius: int,
    occupied: Set[Cell],
    sample_count: int,
    rng: random.Random,
) -> Tuple[int, int, int, int]:
    """Return sampled target counts for zero, one, two-plus legal placements."""
    if not targets:
        return 0, 0, 0, 0

    sample = list(targets)
    rng.shuffle(sample)
    sample = sample[: min(sample_count, len(sample))]

    zero = one = two_plus = 0
    for target in sample:
        options = placements_covering_target(orientations, target, radius, occupied, cap=2)
        if len(options) == 0:
            zero += 1
        elif len(options) == 1:
            one += 1
        else:
            two_plus += 1

    return len(sample), zero, one, two_plus


# -----------------------------------------------------------------------------
# SVG output
# -----------------------------------------------------------------------------

def lattice_to_xy(vertex: Tuple[int, int], scale: float, pad: float, min_x: int, min_y: int) -> Tuple[float, float]:
    a, b = vertex
    x = (a + 0.5 * b - min_x) * scale + pad
    y = (math.sqrt(3.0) * 0.5 * b - min_y) * scale + pad
    return x, y


def tile_boundary_edges(cells: Sequence[Cell]) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    edges: Counter[Tuple[Tuple[int, int], Tuple[int, int]]] = Counter()
    for cell in cells:
        vertices = vertices_of_cell(cell)
        for i in range(3):
            a = vertices[i]
            b = vertices[(i + 1) % 3]
            if a > b:
                a, b = b, a
            edges[(a, b)] += 1
    return [edge for edge, count in edges.items() if count == 1]


def write_svg(
    out_path: Path,
    orientations: Sequence[Sequence[Cell]],
    placements: Sequence[Placement],
    title: str,
    line_width: float = 1.0,
) -> None:
    all_cells: List[Cell] = []
    for placement in placements:
        all_cells.extend(translated_cells(orientations, placement.orientation, placement.tx, placement.ty))

    if not all_cells:
        out_path.write_text("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"100\" height=\"100\"/>\n", encoding="utf-8")
        return

    all_vertices = [vertex for cell in all_cells for vertex in vertices_of_cell(cell)]
    euclidean_x = [a + 0.5 * b for a, b in all_vertices]
    euclidean_y = [math.sqrt(3.0) * 0.5 * b for a, b in all_vertices]

    min_x = math.floor(min(euclidean_x))
    min_y = math.floor(min(euclidean_y))
    max_x = math.ceil(max(euclidean_x))
    max_y = math.ceil(max(euclidean_y))

    pad = 24.0
    max_dim = max(max_x - min_x, max_y - min_y, 1)
    scale = min(16.0, max(1.8, 1800.0 / max_dim))
    width = int((max_x - min_x) * scale + 2 * pad)
    height = int((max_y - min_y) * scale + 2 * pad)

    forced_colour = "#0f766e"
    fallback_colour = "#7c3aed"
    seed_colour = "#b45309"

    lines: List[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="16" y="20" font-family="monospace" font-size="14" fill="#111827">{title}</text>',
    ]

    for placement in placements:
        cells = translated_cells(orientations, placement.orientation, placement.tx, placement.ty)
        if placement.method == "forced":
            colour = forced_colour
        elif placement.method == "seed":
            colour = seed_colour
        else:
            colour = fallback_colour

        for a, b in tile_boundary_edges(cells):
            x1, y1 = lattice_to_xy(a, scale, pad, min_x, min_y)
            x2, y2 = lattice_to_xy(b, scale, pad, min_x, min_y)
            lines.append(
                f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
                f'stroke="{colour}" stroke-width="{line_width}" stroke-linecap="round"/>'
            )

    legend_y = height - 12
    lines.extend([
        f'<text x="16" y="{legend_y}" font-family="monospace" font-size="12" fill="{seed_colour}">seed</text>',
        f'<text x="76" y="{legend_y}" font-family="monospace" font-size="12" fill="{forced_colour}">locally-unique boundary placement</text>',
        f'<text x="340" y="{legend_y}" font-family="monospace" font-size="12" fill="{fallback_colour}">heuristic fallback placement</text>',
        '</svg>',
    ])

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# -----------------------------------------------------------------------------
# Repeated local-cluster signatures
# -----------------------------------------------------------------------------

def axial_distance(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return max(abs(dx), abs(dy), abs(dx + dy))


def local_cluster_counter(placements: Sequence[Placement], radius: int) -> Counter[str]:
    """Count exact translation-normalised neighbourhoods of tile anchors.

    This is deliberately conservative: a signature only matches when the same
    orientation indices and relative anchor positions appear. It is a clue for
    recurring structure, not a detection of a formal substitution rule.
    """
    counter: Counter[str] = Counter()

    anchors = [(placement.tx, placement.ty) for placement in placements]
    for centre_index, centre in enumerate(placements):
        centre_anchor = anchors[centre_index]
        members: List[Tuple[int, int, int]] = []

        for other_index, other in enumerate(placements):
            other_anchor = anchors[other_index]
            if axial_distance(centre_anchor, other_anchor) <= radius:
                members.append(
                    (
                        other_anchor[0] - centre_anchor[0],
                        other_anchor[1] - centre_anchor[1],
                        other.orientation,
                    )
                )

        signature = ";".join(f"{dx},{dy},o{orient}" for dx, dy, orient in sorted(members))
        counter[signature] += 1

    return counter


def write_repeated_cluster_report(out_path: Path, placements: Sequence[Placement]) -> None:
    lines = [
        "Repeated local cluster report",
        "============================",
        "",
        "A repeated signature is a translation-normalised neighbourhood of tile anchors.",
        "It is a heuristic clue only. It does not establish a supertile or proof of aperiodicity.",
        "",
        f"placements analysed: {len(placements)}",
        "",
    ]

    for radius in (1, 2, 3):
        counts = local_cluster_counter(placements, radius)
        lines.append(f"Neighbourhood radius {radius}")
        lines.append("-" * 28)
        for rank, (signature, count) in enumerate(counts.most_common(12), start=1):
            member_count = 0 if not signature else signature.count(";") + 1
            preview = signature[:500]
            lines.append(f"{rank:2d}. occurrences={count:4d} tiles_in_signature={member_count:3d}")
            lines.append(f"    {preview}")
        lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# -----------------------------------------------------------------------------
# Main probe
# -----------------------------------------------------------------------------

def run_trial(
    orientations: Sequence[Sequence[Cell]],
    radius: int,
    max_tiles: int,
    trial: int,
    seed: int,
    reporter: Reporter,
    snapshot_callback,
    boundary_sample_every: int,
    boundary_sample_size: int,
    forced_scan_limit: int,
) -> Tuple[List[Placement], List[Dict[str, int]]]:
    rng = random.Random(seed)
    occupied: Set[Cell] = set()
    occupied_cells: List[Cell] = []
    placements: List[Placement] = []
    boundary_rows: List[Dict[str, int]] = []

    initial_orientation = rng.randrange(len(orientations))
    seed_placement = Placement(initial_orientation, 0, 0, "seed", 1, trial)
    add_placement(seed_placement, orientations, occupied, occupied_cells)
    placements.append(seed_placement)

    fallback_streak = 0
    next_snapshot = snapshot_callback.snapshot_every if snapshot_callback is not None else max_tiles + 1

    while len(placements) < max_tiles:
        step = len(placements) + 1
        reporter.heartbeat(
            "patch growth",
            f"trial={trial} step={step}/{max_tiles} tiles={len(placements)} occupied_cells={len(occupied_cells)}",
        )
        targets = collect_boundary_targets(occupied_cells, occupied, radius)
        if not targets:
            reporter.log(f"[trial {trial}] no boundary targets remain at step={step}")
            break

        rng.shuffle(targets)
        scan_targets = targets[: min(forced_scan_limit, len(targets))]
        chosen: Tuple[int, int, int] | None = None
        method = ""

        # Prefer a boundary cell that has exactly one local legal continuation.
        for scan_index, target in enumerate(scan_targets, start=1):
            if scan_index % 10 == 0:
                reporter.heartbeat(
                    "forced scan",
                    f"trial={trial} step={step} scanned={scan_index}/{len(scan_targets)} boundary={len(targets)}",
                )
            options = placements_covering_target(orientations, target, radius, occupied, cap=2)
            if len(options) == 1:
                chosen = options[0]
                method = "forced"
                fallback_streak = 0
                break

        if chosen is None:
            chosen = choose_fallback_placement(
                orientations,
                scan_targets,
                radius,
                occupied,
                rng,
                scan_limit=len(scan_targets),
                reporter=reporter,
                trial=trial,
                step=step,
            )
            method = "fallback"
            fallback_streak += 1

        if chosen is None:
            reporter.log(f"[trial {trial}] no legal placement found at step={step}")
            break

        orientation, tx, ty = chosen
        placement = Placement(orientation, tx, ty, method, step, trial)
        add_placement(placement, orientations, occupied, occupied_cells)
        placements.append(placement)

        if step % 100 == 0 or method == "forced" and step % 25 == 0:
            forced_count = sum(1 for item in placements if item.method == "forced")
            reporter.heartbeat(
                "patch growth",
                f"trial={trial} step={step}/{max_tiles} occupied_tiles={len(placements)} "
                f"forced={forced_count} fallback_streak={fallback_streak} boundary={len(targets)}",
            )

        if boundary_sample_every > 0 and step % boundary_sample_every == 0:
            sample_count, zero, one, two_plus = sampled_boundary_options(
                orientations,
                targets,
                radius,
                occupied,
                boundary_sample_size,
                rng,
            )
            boundary_rows.append({
                "trial": trial,
                "step": step,
                "boundary_cells": len(targets),
                "sampled_targets": sample_count,
                "zero_options": zero,
                "one_option": one,
                "two_or_more_options": two_plus,
                "forced_placements_so_far": sum(1 for item in placements if item.method == "forced"),
                "fallback_placements_so_far": sum(1 for item in placements if item.method == "fallback"),
            })
            reporter.log(
                f"[boundary sample] trial={trial} step={step} sampled={sample_count} "
                f"zero={zero} one={one} two_plus={two_plus} boundary={len(targets)}"
            )

        if snapshot_callback is not None and len(placements) >= next_snapshot:
            snapshot_callback(placements)
            next_snapshot += max(1, snapshot_callback.snapshot_every)

        # No valid placement has been found in a while; continuing is unlikely
        # to be useful for a finite greedy probe.
        if fallback_streak >= 600:
            reporter.log(f"[trial {trial}] stopping after {fallback_streak} consecutive fallback placements")
            break

    return placements, boundary_rows


def write_csv_outputs(out_dir: Path, placements: Sequence[Placement], boundary_rows: Sequence[Dict[str, int]]) -> None:
    with (out_dir / "placements.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["trial", "step", "method", "orientation", "tx", "ty"],
        )
        writer.writeheader()
        for item in placements:
            writer.writerow({
                "trial": item.trial,
                "step": item.step,
                "method": item.method,
                "orientation": item.orientation,
                "tx": item.tx,
                "ty": item.ty,
            })

    with (out_dir / "forced_events.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["trial", "step", "orientation", "tx", "ty"],
        )
        writer.writeheader()
        for item in placements:
            if item.method == "forced":
                writer.writerow({
                    "trial": item.trial,
                    "step": item.step,
                    "orientation": item.orientation,
                    "tx": item.tx,
                    "ty": item.ty,
                })

    with (out_dir / "boundary_option_counts.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "trial", "step", "boundary_cells", "sampled_targets", "zero_options",
            "one_option", "two_or_more_options", "forced_placements_so_far",
            "fallback_placements_so_far",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(boundary_rows)


class SnapshotWriter:
    def __init__(self, out_dir: Path, orientations: Sequence[Sequence[Cell]], every: int) -> None:
        self.out_dir = out_dir
        self.orientations = orientations
        self.snapshot_every = max(1, every)
        self.last_written_tiles = 0

    def __call__(self, placements: Sequence[Placement]) -> None:
        count = len(placements)
        if count <= self.last_written_tiles:
            return
        self.last_written_tiles = count
        path = self.out_dir / f"patch_{count:05d}.svg"
        write_svg(
            path,
            self.orientations,
            placements,
            f"858 forced-structure probe snapshot: {count} tiles",
            line_width=0.8,
        )
        print(f"[snapshot] wrote {path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Independent local forced-structure probe for a triangular-lattice monotile candidate."
    )
    parser.add_argument("candidate_file", type=Path)
    parser.add_argument("--out", type=Path, default=Path("858_structure_probe"))
    parser.add_argument("--radius", type=int, default=90)
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--max-tiles", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=858)
    parser.add_argument("--allow-reflections", type=int, choices=[0, 1], default=1)
    parser.add_argument("--heartbeat-sec", type=int, default=15)
    parser.add_argument("--snapshot-every", type=int, default=500)
    parser.add_argument("--boundary-sample-every", type=int, default=200)
    parser.add_argument("--boundary-sample-size", type=int, default=80)
    parser.add_argument("--forced-scan-limit", type=int, default=160)
    args = parser.parse_args()

    if args.radius < 10:
        raise ValueError("--radius must be at least 10")
    if args.trials < 1:
        raise ValueError("--trials must be at least 1")
    if args.max_tiles < 2:
        raise ValueError("--max-tiles must be at least 2")

    out_dir: Path = args.out
    snapshots_dir = out_dir / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    reporter = Reporter(args.heartbeat_sec)
    shape = parse_candidate(args.candidate_file)
    orientations = all_orientations(shape, bool(args.allow_reflections))

    reporter.log("=== Independent Forced-Structure Probe ===")
    reporter.log(f"candidate_file: {args.candidate_file}")
    reporter.log(f"out: {out_dir}")
    reporter.log(f"cells: {len(shape)}")
    reporter.log(f"connected: {int(is_connected(shape))}")
    reporter.log(f"orientations: {len(orientations)}")
    reporter.log(f"allow_reflections: {args.allow_reflections}")
    reporter.log(f"radius: {args.radius}")
    reporter.log(f"trials: {args.trials}")
    reporter.log(f"max_tiles_per_trial: {args.max_tiles}")
    reporter.log(f"seed: {args.seed}")
    reporter.log("")
    reporter.log("NOTE: 'forced' means locally unique relative to this finite patch boundary.")
    reporter.log("It is evidence for structure, not a proof of a globally forced infinite tiling.")
    reporter.log("")

    # One-tile SVG uses a placement at the origin.
    write_svg(out_dir / "tile.svg", orientations, [Placement(0, 0, 0, "seed", 1, 0)], "Candidate tile")

    snapshot_writer = SnapshotWriter(snapshots_dir, orientations, args.snapshot_every)

    best_placements: List[Placement] = []
    best_boundary_rows: List[Dict[str, int]] = []
    best_forced_count = -1

    for trial in range(1, args.trials + 1):
        trial_seed = args.seed + trial * 1_000_003
        reporter.log(f"[trial start] {trial}/{args.trials} seed={trial_seed}")
        trial_start = time.time()

        placements, boundary_rows = run_trial(
            orientations=orientations,
            radius=args.radius,
            max_tiles=args.max_tiles,
            trial=trial,
            seed=trial_seed,
            reporter=reporter,
            snapshot_callback=snapshot_writer,
            boundary_sample_every=args.boundary_sample_every,
            boundary_sample_size=args.boundary_sample_size,
            forced_scan_limit=args.forced_scan_limit,
        )

        forced_count = sum(1 for item in placements if item.method == "forced")
        fallback_count = sum(1 for item in placements if item.method == "fallback")
        reporter.log(
            f"[trial done] {trial}/{args.trials} tiles={len(placements)} forced={forced_count} "
            f"fallback={fallback_count} elapsed={reporter.elapsed_text(time.time() - trial_start)}"
        )

        # Primary criterion is number of tiles; tie-break by forced placements.
        if (len(placements), forced_count) > (len(best_placements), best_forced_count):
            best_placements = list(placements)
            best_boundary_rows = list(boundary_rows)
            best_forced_count = forced_count
            write_svg(
                out_dir / "best_patch.svg",
                orientations,
                best_placements,
                f"Best patch: {len(best_placements)} tiles, {best_forced_count} locally-unique placements",
                line_width=0.8,
            )
            reporter.log(
                f"[new best] trial={trial} tiles={len(best_placements)} forced={best_forced_count} "
                f"wrote={out_dir / 'best_patch.svg'}"
            )

    write_csv_outputs(out_dir, best_placements, best_boundary_rows)
    write_repeated_cluster_report(out_dir / "repeated_cluster_report.txt", best_placements)

    method_counts = Counter(item.method for item in best_placements)
    summary = [
        "Independent forced-structure probe summary",
        "==========================================",
        "",
        f"candidate_file: {args.candidate_file}",
        f"cells: {len(shape)}",
        f"connected: {int(is_connected(shape))}",
        f"orientations: {len(orientations)}",
        f"allow_reflections: {args.allow_reflections}",
        f"radius: {args.radius}",
        f"trials: {args.trials}",
        f"max_tiles_per_trial: {args.max_tiles}",
        f"seed: {args.seed}",
        "",
        f"best_patch_tiles: {len(best_placements)}",
        f"seed_placements: {method_counts.get('seed', 0)}",
        f"locally_unique_boundary_placements: {method_counts.get('forced', 0)}",
        f"heuristic_fallback_placements: {method_counts.get('fallback', 0)}",
        "",
        "Interpretation:",
        "- A large best_patch_tiles value is finite patch-growth evidence only.",
        "- Locally unique placements are conditional on the current finite patch and probe radius.",
        "- Repeated cluster signatures are leads for manual inspection, not a substitution proof.",
        "- This script does not test every periodic tiling and cannot prove an einstein tile.",
        "",
        "Important files:",
        "- tile.svg: candidate tile drawing",
        "- best_patch.svg: largest probe patch; teal=locally unique, purple=fallback, brown=seed",
        "- placements.csv: all placements in the best patch",
        "- forced_events.csv: placements that were locally unique when selected",
        "- boundary_option_counts.csv: sampled option counts along the growing boundary",
        "- repeated_cluster_report.txt: repeated translation-normalised local anchor configurations",
    ]
    (out_dir / "summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")

    reporter.log("")
    reporter.log("=== Probe complete ===")
    reporter.log(f"best_patch_tiles: {len(best_placements)}")
    reporter.log(f"locally_unique_boundary_placements: {method_counts.get('forced', 0)}")
    reporter.log(f"heuristic_fallback_placements: {method_counts.get('fallback', 0)}")
    reporter.log(f"results: {out_dir.resolve()}")
    reporter.log("Read summary.txt and repeated_cluster_report.txt first.")


if __name__ == "__main__":
    main()
