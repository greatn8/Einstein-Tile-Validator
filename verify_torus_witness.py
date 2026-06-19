#!/usr/bin/env python3
"""Independent verifier for a periodic torus witness produced by einstein_torus_audit.

It independently reparses the tile, regenerates its allowed orientations, rebuilds
all listed translated placements, verifies the optional cell-level CSV, and checks
that every triangular cell of the W x H torus is covered exactly once.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, List, Tuple

Cell = Tuple[int, int, int]
Vertex = Tuple[int, int]


def normalize(cells: Iterable[Cell]) -> Tuple[Cell, ...]:
    items = list(cells)
    if not items:
        return tuple()
    min_x = min(x for x, _, _ in items)
    min_y = min(y for _, y, _ in items)
    return tuple(sorted(set((x - min_x, y - min_y, o) for x, y, o in items)))


def parse_candidate(path: Path) -> Tuple[Cell, ...]:
    in_cells = False
    cells: List[Cell] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if "normalized_cells" in raw:
            in_cells = True
            continue
        if not in_cells:
            continue
        parts = raw.split()
        if len(parts) != 3:
            continue
        try:
            x, y, o = map(int, parts)
        except ValueError:
            continue
        if o in (0, 1):
            cells.append((x, y, o))
    result = normalize(cells)
    if not result:
        raise ValueError(f"No normalized_cells could be read from {path}")
    return result


def vertices_of(cell: Cell) -> Tuple[Vertex, Vertex, Vertex]:
    x, y, o = cell
    if o == 0:
        return ((x, y), (x + 1, y), (x, y + 1))
    return ((x + 1, y + 1), (x + 1, y), (x, y + 1))


def rotate60(p: Vertex) -> Vertex:
    a, b = p
    return (-b, a + b)


def transform_vertex(p: Vertex, rot: int, reflect: bool) -> Vertex:
    a, b = p
    if reflect:
        a, b = b, a
    for _ in range(rot):
        a, b = rotate60((a, b))
    return a, b


def cell_from_vertices(vertices: Tuple[Vertex, Vertex, Vertex]) -> Cell:
    min_x = min(x for x, _ in vertices)
    min_y = min(y for _, y in vertices)
    up = (min_x, min_y, 0)
    if set(vertices) == set(vertices_of(up)):
        return up
    down = (min_x, min_y, 1)
    if set(vertices) == set(vertices_of(down)):
        return down
    raise ValueError(f"Could not map transformed triangle to a lattice cell: {vertices}")


def transform_shape(shape: Tuple[Cell, ...], rot: int, reflect: bool) -> Tuple[Cell, ...]:
    transformed: List[Cell] = []
    for cell in shape:
        vertices = tuple(transform_vertex(v, rot, reflect) for v in vertices_of(cell))
        transformed.append(cell_from_vertices(vertices))
    return normalize(transformed)


def all_orientations(shape: Tuple[Cell, ...], allow_reflections: bool) -> List[Tuple[Cell, ...]]:
    result: List[Tuple[Cell, ...]] = []
    seen = set()
    for reflect_flag in range(2 if allow_reflections else 1):
        for rotation in range(6):
            orient = transform_shape(shape, rotation, bool(reflect_flag))
            if orient not in seen:
                seen.add(orient)
                result.append(orient)
    return result


def wrapped_cells(orientation: Tuple[Cell, ...], tx: int, ty: int, W: int, H: int) -> List[Cell]:
    return [((x + tx) % W, (y + ty) % H, o) for x, y, o in orientation]


def parse_witness_cells(path: Path) -> dict[int, List[Cell]]:
    by_tile: dict[int, List[Cell]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            by_tile[int(row["tile_index"])].append(
                (int(row["x"]), int(row["y"]), int(row["triangle_orientation"]))
            )
    return dict(by_tile)


def main() -> int:
    parser = argparse.ArgumentParser(description="Independently verify a periodic torus witness.")
    parser.add_argument("--input", required=True, type=Path, help="candidate_0000858.txt")
    parser.add_argument("--placements", required=True, type=Path, help="*_placements.csv")
    parser.add_argument("--cells", required=True, type=Path, help="*_cells.csv")
    parser.add_argument("--W", required=True, type=int)
    parser.add_argument("--H", required=True, type=int)
    parser.add_argument("--mode", required=True, choices=["reflect", "no-reflect"])
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    shape = parse_candidate(args.input)
    orientations = all_orientations(shape, args.mode == "reflect")
    witness_cells = parse_witness_cells(args.cells)

    board_cells = 2 * args.W * args.H
    tile_cells = len(shape)
    expected_tiles, remainder = divmod(board_cells, tile_cells)
    failures: List[str] = []

    if remainder:
        failures.append(f"Board has {board_cells} cells, not divisible by tile area {tile_cells}")

    occupancy: Counter[Cell] = Counter()
    seen_tile_indices = set()
    placement_rows = []
    with args.placements.open(newline="", encoding="utf-8") as handle:
        placement_rows = list(csv.DictReader(handle))

    for row in placement_rows:
        tile_index = int(row["tile_index"])
        orient_index = int(row["orientation_index"])
        tx = int(row["tx"])
        ty = int(row["ty"])
        declared_count = int(row["cell_count"])

        if tile_index in seen_tile_indices:
            failures.append(f"Duplicate tile_index {tile_index} in placement CSV")
            continue
        seen_tile_indices.add(tile_index)

        if not (0 <= orient_index < len(orientations)):
            failures.append(f"Tile {tile_index}: invalid orientation_index {orient_index}")
            continue

        generated = wrapped_cells(orientations[orient_index], tx, ty, args.W, args.H)
        generated_set = set(generated)
        if len(generated) != tile_cells or len(generated_set) != tile_cells:
            failures.append(f"Tile {tile_index}: self-overlap after wrapping")
        if declared_count != tile_cells:
            failures.append(f"Tile {tile_index}: declared cell_count {declared_count}, expected {tile_cells}")

        saved = witness_cells.get(tile_index, [])
        if len(saved) != tile_cells:
            failures.append(f"Tile {tile_index}: cell CSV has {len(saved)} rows, expected {tile_cells}")
        elif set(saved) != generated_set:
            failures.append(f"Tile {tile_index}: regenerated cells do not match exported cell CSV")

        occupancy.update(generated)

    if len(placement_rows) != expected_tiles:
        failures.append(f"Placement count {len(placement_rows)}, expected {expected_tiles}")

    expected_board = {(x, y, o) for x in range(args.W) for y in range(args.H) for o in (0, 1)}
    occupied_board = set(occupancy)
    missing = expected_board - occupied_board
    outside = occupied_board - expected_board
    multiply = {cell: count for cell, count in occupancy.items() if count != 1}

    if missing:
        failures.append(f"{len(missing)} uncovered board cells")
    if outside:
        failures.append(f"{len(outside)} out-of-board cells")
    if multiply:
        failures.append(f"{len(multiply)} cells have coverage count other than 1")

    lines = [
        "Independent periodic witness verification",
        "=======================================",
        f"candidate: {args.input}",
        f"mode: {args.mode}",
        f"W,H: {args.W},{args.H}",
        f"board_cells: {board_cells}",
        f"tile_cells: {tile_cells}",
        f"orientations_regenerated: {len(orientations)}",
        f"placements_checked: {len(placement_rows)}",
        f"expected_placements: {expected_tiles}",
        f"cells_covered_once: {sum(1 for cell in expected_board if occupancy[cell] == 1)}",
        f"missing_cells: {len(missing)}",
        f"multiply_covered_cells: {len(multiply)}",
    ]

    if failures:
        lines.append("RESULT: FAIL")
        lines.append("Failures:")
        lines.extend(f"- {item}" for item in failures)
        exit_code = 1
    else:
        lines.append("RESULT: PASS")
        lines.append("The listed placements are legal transformed copies of the candidate and cover each wrapped triangular cell exactly once.")
        exit_code = 0

    report = "\n".join(lines) + "\n"
    print(report, end="")
    if args.report:
        args.report.write_text(report, encoding="utf-8")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
