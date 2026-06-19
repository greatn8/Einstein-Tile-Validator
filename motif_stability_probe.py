#!/usr/bin/env python3
"""
Motif Stability Probe for a triangular-lattice monotile candidate.

Purpose
-------
This is a follow-up research aid for a candidate such as candidate_0000858.txt.
It grows several finite patches under different construction policies and seeds,
then asks whether local placement motifs recur across independent runs.

It is deliberately conservative:
- It does NOT prove that a candidate tiles the plane.
- It does NOT prove that the candidate is an einstein tile.
- "locally unique" means exactly one placement was found in a *larger finite
  local window* around a boundary target; it is not a global forcing proof.
- Repeated motifs are evidence leads only. A real substitution proof would need
  to show that motifs are unavoidable in every valid infinite tiling and form
  an indefinitely extendable hierarchy.

Prerequisite
------------
Place this file in the same directory as forced_structure_probe.py from the
previous step. This script imports the triangular-grid geometry and SVG writer
from that file, which keeps both probes consistent.

Outputs
-------
<out>/
  summary.txt
  run_summary.csv
  stable_motifs.csv
  stable_motifs_report.txt
  runs/<run_label>/placements.csv
  runs/<run_label>/patch.svg
  runs/<run_label>/run_summary.txt
  runs/<run_label>/motifs_radius_<R>.csv

Example
-------
python3 motif_stability_probe.py candidate_0000858.txt \
  --out 858_motif_stability_screen \
  --seeds 858,271828,314159 \
  --modes forced,mixed,random \
  --reflection-modes 1 \
  --radius 60 --max-tiles 800 \
  --motif-radii 3,4,5,6
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Set, Tuple

# The previous probe is intentionally reused so both scripts share the exact
# same triangular-lattice conventions and SVG rendering.
try:
    from forced_structure_probe import (
        Cell,
        Placement,
        Reporter,
        add_placement,
        all_orientations,
        choose_fallback_placement,
        collect_boundary_targets,
        is_connected,
        parse_candidate,
        placement_fits,
        placements_covering_target,
        translated_cells,
        write_svg,
    )
except ImportError as exc:
    raise SystemExit(
        "Could not import forced_structure_probe.py. Put motif_stability_probe.py "
        "in the same directory as forced_structure_probe.py, then run it again.\n"
        f"Original import error: {exc}"
    )


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

VALID_MODES = ("forced", "mixed", "heuristic", "random")


def parse_int_csv(text: str, label: str) -> List[int]:
    values: List[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            values.append(int(part))
        except ValueError as exc:
            raise ValueError(f"{label} contains a non-integer value: {part!r}") from exc
    if not values:
        raise ValueError(f"{label} must contain at least one integer")
    return values


def parse_modes(text: str) -> List[str]:
    modes = [part.strip().lower() for part in text.split(",") if part.strip()]
    if not modes:
        raise ValueError("--modes must contain at least one mode")
    invalid = sorted(set(modes) - set(VALID_MODES))
    if invalid:
        raise ValueError(
            f"Unknown mode(s): {', '.join(invalid)}. "
            f"Allowed modes: {', '.join(VALID_MODES)}"
        )
    # Preserve order while removing accidental duplicates.
    return list(dict.fromkeys(modes))


def parse_reflection_modes(text: str) -> List[int]:
    values = parse_int_csv(text, "--reflection-modes")
    invalid = [value for value in values if value not in (0, 1)]
    if invalid:
        raise ValueError("--reflection-modes accepts only 0 and/or 1")
    return list(dict.fromkeys(values))


def parse_radii(text: str) -> List[int]:
    values = parse_int_csv(text, "--motif-radii")
    if any(value < 1 for value in values):
        raise ValueError("--motif-radii values must be positive")
    return sorted(set(values))


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def axial_distance(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return max(abs(dx), abs(dy), abs(dx + dy))


def axial_norm(point: Tuple[int, int]) -> int:
    x, y = point
    return max(abs(x), abs(y), abs(x + y))


def placement_method_counts(placements: Sequence[Placement]) -> Counter[str]:
    return Counter(item.method for item in placements)


def run_label(reflection_mode: int, mode: str, seed: int) -> str:
    ref_label = "reflect" if reflection_mode else "no_reflect"
    return f"{ref_label}_{mode}_seed{seed}"


# -----------------------------------------------------------------------------
# Construction policies
# -----------------------------------------------------------------------------


def candidate_unique_placements(
    orientations: Sequence[Sequence[Cell]],
    targets: Sequence[Cell],
    occupied: Set[Cell],
    actual_radius: int,
    analysis_radius: int,
    edge_guard: int,
    scan_limit: int,
    rng: random.Random,
) -> List[Tuple[int, int, int]]:
    """Return genuinely local candidates from an interior part of the frontier.

    A target is only considered for a "forced" claim if it lies away from the
    patch cutoff. Its options are then counted using a slightly *larger* finite
    search window. This is conservative: alternatives just outside the patch
    radius can prevent a target from being labelled forced.
    """
    shuffled = list(targets)
    rng.shuffle(shuffled)
    unique: List[Tuple[int, int, int]] = []
    seen: Set[Tuple[int, int, int]] = set()

    for target in shuffled[: min(scan_limit, len(shuffled))]:
        if axial_norm((target[0], target[1])) > actual_radius * 2 - edge_guard:
            continue

        # Count alternatives in an expanded local window to reduce edge artefacts.
        options = placements_covering_target(
            orientations,
            target,
            analysis_radius,
            occupied,
            cap=2,
        )
        if len(options) != 1:
            continue

        orientation, tx, ty = options[0]
        # The selected placement must still fit inside the actual run patch.
        if placement_fits(orientations, orientation, tx, ty, actual_radius, occupied):
            descriptor = (orientation, tx, ty)
            if descriptor not in seen:
                seen.add(descriptor)
                unique.append(descriptor)

    return unique


def choose_random_boundary_placement(
    orientations: Sequence[Sequence[Cell]],
    targets: Sequence[Cell],
    radius: int,
    occupied: Set[Cell],
    rng: random.Random,
    scan_limit: int,
) -> Tuple[int, int, int] | None:
    """Choose a random legal placement from a random sample of the frontier."""
    shuffled = list(targets)
    rng.shuffle(shuffled)

    reservoir: Tuple[int, int, int] | None = None
    seen = 0

    for target in shuffled[: min(scan_limit, len(shuffled))]:
        options = placements_covering_target(
            orientations,
            target,
            radius,
            occupied,
            cap=16,
        )
        for option in options:
            seen += 1
            # Reservoir sampling: one uniformly chosen candidate without
            # storing an enormous list.
            if rng.randrange(seen) == 0:
                reservoir = option

    return reservoir


def select_next_placement(
    mode: str,
    orientations: Sequence[Sequence[Cell]],
    targets: Sequence[Cell],
    radius: int,
    analysis_radius: int,
    edge_guard: int,
    occupied: Set[Cell],
    rng: random.Random,
    forced_scan_limit: int,
    fallback_scan_limit: int,
    mixed_forced_probability: float,
) -> Tuple[Tuple[int, int, int] | None, str]:
    """Choose a placement according to a construction policy.

    forced:
        Prefer a locally-unique continuation, then use a high-contact heuristic.
    mixed:
        Use a locally-unique continuation with a configurable probability;
        otherwise use a random legal boundary continuation.
    heuristic:
        Always use the high-contact fallback heuristic.
    random:
        Always choose a random legal placement from sampled boundary targets.
    """
    unique = candidate_unique_placements(
        orientations=orientations,
        targets=targets,
        occupied=occupied,
        actual_radius=radius,
        analysis_radius=analysis_radius,
        edge_guard=edge_guard,
        scan_limit=forced_scan_limit,
        rng=rng,
    )

    if mode == "forced":
        if unique:
            return rng.choice(unique), "forced"
        selected = choose_fallback_placement(
            orientations,
            targets,
            radius,
            occupied,
            rng,
            scan_limit=min(fallback_scan_limit, len(targets)),
        )
        return selected, "heuristic"

    if mode == "mixed":
        if unique and rng.random() < mixed_forced_probability:
            return rng.choice(unique), "forced"
        selected = choose_random_boundary_placement(
            orientations,
            targets,
            radius,
            occupied,
            rng,
            scan_limit=min(fallback_scan_limit, len(targets)),
        )
        return selected, "random"

    if mode == "heuristic":
        selected = choose_fallback_placement(
            orientations,
            targets,
            radius,
            occupied,
            rng,
            scan_limit=min(fallback_scan_limit, len(targets)),
        )
        return selected, "heuristic"

    if mode == "random":
        selected = choose_random_boundary_placement(
            orientations,
            targets,
            radius,
            occupied,
            rng,
            scan_limit=min(fallback_scan_limit, len(targets)),
        )
        return selected, "random"

    raise ValueError(f"Unhandled construction mode: {mode}")


# -----------------------------------------------------------------------------
# Motif extraction and stability analysis
# -----------------------------------------------------------------------------


def motif_signature(
    placements: Sequence[Placement],
    centre_index: int,
    motif_radius: int,
) -> str:
    """Build a literal translation-normalised local anchor signature.

    Orientation indices are stable for runs using the same reflection setting,
    because all_orientations() enumerates in a deterministic order. For that
    reason stability comparisons are *never* made across reflection settings.
    """
    centre = placements[centre_index]
    cx, cy = centre.tx, centre.ty
    members: List[Tuple[int, int, int]] = []

    for other in placements:
        if axial_distance((cx, cy), (other.tx, other.ty)) <= motif_radius:
            members.append((other.tx - cx, other.ty - cy, other.orientation))

    return ";".join(
        f"{dx},{dy},o{orientation}"
        for dx, dy, orientation in sorted(members)
    )


def motif_counter(
    placements: Sequence[Placement],
    motif_radius: int,
    interior_limit: int,
    min_members: int,
) -> Counter[str]:
    """Count interior-centred, translation-normalised local anchor motifs."""
    counter: Counter[str] = Counter()

    for centre_index, centre in enumerate(placements):
        # Exclude seed and centres near the patch cutoff. This reduces motifs
        # created simply because the greedy patch is truncated at its boundary.
        if centre.method == "seed":
            continue
        if axial_norm((centre.tx, centre.ty)) > interior_limit:
            continue

        signature = motif_signature(placements, centre_index, motif_radius)
        member_count = 0 if not signature else signature.count(";") + 1
        if member_count >= min_members:
            counter[signature] += 1

    return counter


def signature_member_count(signature: str) -> int:
    return 0 if not signature else signature.count(";") + 1


@dataclass
class RunResult:
    label: str
    seed: int
    reflection_mode: int
    mode: str
    placements: List[Placement]
    elapsed_seconds: float
    stalled: bool
    motif_counts: Dict[int, Counter[str]]


# -----------------------------------------------------------------------------
# Patch runner
# -----------------------------------------------------------------------------


def run_one_patch(
    *,
    orientations: Sequence[Sequence[Cell]],
    mode: str,
    seed: int,
    radius: int,
    max_tiles: int,
    analysis_margin: int,
    edge_guard: int,
    forced_scan_limit: int,
    fallback_scan_limit: int,
    mixed_forced_probability: float,
    reporter: Reporter,
    run_display: str,
) -> Tuple[List[Placement], bool]:
    rng = random.Random(seed)
    occupied: Set[Cell] = set()
    occupied_cells: List[Cell] = []
    placements: List[Placement] = []

    initial_orientation = rng.randrange(len(orientations))
    seed_placement = Placement(initial_orientation, 0, 0, "seed", 1, 1)
    add_placement(seed_placement, orientations, occupied, occupied_cells)
    placements.append(seed_placement)

    analysis_radius = radius + analysis_margin
    stalled = False

    while len(placements) < max_tiles:
        step = len(placements) + 1
        targets = collect_boundary_targets(occupied_cells, occupied, radius)

        if not targets:
            reporter.log(f"[run stop] {run_display} no boundary targets at step={step}")
            stalled = True
            break

        selected, method = select_next_placement(
            mode=mode,
            orientations=orientations,
            targets=targets,
            radius=radius,
            analysis_radius=analysis_radius,
            edge_guard=edge_guard,
            occupied=occupied,
            rng=rng,
            forced_scan_limit=forced_scan_limit,
            fallback_scan_limit=fallback_scan_limit,
            mixed_forced_probability=mixed_forced_probability,
        )

        if selected is None:
            reporter.log(f"[run stop] {run_display} no legal placement at step={step}")
            stalled = True
            break

        orientation, tx, ty = selected
        placement = Placement(orientation, tx, ty, method, step, 1)
        add_placement(placement, orientations, occupied, occupied_cells)
        placements.append(placement)

        if step % 100 == 0:
            counts = placement_method_counts(placements)
            reporter.log(
                f"[run progress] {run_display} step={step}/{max_tiles} "
                f"boundary={len(targets)} forced={counts.get('forced', 0)} "
                f"heuristic={counts.get('heuristic', 0)} random={counts.get('random', 0)}"
            )

        reporter.heartbeat(
            "motif stability growth",
            f"run={run_display} step={step}/{max_tiles} tiles={len(placements)} "
            f"boundary={len(targets)} method={method}",
        )

    return placements, stalled


# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------


def write_run_outputs(
    run_dir: Path,
    result: RunResult,
    orientations: Sequence[Sequence[Cell]],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)

    write_csv(
        run_dir / "placements.csv",
        ["step", "method", "orientation", "tx", "ty"],
        (
            {
                "step": item.step,
                "method": item.method,
                "orientation": item.orientation,
                "tx": item.tx,
                "ty": item.ty,
            }
            for item in result.placements
        ),
    )

    counts = placement_method_counts(result.placements)
    write_svg(
        run_dir / "patch.svg",
        orientations,
        result.placements,
        (
            f"{result.label}: {len(result.placements)} tiles; "
            f"forced={counts.get('forced', 0)}; "
            f"heuristic={counts.get('heuristic', 0)}; random={counts.get('random', 0)}"
        ),
        line_width=0.8,
    )

    for radius, counter in result.motif_counts.items():
        rows = []
        for rank, (signature, occurrences) in enumerate(counter.most_common(80), start=1):
            rows.append(
                {
                    "rank": rank,
                    "occurrences": occurrences,
                    "tiles_in_signature": signature_member_count(signature),
                    "signature": signature,
                }
            )
        write_csv(
            run_dir / f"motifs_radius_{radius}.csv",
            ["rank", "occurrences", "tiles_in_signature", "signature"],
            rows,
        )

    run_summary = [
        "Motif stability run summary",
        "==========================",
        "",
        f"label: {result.label}",
        f"seed: {result.seed}",
        f"allow_reflections: {result.reflection_mode}",
        f"construction_mode: {result.mode}",
        f"tiles: {len(result.placements)}",
        f"stalled_before_cap: {int(result.stalled)}",
        f"elapsed_seconds: {result.elapsed_seconds:.2f}",
        f"locally_unique_placements: {counts.get('forced', 0)}",
        f"heuristic_placements: {counts.get('heuristic', 0)}",
        f"random_placements: {counts.get('random', 0)}",
        "",
        "Interpretation:",
        "- This is one finite greedy construction, not an exhaustive tiling search.",
        "- Locally unique means unique in a conservative expanded finite local window.",
        "- The patch.svg is useful for visual comparison across seeds/modes.",
    ]
    (run_dir / "run_summary.txt").write_text("\n".join(run_summary) + "\n", encoding="utf-8")


def stable_motif_rows(
    results: Sequence[RunResult],
    motif_radii: Sequence[int],
    per_run_min_occurrences: int,
    stability_fraction: float,
) -> List[Dict[str, object]]:
    """Find motifs that recur across runs in comparable condition groups.

    We compare only runs sharing allow_reflections, because orientation-index
    numbering is deterministic within that setting but not intended to be
    compared across the two setting families.

    Reports include:
    - reflect_<0/1>_all_modes: strongest cross-policy test.
    - reflect_<0/1>_<mode>: repeatability within one policy.
    """
    grouped: Dict[Tuple[int, str], List[RunResult]] = defaultdict(list)

    for result in results:
        grouped[(result.reflection_mode, "all_modes")].append(result)
        grouped[(result.reflection_mode, result.mode)].append(result)

    rows: List[Dict[str, object]] = []

    for (reflection_mode, group_name), group_results in sorted(grouped.items()):
        total_runs = len(group_results)
        required_runs = max(1, math.ceil(total_runs * stability_fraction))

        for radius in motif_radii:
            presence: Dict[str, List[int]] = defaultdict(list)

            for result in group_results:
                counter = result.motif_counts.get(radius, Counter())
                for signature, occurrences in counter.items():
                    if occurrences >= per_run_min_occurrences:
                        presence[signature].append(occurrences)

            for signature, occurrences_across_runs in presence.items():
                presence_runs = len(occurrences_across_runs)
                if presence_runs < required_runs:
                    continue

                rows.append(
                    {
                        "reflection_mode": reflection_mode,
                        "group": group_name,
                        "radius": radius,
                        "presence_runs": presence_runs,
                        "total_runs": total_runs,
                        "presence_fraction": f"{presence_runs / total_runs:.3f}",
                        "required_runs": required_runs,
                        "min_occurrences_in_present_runs": min(occurrences_across_runs),
                        "mean_occurrences_in_present_runs": f"{sum(occurrences_across_runs) / presence_runs:.2f}",
                        "max_occurrences_in_present_runs": max(occurrences_across_runs),
                        "tiles_in_signature": signature_member_count(signature),
                        "signature": signature,
                    }
                )

    rows.sort(
        key=lambda row: (
            -int(row["presence_runs"]),
            -float(row["presence_fraction"]),
            -float(row["mean_occurrences_in_present_runs"]),
            -int(row["tiles_in_signature"]),
            int(row["radius"]),
        )
    )
    return rows


def write_stable_report(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    lines = [
        "Motif stability report",
        "======================",
        "",
        "A motif is a translation-normalised neighbourhood of tile placement anchors.",
        "It is considered stable here when it appears at least the configured number", 
        "of times in a configured fraction of independent finite construction runs.",
        "",
        "This is a screening result only:",
        "- It does NOT show the motif is unavoidable in every tiling.",
        "- It does NOT establish a substitution rule or aperiodicity proof.",
        "- Strongest rows are those present in reflect_1_all_modes or reflect_0_all_modes.",
        "",
    ]

    if not rows:
        lines.append("No motifs met the configured cross-run stability threshold.")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    current_group = None
    rank = 0

    for row in rows[:120]:
        heading = f"reflections={row['reflection_mode']} group={row['group']} radius={row['radius']}"
        if heading != current_group:
            lines.extend(["", heading, "-" * len(heading)])
            current_group = heading
            rank = 0

        rank += 1
        preview = str(row["signature"])[:700]
        lines.append(
            f"{rank:2d}. presence={row['presence_runs']}/{row['total_runs']} "
            f"mean_occ={row['mean_occurrences_in_present_runs']} "
            f"tiles={row['tiles_in_signature']}"
        )
        lines.append(f"    {preview}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare repeated local tile motifs across independent finite patch-growth runs."
    )
    parser.add_argument("candidate_file", type=Path)
    parser.add_argument("--out", type=Path, default=Path("858_motif_stability"))
    parser.add_argument("--seeds", default="858,271828,314159")
    parser.add_argument("--modes", default="forced,mixed,random")
    parser.add_argument("--reflection-modes", default="1")
    parser.add_argument("--radius", type=int, default=60)
    parser.add_argument("--max-tiles", type=int, default=800)
    parser.add_argument("--analysis-margin", type=int, default=12)
    parser.add_argument("--edge-guard", type=int, default=10)
    parser.add_argument("--forced-scan-limit", type=int, default=80)
    parser.add_argument("--fallback-scan-limit", type=int, default=70)
    parser.add_argument("--mixed-forced-probability", type=float, default=0.55)
    parser.add_argument("--motif-radii", default="3,4,5,6")
    parser.add_argument("--motif-margin", type=int, default=14)
    parser.add_argument("--min-motif-members", type=int, default=3)
    parser.add_argument("--per-run-min-occurrences", type=int, default=3)
    parser.add_argument("--stability-fraction", type=float, default=0.67)
    parser.add_argument("--heartbeat-sec", type=int, default=15)
    args = parser.parse_args()

    seeds = parse_int_csv(args.seeds, "--seeds")
    modes = parse_modes(args.modes)
    reflection_modes = parse_reflection_modes(args.reflection_modes)
    motif_radii = parse_radii(args.motif_radii)

    if args.radius < 20:
        raise ValueError("--radius must be at least 20")
    if args.max_tiles < 50:
        raise ValueError("--max-tiles must be at least 50")
    if args.analysis_margin < 0 or args.edge_guard < 0:
        raise ValueError("--analysis-margin and --edge-guard must be non-negative")
    if not (0.0 <= args.mixed_forced_probability <= 1.0):
        raise ValueError("--mixed-forced-probability must be between 0 and 1")
    if args.min_motif_members < 2:
        raise ValueError("--min-motif-members must be at least 2")
    if args.per_run_min_occurrences < 1:
        raise ValueError("--per-run-min-occurrences must be at least 1")
    if not (0.0 < args.stability_fraction <= 1.0):
        raise ValueError("--stability-fraction must be in (0, 1]")

    out_dir: Path = args.out
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    reporter = Reporter(args.heartbeat_sec)
    shape = parse_candidate(args.candidate_file)

    total_runs = len(seeds) * len(modes) * len(reflection_modes)
    reporter.log("=== Motif Stability Probe ===")
    reporter.log(f"candidate_file: {args.candidate_file}")
    reporter.log(f"out: {out_dir}")
    reporter.log(f"cells: {len(shape)}")
    reporter.log(f"connected: {int(is_connected(shape))}")
    reporter.log(f"seeds: {seeds}")
    reporter.log(f"modes: {modes}")
    reporter.log(f"reflection_modes: {reflection_modes}")
    reporter.log(f"radius: {args.radius}")
    reporter.log(f"max_tiles_per_run: {args.max_tiles}")
    reporter.log(f"motif_radii: {motif_radii}")
    reporter.log(f"total_runs: {total_runs}")
    reporter.log("")
    reporter.log("NOTE: Stable motifs are proof-hunting leads only, not a proof of aperiodicity.")
    reporter.log("")

    results: List[RunResult] = []
    run_number = 0

    for reflection_mode in reflection_modes:
        orientations = all_orientations(shape, bool(reflection_mode))
        reporter.log(
            f"[orientation family] allow_reflections={reflection_mode} "
            f"orientations={len(orientations)}"
        )

        for mode in modes:
            for seed in seeds:
                run_number += 1
                label = run_label(reflection_mode, mode, seed)
                run_dir = runs_dir / label
                reporter.log(f"[run start] {run_number}/{total_runs} {label}")
                started = time.time()

                # Mix in mode and reflection setting so each condition gets a
                # deterministic but different pseudo-random sequence.
                derived_seed = (
                    seed
                    + (101 if reflection_mode else 907)
                    + {"forced": 100_003, "mixed": 200_003, "heuristic": 300_007, "random": 400_009}[mode]
                )

                placements, stalled = run_one_patch(
                    orientations=orientations,
                    mode=mode,
                    seed=derived_seed,
                    radius=args.radius,
                    max_tiles=args.max_tiles,
                    analysis_margin=args.analysis_margin,
                    edge_guard=args.edge_guard,
                    forced_scan_limit=args.forced_scan_limit,
                    fallback_scan_limit=args.fallback_scan_limit,
                    mixed_forced_probability=args.mixed_forced_probability,
                    reporter=reporter,
                    run_display=label,
                )
                elapsed = time.time() - started

                interior_limit = max(0, args.radius * 2 - args.motif_margin)
                counters = {
                    motif_radius: motif_counter(
                        placements,
                        motif_radius=motif_radius,
                        interior_limit=interior_limit,
                        min_members=args.min_motif_members,
                    )
                    for motif_radius in motif_radii
                }

                result = RunResult(
                    label=label,
                    seed=seed,
                    reflection_mode=reflection_mode,
                    mode=mode,
                    placements=placements,
                    elapsed_seconds=elapsed,
                    stalled=stalled,
                    motif_counts=counters,
                )
                write_run_outputs(run_dir, result, orientations)
                results.append(result)

                counts = placement_method_counts(placements)
                reporter.log(
                    f"[run done] {run_number}/{total_runs} {label} "
                    f"tiles={len(placements)} stalled={int(stalled)} "
                    f"forced={counts.get('forced', 0)} heuristic={counts.get('heuristic', 0)} "
                    f"random={counts.get('random', 0)} elapsed={elapsed:.1f}s"
                )

    # Overall run matrix.
    run_rows: List[Dict[str, object]] = []
    for result in results:
        counts = placement_method_counts(result.placements)
        run_rows.append(
            {
                "label": result.label,
                "seed": result.seed,
                "reflection_mode": result.reflection_mode,
                "mode": result.mode,
                "tiles": len(result.placements),
                "stalled_before_cap": int(result.stalled),
                "elapsed_seconds": f"{result.elapsed_seconds:.2f}",
                "forced": counts.get("forced", 0),
                "heuristic": counts.get("heuristic", 0),
                "random": counts.get("random", 0),
                "seed_placements": counts.get("seed", 0),
            }
        )

    write_csv(
        out_dir / "run_summary.csv",
        [
            "label", "seed", "reflection_mode", "mode", "tiles",
            "stalled_before_cap", "elapsed_seconds", "forced", "heuristic",
            "random", "seed_placements",
        ],
        run_rows,
    )

    stable_rows = stable_motif_rows(
        results=results,
        motif_radii=motif_radii,
        per_run_min_occurrences=args.per_run_min_occurrences,
        stability_fraction=args.stability_fraction,
    )
    write_csv(
        out_dir / "stable_motifs.csv",
        [
            "reflection_mode", "group", "radius", "presence_runs", "total_runs",
            "presence_fraction", "required_runs", "min_occurrences_in_present_runs",
            "mean_occurrences_in_present_runs", "max_occurrences_in_present_runs",
            "tiles_in_signature", "signature",
        ],
        stable_rows,
    )
    write_stable_report(out_dir / "stable_motifs_report.txt", stable_rows)

    summary = [
        "Motif Stability Probe Summary",
        "============================",
        "",
        f"candidate_file: {args.candidate_file}",
        f"cells: {len(shape)}",
        f"connected: {int(is_connected(shape))}",
        f"seeds: {','.join(str(seed) for seed in seeds)}",
        f"modes: {','.join(modes)}",
        f"reflection_modes: {','.join(str(value) for value in reflection_modes)}",
        f"radius: {args.radius}",
        f"max_tiles_per_run: {args.max_tiles}",
        f"motif_radii: {','.join(str(value) for value in motif_radii)}",
        f"total_runs: {total_runs}",
        f"per_run_min_occurrences: {args.per_run_min_occurrences}",
        f"stability_fraction: {args.stability_fraction}",
        "",
        f"stable_motif_rows: {len(stable_rows)}",
        "",
        "Read these files in order:",
        "1. run_summary.csv — did all construction policies reach the tile cap?",
        "2. stable_motifs_report.txt — which literal motifs recur across runs?",
        "3. stable_motifs.csv — complete machine-readable motif table.",
        "4. runs/*/patch.svg — visually compare the construction families.",
        "",
        "Interpretation:",
        "- A motif stable across reflect_1_all_modes is more persuasive than one only stable in forced mode.",
        "- A motif stable across different seeds and all policies is a candidate local rule, not a theorem.",
        "- If random/heuristic runs diverge wildly while forced runs agree, the prior forced motif may be growth-policy bias.",
        "- If the same larger motifs survive all runs, the next step is to test whether they assemble into candidate supertiles.",
        "",
        "This program is not a mathematical proof engine and does not establish an einstein tile.",
    ]
    (out_dir / "summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")

    reporter.log("")
    reporter.log("=== Motif Stability Probe Complete ===")
    reporter.log(f"runs: {len(results)}")
    reporter.log(f"stable_motif_rows: {len(stable_rows)}")
    reporter.log(f"results: {out_dir.resolve()}")
    reporter.log("Read summary.txt, run_summary.csv, and stable_motifs_report.txt first.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr, flush=True)
        raise SystemExit(130)
