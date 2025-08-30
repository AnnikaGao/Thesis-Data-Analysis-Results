#!/Users/donyin/miniconda3/bin/python
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from rich import print
from rich.table import Table
from rich.console import Console
from src.parsers import Script


@dataclass
class GroupSummary:
    label: str
    count: int
    mean: float
    std: float
    median: float
    min: float
    max: float


def _collect_interviewee_depths(folder: Path) -> np.ndarray:
    depths: List[float] = []
    for csv_file in sorted(folder.glob("*.csv")):
        conv = Script(csv_file)
        for msg in conv.rounds:
            if str(msg.speaker).strip().lower() == "interviewee":
                val = msg.depth
                if pd.notna(val):
                    depths.append(float(val))
    return np.array(depths, dtype=float)


def _summarize(label: str, values: np.ndarray) -> GroupSummary:
    clean = values[np.isfinite(values)]
    return GroupSummary(
        label=label,
        count=int(clean.size),
        mean=float(np.mean(clean)) if clean.size else float("nan"),
        std=float(np.std(clean, ddof=1)) if clean.size > 1 else float("nan"),
        median=float(np.median(clean)) if clean.size else float("nan"),
        min=float(np.min(clean)) if clean.size else float("nan"),
        max=float(np.max(clean)) if clean.size else float("nan"),
    )


def _hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return float("nan")
    sa2 = float(np.var(a, ddof=1))
    sb2 = float(np.var(b, ddof=1))
    na, nb = a.size, b.size
    denom_df = na + nb - 2
    if denom_df <= 0:
        return float("nan")
    s_pooled2 = ((na - 1) * sa2 + (nb - 1) * sb2) / denom_df
    if not np.isfinite(s_pooled2) or s_pooled2 <= 0:
        return float("nan")
    s_pooled = s_pooled2**0.5
    d = (np.mean(a) - np.mean(b)) / s_pooled
    J = 1 - (3 / (4 * (na + nb) - 9)) if (na + nb) > 2 else 1.0
    return float(d * J)


def run_welch_t_test(include_zeros: bool = True) -> Tuple[GroupSummary, GroupSummary, dict]:
    """
    NOTE don: why this choice of t-test:
    we use Welch's t-test because sample sizes may differ and variances can be
    unequal between human and AI depth distributions. Welch's test does not
    assume equal variances, is robust to heteroscedasticity, and remains valid
    with unequal n. Assumptions: independent observations and approximately
    normal sampling distributions of the means (via CLT). We report Hedges' g
    as an unbiased effect size suitable for small-to-moderate samples.
    """
    human_depths = _collect_interviewee_depths(Path("data/human"))
    ai_depths = _collect_interviewee_depths(Path("data/ai"))

    if not include_zeros:
        human_depths = human_depths[human_depths > 0]
        ai_depths = ai_depths[ai_depths > 0]

    # welch's t-test
    t_stat, p_value = stats.ttest_ind(human_depths, ai_depths, equal_var=False, nan_policy="omit")

    # effect size
    g = _hedges_g(human_depths, ai_depths)

    human_summary = _summarize("Human (Interviewee)", human_depths)
    ai_summary = _summarize("AI (Interviewee)", ai_depths)

    result = {"t_stat": float(t_stat) if pd.notna(t_stat) else float("nan"), "p_value": float(p_value) if pd.notna(p_value) else float("nan"), "hedges_g": g, "include_zeros": include_zeros}

    return human_summary, ai_summary, result


def _print_console_report(human: GroupSummary, ai: GroupSummary, result: dict) -> None:
    console = Console()
    table = Table(title="Human vs AI Interviewee Depths (Welch's t-test)")
    table.add_column("Group", style="cyan")
    table.add_column("N", justify="right")
    table.add_column("Mean", justify="right")
    table.add_column("Std", justify="right")
    table.add_column("Median", justify="right")
    table.add_column("Min", justify="right")
    table.add_column("Max", justify="right")

    for g in [human, ai]:
        table.add_row(
            g.label,
            str(g.count),
            f"{g.mean:.3f}" if g.mean == g.mean else "nan",
            f"{g.std:.3f}" if g.std == g.std else "nan",
            f"{g.median:.3f}" if g.median == g.median else "nan",
            f"{g.min:.3f}" if g.min == g.min else "nan",
            f"{g.max:.3f}" if g.max == g.max else "nan",
        )

    console.print(table)
    print(f"\n[bold yellow]Welch t-test[/bold yellow]: t={result['t_stat']:.3f}, p={result['p_value']:.6f} (include_zeros={result['include_zeros']})")
    print(f"Hedges' g: {result['hedges_g']:.3f}")


def _write_report(human: GroupSummary, ai: GroupSummary, result: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("Human vs AI Interviewee Depths (Welch's t-test)\n")
    lines.append(f"Include zeros: {result['include_zeros']}\n")
    lines.append("\nGroup summaries:\n")
    for g in [human, ai]:
        lines.append(f"- {g.label}: N={g.count}, mean={g.mean:.3f}, std={g.std:.3f}, median={g.median:.3f}, min={g.min:.3f}, max={g.max:.3f}")
    lines.append(f"\nWelch t-test: t={result['t_stat']:.6f}, p={result['p_value']:.8f}\nHedges' g: {result['hedges_g']:.6f}\n")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    include_zeros = True
    out_file = Path("results/human_vs_ai_depth_ttest.txt")

    human, ai, result = run_welch_t_test(include_zeros=include_zeros)
    _print_console_report(human, ai, result)
    _write_report(human, ai, result, out_file)
    print(f"\n[green]Report written to[/green] {out_file}")


if __name__ == "__main__":
    main()
