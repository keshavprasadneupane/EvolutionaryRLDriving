
"""
Turns the runs of one experiment into separate completed and interrupted
results tables and learning/evaluation curves.

    python -m evaluation.aggregate dqn_baseline
    python -m evaluation.aggregate dqn_baseline --window 50 --target 300

Reads:
    runs/<experiment>/*/{metrics.csv,eval.csv,run_info.json}

Writes:
    results/<experiment>/{completed,interrupted}/
        summary.csv
        summary.md
        episode_learn_curve.png
        env_step_learn_curve.png
        evaluation_curve.png

Everything is computed from the run folders, so results can be regenerated
at any time. Never copy numbers into the report by hand.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []

    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def moving_average(
    x: np.ndarray,
    window: int,
) -> tuple[np.ndarray, int]:
    """Trailing moving average. Returns (smoothed, window actually used)."""
    window = max(1, min(window, len(x)))
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid"), window


def load_run(
    run_dir: Path,
    window: int,
    target: float | None,
) -> dict | None:
    rows = _read_csv(run_dir / "metrics.csv")

    if not rows:
        return None

    steps = np.array([float(r["env_steps"]) for r in rows])
    rewards = np.array([float(r["reward"]) for r in rows])
    lengths = np.array([float(r["length"]) for r in rows])

    # Use the episode column when available. Fall back to row indices.
    episodes = np.array([
        float(r["episode"]) if r.get("episode") not in (None, "") else i
        for i, r in enumerate(rows)
    ])

    smooth, used = moving_average(rewards, window)

    # Each smoothed reward is positioned at the final episode in its window.
    smooth_episodes = episodes[used - 1:]
    smooth_steps = steps[used - 1:]

    info_path = run_dir / "run_info.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}

    steps_to_target = None

    if target is not None:
        hit = np.nonzero(smooth >= target)[0]

        if len(hit):
            steps_to_target = float(smooth_steps[hit[0]])

    eval_rows = _read_csv(run_dir / "eval.csv")

    eval_rewards = np.array([
        float(r["reward"]) for r in eval_rows
    ])

    eval_steps = (
        np.array([
            float(r["env_steps"]) for r in eval_rows
        ])
        if eval_rows and "env_steps" in eval_rows[0]
        else np.arange(1, len(eval_rewards) + 1)
    )

    return {
        "name": run_dir.name,
        "seed": info.get("seed"),
        "status": info.get("status", "unknown"),
        "git_dirty": info.get("git_dirty"),
        "episodes": len(rows),
        "env_steps": float(steps[-1]),
        "final_reward": float(smooth[-1]),
        "best_episode": float(rewards.max()),
        "worst_episode": float(rewards.min()),
        "mean_length": float(lengths.mean()),
        "steps_to_target": steps_to_target,
        "train_seconds": info.get("duration_seconds"),
        "eval_mean": (
            float(eval_rewards.mean()) if len(eval_rewards) else None
        ),
        "eval_std": (
            float(eval_rewards.std()) if len(eval_rewards) else None
        ),
        "curve": {
            "episode": smooth_episodes,
            "env_steps": smooth_steps,
            "reward": smooth,
        },
        "eval_curve": (eval_steps, eval_rewards),
    }


def _mean_std(
    values: list[float | None],
) -> tuple[float, float] | None:
    clean = [v for v in values if v is not None]

    if not clean:
        return None

    return float(np.mean(clean)), float(np.std(clean))


def _fmt(
    ms: tuple[float, float] | None,
    digits: int = 1,
) -> str:
    return (
        "n/a"
        if ms is None
        else f"{ms[0]:.{digits}f} ± {ms[1]:.{digits}f}"
    )


def summarize(
    runs: list[dict],
    target: float | None,
) -> list[tuple[str, str]]:
    """Rows of (metric, 'mean ± std over runs')."""
    rows = [
        ("runs (seeds)", str(len(runs))),
        (
            "final reward (last window)",
            _fmt(_mean_std([r["final_reward"] for r in runs])),
        ),
        (
            "best episode reward",
            _fmt(_mean_std([r["best_episode"] for r in runs])),
        ),
        (
            "worst episode reward",
            _fmt(_mean_std([r["worst_episode"] for r in runs])),
        ),
        (
            "mean episode length",
            _fmt(_mean_std([r["mean_length"] for r in runs])),
        ),
        (
            "env steps trained",
            _fmt(_mean_std([r["env_steps"] for r in runs]), 0),
        ),
        (
            "train time (s)",
            _fmt(_mean_std([r["train_seconds"] for r in runs]), 0),
        ),
        (
            "eval reward (mean of eval episodes)",
            _fmt(_mean_std([r["eval_mean"] for r in runs])),
        ),
    ]

    if target is not None:
        reached = [
            r["steps_to_target"]
            for r in runs
            if r["steps_to_target"] is not None
        ]

        rows.append((
            f"runs reaching target {target:g}",
            f"{len(reached)}/{len(runs)}",
        ))

        if reached:
            rows.append((
                "env steps to reach target",
                _fmt(_mean_std(reached), 0),
            ))

    return rows


def plot_learning_curve(
    runs: list[dict],
    experiment: str,
    out: Path,
    x_axis: str,
) -> None:
    """Plot seed learning curves, their mean, and the standard deviation."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba

    # Reserved colors for aggregate statistics.
    MEAN_COLOR = "#E63946"
    STD_COLOR = "#457B9D"

    # Individual seeds never use the mean or standard-deviation colors.
    SEED_COLORS = [
        "#2A9D8F",
        "#F4A261",
        "#9B5DE5",
        "#00B4D8",
        "#F15BB5",
        "#80B918",
        "#FF6B35",
        "#6C757D",
        "#BC6C25",
        "#3A86FF",
        "#8338EC",
        "#588157",
        "#E76F51",
        "#4361EE",
        "#B5179E",
        "#606C38",
    ]

    MEAN_ALPHA = 0.80
    MEAN_WIDTH = 1.2
    SEED_ALPHA = 1.00
    SEED_WIDTH = 1.0
    STD_ALPHA = 0.60

    curves = [
        (r["curve"][x_axis], r["curve"]["reward"])
        for r in runs
    ]

    fig, ax = plt.subplots(figsize=(7, 4))

    # Individual seed curves.
    for i, (run, (x, rewards)) in enumerate(zip(runs, curves)):
        color = SEED_COLORS[i % len(SEED_COLORS)]

        ax.plot(
            x,
            rewards,
            color=color,
            alpha=SEED_ALPHA,
            linewidth=SEED_WIDTH,
            label=f"seed {run['seed']}",
            zorder=2,
        )

    # Compute aggregate statistics over the common x-axis range.
    if len(curves) >= 2:
        grid_min = max(x[0] for x, _ in curves)
        grid_max = min(x[-1] for x, _ in curves)

        if grid_max > grid_min:
            grid = np.linspace(grid_min, grid_max, 300)

            interpolated = np.stack([
                np.interp(grid, x, rewards)
                for x, rewards in curves
            ])

            mean = interpolated.mean(axis=0)
            std = interpolated.std(axis=0)

            # Standard deviation band.
            ax.fill_between(
                grid,
                mean - std,
                mean + std,
                color=to_rgba(STD_COLOR, STD_ALPHA),
                linewidth=0,
                label="± 1 std",
                zorder=1,
            )

            # Mean curve.
            ax.plot(
                grid,
                mean,
                color=MEAN_COLOR,
                alpha=MEAN_ALPHA,
                linewidth=MEAN_WIDTH,
                label=f"mean of {len(runs)} runs",
                zorder=3,
            )

    ax.set_xlabel(
        "episode"
        if x_axis == "episode"
        else "environment steps"
    )
    ax.set_ylabel("episode reward (moving average)")
    ax.set_title(
        f"{experiment} — training by "
        f"{'episode' if x_axis == 'episode' else 'environment steps'}"
    )
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_evaluation_curve(
    runs: list[dict],
    experiment: str,
    out: Path,
) -> None:
    """Plot evaluation rewards separately from training rewards."""
    runs = [
        r for r in runs
        if len(r["eval_curve"][1])
    ]

    if not runs:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))

    for r in runs:
        ax.plot(
            *r["eval_curve"],
            marker="o",
            alpha=0.8,
            label=f"seed {r['seed']}",
        )

    ax.set_xlabel(
        "environment steps"
        if all(
            len(r["eval_curve"][0]) == len(r["eval_curve"][1])
            and "env_steps" in _read_csv(
                Path("runs") / experiment / r["name"] / "eval.csv"
            )[0]
            for r in runs
        )
        else "evaluation checkpoint"
    )
    ax.set_ylabel("evaluation reward")
    ax.set_title(f"{experiment} — evaluation")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    assert __doc__ is not None

    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[1]
    )
    parser.add_argument("experiment")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument(
        "--window",
        type=int,
        default=20,
        help="episodes in the moving average",
    )
    parser.add_argument(
        "--target",
        type=float,
        default=None,
        help="reward threshold for steps-to-target",
    )

    args = parser.parse_args()

    if args.window < 1:
        parser.error("--window must be at least 1")

    exp_dir = Path(args.runs_dir) / args.experiment
    run_dirs = sorted(
        p for p in exp_dir.glob("*")
        if p.is_dir()
    )

    runs = [
        r
        for p in run_dirs
        if (r := load_run(p, args.window, args.target))
    ]

    if not runs:
        raise SystemExit(f"No usable runs found in {exp_dir}")

    out_root = Path(args.results_dir) / args.experiment

    for status in ("completed", "interrupted"):
        # Treat all non-completed statuses as interrupted.
        group = [
            r for r in runs
            if (r["status"] == "completed")
            == (status == "completed")
        ]

        if not group:
            print(f"No {status} runs found.")
            continue

        out_dir = out_root / status
        out_dir.mkdir(parents=True, exist_ok=True)

        table = summarize(group, args.target)

        with open(out_dir / "summary.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "mean ± std over runs"])
            writer.writerows(table)

        lines = [
            f"# {args.experiment} — {status.title()} Runs",
            "",
            "| metric | mean ± std over runs |",
            "|---|---|",
        ]

        lines += [
            f"| {k} | {v} |"
            for k, v in table
        ]

        lines += [
            "",
            "## Runs",
            "",
            "| run | seed | final reward | eval mean | status | git dirty |",
            "|---|---|---|---|---|---|",
        ]

        for r in group:
            eval_mean = (
                "n/a"
                if r["eval_mean"] is None
                else f"{r['eval_mean']:.1f}"
            )

            lines.append(
                f"| {r['name']} | {r['seed']} "
                f"| {r['final_reward']:.1f} "
                f"| {eval_mean} | {r['status']} "
                f"| {r['git_dirty']} |"
            )

        if any(r["git_dirty"] for r in group):
            lines += [
                "",
                "Warning: some runs used uncommitted code changes.",
            ]

        (out_dir / "summary.md").write_text(
            "\n".join(lines) + "\n"
        )

        # Generate both training learning curves automatically.
        plot_learning_curve(
            group,
            args.experiment,
            out_dir / "train_learn_curve.png",
            x_axis="episode",
        )
        plot_evaluation_curve(
            group,
            args.experiment,
            out_dir / "evaluation_curve.png",
        )

        print(f"Wrote results to {out_dir}")
        print("\n".join(lines))


if __name__ == "__main__":
    main()
