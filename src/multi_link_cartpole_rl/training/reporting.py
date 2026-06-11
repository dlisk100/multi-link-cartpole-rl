"""Reporting helpers for reproducible cartpole experiment evidence."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MonitorEpisode:
    """One episode row from a Stable-Baselines3 Monitor CSV file."""

    reward: float
    length: int
    elapsed_time: float


def read_monitor_csv(path: str | Path) -> list[MonitorEpisode]:
    """Read episode metrics from an SB3 Monitor CSV file."""
    monitor_path = Path(path)
    if not monitor_path.exists():
        raise FileNotFoundError(f"Monitor CSV not found: {monitor_path}")

    with monitor_path.open("r", encoding="utf-8") as monitor_file:
        lines = [line for line in monitor_file if not line.startswith("#")]

    rows: list[MonitorEpisode] = []
    for row in csv.DictReader(lines):
        rows.append(
            MonitorEpisode(
                reward=float(row["r"]),
                length=int(row["l"]),
                elapsed_time=float(row["t"]),
            )
        )

    if not rows:
        raise ValueError(f"Monitor CSV has no episode rows: {monitor_path}")

    return rows


def monitor_summary(episodes: list[MonitorEpisode]) -> dict[str, Any]:
    """Summarize training episodes for a compact run report."""
    lengths = np.array([episode.length for episode in episodes], dtype=np.float32)
    rewards = np.array([episode.reward for episode in episodes], dtype=np.float32)

    return {
        "episode_count": int(len(episodes)),
        "mean_episode_length": float(np.mean(lengths)),
        "best_episode_length": int(np.max(lengths)),
        "final_episode_length": int(episodes[-1].length),
        "mean_episode_reward": float(np.mean(rewards)),
        "best_episode_reward": float(np.max(rewards)),
        "final_episode_reward": float(episodes[-1].reward),
    }


def write_json_report(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write a deterministic JSON report and return its path."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    return output_path


def read_json_report(path: str | Path) -> dict[str, Any]:
    """Load a JSON report as a dictionary."""
    with Path(path).open("r", encoding="utf-8") as report_file:
        data = json.load(report_file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in report: {path}")
    return data


def build_run_summary(
    *,
    monitor_csv: str | Path,
    evaluation_json: str | Path,
) -> dict[str, Any]:
    """Combine training and evaluation evidence into one compact run summary."""
    episodes = read_monitor_csv(monitor_csv)
    evaluation = read_json_report(evaluation_json)

    return {
        "task": evaluation.get("task", "single_link_stabilization"),
        "config_path": evaluation.get("config_path"),
        "model_path": evaluation.get("model_path"),
        "episodes": evaluation.get("episodes"),
        "seed": evaluation.get("seed"),
        "training": monitor_summary(episodes),
        "evaluation": evaluation.get("policies", {}),
    }


def plot_training_curve(
    *,
    monitor_csv: str | Path,
    output_path: str | Path,
) -> Path:
    """Plot episode length over training from an SB3 Monitor CSV."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    episodes = read_monitor_csv(monitor_csv)
    episode_indices = np.arange(1, len(episodes) + 1)
    episode_lengths = np.array([episode.length for episode in episodes])

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(episode_indices, episode_lengths, color="#4c78a8", linewidth=1.8)
    ax.set_title("Single-Link PPO Training Progress")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Episode length")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def plot_random_vs_ppo(
    *,
    evaluation_json: str | Path,
    output_path: str | Path,
) -> Path:
    """Plot random policy versus trained PPO mean episode length."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = read_json_report(evaluation_json)
    policies = report["policies"]
    labels = ["Random", "Trained PPO"]
    values = [
        policies["random"]["mean_episode_length"],
        policies["trained_ppo"]["mean_episode_length"],
    ]

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bars = ax.bar(labels, values, color=["#bab0ac", "#4c78a8"])
    ax.set_title("Single-Link Stabilization: Random vs PPO")
    ax.set_ylabel("Mean episode length")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.1f}",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output
