from __future__ import annotations

from pathlib import Path

# Safe backend for servers / headless environments (e.g., Flask demo).
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_fitness(history_best: list[float], history_mean: list[float], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7.5, 4.2))
    plt.plot(history_best, label="Best Fitness")
    plt.plot(history_mean, label="Mean Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Fitness Evolution Over Generations")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_before_after_bar(before: dict, after: dict, key: str, title: str, out_path: str | Path, ylabel: str) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    labels = ["Before", "After"]
    vals = [before[key], after[key]]

    plt.figure(figsize=(5.8, 4.2))
    plt.bar(labels, vals)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_multi_before_after(before: dict, after: dict, keys: list[str], title: str, out_path: str | Path, ylabel: str) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    labels = keys
    before_vals = [before[k] for k in keys]
    after_vals = [after[k] for k in keys]

    x = list(range(len(keys)))
    width = 0.35

    plt.figure(figsize=(7.8, 4.2))
    plt.bar([i - width / 2 for i in x], before_vals, width=width, label="Before")
    plt.bar([i + width / 2 for i in x], after_vals, width=width, label="After")
    plt.xticks(x, labels)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
