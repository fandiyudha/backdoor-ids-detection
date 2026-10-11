"""Figure 3 of the article: the intrinsically easy class, by (a) model-level
reachability and (b) input-level homogeneity, on CICIDS2017 and UNSW-NB15.

Plots committed values only (no models are loaded). Every bar is read from the
committed metrics file and field listed in ``SOURCES``; the plotted values and
their sources are written next to the figure.

Outputs: ``experiments/figure3/figure3.png``, ``experiments/figure3/figure3_data.json``.

Run:
    python -m src.eval.plot_figure3
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.utils import resolve_path  # noqa: E402

# (panel, dataset, class) -> (metrics file, path of keys inside it, note)
SOURCES = {
    ("reachability", "CICIDS2017", "normal/benign"): (
        "experiments/phase6d/metrics.json",
        ["per_condition", "B_imbal_tgtBenign", "mask_clean_benign", "mean"],
        "clean models, natural class ratio, mean over 5 seeds"),
    ("reachability", "CICIDS2017", "attack"): (
        "experiments/phase6d/metrics.json",
        ["per_condition", "B_imbal_tgtBenign", "mask_clean_attack", "mean"],
        "clean models, natural class ratio, mean over 5 seeds"),
    ("reachability", "UNSW-NB15", "normal/benign"): (
        "experiments/phase7d/metrics.json",
        ["intrinsic", "measures", "nc_mask_normal", "mean"],
        "clean models, mean over 5 seeds"),
    ("reachability", "UNSW-NB15", "attack"): (
        "experiments/phase7d/metrics.json",
        ["intrinsic", "measures", "nc_mask_attack", "mean"],
        "clean models, mean over 5 seeds"),
    ("homogeneity", "CICIDS2017", "normal/benign"): (
        "experiments/phase4/metrics.json",
        ["class_breakdown", "mean_entropy", "clean_benign"],
        "clean benign inputs, representative p0.05 model"),
    ("homogeneity", "CICIDS2017", "attack"): (
        "experiments/phase4/metrics.json",
        ["class_breakdown", "mean_entropy", "clean_attack"],
        "clean attack inputs, representative p0.05 model"),
    ("homogeneity", "UNSW-NB15", "normal/benign"): (
        "experiments/phase7d/metrics.json",
        ["intrinsic", "measures", "ent_normal", "mean"],
        "clean models, mean over 5 seeds"),
    ("homogeneity", "UNSW-NB15", "attack"): (
        "experiments/phase7d/metrics.json",
        ["intrinsic", "measures", "ent_attack", "mean"],
        "clean models, mean over 5 seeds"),
}
DATASETS = ["CICIDS2017", "UNSW-NB15"]
CLASSES = ["normal/benign", "attack"]
COLORS = {"normal/benign": "#4C72B0", "attack": "#DD8452"}
MAJORITY = {"CICIDS2017": "normal/benign", "UNSW-NB15": "attack"}


def load_values() -> dict:
    cache, out = {}, {}
    for key, (path, keys, note) in SOURCES.items():
        if path not in cache:
            cache[path] = json.loads(resolve_path(path).read_text(encoding="utf-8"))
        v = cache[path]
        for k in keys:
            v = v[k]
        out[key] = {"value": float(v), "source": f"{path}: {'.'.join(keys)}", "note": note}
    return out


def plot(values: dict, png_path) -> None:
    plt.rcParams.update({"font.family": "serif", "font.size": 11})
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
    width = 0.36
    panels = [("reachability", "(a) Model-level reachability",
               r"min. mask $L_1$ norm (clean model)", 7.5),
              ("homogeneity", "(b) Input-level homogeneity",
               "mean prediction entropy (clean)", 0.27)]
    for ax, (panel, title, ylabel, ymax) in zip(axes, panels):
        for i, cls in enumerate(CLASSES):
            xs = [d + (i - 0.5) * width for d in range(len(DATASETS))]
            ys = [values[(panel, d, cls)]["value"] for d in DATASETS]
            ax.bar(xs, ys, width, color=COLORS[cls], edgecolor="black", linewidth=0.8, label=cls)
        if panel == "reachability":
            for d_i, d in enumerate(DATASETS):
                cls = MAJORITY[d]; i = CLASSES.index(cls)
                y = values[(panel, d, cls)]["value"]
                ax.text(d_i + (i - 0.5) * width, y + 0.1, f"{cls.split('/')[-1]}\n(majority)",
                        ha="center", va="bottom", fontsize=8.5)
        ax.set_xticks(range(len(DATASETS)), DATASETS)
        ax.set_ylim(0, ymax)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, loc="upper right" if panel == "reachability" else "upper left")
    fig.tight_layout()
    fig.savefig(png_path, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Regenerate Figure 3 from committed metrics")
    ap.add_argument("--out-dir", default="experiments/figure3")
    args = ap.parse_args()
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    values = load_values()
    plot(values, out_dir / "figure3.png")
    rows = [{"panel": p, "dataset": d, "class": c, **v} for (p, d, c), v in values.items()]
    (out_dir / "figure3_data.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    for r in rows:
        print(f"  {r['panel']:<12} {r['dataset']:<10} {r['class']:<13} {r['value']:.4f}  <- {r['source']}")
    print(f"  wrote {out_dir / 'figure3.png'}")


if __name__ == "__main__":
    main()
