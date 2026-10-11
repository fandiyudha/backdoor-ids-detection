"""Figure 3 of the article: the intrinsically easy class, by (a) model-level
reachability and (b) input-level homogeneity, on CICIDS2017 and UNSW-NB15.

Plots committed values only (no models are loaded). Every bar is the mean of a
committed ``{mean, std}`` statistic over clean models; the file, field, number of
models (n), and sd of each bar are listed in ``SOURCES`` and written next to the
figure. The visual style (size, fonts, colors, legend, annotations) matches the
figure in the article; PNG at 300 dpi.

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

# (panel, dataset, class) -> (metrics file, key path to a {mean, std} statistic,
#                             key path to n (None = len(stat["values"])), models averaged)
_6D = "experiments/phase6d/metrics.json"
_7D = "experiments/phase7d/metrics.json"
_CE = "experiments/clean_entropy/metrics.json"
_6D_N = ["per_condition", "B_imbal_tgtBenign", "n_seeds"]
_6D_MODELS = "clean models of controlled-experiment condition B (natural class ratio), seeds 42-46"
_7D_MODELS = "clean UNSW-NB15 models, seeds 42-46"
_CE_MODELS = "clean CICIDS2017 reference models, seeds 43-47"
SOURCES = {
    ("reachability", "CICIDS2017", "normal/benign"): (
        _6D, ["per_condition", "B_imbal_tgtBenign", "mask_clean_benign"], _6D_N, _6D_MODELS),
    ("reachability", "CICIDS2017", "attack"): (
        _6D, ["per_condition", "B_imbal_tgtBenign", "mask_clean_attack"], _6D_N, _6D_MODELS),
    ("reachability", "UNSW-NB15", "normal/benign"): (
        _7D, ["intrinsic", "measures", "nc_mask_normal"], None, _7D_MODELS),
    ("reachability", "UNSW-NB15", "attack"): (
        _7D, ["intrinsic", "measures", "nc_mask_attack"], None, _7D_MODELS),
    ("homogeneity", "CICIDS2017", "normal/benign"): (
        _CE, ["ent_benign"], None, _CE_MODELS),
    ("homogeneity", "CICIDS2017", "attack"): (
        _CE, ["ent_attack"], None, _CE_MODELS),
    ("homogeneity", "UNSW-NB15", "normal/benign"): (
        _7D, ["intrinsic", "measures", "ent_normal"], None, _7D_MODELS),
    ("homogeneity", "UNSW-NB15", "attack"): (
        _7D, ["intrinsic", "measures", "ent_attack"], None, _7D_MODELS),
}
DATASETS = ["CICIDS2017", "UNSW-NB15"]
CLASSES = ["normal/benign", "attack"]
COLORS = {"normal/benign": "#4C72B0", "attack": "#DD8452"}
MAJORITY = {"CICIDS2017": "normal/benign", "UNSW-NB15": "attack"}

# Style of the figure in the article.
FIGSIZE = (7.0, 2.72)
DPI = 300
FONT_SIZE = 8
LEGEND_SIZE = 7
ANNOTATION_SIZE = 5.75
BAR_WIDTH = 0.36
BAR_EDGE_WIDTH = 0.6


def _get(obj, keys):
    for k in keys:
        obj = obj[k]
    return obj


def load_values() -> dict:
    cache, out = {}, {}
    for key, (path, stat_keys, n_keys, models) in SOURCES.items():
        if path not in cache:
            cache[path] = json.loads(resolve_path(path).read_text(encoding="utf-8"))
        stat = _get(cache[path], stat_keys)
        n = _get(cache[path], n_keys) if n_keys else len(stat["values"])
        out[key] = {"value": float(stat["mean"]), "sd": float(stat["std"]), "n": int(n),
                    "models": models, "source": f"{path}: {'.'.join(stat_keys)}"}
    return out


def plot(values: dict, png_path) -> None:
    plt.rcParams.update({"font.family": "serif", "font.size": FONT_SIZE})
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
    panels = [("reachability", "(a) Model-level reachability",
               r"min. mask $L_1$ norm (clean model)", 7.5),
              ("homogeneity", "(b) Input-level homogeneity",
               "mean prediction entropy (clean)", 0.27)]
    for ax, (panel, title, ylabel, ymax) in zip(axes, panels):
        for i, cls in enumerate(CLASSES):
            xs = [d + (i - 0.5) * BAR_WIDTH for d in range(len(DATASETS))]
            ys = [values[(panel, d, cls)]["value"] for d in DATASETS]
            ax.bar(xs, ys, BAR_WIDTH, color=COLORS[cls], edgecolor="black",
                   linewidth=BAR_EDGE_WIDTH, label=cls)
        if panel == "reachability":
            for d_i, d in enumerate(DATASETS):
                cls = MAJORITY[d]; i = CLASSES.index(cls)
                y = values[(panel, d, cls)]["value"]
                ax.text(d_i + (i - 0.5) * BAR_WIDTH, y + 0.1, f"{cls.split('/')[-1]}\n(majority)",
                        ha="center", va="bottom", fontsize=ANNOTATION_SIZE)
        ax.set_xticks(range(len(DATASETS)), DATASETS)
        ax.set_ylim(0, ymax)
        ax.set_title(title, fontsize=FONT_SIZE)
        ax.set_ylabel(ylabel)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=LEGEND_SIZE,
                  loc="upper right" if panel == "reachability" else "upper left")
    fig.tight_layout()
    fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
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
        print(f"  {r['panel']:<12} {r['dataset']:<10} {r['class']:<13} "
              f"{r['value']:.4f} ± {r['sd']:.4f} (n={r['n']})  <- {r['source']}")
    print(f"  wrote {out_dir / 'figure3.png'}")


if __name__ == "__main__":
    main()
