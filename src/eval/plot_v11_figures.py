"""Publication figures for the extended (v1.1) manuscript (outputs ``experiments/v11_figures/``).

Every panel is drawn from committed files; nothing is retrained or recomputed
except simple summaries of the stored values (ROC of stored per-input scores,
means). Each figure is written as PNG (600 dpi), PDF, and SVG at its final
printed size (double column 17.5 cm wide), with a serif font, the Okabe-Ito
colorblind-safe palette, no titles inside the image, and a ``<name>_data.json``
listing the source files and fields of every plotted value.

Figures:
  fig1_threat_model     threat model with the four detectors in their two regimes
  fig2_strip_entropy    CICIDS2017 MLP STRIP entropy distributions + ROC (article Figure 2, restyled)
  fig3_class_properties per-class NC mask, STRIP entropy, MM-BD margin on the 10 clean
                        screening models of each dataset x architecture cell
  fig4_contrast         target-class NC mask and MM-BD margin, clean vs poisoned, with ROC-AUCs

Run:
    python -m src.eval.plot_v11_figures [--only fig2_strip_entropy,...]
"""

from __future__ import annotations

import argparse
import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

from src.utils import resolve_path  # noqa: E402

OUT = "experiments/v11_figures"
CM = 1 / 2.54
DOUBLE = 17.5 * CM
SINGLE = 8.5 * CM

# Okabe-Ito (colorblind-safe); checked with the dataviz palette validator
C_BENIGN, C_ATTACK, C_TRIG = "#0072B2", "#E69F00", "#CC79A7"
C_CLEAN, C_POIS = "#009E73", "#D55E00"
INK, MUTED, GRID = "#1a1a1a", "#555555", "#d9d9d9"

CELLS = (("CICIDS2017", "mlp", "CICIDS2017\nMLP"), ("CICIDS2017", "cnn1d", "CICIDS2017\n1D-CNN"),
         ("UNSW-NB15", "mlp", "UNSW-NB15\nMLP"), ("UNSW-NB15", "cnn1d", "UNSW-NB15\n1D-CNN"))


def style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "legend.frameon": False,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.edgecolor": INK,
        "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.2,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})


def _j(p):
    return json.loads(resolve_path(p).read_text(encoding="utf-8"))


def save(fig, name, data, tight=True):
    out = resolve_path(OUT)
    out.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"savefig.bbox": "tight" if tight else "standard"}):
        for ext in ("png", "pdf", "svg"):
            fig.savefig(out / f"{name}.{ext}", metadata={"Date": None} if ext in ("pdf", "svg") else None)
    plt.close(fig)
    from PIL import Image

    with Image.open(out / f"{name}.png") as im:
        w, h = im.size
    data = {"figure": name, "command": "python -m src.eval.plot_v11_figures",
            "printed_size_cm": [round(w / 600 * 2.54, 2), round(h / 600 * 2.54, 2)], "png_dpi": 600,
            "formats": ["png", "pdf", "svg"], **data}
    (out / f"{name}_data.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"  wrote {name}.png/.pdf/.svg + {name}_data.json")


def panel_label(ax, s, x=-0.14, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")


# ---------------------------------------------------------------------------
# Figure 1: threat model
# ---------------------------------------------------------------------------
def _box(ax, x, y, w, h, text=None, fc="white", ec=INK, lw=0.8, ls="-", fs=7.5, weight="normal",
         r=1.2, va="center", ty=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, ls=ls))
    if text:
        ax.text(x + w / 2, ty if ty is not None else y + h / 2, text, ha="center", va=va, fontsize=fs,
                weight=weight, linespacing=1.25)


def _arrow(ax, p, q, text=None, tx=None, ty=None, ls="-", color=INK, fs=6.5, ha="center", rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, lw=0.8, color=color,
                                 ls=ls, shrinkA=0, shrinkB=0, connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text(tx, ty, text, fontsize=fs, ha=ha, va="center", color=MUTED, style="italic",
                linespacing=1.2)


def fig1():
    W, H = 175, 74
    fig = plt.figure(figsize=(DOUBLE, DOUBLE * H / W))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    model_fc, run_fc = "#e6f1f8", "#f6e9f1"    # light tints of the model-level / runtime hues

    # supply chain (top lane)
    _box(ax, 1, 40, 33, 27, "Untrusted supplier\n(attacker)", fs=7.5, weight="bold", va="top", ty=64.5)
    ax.text(17.5, 50, "dirty-label backdoor:\ntrigger on 4 forward-flow\nfeatures; triggered\nattack flows → benign",
            ha="center", va="center", fontsize=6.5, color=MUTED, linespacing=1.2)
    _arrow(ax, (34, 53.5), (43, 53.5))
    _box(ax, 43, 43, 27, 21, "Imported IDS model\n(possibly backdoored)\n\nMLP or 1D-CNN", fs=7.5)
    ax.text(56.5, 39.5, "importing party\n(individual or unit)", ha="center", va="top", fontsize=6.5,
            color=MUTED, style="italic", linespacing=1.2)
    _arrow(ax, (70, 53.5), (78, 53.5), "intake", 74, 56.5)

    # Regime 1: model-intake screening gate
    _box(ax, 78, 36, 52, 34, fc="#f7f7f7", ec=INK, lw=0.9)
    ax.text(104, 67.2, "Model-intake screening gate", ha="center", va="top", fontsize=7.5, weight="bold")
    ax.text(104, 63.2, "regime 1: pre-deployment, per model (institution)", ha="center", va="top",
            fontsize=6.5, color=MUTED, style="italic")
    _box(ax, 81, 46, 22, 11.5, "Neural Cleanse", fc=model_fc, ec=C_BENIGN, lw=1.0, fs=7.5)
    _box(ax, 105, 46, 22, 11.5, "MM-BD", fc=model_fc, ec=C_BENIGN, lw=1.0, fs=7.5)
    ax.text(104, 41.2, "clean holdout + clean reference models\n→ accept / reject the model",
            ha="center", va="center", fontsize=6.5, color=MUTED, linespacing=1.2)
    _arrow(ax, (130, 53.5), (139, 53.5), "accept", 134.5, 56.5)
    _box(ax, 139, 43, 34, 21, "Deployed NIDS\n\nflow → benign / attack", fs=7.5)

    # Regime 2: runtime monitoring
    _box(ax, 78, 2, 52, 28, fc="#f7f7f7", ec=INK, lw=0.9)
    ax.text(104, 27.2, "Runtime monitoring", ha="center", va="top", fontsize=7.5, weight="bold")
    ax.text(104, 23.2, "regime 2: after deployment, per input", ha="center", va="top", fontsize=6.5,
            color=MUTED, style="italic")
    _box(ax, 81, 9.5, 22, 10.5, "STRIP", fc=run_fc, ec=C_TRIG, lw=1.0, fs=7.5)
    _box(ax, 105, 9.5, 22, 10.5, "IBD-PSC", fc=run_fc, ec=C_TRIG, lw=1.0, fs=7.5)
    ax.text(104, 5.3, "clean holdout → flag each suspicious flow", ha="center", va="center", fontsize=6.5,
            color=MUTED)
    _box(ax, 139, 6, 34, 20, "Network traffic\n\nbenign · attack ·\ntriggered attack", fs=7.5)
    _arrow(ax, (156, 26), (156, 43), "flows", 160.5, 34.5)
    _arrow(ax, (139, 16), (130, 16))
    ax.add_patch(FancyArrowPatch((139, 46), (127, 25), arrowstyle="<|-|>", mutation_scale=6, lw=0.7,
                                 color=MUTED, ls=(0, (2, 1.5)), connectionstyle="arc3,rad=0.15"))
    ax.text(139.5, 33.5, "queries the\ndeployed model", fontsize=6.5, color=MUTED, style="italic",
            ha="left", va="center", linespacing=1.2)

    # defender assumptions
    _box(ax, 1, 2, 69, 28, fc="white", ec=MUTED, lw=0.7, ls=(0, (3, 2)))
    ax.text(35.5, 26.8, "Defender", ha="center", va="top", fontsize=7.5, weight="bold")
    ax.text(4, 21.5, "holds:  the trained model (white-box: parameters,\n            gradients) and a small clean holdout",
            ha="left", va="top", fontsize=6.5, linespacing=1.25)
    ax.text(4, 14.5, "lacks:   training data, trigger, triggered inputs",
            ha="left", va="top", fontsize=6.5, linespacing=1.25)
    ax.text(4, 8.0, "model-level detectors need no triggered input",
            ha="left", va="top", fontsize=6.5, color=MUTED, style="italic")
    save(fig, "fig1_threat_model", {
        "description": "Threat model: an untrusted supplier delivers a possibly backdoored IDS model (MLP or "
                       "1D-CNN); the institution's model-intake screening gate (regime 1) runs Neural Cleanse "
                       "and MM-BD with a clean holdout and clean reference models; after deployment, runtime "
                       "monitoring (regime 2) runs STRIP and IBD-PSC on each flow. Conceptual; no data.",
        "sources": ["article Section 2.1 (threat model and defender assumptions)",
                    "src/gate/screen_model.py (model-intake gate)", "src/detectors/base.py (detector regimes)"],
        "size_cm": [17.5, round(17.5 * H / W, 2)]}, tight=False)


# ---------------------------------------------------------------------------
# Figure 2: STRIP entropy (article Figure 2, restyled)
# ---------------------------------------------------------------------------
def fig2():
    src = "experiments/phase7f/fig3_entropy_raw.csv"
    groups = {"clean_benign": [], "clean_attack": [], "triggered": []}
    with open(resolve_path(src), encoding="utf-8") as fh:
        r = csv.reader(fh)
        next(r)
        for g, v in r:
            groups[g].append(float(v))
    e = {g: np.asarray(v) for g, v in groups.items()}
    ref = _j("experiments/phase7f/metrics.json")["CICIDS2017"]["part4_real_entropy"]

    fig, (a, b) = plt.subplots(1, 2, figsize=(DOUBLE, 6.2 * CM), gridspec_kw={"width_ratios": [1.65, 1]})
    hi = max(v.max() for v in e.values())
    bins = np.linspace(0, hi, 61)
    grid = np.linspace(0, hi, 400)
    spec = (("clean_benign", C_BENIGN, "-", "Clean benign"), ("clean_attack", C_ATTACK, "--", "Clean attack"),
            ("triggered", C_TRIG, ":", "Triggered attack"))
    for g, col, ls, lbl in spec:
        a.hist(e[g], bins=bins, density=True, histtype="step", color=col, lw=0.6, alpha=0.75)
        # entropy is >= 0: reflect the Gaussian KDE at 0 so no mass leaks below the boundary
        k = gaussian_kde(e[g])
        a.plot(grid, k(grid) + k(-grid), color=col, ls=ls, lw=1.4,
               label=f"{lbl} (n = {len(e[g]):,}, mean {e[g].mean():.3f})")
    a.set_xlabel("STRIP entropy (nats)")
    a.set_ylabel("Density")
    a.set_xlim(0, hi)
    a.set_ylim(bottom=0)
    a.legend(loc="upper right", handlelength=2.4)
    panel_label(a, "(a)", x=-0.1)

    y = np.r_[np.ones(len(e["triggered"])), np.zeros(len(e["clean_benign"]))]
    s = -np.r_[e["triggered"], e["clean_benign"]]
    fpr, tpr, _ = roc_curve(y, s)
    auc = float(roc_auc_score(y, s))
    b.plot([0, 1], [0, 1], color=GRID, lw=0.8, ls="--")
    b.plot(fpr, tpr, color=C_TRIG, lw=1.4)
    b.text(0.97, 0.06, f"ROC-AUC = {auc:.3f}", ha="right", va="bottom", fontsize=7.5, transform=b.transAxes)
    b.set_xlabel("False-positive rate (clean benign)")
    b.set_ylabel("True-positive rate (triggered)")
    b.set_xlim(0, 1); b.set_ylim(0, 1.01); b.set_aspect("equal")
    panel_label(b, "(b)", x=-0.3)
    fig.tight_layout(w_pad=1.5)
    # the CSV stores 6 decimals, so ties shift the AUC in the 7th decimal
    assert abs(auc - ref["roc_auc_triggered_vs_clean_benign"]) < 1e-4
    save(fig, "fig2_strip_entropy", {
        "description": "STRIP entropy (blend, alpha 0.7, N = 100) per input on the representative "
                       "CICIDS2017 benign-target MLP (causal_ids_B_imbal_tgtBenign_seed42): (a) histogram "
                       "(outline, 60 bins) and Gaussian KDE reflected at 0 (entropy >= 0) per group, (b) ROC of triggered vs clean benign inputs (lower entropy = "
                       "more suspicious). Same data as the article's Figure 2.",
        "sources": [{"file": src, "fields": "group, entropy (5,000 rows per group)"},
                    {"file": "experiments/phase7f/metrics.json",
                     "fields": "CICIDS2017.part4_real_entropy (n, entropy_stats, roc_auc_triggered_vs_clean_benign)"}],
        "values": {"means": {g: float(v.mean()) for g, v in e.items()}, "n": {g: int(len(v)) for g, v in e.items()},
                   "roc_auc": auc, "roc_auc_reference": ref["roc_auc_triggered_vs_clean_benign"],
                   "note": "roc_auc is computed from the 6-decimal CSV; roc_auc_reference is the full-precision value in phase7f"},
        "size_cm": [17.5, 6.2]})


# ---------------------------------------------------------------------------
# Figure 3: class properties on clean models
# ---------------------------------------------------------------------------
def fig3():
    cp = _j("experiments/v11_cnn/metrics.json")["class_properties"]
    props = (("nc_mask_l1", "Minimal mask L1 (NC)", "(a)"),
             ("strip_entropy", "Mean STRIP entropy (nats)", "(b)"),
             ("mmbd_margin", r"Max margin $r_c$ (MM-BD)", "(c)"))
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 6.4 * CM))
    x = np.arange(len(CELLS))
    w = 0.36
    values = {}
    for ax, (p, ylab, lab) in zip(axes, props):
        for k, (cls, col, off, name) in enumerate((("0", C_BENIGN, -w / 2 - 0.01, "Benign/normal"),
                                                   ("1", C_ATTACK, w / 2 + 0.01, "Attack"))):
            means, lo, hi, maj = [], [], [], []
            for d, arch, _ in CELLS:
                v = cp[d][arch][p][f"class{cls}"]
                means.append(v["mean"]); lo.append(v["boot_ci95"][0]); hi.append(v["boot_ci95"][1])
                maj.append(cp[d]["majority_class"] == cls)
                values.setdefault(f"{d}/{arch}", {})[f"{p}/class{cls}"] = {
                    "mean": v["mean"], "boot_ci95": v["boot_ci95"], "n_models": v["n"]}
            means, lo, hi = map(np.asarray, (means, lo, hi))
            bars = ax.bar(x + off, means, w, color=col, edgecolor=INK, linewidth=0.5,
                          label=name, zorder=2)
            for bar, m in zip(bars, maj):
                if m:
                    bar.set_hatch("////")
            ax.errorbar(x + off, means, yerr=[means - lo, hi - means], fmt="none", ecolor=INK,
                        elinewidth=0.7, capsize=1.6, capthick=0.7, zorder=3)
        ax.set_xticks(x, ["MLP" if c[1] == "mlp" else "1D-CNN" for c in CELLS], fontsize=6.5)
        for xc, dname in ((0.5, "CICIDS2017"), (2.5, "UNSW-NB15")):
            ax.text(xc, -0.17, dname, transform=ax.get_xaxis_transform(), ha="center", va="top",
                    fontsize=7)
        ax.set_ylabel(ylab)
        ax.set_ylim(bottom=0)
        ax.yaxis.grid(True, color=GRID, lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.axvline(1.5, color=GRID, lw=0.6)
        panel_label(ax, lab, x=-0.24)
    plt.rcParams["hatch.linewidth"] = 0.5
    from matplotlib.patches import Patch
    handles = [Patch(fc=C_BENIGN, ec=INK, lw=0.5, label="Benign/normal"),
               Patch(fc=C_ATTACK, ec=INK, lw=0.5, label="Attack"),
               Patch(fc="white", ec=INK, lw=0.5, hatch="////", label="Majority class (hatched)")]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig3_class_properties", {
        "description": "Per-class properties of the 10 clean screening models in each dataset x architecture "
                       "cell: (a) minimal Neural Cleanse mask L1 (target-excluded, count-matched; smaller = "
                       "easier to reach), (b) mean STRIP entropy of clean test inputs (blend, alpha 0.7, N = 100; "
                       "smaller = more homogeneous), (c) MM-BD maximum margin (min/max box, holdout-seeded starts; "
                       "larger = easier to reach). Bars: mean over 10 models; error bars: bootstrap 95% CI over "
                       "models (10,000 resamples); hatched: majority class (CICIDS2017 benign 83%, UNSW-NB15 "
                       "attack 68%).",
        "sources": [{"file": "experiments/v11_cnn/metrics.json",
                     "fields": "class_properties.<dataset>.<mlp|cnn1d>.<nc_mask_l1|strip_entropy|mmbd_margin>"
                               ".class0/class1 (mean, boot_ci95, n, values); class_properties.<dataset>.majority_class"},
                    "models: clean lists of configs/v11_screening.yaml (MLP) and configs/v11_cnn.yaml (1D-CNN)"],
        "values": values,
        "majority_class": {d: cp[d]["majority_class"] for d in ("CICIDS2017", "UNSW-NB15")},
        "size_cm": [17.5, 6.4]})


# ---------------------------------------------------------------------------
# Figure 4: target-class contrast
# ---------------------------------------------------------------------------
def _cell_sources(d, arch):
    if arch == "mlp":
        nc = _j("experiments/v11_screening/metrics.json")["datasets"][d]
        mm = _j("experiments/v11_mmbd/metrics.json")["datasets"][d]
        return nc, mm, "experiments/v11_screening/metrics.json", "experiments/v11_mmbd/metrics.json"
    nc = _j("experiments/v11_cnn/nc/metrics.json")["datasets"][d]
    mm = _j("experiments/v11_cnn/mmbd/metrics.json")["datasets"][d]
    return nc, mm, "experiments/v11_cnn/nc/metrics.json", "experiments/v11_cnn/mmbd/metrics.json"


def fig4():
    fig, axes = plt.subplots(2, 4, figsize=(DOUBLE, 10.5 * CM))
    rng = np.random.default_rng(0)
    values, srcs = {}, set()
    for j, (d, arch, lbl) in enumerate(CELLS):
        nc, mm, f_nc, f_mm = _cell_sources(d, arch)
        srcs.update([f_nc, f_mm])
        a = mm["boxes"]["minmax/seeded"]
        rows = (("nc", nc["per_model"], lambda p: p["l1"]["0"], nc["n10"]["auc"], "NC"),
                ("mmbd", a["per_model"], lambda p: p["margins"]["0"], a["auc"], "MM-BD"))
        for i, (key, per, get, auc, det) in enumerate(rows):
            ax = axes[i][j]
            data = [[get(p) for p in per.values() if p["role"] == role] for role in ("clean", "poisoned")]
            bp = ax.boxplot(data, positions=[0, 1], widths=0.5, showfliers=False, patch_artist=True,
                            medianprops={"color": INK, "lw": 0.9}, whiskerprops={"lw": 0.6, "color": MUTED},
                            capprops={"lw": 0.6, "color": MUTED}, boxprops={"lw": 0.6, "edgecolor": MUTED})
            for patch, col in zip(bp["boxes"], (C_CLEAN, C_POIS)):
                patch.set_facecolor(col); patch.set_alpha(0.22)
            for k, (vals, col, mk) in enumerate(zip(data, (C_CLEAN, C_POIS), ("o", "s"))):
                ax.scatter(k + rng.uniform(-0.13, 0.13, len(vals)), vals, s=11, marker=mk, color=col,
                           edgecolor="white", linewidth=0.4, zorder=3)
            ax.set_xticks([0, 1], ["Clean", "Poisoned"])
            ax.set_xlim(-0.6, 1.6)
            ax.yaxis.grid(True, color=GRID, lw=0.5)
            ax.set_axisbelow(True)
            top = max(max(v) for v in data)
            ax.set_ylim(0, top * 1.32)
            ax.text(0.5, 0.97, f"{det} ROC-AUC {auc['roc_auc']:.2f}\n[{auc['roc_ci95'][0]:.2f}, "
                               f"{auc['roc_ci95'][1]:.2f}]", transform=ax.transAxes, ha="center", va="top",
                    fontsize=6.5, linespacing=1.15)
            if j == 0:
                ax.set_ylabel("Target-class mask L1 (NC)" if key == "nc" else r"Target-class margin $r_0$ (MM-BD)")
            if i == 0:
                ax.text(0.5, 1.08, lbl.replace("\n", ", "), transform=ax.transAxes, ha="center", va="bottom",
                        fontsize=7.5, weight="bold")
            panel_label(ax, f"({'abcdefgh'[i * 4 + j]})", x=-0.32 if j == 0 else -0.2, y=1.0)
            values[f"{d}/{arch}/{key}"] = {"clean": data[0], "poisoned": data[1],
                                           "detector_roc_auc": auc["roc_auc"], "detector_roc_ci95": auc["roc_ci95"]}
    fig.tight_layout(h_pad=1.6, w_pad=0.9)
    save(fig, "fig4_contrast", {
        "description": "Target-class (benign/normal) statistic of the 10 clean and 10 poisoned models per cell: "
                       "top, Neural Cleanse minimal mask L1 of the target class; bottom, MM-BD maximum margin of "
                       "the target class (min/max box, holdout-seeded starts). Boxes: quartiles and whiskers "
                       "(1.5 IQR); points: individual models. The annotated ROC-AUC [bootstrap 95% CI] is that of "
                       "each detector's screening statistic (NC: mask-norm ratio max/min; MM-BD: S = max_c r_c / "
                       "calibration median), not of the plotted target-class value.",
        "sources": [{"file": f, "fields": ("datasets.<dataset>.per_model.<model>.l1.0, .role; datasets.<dataset>.n10.auc"
                                           if "mmbd" not in f else
                                           "datasets.<dataset>.boxes['minmax/seeded'].per_model.<model>.margins.0, "
                                           ".role; datasets.<dataset>.boxes['minmax/seeded'].auc")} for f in sorted(srcs)],
        "values": values, "size_cm": [17.5, 10.5]})


FIGS = {"fig1_threat_model": fig1, "fig2_strip_entropy": fig2, "fig3_class_properties": fig3,
        "fig4_contrast": fig4}


def main():
    ap = argparse.ArgumentParser(description="v1.1 publication figures")
    ap.add_argument("--only", default=None, help="comma-separated figure names")
    a = ap.parse_args()
    style()
    for name in (a.only.split(",") if a.only else FIGS):
        FIGS[name]()


if __name__ == "__main__":
    main()
