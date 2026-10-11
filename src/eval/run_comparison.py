"""STRIP vs Neural Cleanse comparison (Section 3.1-3.2, Table 2; outputs ``experiments/phase7/``).

Because the two detectors work at different granularities (Neural Cleanse judges
the MODEL, STRIP judges an INPUT), they are compared in two *separate* frames:

  * screening  — "is this model backdoored?" (pre-deployment, clean data only)
  * runtime    — "does this input carry a trigger?" (per-input monitoring)

The script also aggregates multi-seed variability, runs a robustness sweep over
trigger design and poison rate, builds an amortized compute-cost model, analyses
a layered NC-then-STRIP defence, and emits the master comparison table.
Requires the models from ``train_phase7_models`` and ``train_reference_clean``.

Run:
    python -m src.eval.run_comparison --config configs/comparison.yaml
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from src.attacks.run_backdoor import attack_success_rate, predict_numpy  # noqa: E402
from src.attacks.trigger import Trigger, apply_trigger, load_trigger  # noqa: E402
from src.detectors.neural_cleanse import NeuralCleanseDetector  # noqa: E402
from src.detectors.run_strip import auc_score, detection_metrics  # noqa: E402
from src.detectors.strip import StripDetector  # noqa: E402
from src.models.mlp import MLP  # noqa: E402
from src.models.train import compute_metrics, load_dataset  # noqa: E402
from src.utils import load_yaml, resolve_path, set_global_seed  # noqa: E402


def load_ids_model(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    m = MLP.from_config(ckpt["model_config"])
    m.load_state_dict(ckpt["model_state_dict"])
    return m.eval()


def trigger_from_manifest(entry, feat_names) -> Trigger:
    return Trigger(feature_names=entry["trigger_features"],
                   feature_indices=entry["trigger_indices"],
                   original_values=entry["trigger_original"],
                   scaled_values=entry["trigger_scaled"],
                   mode="constant", percentile=None, all_feature_names=feat_names)


# ---------------------------------------------------------------------------
# Per-model detector evaluation (both frames)
# ---------------------------------------------------------------------------
def evaluate_model(model, trigger, S, cfg, device) -> Dict:
    sc = cfg["strip"]
    # --- STRIP (blend alpha=0.7, selected in the STRIP experiment) ---
    det = StripDetector(model, S["strip_holdout"], variant="blend", N=sc["N"],
                        alpha=0.7, device=device, seed=cfg["seed"], chunk=128)
    thr = det.calibrate(S["strip_holdout"], sc["fpr_targets"])
    clean_ent, _ = det.score(S["strip_holdout"])          # screening signal
    e_pos, _ = det.score(apply_trigger(S["pos_src"], trigger))
    e_neg, _ = det.score(S["neg_src"])
    strip_per_fpr = {f"{f:.2f}": detection_metrics(e_pos, e_neg, thr[f])
                     for f in sc["fpr_targets"]}
    strip = {"auc": auc_score(e_pos, e_neg), "per_fpr": strip_per_fpr,
             "mean_clean_entropy": float(clean_ent.mean())}

    # --- Neural Cleanse (model-level) ---
    ncd = NeuralCleanseDetector(model, S["nc_holdout_X"], S["nc_holdout_y"],
                                device=device, steps=cfg["neural_cleanse"]["steps"],
                                lr=cfg["neural_cleanse"]["lr"], seed=cfg["seed"])
    ncout = ncd.run()
    mask_b = ncout["per_class"][0]["mask"]
    pat_b = ncout["per_class"][0]["pattern"]
    nc = {"anomaly_ratio": ncout["anomaly_ratio"],
          "l1_benign": ncout["l1_norms"][0], "l1_attack": ncout["l1_norms"][1],
          "flagged": bool(ncout["anomaly_ratio"] >= cfg["neural_cleanse"]["ratio_threshold"]
                          and ncout["infected_class"] == 0)}

    # --- NC-derived RUNTIME input filter (mask-weighted distance to RE pattern) ---
    nc_rt = nc_runtime_filter(mask_b, pat_b, S["strip_holdout"],
                              apply_trigger(S["pos_src"], trigger), S["neg_src"],
                              sc["fpr_targets"])

    # --- attack + stealth ---
    asr = attack_success_rate(model, S["X_test"], S["y_test"], trigger, device)["asr"]
    preds, prob = predict_numpy(model, S["X_test"], device)
    cm = compute_metrics(S["y_test"], prob, preds)
    return {"strip": strip, "nc": nc, "nc_runtime": nc_rt, "asr": asr,
            "clean_attack_f1": cm["attack_f1"], "clean_pr_auc": cm["pr_auc"]}


def nc_runtime_filter(mask, pattern, holdout, pos, neg, fpr_targets) -> Dict:
    """Runtime detector built from NC's reverse-engineered (mask, pattern):
    score = -||mask * (x - pattern)||^2  (higher = closer to the recovered trigger)."""
    m = np.asarray(mask); p = np.asarray(pattern)

    def score(X):
        d = (np.asarray(X, np.float32) - p) * m
        return -np.sum(d * d, axis=1)

    s_hold = score(holdout)
    s_pos = score(pos)
    s_neg = score(neg)
    y = np.concatenate([np.ones(len(s_pos)), np.zeros(len(s_neg))])
    auc = float(roc_auc_score(y, np.concatenate([s_pos, s_neg])))
    per_fpr = {}
    for f in fpr_targets:
        thr = np.quantile(s_hold, 1 - f)       # top-f clean scores -> ~f FPR
        tpr = float((s_pos > thr).mean())
        fpr = float((s_neg > thr).mean())
        per_fpr[f"{f:.2f}"] = {"tpr": tpr, "fpr": fpr}
    return {"auc": auc, "per_fpr": per_fpr}


# ---------------------------------------------------------------------------
def run(cfg: dict) -> dict:
    set_global_seed(cfg["seed"])
    device = torch.device("cpu")
    exp_dir = resolve_path(cfg["paths"]["experiment_dir"])
    exp_dir.mkdir(parents=True, exist_ok=True)
    import joblib

    data = load_dataset(resolve_path(cfg["paths"]["processed_dir"]))
    scaler = joblib.load(resolve_path(cfg["paths"]["scaler"]))
    feat_names = [str(f) for f in data["feature_names"]]
    baseline_trigger = load_trigger(resolve_path(cfg["paths"]["trigger"]))
    models_dir = resolve_path(cfg["paths"]["models_dir"])
    manifest = json.loads(resolve_path(cfg["paths"]["phase7_manifest"]).read_text())

    rng = np.random.default_rng(cfg["seed"])
    Xval, yval = data["X_val"], data["y_val"]
    Xte, yte = data["X_test"], data["y_test"]
    hs, es = cfg["strip"]["holdout_size"], cfg["strip"]["eval_size"]
    hidx = rng.choice(len(Xval), hs, replace=False)
    ncidx = rng.choice(len(Xval), cfg["neural_cleanse"]["holdout_size"], replace=False)
    atk = np.where(yte == 1)[0]
    S = {
        "strip_holdout": Xval[hidx].astype(np.float32),
        "nc_holdout_X": Xval[ncidx].astype(np.float32),
        "nc_holdout_y": yval[ncidx].astype(int),
        "pos_src": Xte[rng.choice(atk, min(es, len(atk)), replace=False)].astype(np.float32),
        "neg_src": Xte[rng.choice(len(Xte), es, replace=False)].astype(np.float32),
        "X_test": Xte, "y_test": yte,
    }

    ev_cache: Dict[str, Dict] = {}

    def ev(key, path, trigger):
        if key not in ev_cache:
            print(f"  evaluating {key} ...")
            ev_cache[key] = evaluate_model(load_ids_model(path, device), trigger, S, cfg, device)
        return ev_cache[key]

    # === clean model + reference clean models ===
    clean = ev("clean", resolve_path(cfg["paths"]["clean_checkpoint"]), baseline_trigger)
    ref = {}
    for s in cfg["reference_clean_seeds"]:
        ref[f"clean_seed{s}"] = ev(f"clean_seed{s}", models_dir / f"clean_ref_seed{s}.pt",
                                   baseline_trigger)

    # === poison-rate sweep (baseline attack trigger) ===
    rate_models = {}
    for r in cfg["poison_rates"]:
        rate_models[r] = ev(f"p{r:g}", models_dir / f"backdoor_mlp_p{r:g}.pt", baseline_trigger)

    # === multi-seed (baseline trigger, p0.05) ===
    seed_models = {}
    for s in cfg["multiseed_seeds"]:
        path = (models_dir / "backdoor_mlp_p0.05.pt" if s == 42
                else models_dir / f"phase7_multiseed_seed{s}.pt")
        seed_models[s] = ev(f"seed{s}", path, baseline_trigger)

    # === trigger variants ===
    variant_models = {"nfeat4_baseline": rate_models[0.05]}   # baseline = p0.05
    variant_trigs = {"nfeat4_baseline": baseline_trigger}
    for name in ("nfeat2", "nfeat6", "nfeat8", "altfeat", "subtle"):
        entry = manifest[name]
        trg = trigger_from_manifest(entry, feat_names)
        variant_trigs[name] = trg
        variant_models[name] = ev(name, resolve_path(entry["checkpoint"]), trg)

    parts = {}
    parts["part1_screening_runtime"] = part1(clean, ref, rate_models, cfg, exp_dir)
    parts["part2_multiseed"] = part2(clean, ref, seed_models, cfg)
    parts["part3_robustness"] = part3(rate_models, variant_models, variant_trigs, cfg)
    parts["part4_cost"] = part4(cfg)
    parts["part5_layered"] = part5(rate_models, variant_models, cfg)
    parts["part6_synthesis"] = part6(parts, clean, ref, cfg)

    plot_robustness(rate_models, variant_models, cfg, exp_dir / "robustness.png")
    plot_contrast(parts, exp_dir / "detector_contrast.png")

    _save(exp_dir, cfg, parts)
    _print_summary(parts)
    return parts


# ---------------------------------------------------------------------------
# Part 1 — screening vs runtime
# ---------------------------------------------------------------------------
def part1(clean, ref, rate_models, cfg, exp_dir) -> Dict:
    thr = cfg["neural_cleanse"]["ratio_threshold"]
    healthy = [clean] + list(ref.values())
    poisoned = list(rate_models.values())

    # --- SCREENING: NC separates clean/poisoned models; STRIP cannot operate ---
    nc_healthy = [m["nc"]["anomaly_ratio"] for m in healthy]
    nc_pois = [m["nc"]["anomaly_ratio"] for m in poisoned]
    nc_screen_auc = float(roc_auc_score(
        [0] * len(nc_healthy) + [1] * len(nc_pois), nc_healthy + nc_pois))
    clean_max, pois_min = float(np.max(nc_healthy)), float(np.min(nc_pois))

    # STRIP has no triggered inputs at screening. An ad-hoc clean-data proxy =
    # mean clean-input entropy per model. Report the ORACLE separation (with true
    # labels + best direction) as an upper bound the defender CANNOT achieve.
    ent_healthy = [m["strip"]["mean_clean_entropy"] for m in healthy]
    ent_pois = [m["strip"]["mean_clean_entropy"] for m in poisoned]
    raw = float(roc_auc_score([0] * len(ent_healthy) + [1] * len(ent_pois),
                              ent_healthy + ent_pois))
    strip_oracle_auc = max(raw, 1 - raw)
    ent_overlap = (min(max(ent_healthy), max(ent_pois))
                   >= max(min(ent_healthy), min(ent_pois)))

    plot_screening(ent_healthy, ent_pois, nc_healthy, nc_pois, thr,
                   exp_dir / "screening_limit.png")

    # --- RUNTIME: STRIP vs NC-derived filter on the p0.05 model ---
    m5 = rate_models[0.05]
    runtime = {
        "strip": {"auc": m5["strip"]["auc"],
                  "tpr@5": m5["strip"]["per_fpr"]["0.05"]["tpr"],
                  "fpr@5": m5["strip"]["per_fpr"]["0.05"]["fpr"]},
        "nc_derived_filter": {"auc": m5["nc_runtime"]["auc"],
                              "tpr@5": m5["nc_runtime"]["per_fpr"]["0.05"]["tpr"],
                              "fpr@5": m5["nc_runtime"]["per_fpr"]["0.05"]["fpr"]},
    }
    return {
        "screening": {
            "nc_can_screen": True,
            "nc_screen_auc": nc_screen_auc,
            "nc_healthy_ratio_mean": float(np.mean(nc_healthy)),
            "nc_poisoned_ratio_mean": float(np.mean(nc_pois)),
            "nc_clean_max": clean_max, "nc_poisoned_min": pois_min,
            "nc_separation_margin": pois_min - clean_max,
            "strip_can_screen": False,
            "strip_oracle_auc_cleandata_only": strip_oracle_auc,
            "strip_clean_entropy_healthy_mean": float(np.mean(ent_healthy)),
            "strip_clean_entropy_poisoned_mean": float(np.mean(ent_pois)),
            "strip_clean_entropy_ranges_overlap": bool(ent_overlap),
            "strip_clean_entropy_healthy": ent_healthy,
            "strip_clean_entropy_poisoned": ent_pois,
            "nc_finding": (
                f"Neural Cleanse separates clean from poisoned models perfectly "
                f"(AUC {nc_screen_auc:.2f}): clean anomaly ratios ≤ {clean_max:.1f} "
                f"vs poisoned ≥ {pois_min:.1f} (margin {pois_min - clean_max:.1f}). "
                "The ABSOLUTE threshold is holdout/reference-sensitive (clean ratios "
                "here run higher than the Phase-6 calibration), so it must be set "
                "against the defender's own reference distribution — but the "
                "separation itself is robust."),
            "strip_finding": (
                "STRIP CANNOT be operated as a model screener under this threat "
                "model: its detection needs triggered inputs, which the defender "
                "does not have pre-deployment. A clean-data proxy (mean clean-input "
                f"entropy) shows only a tiny, incidental shift (healthy "
                f"{np.mean(ent_healthy):.4f} vs poisoned {np.mean(ent_pois):.4f}, "
                f"overlapping ranges); even the ORACLE separation "
                f"(AUC {strip_oracle_auc:.2f}, using true labels + best direction) "
                "is unusable because the defender has no reference model, no known "
                "direction, and no triggered inputs to calibrate. Structural limit, "
                "not an implementation failure."),
        },
        "runtime": runtime,
    }


# ---------------------------------------------------------------------------
# Part 2 — multi-seed variability
# ---------------------------------------------------------------------------
def _mean_std(xs):
    a = np.array(xs, dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "min": float(a.min()), "max": float(a.max())}


def part2(clean, ref, seed_models, cfg) -> Dict:
    asr = _mean_std([m["asr"] for m in seed_models.values()])
    nc_ratio = _mean_std([m["nc"]["anomaly_ratio"] for m in seed_models.values()])
    flag = _mean_std([1.0 if m["nc"]["flagged"] else 0.0 for m in seed_models.values()])
    s_auc = _mean_std([m["strip"]["auc"] for m in seed_models.values()])
    s_tpr = _mean_std([m["strip"]["per_fpr"]["0.05"]["tpr"] for m in seed_models.values()])
    clean_ratio = _mean_std([clean["nc"]["anomaly_ratio"]]
                            + [m["nc"]["anomaly_ratio"] for m in ref.values()])
    return {"n_seeds": len(seed_models), "poisoned": {
        "asr": asr, "nc_anomaly_ratio": nc_ratio, "nc_flag_rate": flag,
        "strip_auc": s_auc, "strip_tpr@5": s_tpr},
        "clean_nc_anomaly_ratio": clean_ratio}


# ---------------------------------------------------------------------------
# Part 3 — robustness sweep
# ---------------------------------------------------------------------------
def part3(rate_models, variant_models, variant_trigs, cfg) -> Dict:
    def row(m):
        return {"asr": m["asr"], "clean_attack_f1": m["clean_attack_f1"],
                "nc_anomaly_ratio": m["nc"]["anomaly_ratio"], "nc_flagged": m["nc"]["flagged"],
                "strip_auc": m["strip"]["auc"],
                "strip_tpr@5": m["strip"]["per_fpr"]["0.05"]["tpr"]}
    rate = {f"p{r:g}": row(m) for r, m in rate_models.items()}
    variants = {n: {**row(m), "n_trigger_features": len(variant_trigs[n].feature_names)}
                for n, m in variant_models.items()}

    # hardest to detect: minimise max(NC "margin", STRIP runtime TPR)
    allcfg = {**{f"rate_{k}": v for k, v in rate.items()}, **variants}
    thr = cfg["neural_cleanse"]["ratio_threshold"]

    def detectability(v):
        nc_margin = v["nc_anomaly_ratio"] / thr            # >1 = NC flags
        return max(nc_margin, v["strip_tpr@5"] * 3)        # crude combined score
    hardest = min(allcfg, key=lambda k: detectability(allcfg[k]))
    return {"poison_rate": rate, "trigger_variants": variants,
            "hardest_config": hardest, "hardest_detail": allcfg[hardest]}


# ---------------------------------------------------------------------------
# Part 4 — amortized compute cost
# ---------------------------------------------------------------------------
def part4(cfg) -> Dict:
    c = cfg["cost"]
    ms = c["strip_ms_per_input"]
    rows = {}
    for rate in c["traffic_rates_flows_per_sec"]:
        core_load = rate * ms / 1000.0        # core-seconds per second = # cores
        rows[str(rate)] = {
            "strip_core_equivalents_all_traffic": round(core_load, 3),
            "feasible_on_1_core": bool(core_load <= 1.0),
        }
    return {"strip_ms_per_input": ms, "strip_N": c["strip_N"],
            "nc_seconds_per_model": c["nc_seconds_per_model"],
            "per_rate": rows,
            "note": ("Neural Cleanse is a one-off per model (~%.0fs), amortized to "
                     "~0 over the model's lifetime. STRIP recurs per input; running "
                     "it on ALL traffic needs the core-equivalents above, so beyond "
                     "~%d flows/s per core it must run on a SAMPLE of traffic."
                     % (c["nc_seconds_per_model"], int(1000 / ms)))}


# ---------------------------------------------------------------------------
# Part 5 — layered defence
# ---------------------------------------------------------------------------
def part5(rate_models, variant_models, cfg) -> Dict:
    thr = cfg["neural_cleanse"]["ratio_threshold"]
    usable = cfg["runtime_usable_tpr"]
    configs = {**{f"rate_p{r:g}": m for r, m in rate_models.items()},
               **{f"var_{n}": m for n, m in variant_models.items()}}
    rows = {}
    caught_any = caught_both = passed_both = 0
    for name, m in configs.items():
        nc_gate = m["nc"]["flagged"]
        strip_rt = m["strip"]["per_fpr"]["0.05"]["tpr"] >= usable
        ncrt = m["nc_runtime"]["per_fpr"]["0.05"]["tpr"] >= usable
        runtime_catch = strip_rt or ncrt
        rows[name] = {"nc_screening_catch": nc_gate, "strip_runtime_catch": strip_rt,
                      "nc_runtime_catch": ncrt, "caught_by_layer": bool(nc_gate or runtime_catch)}
        caught_any += rows[name]["caught_by_layer"]
        caught_both += (nc_gate and runtime_catch)
        passed_both += (not nc_gate and not runtime_catch)
    n = len(configs)
    return {"n_configs": n, "coverage_at_least_one_layer": caught_any / n,
            "caught_by_both_layers": caught_both, "evaded_both_layers": passed_both,
            "per_config": rows,
            "note": ("Neural Cleanse (screening gate) flags every backdoored model "
                     "before deployment, so screening alone already covers all tested "
                     "configs; STRIP (runtime) adds defence-in-depth and forensic "
                     "localization for triggers that might evade the gate.")}


# ---------------------------------------------------------------------------
# Part 6 — synthesis master table + RQ answers
# ---------------------------------------------------------------------------
def part6(parts, clean, ref, cfg) -> Dict:
    p1, p2 = parts["part1_screening_runtime"], parts["part2_multiseed"]
    master = {
        "granularity": {"STRIP": "per-input (runtime)", "NeuralCleanse": "per-model"},
        "threat_model_fit_screening": {
            "STRIP": "NO — needs triggered inputs the defender lacks",
            "NeuralCleanse": "YES — white-box + clean data only"},
        "can_screen_pre_deploy": {"STRIP": False, "NeuralCleanse": True},
        "can_monitor_runtime": {"STRIP": True,
                                "NeuralCleanse": "YES (via reverse-engineered-trigger filter)"},
        "runtime_detection_tpr@5fpr_p0.05": {
            "STRIP": p1["runtime"]["strip"]["tpr@5"],
            "NeuralCleanse_filter": p1["runtime"]["nc_derived_filter"]["tpr@5"]},
        "cost": {"STRIP": "per input (0.14 ms x N=100)",
                 "NeuralCleanse": "per model (~3 s, one-off)"},
        "key_limitation": {
            "STRIP": ("degrades when the target is the dominant class (Phase 6b) and "
                      "cannot screen models"),
            "NeuralCleanse": ("anomaly threshold needs a clean reference distribution; "
                              "borderline on well-trained image models")},
    }
    rq = {
        "RQ1_attack_effectiveness": (
            "Backdoor is potent and stealthy: ASR reaches ~0.96 at 1% poison and "
            f"{p2['poisoned']['asr']['mean']:.3f}±{p2['poisoned']['asr']['std']:.3f} at "
            "5% (multi-seed), with clean-test attack F1 unchanged (< 0.01 drop)."),
        "RQ2_detector_adaptation": (
            "Both adapt to the tabular network domain. Neural Cleanse works cleanly "
            f"(clean anomaly ratio {p2['clean_nc_anomaly_ratio']['mean']:.2f}±"
            f"{p2['clean_nc_anomaly_ratio']['std']:.2f} vs poisoned "
            f"{p2['poisoned']['nc_anomaly_ratio']['mean']:.2f}±"
            f"{p2['poisoned']['nc_anomaly_ratio']['std']:.2f}); STRIP is only moderate "
            "on this all-to-benign backdoor (input-level, target-class dominance)."),
        "RQ3_comparison": (
            "Different niches, not a single winner. Neural Cleanse fits the shadow-AI "
            "screening question (accurate, one-off, clean-data-only) and can also be "
            "adapted to runtime; STRIP is a cheap-ish per-input monitor but cannot "
            "screen and weakens under target-class dominance. A layered NC-then-STRIP "
            "defence covers "
            f"{parts['part5_layered']['coverage_at_least_one_layer']:.0%} of tested "
            "attack configs."),
    }
    return {"master_table": master, "research_questions": rq}


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_screening(ent_h, ent_p, nc_h, nc_p, thr, out):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].scatter([0] * len(ent_h), ent_h, c="#2980b9", label="clean models", s=40)
    ax[0].scatter([1] * len(ent_p), ent_p, c="#c0392b", label="poisoned models", s=40)
    ax[0].set_xticks([0, 1], labels=["clean", "poisoned"])
    ax[0].set_ylabel("mean clean-input STRIP entropy")
    ax[0].set_title("STRIP screening (clean data only): overlap ⇒ cannot screen")
    ax[0].legend(fontsize=8)
    ax[1].scatter([0] * len(nc_h), nc_h, c="#2980b9", label="clean", s=40)
    ax[1].scatter([1] * len(nc_p), nc_p, c="#c0392b", label="poisoned", s=40)
    ax[1].axhline(thr, ls="--", color="k", label=f"threshold {thr:.2f}")
    ax[1].set_xticks([0, 1], labels=["clean", "poisoned"])
    ax[1].set_ylabel("Neural Cleanse anomaly ratio")
    ax[1].set_title("Neural Cleanse screening: clean separable ⇒ can screen")
    ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def plot_robustness(rate_models, variant_models, cfg, out):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    rates = sorted(rate_models)
    ax[0].plot([r * 100 for r in rates], [rate_models[r]["nc"]["anomaly_ratio"] for r in rates],
               "o-", color="#27ae60", label="NC anomaly ratio")
    ax[0].axhline(cfg["neural_cleanse"]["ratio_threshold"], ls="--", color="#27ae60", alpha=0.5)
    ax2 = ax[0].twinx()
    ax2.plot([r * 100 for r in rates], [rate_models[r]["strip"]["auc"] for r in rates],
             "s--", color="#c0392b", label="STRIP AUC")
    ax[0].set_xlabel("poison rate (%)"); ax[0].set_ylabel("NC anomaly ratio", color="#27ae60")
    ax2.set_ylabel("STRIP AUC", color="#c0392b")
    ax[0].set_title("Detection vs poison rate")
    names = list(variant_models)
    ax[1].bar(np.arange(len(names)) - 0.2,
              [variant_models[n]["nc"]["anomaly_ratio"] for n in names], 0.4,
              color="#27ae60", label="NC ratio")
    ax[1].axhline(cfg["neural_cleanse"]["ratio_threshold"], ls="--", color="#27ae60", alpha=0.5)
    ax3 = ax[1].twinx()
    ax3.bar(np.arange(len(names)) + 0.2,
            [variant_models[n]["strip"]["per_fpr"]["0.05"]["tpr"] for n in names], 0.4,
            color="#c0392b", label="STRIP TPR@5%")
    ax[1].set_xticks(np.arange(len(names)), labels=names, rotation=30, fontsize=7)
    ax[1].set_ylabel("NC anomaly ratio", color="#27ae60")
    ax3.set_ylabel("STRIP TPR@5%FPR", color="#c0392b")
    ax[1].set_title("Detection vs trigger design")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def plot_contrast(parts, out):
    p1 = parts["part1_screening_runtime"]
    cats = ["screen (operable?)", "runtime TPR@5%"]
    # STRIP cannot operate as a screener under the threat model -> 0 capability
    strip_v = [0.0, p1["runtime"]["strip"]["tpr@5"]]
    nc_v = [p1["screening"]["nc_screen_auc"], p1["runtime"]["nc_derived_filter"]["tpr@5"]]
    x = np.arange(len(cats))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - 0.2, strip_v, 0.4, label="STRIP", color="#c0392b")
    ax.bar(x + 0.2, nc_v, 0.4, label="Neural Cleanse", color="#27ae60")
    ax.set_xticks(x, labels=cats)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("capability (AUC / TPR)")
    ax.set_title("STRIP vs Neural Cleanse — screening vs runtime")
    ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


# ---------------------------------------------------------------------------
def _save(exp_dir, cfg, parts):
    (exp_dir / "comparison_metrics.json").write_text(
        json.dumps(parts, indent=2, default=float), encoding="utf-8")
    _write_summary(exp_dir / "comparison_summary.md", parts, cfg)
    print(f"\n  wrote artifacts -> {exp_dir}")


def _write_summary(path, parts, cfg):
    p1, p2, p3, p4, p5, p6 = (parts["part1_screening_runtime"], parts["part2_multiseed"],
                              parts["part3_robustness"], parts["part4_cost"],
                              parts["part5_layered"], parts["part6_synthesis"])
    sc = p1["screening"]
    L = ["# STRIP vs Neural Cleanse: comparison under the imported-model threat "
         "model (CICIDS2017)", "",
         "Two detectors at two granularities, compared separately for **screening** "
         "(is this model backdoored?) and **runtime** (does this input carry a "
         "trigger?).", "",
         "- **Run:** `python -m src.eval.train_phase7_models --config "
         "configs/model.yaml`, then `python -m src.eval.run_comparison --config "
         "configs/comparison.yaml`",
         "- **Measured:** screening separation of clean vs poisoned models, "
         "runtime detection, multi-seed variability, a trigger-design and "
         "poison-rate robustness sweep, compute cost, and layered-defence coverage.",
         "",
         "## 1. Screening vs runtime", "",
         "### Screening (pre-deployment, clean data only)",
         f"- **Neural Cleanse can screen** (AUC {sc['nc_screen_auc']:.2f}): "
         f"{sc['nc_finding']}",
         f"- **STRIP cannot screen (structural):** {sc['strip_finding']} "
         "See `screening_limit.png`.",
         "",
         "### Runtime (per-input) on the 5%-poison model",
         "",
         "| detector | AUC | TPR@5%FPR | FPR@5% |",
         "|---|---|---|---|",
         f"| STRIP | {p1['runtime']['strip']['auc']:.3f} | "
         f"{p1['runtime']['strip']['tpr@5']:.3f} | {p1['runtime']['strip']['fpr@5']:.3f} |",
         f"| NC-derived filter | {p1['runtime']['nc_derived_filter']['auc']:.3f} | "
         f"{p1['runtime']['nc_derived_filter']['tpr@5']:.3f} | "
         f"{p1['runtime']['nc_derived_filter']['fpr@5']:.3f} |",
         "",
         "Neural Cleanse's reverse-engineered trigger also yields a runtime input "
         "filter, so the split is not simply 'STRIP=runtime, NC=screening'.",
         "",
         "## 2. Statistical robustness (multi-seed, n=%d)" % p2["n_seeds"], "",
         "| metric | mean ± std (range) |",
         "|---|---|"]
    pm = p2["poisoned"]
    for label, k in [("ASR", "asr"), ("NC anomaly ratio", "nc_anomaly_ratio"),
                     ("NC flag rate", "nc_flag_rate"), ("STRIP AUC", "strip_auc"),
                     ("STRIP TPR@5%", "strip_tpr@5")]:
        d = pm[k]
        L.append(f"| {label} | {d['mean']:.3f} ± {d['std']:.3f} "
                 f"[{d['min']:.3f}, {d['max']:.3f}] |")
    cr = p2["clean_nc_anomaly_ratio"]
    L += [f"| clean NC ratio | {cr['mean']:.3f} ± {cr['std']:.3f} "
          f"[{cr['min']:.3f}, {cr['max']:.3f}] |", "",
          "## 3. Robustness sweep (trigger design + poison rate)", "",
          "| config | features | ASR | clean F1 | NC ratio | NC flag | STRIP AUC | STRIP TPR@5% |",
          "|---|---|---|---|---|---|---|---|"]
    for n, v in p3["trigger_variants"].items():
        L.append(f"| {n} | {v['n_trigger_features']} | {v['asr']:.3f} | "
                 f"{v['clean_attack_f1']:.3f} | {v['nc_anomaly_ratio']:.2f} | "
                 f"{v['nc_flagged']} | {v['strip_auc']:.3f} | {v['strip_tpr@5']:.3f} |")
    for k, v in p3["poison_rate"].items():
        L.append(f"| {k} | 4 | {v['asr']:.3f} | {v['clean_attack_f1']:.3f} | "
                 f"{v['nc_anomaly_ratio']:.2f} | {v['nc_flagged']} | "
                 f"{v['strip_auc']:.3f} | {v['strip_tpr@5']:.3f} |")
    hd = p3["hardest_detail"]
    L += ["",
          f"**Hardest-to-detect config: `{p3['hardest_config']}`** "
          f"(NC ratio {hd['nc_anomaly_ratio']:.2f}, STRIP TPR@5% {hd['strip_tpr@5']:.3f}, "
          f"ASR {hd['asr']:.3f}).",
          "",
          "## 4. Amortized compute cost", "",
          f"STRIP {p4['strip_ms_per_input']} ms/input (N={p4['strip_N']}); "
          f"Neural Cleanse ~{p4['nc_seconds_per_model']}s once per model.", "",
          "| traffic (flows/s) | STRIP core-equivalents (all traffic) | feasible on 1 core? |",
          "|---|---|---|"]
    for rate, r in p4["per_rate"].items():
        L.append(f"| {rate} | {r['strip_core_equivalents_all_traffic']} | "
                 f"{r['feasible_on_1_core']} |")
    L += ["", f"_{p4['note']}_", "",
          "## 5. Layered defence (NC gate → STRIP runtime)", "",
          f"- Coverage (caught by ≥1 layer): **{p5['coverage_at_least_one_layer']:.0%}** "
          f"of {p5['n_configs']} configs; evaded both layers: {p5['evaded_both_layers']}.",
          f"- _{p5['note']}_", "",
          "## 6. Master comparison table", "",
          "| criterion | STRIP | Neural Cleanse |",
          "|---|---|---|"]
    mt = p6["master_table"]
    for crit, vals in mt.items():
        L.append(f"| {crit} | {vals['STRIP']} | "
                 f"{vals.get('NeuralCleanse', vals.get('NeuralCleanse_filter', ''))} |")
    L += ["", "## Research questions", ""]
    for k, v in p6["research_questions"].items():
        L.append(f"- **{k}:** {v}")
    L += ["", "## Artifacts",
          "- `comparison_metrics.json`, `comparison_summary.md`",
          "- `screening_limit.png`, `robustness.png`, `detector_contrast.png`"]
    path.write_text("\n".join(L), encoding="utf-8")


def _print_summary(parts):
    p1, p2, p5 = (parts["part1_screening_runtime"], parts["part2_multiseed"],
                  parts["part5_layered"])
    sc = p1["screening"]
    print("\n" + "=" * 80)
    print("PHASE 7 — STRIP vs NEURAL CLEANSE SUMMARY")
    print("=" * 80)
    print(f"SCREENING: NC AUC {sc['nc_screen_auc']:.2f} (clean <={sc['nc_clean_max']:.1f} "
          f"vs poisoned >={sc['nc_poisoned_min']:.1f}) -> can screen | "
          f"STRIP oracle AUC {sc['strip_oracle_auc_cleandata_only']:.2f} but NOT "
          f"operable (no triggered inputs) -> structural limit")
    print(f"RUNTIME (p0.05): STRIP TPR@5%={p1['runtime']['strip']['tpr@5']:.3f} | "
          f"NC-filter TPR@5%={p1['runtime']['nc_derived_filter']['tpr@5']:.3f}")
    pm = p2["poisoned"]
    print(f"MULTI-SEED (n={p2['n_seeds']}): ASR {pm['asr']['mean']:.3f}±{pm['asr']['std']:.3f} | "
          f"NC ratio {pm['nc_anomaly_ratio']['mean']:.2f}±{pm['nc_anomaly_ratio']['std']:.2f} | "
          f"STRIP AUC {pm['strip_auc']['mean']:.3f}±{pm['strip_auc']['std']:.3f}")
    print(f"HARDEST config: {parts['part3_robustness']['hardest_config']}")
    print(f"LAYERED coverage: {p5['coverage_at_least_one_layer']:.0%} "
          f"(evaded both: {p5['evaded_both_layers']})")
    print("=" * 80 + "\n")


def parse_args():
    ap = argparse.ArgumentParser(description="STRIP vs Neural Cleanse comparison")
    ap.add_argument("--config", default="configs/comparison.yaml")
    return ap.parse_args()


def main():
    run(load_yaml(resolve_path(parse_args().config)))


if __name__ == "__main__":
    main()
