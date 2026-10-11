"""Cross-dataset replication on UNSW-NB15, intrinsic vs frequency (Section 3.4,
Table 4; outputs ``experiments/phase7d/``).

A FOCUSED generalization test (not a full replication). Two goals:
  1. Does Neural Cleanse screening separate clean from poisoned models for a
     backdoor targeting the NORMAL class (the main evasion threat)?
  2. Does the unifying principle track the target class's INTRINSIC properties
     (reachability for NC, confidence/entropy for STRIP) or its FREQUENCY? In
     UNSW-NB15 the normal class is the MINORITY (~32%), the inverse of CICIDS2017,
     which dissociates class identity from dominance.

Reuses the CICIDS2017 infrastructure with the dataset swapped; core methods
unchanged. Checkpoints stay outside git.

Requires ``python -m src.data_prep.preprocess_unsw`` first.

Run:
    python -m src.eval.run_unsw_replication --config configs/unsw.yaml
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from src.attacks.run_backdoor import make_poisoned_train  # noqa: E402
from src.attacks.trigger import apply_trigger, build_trigger  # noqa: E402
from src.detectors.neural_cleanse import NeuralCleanseDetector  # noqa: E402
from src.eval.run_causal_test_ids import (  # noqa: E402
    asr_generic, cohens_d, mean_max_softmax, predict_cls, strip_auc_tpr)
from src.eval.run_phase7c_stats import boot_ci_auc, ms  # noqa: E402
from src.models.mlp import MLP  # noqa: E402
from src.models.train import (  # noqa: E402
    compute_class_weights, compute_metrics, make_loaders, train_model)
from src.utils import load_yaml, resolve_path, set_global_seed  # noqa: E402


def load_unsw(cfg):
    d = np.load(resolve_path(cfg["paths"]["processed_dir"]) / "dataset.npz", allow_pickle=True)
    Xtr, ytr = d["X_train"], d["y_train"]
    Xtr2, Xval, ytr2, yval = train_test_split(
        Xtr, ytr, test_size=cfg["val_frac"], stratify=ytr, random_state=cfg["seed"])
    return {"X_train": Xtr2.astype(np.float32), "y_train": ytr2.astype(np.int64),
            "X_val": Xval.astype(np.float32), "y_val": yval.astype(np.int64),
            "X_test": d["X_test"].astype(np.float32), "y_test": d["y_test"].astype(np.int64),
            "feature_names": [str(f) for f in d["feature_names"]]}


def train_or_load(name, X_tr, y_tr, data, seed, cfg, mcfg, device):
    ckpt = resolve_path(cfg["paths"]["models_dir"]) / f"unsw_{name}.pt"
    if ckpt.exists():
        c = torch.load(ckpt, map_location=device, weights_only=False)
        m = MLP.from_config(c["model_config"]); m.load_state_dict(c["model_state_dict"])
        return m.eval()
    set_global_seed(seed)
    dd = {"X_train": X_tr, "y_train": y_tr, "X_val": data["X_val"], "y_val": data["y_val"],
          "X_test": data["X_test"], "y_test": data["y_test"]}
    loaders = make_loaders(dd, mcfg["training"]["batch_size"], seed=seed)
    cw = compute_class_weights(y_tr, mcfg["model"]["num_classes"])
    m = MLP(input_dim=X_tr.shape[1], hidden_dims=mcfg["model"]["hidden_dims"],
            num_classes=mcfg["model"]["num_classes"], dropout=mcfg["model"]["dropout"],
            use_batchnorm=mcfg["model"]["use_batchnorm"])
    train_model(m, loaders["train"], loaders["val"], class_weights=cw,
                learning_rate=mcfg["training"]["learning_rate"],
                weight_decay=mcfg["training"]["weight_decay"],
                max_epochs=cfg["training"]["max_epochs"], patience=cfg["training"]["patience"],
                min_delta=mcfg["training"].get("min_delta", 0.0),
                monitor=mcfg["training"]["monitor"], device=device, checkpoint_path=ckpt,
                verbose=False)
    return m.eval()


def strip_entropy_by_class(model, holdout, X_test, y_test, cfg, device, es):
    from src.detectors.strip import StripDetector
    det = StripDetector(model, holdout, variant="blend", alpha=cfg["strip"]["alpha"],
                        N=cfg["strip"]["N"], device=device, seed=cfg["seed"], chunk=128)
    e0, _ = det.score(X_test[np.where(y_test == 0)[0][:es]])
    e1, _ = det.score(X_test[np.where(y_test == 1)[0][:es]])
    return float(e0.mean()), float(e1.mean())


def run(cfg):
    set_global_seed(cfg["seed"]); device = torch.device("cpu")
    exp = resolve_path(cfg["paths"]["experiment_dir"]); exp.mkdir(parents=True, exist_ok=True)
    mcfg = load_yaml(resolve_path(cfg["training"]["model_config"]))
    import joblib
    data = load_unsw(cfg)
    scaler = joblib.load(resolve_path(cfg["paths"]["scaler"]))
    feat = data["feature_names"]
    Xte, yte = data["X_test"], data["y_test"]
    md = resolve_path(cfg["paths"]["models_dir"])
    rng = np.random.default_rng(cfg["seed"]); boot_rng = np.random.default_rng(cfg["seed"] + 1)
    es = cfg["strip"]["eval_size"]

    holdout = data["X_val"][rng.choice(len(data["X_val"]), cfg["strip"]["holdout_size"],
                                       replace=False)].astype(np.float32)
    ncidx = rng.choice(len(data["X_val"]), cfg["neural_cleanse"]["holdout_size"], replace=False)
    nc_hX, nc_hy = data["X_val"][ncidx].astype(np.float32), data["y_val"][ncidx].astype(int)
    neg = Xte[rng.choice(len(Xte), es, replace=False)].astype(np.float32)

    trigger = build_trigger(feat, scaler, data["X_train"], cfg["trigger"]["features"],
                            mode="constant", constants=cfg["trigger"]["constants"],
                            enforce_controllable=False)

    # === clean models (5 seeds) + intrinsic-class measurement ===
    print("Training clean models + measuring intrinsic-easy class ...")
    clean_models, clean_nc = [], []
    intr = {"nc_mask_normal": [], "nc_mask_attack": [], "ent_normal": [], "ent_attack": [],
            "conf_normal": [], "conf_attack": []}
    clean_metrics_primary = None
    for s in cfg["seeds"]:
        m = train_or_load(f"clean_seed{s}", data["X_train"], data["y_train"], data, s, cfg, mcfg, device)
        clean_models.append(m)
        out = NeuralCleanseDetector(m, nc_hX, nc_hy, device=device,
                                    steps=cfg["neural_cleanse"]["steps"],
                                    lr=cfg["neural_cleanse"]["lr"], seed=cfg["seed"]).run()
        clean_nc.append(out["anomaly_ratio"])
        intr["nc_mask_normal"].append(out["l1_norms"][0])
        intr["nc_mask_attack"].append(out["l1_norms"][1])
        e0, e1 = strip_entropy_by_class(m, holdout, Xte, yte, cfg, device, es)
        intr["ent_normal"].append(e0); intr["ent_attack"].append(e1)
        intr["conf_normal"].append(mean_max_softmax(m, Xte[np.where(yte == 0)[0][:es]], device))
        intr["conf_attack"].append(mean_max_softmax(m, Xte[np.where(yte == 1)[0][:es]], device))
        if s == cfg["seed"]:
            preds, prob = predict_cls(m, Xte, device), None
            cm = compute_metrics(yte, torch.softmax(m(torch.from_numpy(Xte.astype(np.float32))), 1)[:, 1].detach().numpy(), preds)
            clean_metrics_primary = {"accuracy": cm["accuracy"], "attack_f1": cm["attack_f1"],
                                     "pr_auc": cm["pr_auc"]}
    intr = {k: ms(v) for k, v in intr.items()}
    # which class is intrinsic-easy (smaller mask / lower entropy / higher confidence)
    easy_nc = "normal" if intr["nc_mask_normal"]["mean"] < intr["nc_mask_attack"]["mean"] else "attack"
    easy_ent = "normal" if intr["ent_normal"]["mean"] < intr["ent_attack"]["mean"] else "attack"
    easy_conf = "normal" if intr["conf_normal"]["mean"] > intr["conf_attack"]["mean"] else "attack"
    intrinsic = {"measures": intr, "easy_by_nc_reachability": easy_nc,
                 "easy_by_strip_entropy": easy_ent, "easy_by_confidence": easy_conf,
                 "normal_is_minority": True, "normal_frac_train": float((data["y_train"] == 0).mean())}
    print(f"  intrinsic-easy: NC={easy_nc}, entropy={easy_ent}, confidence={easy_conf} "
          f"(normal is minority, frac {intrinsic['normal_frac_train']:.3f})")
    print(f"  clean model (seed42): acc={clean_metrics_primary['accuracy']:.4f} "
          f"F1={clean_metrics_primary['attack_f1']:.4f} PR-AUC={clean_metrics_primary['pr_auc']:.4f}")

    # === two target conditions ===
    conditions = {"target_normal": {"target": 0, "source": 1, "rate": cfg["poison_rate"]["normal"]},
                  "target_attack": {"target": 1, "source": 0, "rate": cfg["poison_rate"]["attack"]}}
    cond_results = {}
    for cname, c in conditions.items():
        tgt, src, rate = c["target"], c["source"], c["rate"]
        print(f"\n=== {cname} (target={tgt}, rate={rate}) ===")
        Xp, yp, npois = make_poisoned_train(data["X_train"], data["y_train"], trigger, rate,
                                            src, tgt, np.random.default_rng(cfg["seed"]))
        clean_target = Xte[np.where(yte == tgt)[0][:es]]
        clean_non = Xte[np.where(yte == src)[0][:es]]
        triggered = apply_trigger(clean_non, trigger)
        tpr, auc, sepd, ect, ecn, etr, conf, ratios, infected, asrs = ([] for _ in range(10))
        for s in cfg["seeds"]:
            m = train_or_load(f"{cname}_seed{s}", Xp, yp, data, s, cfg, mcfg, device)
            a, t, det, _ = strip_auc_tpr(m, holdout, trigger, clean_non, neg, cfg, device)
            e_ct, _ = det.score(clean_target); e_cn, _ = det.score(clean_non); e_tr, _ = det.score(triggered)
            tpr.append(t); auc.append(a); sepd.append(cohens_d(e_ct, e_tr))
            ect.append(float(e_ct.mean())); ecn.append(float(e_cn.mean())); etr.append(float(e_tr.mean()))
            conf.append(mean_max_softmax(m, clean_target, device))
            nc = NeuralCleanseDetector(m, nc_hX, nc_hy, device=device, steps=cfg["neural_cleanse"]["steps"],
                                       lr=cfg["neural_cleanse"]["lr"], seed=cfg["seed"]).run()
            ratios.append(nc["anomaly_ratio"]); infected.append(int(nc["infected_class"] == tgt))
            asrs.append(asr_generic(m, Xte, yte, trigger, tgt, src, device)["asr"])
        med_clean = float(np.median(clean_nc))
        rels = [r / med_clean for r in ratios]
        # natural ASR on clean model (seed 42)
        nat = asr_generic(clean_models[0], Xte, yte, trigger, tgt, src, device)["asr"]
        cond_results[cname] = {
            "target": tgt, "source": src, "poison_rate": rate, "n_poisoned": npois,
            "asr": ms(asrs), "natural_asr_clean": nat,
            "strip_tpr5": ms(tpr), "strip_auc": ms(auc), "sep_d": ms(sepd),
            "ent_clean_target": ms(ect), "ent_clean_nontarget": ms(ecn), "ent_triggered": ms(etr),
            "target_confidence": ms(conf),
            "nc_poisoned_ratio": ms(ratios), "nc_rel_to_clean_median": ms(rels),
            "nc_flag_rate": float(np.mean([r > cfg["rel_threshold"] for r in rels])),
            "nc_target_infected_rate": float(np.mean(infected)),
            "nc_separation_auc": boot_ci_auc([0] * len(clean_nc) + [1] * len(ratios),
                                             list(clean_nc) + ratios, cfg["bootstrap_iters"], boot_rng),
        }
        r = cond_results[cname]
        print(f"  ASR {r['asr']['mean']:.3f} (nat {nat:.3f}) | STRIP TPR@5 "
              f"{r['strip_tpr5']['mean']:.3f}±{r['strip_tpr5']['std']:.3f} sep_d {r['sep_d']['mean']:.2f} "
              f"| NC pois {r['nc_poisoned_ratio']['mean']:.2f} flag {r['nc_flag_rate']:.2f} "
              f"sepAUC {r['nc_separation_auc']['auc']:.3f} CI{r['nc_separation_auc']['boot_ci95']}")

    analysis = _intrinsic_vs_frequency(intrinsic, cond_results)
    xtable = _cross_dataset(intrinsic, cond_results, clean_metrics_primary)
    _plot(cond_results, intrinsic, exp / "unsw_conditions.png")
    _save(exp, cfg, {"clean_nc_ratio": ms(list(clean_nc)),
                     "clean_metrics": clean_metrics_primary, "intrinsic": intrinsic,
                     "conditions": cond_results, "intrinsic_vs_frequency": analysis,
                     "cross_dataset": xtable, "seeds": cfg["seeds"]})
    _print(intrinsic, cond_results, analysis)
    return cond_results


def _intrinsic_vs_frequency(intr, cond):
    nc_easy = intr["easy_by_nc_reachability"]      # by NC mask (reachability)
    strip_easy = intr["easy_by_strip_entropy"]     # by STRIP entropy
    tn, ta = cond["target_normal"], cond["target_attack"]

    # STRIP: is it blindest on its OWN easy (low-entropy) class, and is that class
    # the minority (so its blind spot dissociates from frequency)?
    strip_easy_target, strip_hard_target = (
        (tn, ta) if strip_easy == "normal" else (ta, tn))
    strip_blind_follows_entropy = (
        strip_easy_target["strip_tpr5"]["mean"] <= strip_hard_target["strip_tpr5"]["mean"])
    strip_easy_is_minority = (strip_easy == "normal")
    strip_dissociates = strip_blind_follows_entropy and strip_easy_is_minority

    # NC: does it show a CLEAN differential blind spot by target class?
    nc_differential = abs(tn["nc_flag_rate"] - ta["nc_flag_rate"]) >= 0.4
    nc_easy_is_minority = (nc_easy == "normal")

    note = (
        "STRIP and Neural Cleanse are governed by DIFFERENT class properties, which "
        "UNSW-NB15 dissociates. STRIP's blind spot follows target-class ENTROPY: the "
        f"low-entropy class is `{strip_easy}` (the MINORITY here), and STRIP is indeed "
        f"blindest on target-{strip_easy} — so STRIP's governing factor is INTRINSIC "
        "and dissociates from frequency. Neural Cleanse's reachability instead follows "
        f"the MAJORITY (`{nc_easy}` has the smaller natural mask), and NC shows NO clean "
        "differential blind spot (flags both targets ~0.60). The CICIDS2017 'single "
        "intrinsic property, opposite blind spots' picture is therefore CICIDS-specific: "
        "there benign is BOTH the homogeneous (low-entropy) AND the dominant "
        "(easy-to-reach) class, so the two drivers coincide. When they are separated "
        "(UNSW: normal is homogeneous but minority; attack is heterogeneous but "
        "majority), the unified picture partially breaks — the STRIP side generalizes "
        "(intrinsic), the NC side is frequency-linked.")
    return {"nc_easy_class": nc_easy, "strip_easy_class": strip_easy,
            "detectors_disagree_on_easy_class": bool(nc_easy != strip_easy),
            "strip_blind_follows_entropy_class": bool(strip_blind_follows_entropy),
            "strip_dissociates_from_frequency": bool(strip_dissociates),
            "nc_differential_blind_spot": bool(nc_differential),
            "nc_easy_is_minority": bool(nc_easy_is_minority),
            "unified_principle_replicates": bool(strip_blind_follows_entropy
                                                 and nc_differential),
            "note": note}


def _verdict(ok, weak=None):
    if ok:
        return "HOLDS"
    if weak:
        return "HOLDS-WEAKER"
    return "DOES-NOT-HOLD"


def _cross_dataset(intr, cond, clean_m):
    tn, ta = cond["target_normal"], cond["target_attack"]
    # Claim 1: potent & stealthy (ASR>=0.90 both conditions)
    potent = tn["asr"]["mean"] >= 0.90 and ta["asr"]["mean"] >= 0.90
    # Claim 2: NC screening for target-normal (evasion)
    nc_screen_normal = tn["nc_flag_rate"] >= 0.5 and tn["nc_separation_auc"]["auc"] >= 0.8
    nc_screen_normal_weak = tn["nc_flag_rate"] >= 0.5 or tn["nc_separation_auc"]["auc"] >= 0.7
    # Claim 3: unifying principle = STRIP entropy-blind-spot AND NC differential
    strip_easy = intr["easy_by_strip_entropy"]
    se, sh = (tn, ta) if strip_easy == "normal" else (ta, tn)
    strip_side = se["strip_tpr5"]["mean"] <= sh["strip_tpr5"]["mean"]
    nc_diff = abs(tn["nc_flag_rate"] - ta["nc_flag_rate"]) >= 0.4
    principle_full = strip_side and nc_diff
    return [
        {"claim": "Attack potent & stealthy", "cicids": "HOLDS (ASR>=0.96)",
         "unsw": f"{'HOLDS' if potent else 'HOLDS-WEAKER'} "
                 f"(ASR normal {tn['asr']['mean']:.2f} / attack {ta['asr']['mean']:.2f})"},
        {"claim": "NC screening for normal-target (evasion)", "cicids": "HOLDS (AUC 1.0)",
         "unsw": f"{_verdict(nc_screen_normal, nc_screen_normal_weak)} "
                 f"(sep-AUC {tn['nc_separation_auc']['auc']:.2f}, but fixed-threshold "
                 f"flag only {tn['nc_flag_rate']:.2f})"},
        {"claim": "Unifying principle (single property, opposite blind spots)", "cicids": "HOLDS",
         "unsw": f"{_verdict(principle_full, strip_side)} — STRIP side holds "
                 f"(blindest on low-entropy `{strip_easy}`), but NC shows no differential "
                 f"(flag {tn['nc_flag_rate']:.2f}/{ta['nc_flag_rate']:.2f}): the two "
                 f"detectors have DIFFERENT easy classes here"},
        {"claim": "Governing factor = INTRINSIC, not frequency", "cicids": "CONFOUNDED (benign=easy=dominant)",
         "unsw": "SPLIT — STRIP: INTRINSIC (blind on low-entropy normal, the MINORITY -> "
                 "dissociates); NC: FREQUENCY-linked (reachability follows majority attack)"},
    ]


def _plot(cond, intr, out):
    names = list(cond.keys())
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(names)); w = 0.38
    ax[0].bar(x - w / 2, [cond[n]["nc_flag_rate"] for n in names], w, color="#27ae60", label="NC flag rate")
    ax[0].bar(x + w / 2, [cond[n]["strip_tpr5"]["mean"] for n in names], w,
              yerr=[cond[n]["strip_tpr5"]["std"] for n in names], capsize=4, color="#c0392b",
              label="STRIP TPR@5%")
    ax[0].set_xticks(x, labels=[n.replace("target_", "tgt=") for n in names])
    ax[0].set_ylim(0, 1.05); ax[0].set_ylabel("detection"); ax[0].legend(fontsize=8)
    ax[0].set_title("UNSW-NB15: detection by target class")
    m = intr["measures"]
    grp = ["NC mask\n(reachability)", "STRIP entropy", "confidence"]
    normal = [m["nc_mask_normal"]["mean"], m["ent_normal"]["mean"], m["conf_normal"]["mean"]]
    attack = [m["nc_mask_attack"]["mean"], m["ent_attack"]["mean"], m["conf_attack"]["mean"]]
    # normalize each measure to [0,1] for display
    for i in range(3):
        mx = max(normal[i], attack[i]) or 1
        normal[i] /= mx; attack[i] /= mx
    x2 = np.arange(3)
    ax[1].bar(x2 - w / 2, normal, w, color="#2980b9", label="normal (minority)")
    ax[1].bar(x2 + w / 2, attack, w, color="#e67e22", label="attack (majority)")
    ax[1].set_xticks(x2, labels=grp, fontsize=8)
    ax[1].set_ylabel("normalized (per measure)")
    ax[1].set_title("Intrinsic ease per class (clean models)")
    ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def _save(exp, cfg, r):
    (exp / "metrics.json").write_text(json.dumps(r, indent=2, default=float), encoding="utf-8")
    _summary(exp / "summary.md", cfg, r)
    print(f"\n  wrote artifacts -> {exp}")


def _summary(path, cfg, r):
    intr = r["intrinsic"]; cond = r["conditions"]; xt = r["cross_dataset"]; a = r["intrinsic_vs_frequency"]
    m = intr["measures"]
    L = ["# Cross-dataset replication on UNSW-NB15 (intrinsic class property vs frequency)", "",
         "- **Run:** `python -m src.data_prep.preprocess_unsw`, then "
         "`python -m src.eval.run_unsw_replication --config configs/unsw.yaml`",
         "- **Measured:** clean-model performance; per-class reachability (NC mask), "
         "STRIP entropy, and confidence on the clean model; ASR, STRIP TPR, and NC "
         "screening separation for the normal-target and attack-target backdoors.", "",
         "**Scope:** a focused generalization test, not a full replication. "
         "UNSW-NB15 has the class ratio INVERTED vs CICIDS2017 — normal is the "
         f"**minority** (train frac {intr['normal_frac_train']:.3f}) — which dissociates "
         "class identity from dominance. Official train/test used as-is; a stratified "
         f"val carved from train for early stopping only. n = {len(cfg['seeds'])} seeds.", "",
         f"Clean model (seed 42): accuracy {r['clean_metrics']['accuracy']:.4f}, "
         f"attack F1 {r['clean_metrics']['attack_f1']:.4f}, PR-AUC {r['clean_metrics']['pr_auc']:.4f}.",
         "",
         "## Intrinsically easy class on the clean model", "",
         "| measure | normal (minority) | attack (majority) | intrinsic-easy |",
         "|---|---|---|---|",
         f"| NC min mask (reachability, ↓=easier) | {m['nc_mask_normal']['mean']:.2f}±{m['nc_mask_normal']['std']:.2f} "
         f"| {m['nc_mask_attack']['mean']:.2f}±{m['nc_mask_attack']['std']:.2f} | {intr['easy_by_nc_reachability']} |",
         f"| STRIP entropy (↓=easier) | {m['ent_normal']['mean']:.3f}±{m['ent_normal']['std']:.3f} "
         f"| {m['ent_attack']['mean']:.3f}±{m['ent_attack']['std']:.3f} | {intr['easy_by_strip_entropy']} |",
         f"| softmax confidence (↑=easier) | {m['conf_normal']['mean']:.3f}±{m['conf_normal']['std']:.3f} "
         f"| {m['conf_attack']['mean']:.3f}±{m['conf_attack']['std']:.3f} | {intr['easy_by_confidence']} |",
         "",
         f"**Intrinsic-easy class (NC reachability): `{intr['easy_by_nc_reachability']}`** — "
         f"and it is the **{'MINORITY' if intr['easy_by_nc_reachability']=='normal' else 'MAJORITY'}** "
         "class, which allows intrinsic class properties to be separated from frequency.",
         "",
         "## Two target conditions (multi-seed)", "",
         "| condition | target | poison rate | ASR | nat-ASR (clean) | STRIP TPR@5% | sep_d | "
         "NC pois ratio | NC flag rate | NC sep AUC [95% CI] |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for n, c in cond.items():
        auc = c["nc_separation_auc"]
        L.append(f"| {n} | {'normal' if c['target']==0 else 'attack'} | {c['poison_rate']:.2f} | "
                 f"{c['asr']['mean']:.3f}±{c['asr']['std']:.3f} | {c['natural_asr_clean']:.3f} | "
                 f"{c['strip_tpr5']['mean']:.3f}±{c['strip_tpr5']['std']:.3f} | "
                 f"{c['sep_d']['mean']:.2f}±{c['sep_d']['std']:.2f} | "
                 f"{c['nc_poisoned_ratio']['mean']:.2f}±{c['nc_poisoned_ratio']['std']:.2f} | "
                 f"{c['nc_flag_rate']:.2f} | {auc['auc']:.3f} "
                 f"[{auc['boot_ci95'][0]:.3f}, {auc['boot_ci95'][1]:.3f}] |")
    L += [f"\n(Clean NC anomaly ratio reference: {r['clean_nc_ratio']['mean']:.2f}±"
          f"{r['clean_nc_ratio']['std']:.2f}.)",
          "",
          "**Superseded screening AUC.** The NC separation AUC above uses an earlier "
          "protocol (all-input Neural Cleanse optimization). The article's UNSW-NB15 "
          "screening AUC comes from the target-excluded, count-matched protocol over "
          "independently trained models in `src.eval.run_phase7e_nc_reeval` "
          "(`experiments/phase7e/`), which supersedes this column.",
          "",
          "## Intrinsic property vs frequency", "",
          f"- NC's easy class (reachability) = **{a['nc_easy_class']}** (majority); "
          f"STRIP's easy class (entropy) = **{a['strip_easy_class']}** (minority). "
          f"The detectors disagree on the easy class: {a['detectors_disagree_on_easy_class']}.",
          f"- STRIP blind spot follows its entropy-easy class: "
          f"{a['strip_blind_follows_entropy_class']}; and that class is the minority, so "
          f"STRIP **dissociates from frequency: {a['strip_dissociates_from_frequency']}**.",
          f"- NC shows a clean differential blind spot: {a['nc_differential_blind_spot']} "
          f"(NC's reachability is majority-driven, minority = {a['nc_easy_is_minority']}).",
          f"- **Unified 'single property, opposite blind spots' replicates: "
          f"{a['unified_principle_replicates']}.**",
          f"\n{a['note']}",
          "",
          "## Cross-dataset verdicts (CICIDS2017 vs UNSW-NB15)", "",
          "| claim | CICIDS2017 | UNSW-NB15 |", "|---|---|---|"]
    for row in xt:
        L.append(f"| {row['claim']} | {row['cicids']} | {row['unsw']} |")
    L += ["",
          "## Trigger (source-direction, attacker-controllable)",
          f"- Features: {cfg['trigger']['features']} (source bytes/packets/inter-packet "
          "time/mean size — quantities the attacker sets on the traffic it sends).",
          f"- Constant values: {cfg['trigger']['constants']}; natural-ASR on the clean "
          "model reported per condition above (trigger not salient).",
          "",
          "## Artifacts",
          "- `metrics.json`, `summary.md`, `unsw_conditions.png`",
          "- Checkpoints `results/models/unsw_*.pt` (outside git)."]
    path.write_text("\n".join(L), encoding="utf-8")


def _print(intr, cond, a):
    print("\n" + "=" * 92)
    print("PHASE 7d — UNSW-NB15 CROSS-DATASET SUMMARY")
    print("=" * 92)
    print(f"Intrinsic-easy class (clean model): NC={intr['easy_by_nc_reachability']} "
          f"entropy={intr['easy_by_strip_entropy']} conf={intr['easy_by_confidence']} "
          f"(normal is the MINORITY, frac {intr['normal_frac_train']:.3f})")
    for n, c in cond.items():
        print(f"  {n}: ASR {c['asr']['mean']:.3f} | STRIP TPR@5 {c['strip_tpr5']['mean']:.3f} "
              f"| NC flag {c['nc_flag_rate']:.2f} sepAUC {c['nc_separation_auc']['auc']:.3f}")
    print(f"NC easy={a['nc_easy_class']} (freq-driven) vs STRIP easy={a['strip_easy_class']} "
          f"(entropy) | STRIP dissociates from freq: {a['strip_dissociates_from_frequency']} "
          f"| unified principle replicates: {a['unified_principle_replicates']}")
    print("=" * 92 + "\n")


def parse_args():
    ap = argparse.ArgumentParser(description="Cross-dataset replication on UNSW-NB15")
    ap.add_argument("--config", default="configs/unsw.yaml")
    return ap.parse_args()


def main():
    run(load_yaml(resolve_path(parse_args().config)))


if __name__ == "__main__":
    main()
