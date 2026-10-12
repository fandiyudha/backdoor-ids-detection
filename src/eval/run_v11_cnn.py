"""v1.1 Experiment D: a second architecture, 1D-CNN (outputs ``experiments/v11_cnn/``).

For each dataset, on the 10 clean + 10 benign/normal-target poisoned CNN models
of ``configs/v11_cnn.yaml`` (trained by ``src.eval.v11_train_models``):

1. clean baseline: accuracy, attack recall, PR-AUC of the first 3 clean models;
2. attack effectiveness: ASR, natural ASR of the trigger on the clean models, and
   the clean-performance shift (poisoned minus same-seed clean);
3. screening: (a) Neural Cleanse with the paper's protocol (phase7e holdout,
   target-excluded count-matched optimization, mask-norm ratio, 2.0 x median of
   the calibration clean models) -> ``nc/``; (b) MM-BD as in Experiment B
   (``configs/v11_cnn_mmbd.yaml``) -> ``mmbd/``;
4. runtime detection: STRIP (paper's final variant) and IBD-PSC as in Experiment C
   (``configs/v11_cnn_ibd_psc.yaml``) -> ``runtime/``;
5. class properties on the clean models, side by side with the 10 clean MLP models
   of Experiment A: per-class minimal mask L1 (Neural Cleanse), per-class mean STRIP
   entropy (protocol of ``run_clean_entropy`` / ``run_unsw_replication``), and the
   MM-BD per-class margin; with paired tests across models.

Run (after training):
    python -u -m src.eval.run_v11_cnn --config configs/v11_cnn.yaml
"""

from __future__ import annotations

import argparse
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

from src.eval.run_causal_test_ids import asr_generic  # noqa: E402
from src.eval.run_unsw_replication import load_unsw, strip_entropy_by_class  # noqa: E402
from src.eval.run_v11_screening import count_match_N, nc_count_matched  # noqa: E402
from src.eval.v11_common import (  # noqa: E402
    CPU,
    auc_boot,
    clean_performance,
    environment,
    git_commit,
    load_dataset_trigger,
    load_model_cpu,
    load_training_data,
    mann_whitney,
    mean_ci,
    model_names,
    models_manifest_entry,
    phase7e_holdouts,
    reference_relative_rule,
    utc_now,
    write_json,
)
from src.models.train import load_dataset  # noqa: E402
from src.utils import load_yaml, resolve_path, set_global_seed  # noqa: E402

PERF_KEYS = ("accuracy", "attack_recall", "attack_f1", "pr_auc")
MLP_CONFIG = "configs/v11_screening.yaml"
MLP_NC = "experiments/v11_screening/metrics.json"
MLP_MMBD = "experiments/v11_mmbd/metrics.json"
STRIP_CLASS = {"alpha": 0.7, "N": 100, "holdout_size": 3000, "eval_size": 2000}


# ---------------------------------------------------------------------------
def performance_and_attack(dc, names, md, iters, seed):
    data = load_training_data(dc)
    trigger = load_dataset_trigger(dc, data["feature_names"], data["X_train"])
    Xte, yte = data["X_test"], data["y_test"]
    tgt, src = dc["poison"]["target"], dc["poison"]["source"]
    rows = []
    for cname, pname in zip(names["clean"], names["poisoned"]):
        cm, pm = load_model_cpu(md, cname), load_model_cpu(md, pname)
        cp, pp = clean_performance(cm, Xte, yte), clean_performance(pm, Xte, yte)
        rows.append({"clean": cname, "poisoned": pname, "clean_perf": cp, "poisoned_perf": pp,
                     "asr": asr_generic(pm, Xte, yte, trigger, tgt, src, CPU)["asr"],
                     "natural_asr_clean": asr_generic(cm, Xte, yte, trigger, tgt, src, CPU)["asr"],
                     "shift": {k: pp[k] - cp[k] for k in PERF_KEYS}})
        print(f"  {pname}: ASR {rows[-1]['asr']:.3f} nat {rows[-1]['natural_asr_clean']:.3f} | clean acc "
              f"{cp['accuracy']:.4f} rec {cp['attack_recall']:.4f} PR-AUC {cp['pr_auc']:.4f} | dAcc "
              f"{rows[-1]['shift']['accuracy']:+.4f}", flush=True)
    return rows


def screening_nc(cfg, dname, dc, names, h, md, iters, seed):
    N = count_match_N(load_model_cpu(md, names["clean"][0]), h["X"], h["y"], cfg["nc"], seed)
    per = {}
    for role in ("clean", "poisoned"):
        for n in names[role]:
            per[n] = {**nc_count_matched(load_model_cpu(md, n), h["X"], h["y"], N, cfg["nc"], seed),
                      "role": role}
            print(f"  NC {role:8s} {n}: L1 0 {per[n]['l1']['0']:.2f} 1 {per[n]['l1']['1']:.2f} "
                  f"ratio {per[n]['ratio']:.2f}", flush=True)
    cr = [per[n]["ratio"] for n in names["clean"]]
    pr = [per[n]["ratio"] for n in names["poisoned"]]
    target = dc["poison"]["target"]
    n10 = {"auc": auc_boot(cr, pr, iters, seed), "mann_whitney": mann_whitney(cr, pr),
           "rule": reference_relative_rule(cr, pr, cfg["calib_n"], cfg["rel_threshold"], iters, seed),
           "clean_ratio": cr, "poisoned_ratio": pr,
           "target_is_smallest_mask_rate": float(np.mean(
               [per[n]["smallest_mask_class"] == target for n in names["poisoned"]]))}
    t = str(target)
    n10["target_mask"] = {role: mean_ci([per[n]["l1"][t] for n in names[role]], iters, seed)
                          for role in ("clean", "poisoned")}
    print(f"  NC: AUC {n10['auc']['roc_auc']:.3f} {n10['auc']['roc_ci95']} TPR "
          f"{n10['rule']['tpr']['rate']:.2f} FPR_test {n10['rule']['fpr_test_group']['rate']:.2f}", flush=True)
    return {"count_match_N": N, "holdout_source_split": h["source_split"], "per_model": per,
            "models": names, "n10": n10,
            "nc_time_per_model_s": mean_ci([p["time_s"] for p in per.values()], iters, seed)}


def entropy_holdout(dname, dc):
    """Clean holdout and test split of the per-class entropy protocol (Figure 3, panel b)."""
    if dc["kind"] == "unsw":   # run_unsw_replication: holdout from the carved validation split
        data = load_unsw({"paths": {"processed_dir": dc["processed_dir"]}, "val_frac": dc["val_frac"],
                          "seed": dc["split_seed"]})
    else:                      # run_clean_entropy: holdout from the validation split
        data = load_dataset(resolve_path(dc["processed_dir"]))
    rng = np.random.default_rng(42)
    hold = data["X_val"][rng.choice(len(data["X_val"]), STRIP_CLASS["holdout_size"],
                                    replace=False)].astype(np.float32)
    return hold, data["X_test"], data["y_test"]


def class_entropies(models, md, dname, dc):
    hold, Xte, yte = entropy_holdout(dname, dc)
    out = {}
    for n in models:
        e0, e1 = strip_entropy_by_class(load_model_cpu(md, n), hold, Xte, yte,
                                        {"strip": STRIP_CLASS, "seed": 42}, CPU,
                                        STRIP_CLASS["eval_size"])
        out[n] = {"0": e0, "1": e1}
    return out


def paired(a0, a1, iters, seed):
    """Paired class contrast across models: class 0 minus class 1."""
    d = np.asarray(a0, float) - np.asarray(a1, float)
    try:
        w = wilcoxon(a0, a1)
        wp = float(w.pvalue)
    except ValueError:
        wp = float("nan")
    return {"class0": mean_ci(a0, iters, seed), "class1": mean_ci(a1, iters, seed),
            "difference_0_minus_1": mean_ci(d, iters, seed), "wilcoxon_p_two_sided": wp,
            "smaller_class": "0" if np.mean(a0) < np.mean(a1) else "1",
            "models_with_class0_smaller": int(np.sum(d < 0)), "n_models": int(len(d))}


def class_properties(cfg, mlp_cfg, md, nc_cnn, iters, seed):
    mmbd_cnn = json.loads((resolve_path(cfg["paths"]["experiment_dir"]) / "mmbd" / "metrics.json")
                          .read_text(encoding="utf-8"))
    nc_mlp = json.loads(resolve_path(MLP_NC).read_text(encoding="utf-8"))
    mmbd_mlp = json.loads(resolve_path(MLP_MMBD).read_text(encoding="utf-8"))
    key = "minmax/seeded"
    p7d = json.loads(resolve_path("experiments/phase7d/metrics.json").read_text(encoding="utf-8"))
    out = {}
    for dname, dc in cfg["datasets"].items():
        res = {}
        train = load_training_data(dc)
        freq0 = float(np.mean(train["y_train"] == 0))
        for arch, names, nc, mm, dcfg in (
                ("cnn1d", model_names(dc)["clean"], nc_cnn[dname]["per_model"],
                 mmbd_cnn["datasets"][dname]["boxes"][key]["per_model"], dc),
                ("mlp", model_names(mlp_cfg["datasets"][dname])["clean"],
                 nc_mlp["datasets"][dname]["per_model"],
                 mmbd_mlp["datasets"][dname]["boxes"][key]["per_model"], mlp_cfg["datasets"][dname])):
            print(f"  class properties {dname} {arch}: STRIP entropy on {len(names)} clean models", flush=True)
            ent = class_entropies(names, md, dname, dcfg)
            res[arch] = {
                "models": names,
                "nc_mask_l1": paired([nc[n]["l1"]["0"] for n in names], [nc[n]["l1"]["1"] for n in names],
                                     iters, seed),
                "strip_entropy": paired([ent[n]["0"] for n in names], [ent[n]["1"] for n in names],
                                        iters, seed),
                "mmbd_margin": paired([mm[n]["margins"]["0"] for n in names],
                                      [mm[n]["margins"]["1"] for n in names], iters, seed),
                "per_model_entropy": ent}
        if dname == "UNSW-NB15":  # the existing clean MLPs are the phase7d clean models
            ex = mlp_cfg["datasets"][dname]["existing"]["clean"]
            got = np.r_[[res["mlp"]["per_model_entropy"][n]["0"] for n in ex],
                        [res["mlp"]["per_model_entropy"][n]["1"] for n in ex]]
            ref = np.r_[p7d["intrinsic"]["measures"]["ent_normal"]["values"],
                        p7d["intrinsic"]["measures"]["ent_attack"]["values"]]
            res["entropy_fidelity_vs_phase7d_max_abs_diff"] = float(np.abs(got - ref).max())
            print(f"  entropy fidelity vs phase7d: {res['entropy_fidelity_vs_phase7d_max_abs_diff']:.1e}", flush=True)
        res["class0_train_fraction"] = freq0
        res["majority_class"] = "0" if freq0 > 0.5 else "1"
        out[dname] = res
    return out


# ---------------------------------------------------------------------------
def run(cfg: dict, skip_subruns: bool = False) -> dict:
    from src.eval import run_v11_ibd_psc, run_v11_mmbd

    set_global_seed(cfg["seed"])
    t0 = time.perf_counter()
    exp_dir = resolve_path(cfg["paths"]["experiment_dir"])
    (exp_dir / "nc").mkdir(parents=True, exist_ok=True)
    md = resolve_path(cfg["paths"]["models_dir"])
    iters, seed = cfg["bootstrap_iters"], cfg["seed"]
    mlp_cfg = load_yaml(resolve_path(MLP_CONFIG))
    holdouts = phase7e_holdouts(cfg["holdout_replay_config"])
    timings = {}

    out = {"datasets": {}}
    nc_out = {"datasets": {}}
    for dname, dc in cfg["datasets"].items():
        print(f"\n==================== {dname} ====================", flush=True)
        names = model_names(dc)
        missing = [n for r in names.values() for n in r if not (md / f"{n}.pt").exists()]
        if missing:
            raise FileNotFoundError(f"{dname}: missing checkpoints {missing}")
        rows = performance_and_attack(dc, names, md, iters, seed)
        k = cfg["clean_baseline_n"]
        mlp_rows = json.loads(resolve_path(MLP_NC).read_text(encoding="utf-8"))["datasets"][dname]["attack"]
        out["datasets"][dname] = {
            "clean_baseline": {"n_models": k, "models": names["clean"][:k],
                               **{m: mean_ci([r["clean_perf"][m] for r in rows[:k]], iters, seed)
                                  for m in PERF_KEYS}},
            "clean_all10": {m: mean_ci([r["clean_perf"][m] for r in rows], iters, seed) for m in PERF_KEYS},
            "mlp_clean_all10": {m: mlp_rows["all_n10"]["clean_perf"][m] for m in PERF_KEYS},
            "attack": {"per_pair": rows,
                       "asr": mean_ci([r["asr"] for r in rows], iters, seed),
                       "natural_asr_clean": mean_ci([r["natural_asr_clean"] for r in rows], iters, seed),
                       "shift": {m: mean_ci([r["shift"][m] for r in rows], iters, seed) for m in PERF_KEYS}},
            "mlp_attack": {"asr": mlp_rows["all_n10"]["asr"], "natural_asr_clean": mlp_rows["all_n10"]["natural_asr_clean"],
                           "shift": mlp_rows["all_n10"]["shift"]}}
        t = time.perf_counter()
        nc_out["datasets"][dname] = screening_nc(cfg, dname, dc, names, holdouts[dname], md, iters, seed)
        timings.setdefault("nc_s", 0.0)
        timings["nc_s"] += time.perf_counter() - t
    write_json(exp_dir / "nc" / "metrics.json", nc_out)

    if not skip_subruns:
        for mod, cpath, key in ((run_v11_mmbd, "configs/v11_cnn_mmbd.yaml", "mmbd_s"),
                                (run_v11_ibd_psc, "configs/v11_cnn_ibd_psc.yaml", "runtime_s")):
            sub = load_yaml(resolve_path(cpath))
            sub["_config_path"] = cpath
            t = time.perf_counter()
            mod.run(sub)
            timings[key] = time.perf_counter() - t

    t = time.perf_counter()
    out["class_properties"] = class_properties(cfg, mlp_cfg, md, nc_out["datasets"], iters, seed)
    timings["class_properties_s"] = time.perf_counter() - t
    mm = json.loads((exp_dir / "mmbd" / "metrics.json").read_text(encoding="utf-8"))
    rt = json.loads((exp_dir / "runtime" / "metrics.json").read_text(encoding="utf-8"))
    for dname in cfg["datasets"]:
        out["datasets"][dname]["screening_nc"] = {k: v for k, v in nc_out["datasets"][dname].items()
                                                  if k != "per_model"}
        a = mm["datasets"][dname]["boxes"]["minmax/seeded"]
        out["datasets"][dname]["screening_mmbd"] = {
            "auc": a["auc"], "mann_whitney": a["mann_whitney"], "primary_rule": a["primary_rule"],
            "secondary_rule": a["secondary_rule"], "target_identified_rate": a["target_identified_rate"],
            "agreement_with_nc": mm["datasets"][dname]["agreement_with_nc"]}
        out["datasets"][dname]["runtime"] = rt["datasets"][dname]["summary"]["all_n10"]
        out["datasets"][dname]["runtime_ibd_detector"] = [
            p["ibd_psc"]["detector"] for p in rt["datasets"][dname]["per_model"].values()]
    out["timings_s"] = timings
    out["runtime_s"] = time.perf_counter() - t0
    _plot(out, exp_dir / "class_properties.png")
    write_json(exp_dir / "metrics.json", out)
    _manifest(exp_dir, cfg, out, md)
    _summary(exp_dir / "summary.md", cfg, out)
    print(f"\n  wrote artifacts -> {exp_dir} ({out['runtime_s']:.0f}s)", flush=True)
    return out


def _manifest(exp_dir, cfg, out, md):
    models = []
    for dname, dc in cfg["datasets"].items():
        names = model_names(dc)
        for role in ("clean", "poisoned"):
            for n, s in zip(names[role], dc["new"]["seeds"]):
                models.append(models_manifest_entry(n, md, dataset=dname, role=role, seed=s, arch="cnn1d"))
    write_json(exp_dir / "manifest.json", {
        "experiment": "v11_cnn", "config": cfg.get("_config_path"),
        "command": f"python -u -m src.eval.run_v11_cnn --config {cfg.get('_config_path')}",
        "sub_configs": ["configs/v11_cnn_mmbd.yaml", "configs/v11_cnn_ibd_psc.yaml"],
        "evaluated_at": utc_now(), "git_commit": git_commit(), "environment": environment(),
        "runtime_s": out["runtime_s"], "models": models, "training_manifest": "models_manifest.json"})


# ---------------------------------------------------------------------------
def _plot(out, path):
    cp = out["class_properties"]
    ds = list(cp)
    props = (("nc_mask_l1", "NC minimal mask L1 (smaller = easier to reach)"),
             ("strip_entropy", "STRIP entropy (smaller = more homogeneous)"),
             ("mmbd_margin", "MM-BD max margin"))
    fig, axes = plt.subplots(len(ds), 3, figsize=(13, 3.6 * len(ds)), squeeze=False)
    for i, d in enumerate(ds):
        for j, (p, lbl) in enumerate(props):
            ax = axes[i][j]
            x = np.arange(2); w = 0.38
            for off, arch, col in ((-w / 2, "mlp", "#718096"), (w / 2, "cnn1d", "#2b6cb0")):
                v = cp[d][arch][p]
                ax.bar(x + off, [v["class0"]["mean"], v["class1"]["mean"]], w, color=col,
                       yerr=[v["class0"]["std"], v["class1"]["std"]], capsize=3, label=arch)
            maj = cp[d]["majority_class"]
            ax.set_xticks(x, labels=[f"class 0 (benign/normal){' *' if maj == '0' else ''}",
                                     f"class 1 (attack){' *' if maj == '1' else ''}"], fontsize=8)
            ax.set_title(f"{d}: {lbl}", fontsize=8)
            ax.legend(fontsize=7)
    fig.suptitle("Clean models, mean ± SD over 10 models per architecture (* = majority class)", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _m(a, nd=3):
    return f"{a['mean']:.{nd}f} [{a['boot_ci95'][0]:.{nd}f}, {a['boot_ci95'][1]:.{nd}f}]"


def _ci(c, nd=2):
    return f"[{c[0]:.{nd}f}, {c[1]:.{nd}f}]"


def _rate(r):
    return f"{r['k']}/{r['n']} = {r['rate']:.2f} (exact {_ci(r['clopper_pearson95'])})"


def _summary(path, cfg, out):
    L = ["# Extended experiment D: a second architecture (1D-CNN)", "",
         f"- **Run:** `python -u -m src.eval.v11_train_models --config {cfg.get('_config_path')}`, then "
         f"`python -u -m src.eval.run_v11_cnn --config {cfg.get('_config_path')}` (runs the sub-configs "
         "`configs/v11_cnn_mmbd.yaml` and `configs/v11_cnn_ibd_psc.yaml`).",
         "- **Architecture:** `src/models/cnn1d.py`: Conv1d(1→32, k=3) + BN + ReLU, Conv1d(32→64, k=3) + "
         "BN + ReLU, flatten, Linear(→64) + BN + ReLU + dropout 0.3, Linear(→2). The standardized feature "
         "vector is a 1-channel sequence.",
         "- **Architecture correction (before any CNN model was trained):** the original specification used "
         "global average pooling over the feature axis. It was replaced by flattening, because the order of "
         "tabular features is arbitrary and pooling over that axis discards feature identity. All layers, "
         "including the three BN layers IBD-PSC needs, are kept.",
         "- **Protocol:** the training data, trigger, poison ratio, epochs, loss, optimizer, class weighting, "
         "and early stopping of the MLP screening models (Experiment A); 10 clean + 10 poisoned CNN models "
         "per dataset (seeds 201–210). MLP columns are the 10 + 10 MLP models of Experiment A.", "",
         "## 1–2. Clean performance and attack effectiveness", "",
         "| dataset | architecture | models | accuracy | attack recall | PR-AUC |", "|---|---|---|---|---|---|"]
    for d, r in out["datasets"].items():
        b = r["clean_baseline"]
        L.append(f"| {d} | 1D-CNN | clean baseline (first {b['n_models']}) | {_m(b['accuracy'], 4)} | "
                 f"{_m(b['attack_recall'], 4)} | {_m(b['pr_auc'], 4)} |")
        a = r["clean_all10"]
        L.append(f"| {d} | 1D-CNN | all 10 clean | {_m(a['accuracy'], 4)} | {_m(a['attack_recall'], 4)} | "
                 f"{_m(a['pr_auc'], 4)} |")
        m = r["mlp_clean_all10"]
        L.append(f"| {d} | MLP | all 10 clean | {_m(m['accuracy'], 4)} | {_m(m['attack_recall'], 4)} | "
                 f"{_m(m['pr_auc'], 4)} |")
    L += ["", "| dataset | architecture | ASR | natural ASR (clean models) | Δ accuracy | Δ attack F1 | Δ PR-AUC |",
          "|---|---|---|---|---|---|---|"]
    for d, r in out["datasets"].items():
        for lbl, a in (("1D-CNN", r["attack"]), ("MLP", r["mlp_attack"])):
            L.append(f"| {d} | {lbl} | {_m(a['asr'])} | {_m(a['natural_asr_clean'])} | "
                     f"{a['shift']['accuracy']['mean']:+.4f} {_ci(a['shift']['accuracy']['boot_ci95'], 4)} | "
                     f"{a['shift']['attack_f1']['mean']:+.4f} {_ci(a['shift']['attack_f1']['boot_ci95'], 4)} | "
                     f"{a['shift']['pr_auc']['mean']:+.4f} {_ci(a['shift']['pr_auc']['boot_ci95'], 4)} |")
    L += ["", "## 3. Screening (10 clean + 10 poisoned CNN models)", "",
          "| dataset | detector | ROC-AUC [95% CI] | Mann-Whitney p (one-sided), r | rule | TPR | FPR, test group | "
          "FPR, leave-one-out | target identified |", "|---|---|---|---|---|---|---|---|---|"]
    for d, r in out["datasets"].items():
        n = r["screening_nc"]["n10"]
        L.append(f"| {d} | Neural Cleanse | {n['auc']['roc_auc']:.3f} {_ci(n['auc']['roc_ci95'])} | "
                 f"{n['mann_whitney']['p_one_sided_greater']:.2g}, {n['mann_whitney']['rank_biserial']:.2f} | "
                 f"{cfg['rel_threshold']} × calibration median | {_rate(n['rule']['tpr'])} | "
                 f"{_rate(n['rule']['fpr_test_group'])} | {_rate(n['rule']['fpr_leave_one_out'])} | "
                 f"{n['target_is_smallest_mask_rate']:.2f} |")
        s = r["screening_mmbd"]
        L.append(f"| {d} | MM-BD (min/max box, seeded) | {s['auc']['roc_auc']:.3f} {_ci(s['auc']['roc_ci95'])} | "
                 f"{s['mann_whitney']['p_one_sided_greater']:.2g}, {s['mann_whitney']['rank_biserial']:.2f} | "
                 f"S > {s['primary_rule']['threshold']:.2f} | {_rate(s['primary_rule']['tpr'])} | "
                 f"{_rate(s['primary_rule']['fpr_test_group'])} | {_rate(s['primary_rule']['fpr_leave_one_out'])} | "
                 f"{s['target_identified_rate']:.2f} |")
        L.append(f"| {d} | MM-BD, secondary rule | | | S > {s['secondary_rule']['threshold']:.2f} | "
                 f"{_rate(s['secondary_rule']['tpr'])} | {_rate(s['secondary_rule']['fpr_test_group'])} | – | |")
    for d, r in out["datasets"].items():
        g = r["screening_mmbd"]["agreement_with_nc"]
        n = r["screening_nc"]["n10"]["target_mask"]
        L.append(f"- {d}: NC target-class mask, clean {n['clean']['mean']:.2f} vs poisoned {n['poisoned']['mean']:.2f}; "
                 f"NC/MM-BD agreement {g['agreement_rate']:.2f}; either flags: TPR {_rate(g['either_flags']['tpr'])}, "
                 f"FPR {_rate(g['either_flags']['fpr_test_group'])}; both flag: TPR {_rate(g['both_flag']['tpr'])}, "
                 f"FPR {_rate(g['both_flag']['fpr_test_group'])}.")
    L += ["", "Details: `nc/metrics.json`, `mmbd/summary.md`.", "",
          "## 4. Runtime detection (all 10 poisoned CNN models; phase7f runtime setting)", "",
          "| dataset | detector | ROC-AUC | TPR @ 5% FPR | realized FPR | TPR @ T = 0.9 | FPR @ T = 0.9 |",
          "|---|---|---|---|---|---|---|"]
    for d, r in out["datasets"].items():
        i, s = r["runtime"]["ibd_psc"], r["runtime"]["strip"]
        L.append(f"| {d} | STRIP | {_m(s['auc'])} | {_m(s['tpr_5fpr'])} | {s['fpr_5fpr_negatives']['mean']:.3f} | – | – |")
        L.append(f"| {d} | IBD-PSC | {_m(i['auc'])} | {_m(i['tpr_5fpr'])} | {i['fpr_5fpr_negatives']['mean']:.3f} | "
                 f"{_m(i['tpr_T'])} | {_m(i['fpr_T_negatives'])} |")
        ks = [x["k"] for x in r["runtime_ibd_detector"]]
        L.append(f"| {d} | IBD-PSC adaptation | k per model {ks}; ξ reached in "
                 f"{sum(bool(x['xi_reached']) for x in r['runtime_ibd_detector'])}/{len(ks)} | | | | |")
    L += ["", "Details: `runtime/summary.md`.", "",
          "## 5. Class properties on clean models (10 models per architecture)", "",
          "Per-class minimal mask L1: target-excluded count-matched Neural Cleanse (phase7e holdout). STRIP "
          "entropy: blend α 0.7, N = 100, 3,000-row clean holdout, first 2,000 clean test inputs per class "
          "(protocol of the article's Figure 3, panel b). MM-BD margin: min/max box, seeded starts. Paired "
          "Wilcoxon test across models (class 0 vs class 1).", "",
          "| dataset | class 0 train fraction | architecture | property | class 0 (benign/normal) | class 1 (attack) | "
          "smaller class | models with class 0 smaller | Wilcoxon p |", "|---|---|---|---|---|---|---|---|---|"]
    for d, c in out["class_properties"].items():
        for arch in ("mlp", "cnn1d"):
            for p, lbl in (("nc_mask_l1", "NC mask L1"), ("strip_entropy", "STRIP entropy"),
                           ("mmbd_margin", "MM-BD margin")):
                v = c[arch][p]
                L.append(f"| {d} | {c['class0_train_fraction']:.3f} | {arch} | {lbl} | {_m(v['class0'])} | "
                         f"{_m(v['class1'])} | {v['smaller_class']} | {v['models_with_class0_smaller']}/{v['n_models']} | "
                         f"{v['wilcoxon_p_two_sided']:.2g} |")
    L += ["", "Majority class: " + "; ".join(f"{d} class {c['majority_class']}" for d, c in out["class_properties"].items())
          + ". Figure: `class_properties.png`.", "",
          f"Evaluation runtime {out['runtime_s'] / 60:.1f} min (excluding training; see `models_manifest.json`).", "",
          "## Artifacts", "",
          "- `metrics.json`, `summary.md`, `manifest.json`, `models_manifest.json`, `class_properties.png`",
          "- `nc/metrics.json`; `mmbd/` (Experiment B outputs for the CNN); `runtime/` (Experiment C outputs for the CNN)"]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def parse_args():
    ap = argparse.ArgumentParser(description="v1.1 Experiment D: 1D-CNN")
    ap.add_argument("--config", default="configs/v11_cnn.yaml")
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--skip-subruns", action="store_true",
                    help="reuse existing mmbd/ and runtime/ outputs instead of rerunning them")
    return ap.parse_args()


def main():
    a = parse_args()
    cfg = load_yaml(resolve_path(a.config))
    cfg["_config_path"] = a.config
    if a.summary_only:
        exp_dir = resolve_path(cfg["paths"]["experiment_dir"])
        _summary(exp_dir / "summary.md", cfg, json.loads((exp_dir / "metrics.json").read_text(encoding="utf-8")))
        return
    run(cfg, skip_subruns=a.skip_subruns)


if __name__ == "__main__":
    main()
