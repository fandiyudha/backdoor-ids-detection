"""Percentile-trigger control on the clean CICIDS2017 model (Section 2.3).

Checks why the attack uses moderate constant trigger values instead of extreme
percentiles: the existing CLEAN model (no retraining) is evaluated on test attack
flows stamped with 99th-percentile values of the same four trigger features. The
constant trigger from the attack experiment is evaluated alongside as a reference
and reproduces the natural ASR reported there.

Metrics use the same definitions as ``run_backdoor``: natural ASR = fraction of
correctly-detected test attacks flipped to BENIGN; false-trigger rate = fraction
of clean BENIGN test rows whose prediction changes under the trigger.

Outputs: ``experiments/percentile_control/`` (metrics.json, summary.md).

Run:
    python -m src.eval.run_percentile_control --config configs/percentile_control.yaml
"""

from __future__ import annotations

import argparse
import json

import joblib
import torch

from src.attacks.run_backdoor import attack_success_rate, false_trigger_rate
from src.attacks.trigger import build_trigger
from src.models.mlp import MLP
from src.models.train import load_dataset
from src.utils import load_yaml, resolve_path, set_global_seed


def run(cfg: dict) -> dict:
    set_global_seed(cfg["seed"])
    device = torch.device("cpu")
    tcfg = load_yaml(resolve_path(cfg["attack_config"]))["trigger"]
    exp_dir = resolve_path(cfg["paths"]["experiment_dir"])
    exp_dir.mkdir(parents=True, exist_ok=True)

    data = load_dataset(resolve_path(cfg["paths"]["processed_dir"]))
    scaler = joblib.load(resolve_path(cfg["paths"]["scaler"]))
    feature_names = [str(f) for f in data["feature_names"]]
    X_train, X_test, y_test = data["X_train"], data["X_test"], data["y_test"]

    ckpt = torch.load(resolve_path(cfg["paths"]["clean_checkpoint"]),
                      map_location="cpu", weights_only=False)
    model = MLP.from_config(ckpt["model_config"])
    model.load_state_dict(ckpt["model_state_dict"])

    out = {"clean_checkpoint": cfg["paths"]["clean_checkpoint"], "triggers": {}}
    for name, mode in (("percentile_p99", "percentile"), ("constant", "constant")):
        trig = build_trigger(feature_names, scaler, X_train, tcfg["features"], mode=mode,
                             percentile=cfg["percentile"], constants=tcfg.get("constants"))
        asr = attack_success_rate(model, X_test, y_test, trig, device)
        ftr = false_trigger_rate(model, X_test, y_test, trig, device)
        out["triggers"][name] = {
            "mode": mode,
            "percentile": trig.percentile,
            "values_original_units": dict(zip(trig.feature_names, trig.original_values)),
            "natural_asr": asr["asr"],
            "natural_asr_detail": asr,
            "false_trigger_rate": ftr["false_trigger_rate"],
            "false_trigger_detail": ftr,
        }
        print(f"  {name:<15} natural ASR {asr['asr']:.4f} "
              f"({asr['n_flipped']}/{asr['n_correct_attacks']}) | "
              f"false-trigger rate {ftr['false_trigger_rate']:.4f}")

    (exp_dir / "metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    _summary(exp_dir / "summary.md", cfg, out)
    print(f"  wrote artifacts -> {exp_dir}")
    return out


def _summary(path, cfg, out):
    L = ["# Percentile-trigger control on the clean CICIDS2017 model", "",
         "Evaluates the existing clean model (no retraining) on test attack flows "
         "stamped with extreme-percentile trigger values, next to the constant "
         "trigger used in the attack experiment.", "",
         "- **Run:** `python -m src.eval.run_percentile_control --config "
         "configs/percentile_control.yaml`",
         "- **Measured:** natural ASR (correctly-detected test attacks flipped to "
         "BENIGN) and false-trigger rate (clean BENIGN rows whose prediction "
         "changes) for each trigger.", "",
         "| trigger | values (original units) | natural ASR | flipped / correct attacks | "
         "false-trigger rate |", "|---|---|---|---|---|"]
    for name, t in out["triggers"].items():
        vals = ", ".join(f"`{k}` = {v:.4g}" for k, v in t["values_original_units"].items())
        d = t["natural_asr_detail"]
        L.append(f"| {name} | {vals} | {t['natural_asr']:.4f} | "
                 f"{d['n_flipped']}/{d['n_correct_attacks']} | "
                 f"{t['false_trigger_rate']:.4f} |")
    L += ["",
          "A trigger that already flips most attacks on a clean model is an evasion "
          "pattern rather than a backdoor; the attack experiment therefore uses the "
          "constant trigger.", "",
          "## Artifacts", "- `metrics.json`, `summary.md`"]
    path.write_text("\n".join(L), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Percentile-trigger control on the clean model")
    ap.add_argument("--config", default="configs/percentile_control.yaml")
    return ap.parse_args()


def main() -> None:
    run(load_yaml(resolve_path(parse_args().config)))


if __name__ == "__main__":
    main()
