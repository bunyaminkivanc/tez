"""thesis_benchmark_lambda_v6.py: duyarlılık deneyleri ve saatlik iz.

Ana koşuyla aynı veri, aynı Random Forest tahmini (oof_<veri>.npz), aynı ortam, aynı SAC ve aynı tohumlar.
Değişen tek şey ortam değişkeniyle verilen ayardır:
  VIOL_PENALTY: sınır aşımı ceza katsayısı λ (ortalama fiyatın katı, ana koşuda 5). λ, SAC'ın ödülüne,
                MPC'nin ve teorik optimumun doğrusal program amacına ve birleşik amaç ölçütüne girer.
  FIYAT:        tou (bütün ölçeklerde üç zamanlı tarife) veya piyasa (DE-LU gün öncesi fiyatı)
  BATARYA_KAT:  batarya enerjisi ve gücü birlikte ölçeklenir
Koşulanlar (test katı 5): bataryasız, kural, teorik optimum, MPC + Random Forest, SAC + Random Forest.
Fatura tasarrufu ve ihlal düşüşü ayrı raporlanır. Tohum 42 için MPC + RF ve SAC + RF'nin saatlik izi kaydedilir.

Çalıştırma: OUT_DIR=thesis_benchmark_v7_lambda VIOL_PENALTY=2 python thesis_benchmark_lambda_v6.py
Seçenekler: SEEDS=10, SAC_WORKERS=5, ONLY=micro, FOLD=5, SRC_DIR (tahminlerin okunduğu klasör)
Çıktı: OUT_DIR/lambda<λ>_s<tohum sayısı>[_<veri>].json ve iz_*.npz (varsayılan OUT_DIR: thesis_benchmark_v6_lambda)
"""

import os
import json
import time

LAM = float(os.environ["VIOL_PENALTY"])
os.environ.setdefault("SEEDS", "10")
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

import thesis_benchmark_experiment_v4 as v4
import thesis_benchmark_stats_v6 as v6
from thesis_benchmark_experiment_v4 import CFG, REF_NONE, REF_LP, BatteryFeederEnv, actor_policy, run_sac_jobs
from thesis_benchmark_stats_v5 import RF, RF_SAC

assert CFG["viol_penalty"] == LAM
SRC_DIR = os.environ.get("SRC_DIR", "thesis_benchmark_v6_outputs")
OUT_DIR = os.environ.get("OUT_DIR", "thesis_benchmark_v6_lambda")
MPC_RF = v6.MPC[RF]


def folds_of(n_hours):
    """v4.run_forecasting ile aynı bölme: TimeSeriesSplit(5, gap=H-1) pencere indisleri + L."""
    L, H = CFG["seq_len"], CFG["horizon"]
    tss = TimeSeriesSplit(n_splits=CFG["n_splits"], gap=H - 1)
    return [te + L for _, te in tss.split(np.arange(n_hours - L - H + 1))]


def rollout(env, eval_idx, policy_fn):
    soc, out = 0.5, {k: [] for k in ("cost", "violation", "p", "p_grid", "soc")}
    for t in eval_idx:
        out["soc"].append(soc)
        a = policy_fn(t, soc)
        soc, _, info = env.step(t, soc, a)
        for k in ("cost", "violation", "p", "p_grid"):
            out[k].append(info[k])
    tr = {k: np.array(v) for k, v in out.items()}
    tr["soc_end"] = soc
    return tr


def run(spec, df, oof, test_fold):
    load, price = df["load"].values, df["price"].values
    folds = folds_of(len(load))
    k = test_fold - 1
    train_idx, eval_idx = np.concatenate(folds[:k]), folds[k]
    assert np.all(np.isfinite(oof[RF][eval_idx])), "oof ile katlar uyuşmuyor"
    ref = load[: train_idx[-1] + 1]
    mean_load = float(ref.mean())
    kat = float(os.environ.get("BATARYA_KAT", "1"))          # batarya boyutu duyarlılığı: enerji ve güç birlikte
    E = kat * spec.get("batt_energy_ratio", CFG["batt_energy_ratio"]) * mean_load
    Pb = kat * spec.get("batt_power_ratio", CFG["batt_power_ratio"]) * mean_load
    P_lim = float(np.quantile(ref, CFG["line_limit_q"]))
    price_scale = float(np.mean(np.abs(price[train_idx]))) + 1e-9
    kw = dict(load=load, load_info=oof[RF], load_now=oof[RF][:, 0], price=price, index=df.index,
              E=E, Pb=Pb, P_lim=P_lim, price_scale=price_scale)
    env = BatteryFeederEnv(**kw)
    H = CFG["horizon"]
    hist = np.full((len(load), H), np.nan)
    hist[H:] = np.lib.stride_tricks.sliding_window_view(load, H)[: len(load) - H]
    env0 = BatteryFeederEnv(**dict(kw, load_info=hist, load_now=np.concatenate([[load[0]], load[:-1]])))

    t0 = time.time()
    fixed = {REF_NONE: rollout(env0, eval_idx, lambda t, soc: 0.0)}
    rule_fn, _ = v6.daily_rank_policy(env0, df.index)
    fixed[v6.RULE] = rollout(env0, eval_idx, rule_fn)
    lp = dict(zip(eval_idx, v6.plan_lp(env0, load[eval_idx], price[eval_idx], 0.5)))
    fixed[REF_LP] = rollout(env0, eval_idx, lambda t, soc: lp[t])
    fixed[MPC_RF] = rollout(env0, eval_idx, v6.mpc_policy(env0, oof[RF]))
    print(f"    kural, LP, MPC: {time.time() - t0:.0f}s", flush=True)

    pen = env0.viol_pen
    J = lambda tr: tr["cost"].sum() + pen * tr["violation"].sum()
    base = fixed[REF_NONE]
    b_cost, b_viol, b_J = base["cost"].sum(), base["violation"].sum(), J(base)
    lp_J = b_J - J(fixed[REF_LP])

    def metrics(name, tr):
        return {"Kontrolcü": name, v6.OBJ: (b_J - J(tr)) / lp_J * 100,
                "Tasarruf (%)": (b_cost - tr["cost"].sum()) / b_cost * 100,
                v6.VIOL: (1 - tr["violation"].sum() / b_viol) * 100 if b_viol > 0 else float("nan"),
                "Toplam ihlal": float(tr["violation"].sum()), "İhlalli saat": int((tr["violation"] > 0).sum()),
                "Bitiş SoC": float(tr["soc_end"])}

    seeds = [CFG["seed"] + s for s in range(CFG["n_seeds"])]
    sac = run_sac_jobs([dict(name=RF_SAC, seed=s, env=kw, train_idx=train_idx) for s in seeds])
    rows, sac_traces = [], {}
    for s in seeds:
        sac_traces[s] = rollout(env, eval_idx, actor_policy(sac[(RF_SAC, s)]["actor"], env))
        rows += [dict(metrics(n, tr), Tohum=s) for n, tr in fixed.items()]
        rows.append(dict(metrics(RF_SAC, sac_traces[s]), Tohum=s))

    s0 = seeds[0]
    trace = dict(t=eval_idx, time=df.index[eval_idx].astype(str).values, load=load[eval_idx], price=price[eval_idx],
                 rf_now=oof[RF][eval_idx, 0], rf_dayahead=oof[RF][eval_idx], P_lim=P_lim, E=E, Pb=Pb,
                 **{f"mpc_{k}": fixed[MPC_RF][k] for k in ("p", "p_grid", "soc", "cost", "violation")},
                 **{f"sac_{k}": sac_traces[s0][k] for k in ("p", "p_grid", "soc", "cost", "violation")},
                 seed=s0)
    params = dict(lam=LAM, viol_pen=pen, E=E, Pb=Pb, P_lim=P_lim, seeds=seeds, test_fold=test_fold,
                  base_violation=float(b_viol), base_viol_hours=int((base["violation"] > 0).sum()))
    return pd.DataFrame(rows), params, trace


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    v4.set_seed(CFG["seed"])
    rng = np.random.default_rng(CFG["seed"])
    only, fold = os.environ.get("ONLY"), int(os.environ.get("FOLD", "5"))
    tag = f"lambda{LAM:g}_s{CFG['n_seeds']}" + (f"_{only}" if only else "")
    print(f"λ={LAM:g} · tohum={CFG['n_seeds']} · kat={fold} · SAC adımı={CFG['sac_steps']:,}", flush=True)
    results = []
    for spec in v4.DATASETS:
        if only and spec["key"] != only:
            continue
        print(f"\n=== {spec['name']} ===", flush=True)
        df, source, synthetic, spec = v4.prepare_dataset(spec, rng)
        if synthetic:
            raise RuntimeError("Gerçek veri yüklenemedi.")
        oof = dict(np.load(os.path.join(SRC_DIR, f"oof_{spec['key']}.npz")))
        t0 = time.time()
        long, params, trace = run(spec, df, oof, fold)
        np.savez_compressed(os.path.join(OUT_DIR, f"iz_{tag}_{spec['key']}.npz"), **trace)
        g = long.groupby("Kontrolcü", sort=False)
        results.append(dict(key=spec["key"], name=spec["name"], params=params, per_seed=long.to_dict("records"),
                            mean=g.mean(numeric_only=True).drop(columns="Tohum").reset_index().to_dict("records"),
                            std=g.std(ddof=1, numeric_only=True).drop(columns="Tohum").reset_index().to_dict("records")))
        with open(os.path.join(OUT_DIR, f"{tag}.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=float)
        print(g[[v6.OBJ, "Tasarruf (%)", v6.VIOL]].mean().round(2).to_string(), flush=True)
        print(f"    süre: {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
