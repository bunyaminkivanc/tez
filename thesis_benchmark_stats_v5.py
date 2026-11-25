"""thesis_benchmark_stats_v5.py: istatistiksel değerlendirme.

thesis_benchmark_experiment_v4.py ile aynı veri, aynı tahmin katmanı, aynı ortam ve aynı SAC kullanılır.
Değerlendirme iki yönde genişletilir:
  1) Tohum sayısı 10'a çıkarılır (42-51); kontrolcüler arasındaki farklar Wilcoxon işaretli sıra testi ve
     eşleştirilmiş t testiyle sınanır, farklar için %95 güven aralığı verilir.
  2) Random Forest ve XGBoost tahminleri de aynı SAC'a bağlanır; ağaç tabanlı yöntemlerin tahmin ölçütleri,
     saatlik maliyeti ve fiziksel ihlali aynı tabloda yer alır. Tahmin modelleri arasındaki farklar
     Diebold-Mariano testiyle sınanır.

Çalıştırma: python thesis_benchmark_stats_v5.py
Seçenekler: QUICK=1 (kısa deneme), SEEDS=10, SAC_WORKERS=6, ONLY=meso (tek veri kümesi; mevcut çıktıya eklenir)
Çıktılar: thesis_benchmark_v5_outputs/ (stats_v5_tables.md, stats_v5_raw.json, controllers_*.csv)
"""

import os
import json
import time

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("SEEDS", "10")
import thesis_benchmark_experiment_v4 as v4
from thesis_benchmark_experiment_v4 import (CFG, FULL, REF_NONE, REF_LP, RANDOM, RULE, SAC_PLAIN, HYBRID,
                                            BatteryFeederEnv, rollout, actor_policy, rule_based_policy,
                                            perfect_foresight_lp, run_sac_jobs, forecast_metrics,
                                            near_zero_threshold, calendar_features)

OUT_DIR = "thesis_benchmark_v5_outputs"
RF, XGB, NAIVE = "Random Forest", "XGBoost", "Son 24 saat (naif)"
RF_SAC, XGB_SAC = "Random Forest + SAC", "XGBoost + SAC"
ORDER = [REF_NONE, RANDOM, RULE, SAC_PLAIN, RF_SAC, XGB_SAC, HYBRID, REF_LP]
METRICS = ["Yakalanan potansiyel (%)", "İhlal düşüşü (%)", "Tasarruf (%)"]


# 1) Ağaç tabanlı tahmin modelleri: v4 ile aynı katlar, aynı standartlaştırma
def run_tree_forecasting(df):
    import xgboost as xgb
    L, H = CFG["seq_len"], CFG["horizon"]
    load = df["load"].values.astype(np.float64)
    n = len(load) - L - H + 1
    win = np.lib.stride_tricks.sliding_window_view(load, L)[:n].astype(np.float32)
    target = np.lib.stride_tricks.sliding_window_view(load[L:], H)[:n].astype(np.float32)
    cal_target = calendar_features(df.index)[L:L + n]
    quick = v4.QUICK
    fold_metrics = {m: [] for m in (RF, XGB, NAIVE)}
    oof = {m: np.full((len(load), H), np.nan) for m in (RF, XGB, NAIVE)}
    for k, (tr, te) in enumerate(TimeSeriesSplit(n_splits=CFG["n_splits"], gap=H - 1).split(np.arange(n))):
        scaler = StandardScaler().fit(load[: tr[-1] + L + H].reshape(-1, 1))
        mu, sd = scaler.mean_[0], scaler.scale_[0]
        X = np.concatenate([(win - mu) / sd, cal_target], axis=1)
        yz = (target - mu) / sd
        n_val = max(int(0.1 * len(tr)), L)
        tr_fit, va = tr[:-(n_val + H - 1)], tr[-n_val:]
        tau = near_zero_threshold(target[tr])

        rf = RandomForestRegressor(n_estimators=40 if quick else 200, min_samples_leaf=5, max_features=0.5,
                                   n_jobs=-1, random_state=CFG["seed"])
        rf.fit(X[tr], yz[tr])
        preds = {RF: np.clip(rf.predict(X[te]) * sd + mu, 0, None)}
        xg = xgb.XGBRegressor(n_estimators=80 if quick else 400, learning_rate=0.05, max_depth=6, subsample=0.8,
                              colsample_bytree=0.8, early_stopping_rounds=30, random_state=CFG["seed"],
                              n_jobs=-1, tree_method="hist")
        xg.fit(X[tr_fit], yz[tr_fit], eval_set=[(X[va], yz[va])], verbose=False)
        preds[XGB] = np.clip(xg.predict(X[te]) * sd + mu, 0, None)
        preds[NAIVE] = win[te]                                 # son 24 saatin ölçümü, ertesi günün tahmini
        for m, p in preds.items():
            fold_metrics[m].append(forecast_metrics(target[te], p, tau))
            oof[m][te + L] = p
        print(f"    kat {k + 1}/{CFG['n_splits']}  MAPE_f  " +
              "  ".join(f"{m}={fold_metrics[m][-1]['MAPE_filtreli']:.2f}" for m in preds), flush=True)
    return fold_metrics, oof


# 2) İstatistik testleri
def diebold_mariano(loss_a, loss_b, h):
    """Diebold–Mariano testi (Harvey–Leybourne–Newbold düzeltmeli). d = loss_a − loss_b; negatif DM: a daha iyi.
    Uzun dönem varyansı h−1 gecikmeli Bartlett çekirdeğiyle (Newey–West) kestirilir."""
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    T, dbar = len(d), d.mean()
    dc = d - dbar
    var = dc @ dc / T
    for lag in range(1, h):
        var += 2 * (1 - lag / h) * (dc[lag:] @ dc[:-lag]) / T
    dm = dbar / np.sqrt(var / T)
    dm *= np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    return dict(DM=float(dm), p=float(2 * stats.t.sf(abs(dm), T - 1)), T=int(T))


def paired_test(a, b):
    """Aynı tohumdaki a − b farkı: ortalama, %95 güven aralığı, Wilcoxon işaretli sıra ve eşleştirilmiş t testi."""
    d = np.asarray(a, float) - np.asarray(b, float)
    n, mean, sd = len(d), d.mean(), d.std(ddof=1) if len(d) > 1 else np.nan
    out = dict(n=int(n), fark=float(mean), std=float(sd), ci_alt=np.nan, ci_ust=np.nan, p_wilcoxon=np.nan,
               p_t=np.nan, pozitif=int((d > 0).sum()))
    if n > 1 and sd > 0:
        half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
        out.update(ci_alt=float(mean - half), ci_ust=float(mean + half),
                   p_t=float(stats.ttest_1samp(d, 0.0).pvalue),
                   p_wilcoxon=float(stats.wilcoxon(d, alternative="two-sided").pvalue))
    return out


# 3) Operasyon: v4'teki ortam ve SAC; yük bilgisinin kaynağı dört seçenek
def run_operation(df, oof, folds, spec):
    load, price = df["load"].values, df["price"].values
    H = CFG["horizon"]
    train_idx, eval_idx = np.concatenate(folds[:-1]), folds[-1]
    ref = load[: train_idx[-1] + 1]
    mean_load = float(ref.mean())
    E = spec.get("batt_energy_ratio", CFG["batt_energy_ratio"]) * mean_load
    Pb = spec.get("batt_power_ratio", CFG["batt_power_ratio"]) * mean_load
    P_lim = float(np.quantile(ref, CFG["line_limit_q"]))
    price_scale = float(np.mean(np.abs(price[train_idx]))) + 1e-9

    history = np.full((len(load), H), np.nan)
    history[H:] = np.lib.stride_tricks.sliding_window_view(load, H)[: len(load) - H]
    last_meas = np.concatenate([[load[0]], load[:-1]])
    infos = {"none": (history, last_meas)}
    infos.update({m: (oof[m], oof[m][:, 0]) for m in (FULL, RF, XGB)})
    env_kw = {key: dict(load=load, load_info=info, load_now=now, price=price, index=df.index,
                        E=E, Pb=Pb, P_lim=P_lim, price_scale=price_scale) for key, (info, now) in infos.items()}
    envs = {key: BatteryFeederEnv(**kw) for key, kw in env_kw.items()}
    env_none = envs["none"]

    fixed = {REF_NONE: rollout(env_none, eval_idx, lambda t, soc: 0.0)}
    lo, hi = np.quantile(price[train_idx], [0.30, 0.70])
    fixed[RULE] = rollout(env_none, eval_idx, rule_based_policy(env_none, lo, hi))
    lp_actions = dict(zip(eval_idx, perfect_foresight_lp(env_none, eval_idx)))
    fixed[REF_LP] = rollout(env_none, eval_idx, lambda t, soc: lp_actions[t])
    base = fixed[REF_NONE]
    base_viol = base["violation"].sum()
    lp_saving = base["cost"].sum() - fixed[REF_LP]["cost"].sum()

    def metrics(name, tr):
        saving = base["cost"].sum() - tr["cost"].sum()
        return {"Kontrolcü": name, "Ort. saatlik maliyet": tr["cost"].mean(),
                "Tasarruf (%)": saving / base["cost"].sum() * 100,
                "Yakalanan potansiyel (%)": saving / lp_saving * 100,
                "Toplam ihlal": tr["violation"].sum(), "İhlalli saat (%)": (tr["violation"] > 0).mean() * 100,
                "İhlal düşüşü (%)": (1 - tr["violation"].sum() / base_viol) * 100 if base_viol > 0 else np.nan,
                "Min. gerilim (pu)": tr["v"].min()}

    learners = [(SAC_PLAIN, "none"), (RF_SAC, RF), (XGB_SAC, XGB), (HYBRID, FULL)]
    seeds = [CFG["seed"] + k for k in range(CFG["n_seeds"])]
    jobs = [dict(name=name, seed=seed, env=env_kw[key], train_idx=train_idx) for seed in seeds for name, key in learners]
    sac = run_sac_jobs(jobs)

    rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        traces = dict(fixed)
        traces[RANDOM] = rollout(env_none, eval_idx, lambda t, soc: rng.uniform(-1.0, 1.0))
        for name, key in learners:
            traces[name] = rollout(envs[key], eval_idx, actor_policy(sac[(name, seed)]["actor"], envs[key]))
        rows += [dict(metrics(name, tr), Tohum=seed) for name, tr in traces.items()]
    long = pd.DataFrame(rows)
    params = dict(E=E, Pb=Pb, P_lim=P_lim, eval_hours=len(eval_idx), train_hours=len(train_idx), seeds=seeds)
    return long, params


def summarize(long):
    num = [c for c in long.columns if c not in ("Kontrolcü", "Tohum")]
    g = long.groupby("Kontrolcü", sort=False)[num]
    table = g.mean().join(g.std(ddof=1).add_suffix(" std"))
    return table.reindex([c for c in ORDER if c in table.index]).reset_index()


def control_tests(long):
    pairs = [(HYBRID, RULE), (HYBRID, RANDOM), (HYBRID, SAC_PLAIN), (HYBRID, RF_SAC), (HYBRID, XGB_SAC),
             (SAC_PLAIN, RULE)]
    rows = []
    for metric in METRICS:
        piv = long.pivot(index="Tohum", columns="Kontrolcü", values=metric)
        for a, b in pairs:
            rows.append(dict(Karşılaştırma=f"{a} − {b}", Metrik=metric, **paired_test(piv[a], piv[b])))
    return pd.DataFrame(rows)


def forecast_tests(df, oof, folds):
    """Bütün katların test başlangıç saatlerinde, 24 saatlik tahmin vektörünün ortalama karesel hatası üzerinden DM."""
    H = CFG["horizon"]
    load = df["load"].values
    idx = np.concatenate(folds)
    truth = np.lib.stride_tricks.sliding_window_view(load, H)[idx]
    loss = {m: ((oof[m][idx] - truth) ** 2).mean(axis=1) for m in oof}
    return pd.DataFrame([dict(Karşılaştırma=f"{FULL} − {m}", **diebold_mariano(loss[FULL], loss[m], H))
                         for m in (RF, XGB, NAIVE)])


# 4) Ana akış
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(CFG["data_dir"], exist_ok=True)
    v4.set_seed(CFG["seed"])
    rng = np.random.default_rng(CFG["seed"])
    print(f"cihaz={v4.DEVICE} · QUICK={v4.QUICK} · tohum={CFG['n_seeds']} · SAC süreçleri={CFG['sac_workers']}")
    results, report = [], ["# İstatistiksel değerlendirme (10 tohum)", ""]
    only = os.environ.get("ONLY")                              # yarıda kalan koşuyu tek veri kümesiyle tamamlamak için
    raw_path, md_path = os.path.join(OUT_DIR, "stats_v5_raw.json"), os.path.join(OUT_DIR, "stats_v5_tables.md")
    if only and os.path.exists(raw_path):
        results = [r for r in json.load(open(raw_path, encoding="utf-8")) if r["key"] != only]
        report = open(md_path, encoding="utf-8").read().split("\n")
    for spec in v4.DATASETS:
        if only and spec["key"] != only:
            continue
        t0 = time.time()
        print(f"\n=== {spec['name']} ===")
        df, source, synthetic, spec = v4.prepare_dataset(spec, rng)
        print(f"    kaynak: {source} · {len(df):,} saat")
        print("  [A] Tahmin: CNN-BiLSTM, Random Forest, XGBoost")
        fm_dl, _, oof, folds = v4.run_forecasting_isolated(df)
        fm_tree, oof_tree = run_tree_forecasting(df)
        fold_metrics = dict(fm_dl, **fm_tree)
        oof = dict(oof, **oof_tree)
        dm = forecast_tests(df, oof, folds)

        print("  [B] Kontrolcüler")
        long, params = run_operation(df, oof, folds, spec)
        table, tests = summarize(long), control_tests(long)
        long.to_csv(os.path.join(OUT_DIR, f"controllers_{spec['key']}.csv"), index=False)
        fc = pd.DataFrame({m: pd.DataFrame(v)[["RMSE", "MAE", "R2", "MAPE_filtreli"]].agg(["mean", "std"]).T.stack()
                           for m, v in fold_metrics.items()}).T
        report += [f"## {spec['name']}", "", f"Kaynak: {source} · tohumlar {params['seeds']}", "",
                   "**Tahmin (5 kat ort/std)**", "", "```", fc.round(4).to_string(), "```", "",
                   "**Diebold–Mariano**", "", "```", dm.round(4).to_string(index=False), "```", "",
                   "**Kontrolcüler (ort, std: tohumlar arası, n−1)**", "", "```",
                   table.round(4).to_string(index=False), "```", "",
                   "**Eşleştirilmiş testler**", "", "```", tests.round(4).to_string(index=False), "```", ""]
        results.append(dict(name=spec["name"], key=spec["key"], source=source, synthetic=synthetic, params=params,
                            fold_metrics=fold_metrics, dm=dm.to_dict("records"),
                            controllers=table.to_dict("records"), per_seed=long.to_dict("records"),
                            tests=tests.to_dict("records")))
        with open(os.path.join(OUT_DIR, "stats_v5_raw.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=float)
        with open(os.path.join(OUT_DIR, "stats_v5_tables.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(report))
        print(table[["Kontrolcü", "Yakalanan potansiyel (%)", "Yakalanan potansiyel (%) std", "İhlal düşüşü (%)",
                     "İhlal düşüşü (%) std"]].round(2).to_string(index=False))
        print(f"    süre: {time.time() - t0:.0f}s", flush=True)
    print("\n" + "\n".join(report), flush=True)


if __name__ == "__main__":
    main()
