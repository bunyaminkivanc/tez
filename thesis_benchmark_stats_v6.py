"""thesis_benchmark_stats_v6.py: tezin ana sonuçlarını üreten karşılaştırma.

v5 ile aynı veri, aynı tahmin katmanı, aynı ortam, aynı SAC ve aynı on tohum (42-51). Değişenler:
  1) Kural tabanlı yönetim: her takvim gününde fiyatı en düşük k saatte şarj, en yüksek k saatte deşarj
     (k = kullanılabilir enerji / güç, yukarı yuvarlanmış = 4 saat). Üç zamanlı tarifede bu, gece şarj ve
     puantta deşarj demektir. v5'teki yüzdelik eşikli kural yalnızca karşılaştırma için tutulur.
  2) Model öngörülü kontrol (MPC): her saatte 24 saatlik tahminle doğrusal program çözülür, ilk eylem uygulanır.
     CNN-BiLSTM, Random Forest, XGBoost ve naif tahminle; ayrıca gerçek yükle (kusursuz tahmin) çalıştırılır.
  3) Teorik optimum ve MPC'de bitiş koşulu: dönem sonunda doluluk en az 0,5 (başlangıç değeri).
  4) Ana ölçüt birleşik amaçtır: fatura + bozunma + ceza × hat ihlali (SAC'ın ve doğrusal programın en aza
     indirdiği büyüklük). Teorik optimum bu amaçta %100'dür. Fatura tasarrufu ve ihlal düşüşü ayrıca raporlanır.
  5) Çoklu karşılaştırma: her ölçütte bütün karşılaştırmalara (üç veri kümesi) Holm düzeltmesi uygulanır.
  6) Zamana genelleme: değerlendirme beşinci katta (ana sonuç) ve dördüncü katta tekrarlanır.

Çalıştırma: OOF_FROM=thesis_benchmark_v6_outputs OUT_DIR=thesis_benchmark_v7_outputs python thesis_benchmark_stats_v6.py
Seçenekler: QUICK=1, SEEDS=10, SAC_WORKERS=7, ONLY=meso, FOLDS=5,4, OOF_FROM (tahminleri önceki koşudan al)
Çıktılar: OUT_DIR/ (stats_v6_raw.json, stats_v6_tables.md, oof_<veri>.npz); varsayılan thesis_benchmark_v6_outputs/
"""

import os
import json
import math
import time

os.environ.setdefault("SEEDS", "10")
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog
from sklearn.model_selection import TimeSeriesSplit

import thesis_benchmark_experiment_v4 as v4
import thesis_benchmark_stats_v5 as v5
from thesis_benchmark_experiment_v4 import (CFG, FULL, REF_NONE, REF_LP, RANDOM, SAC_PLAIN, HYBRID, BATT_ETA,
                                            SOC_MIN, SOC_MAX, BatteryFeederEnv, actor_policy, run_sac_jobs)
from thesis_benchmark_stats_v5 import RF, XGB, NAIVE, RF_SAC, XGB_SAC, paired_test

OUT_DIR = os.environ.get("OUT_DIR", "thesis_benchmark_v6_outputs")
RULE = "Kural tabanlı yönetim"
RULE_OLD = "Eski kural (yüzdelik eşik)"
MPC = {FULL: "MPC + CNN-BiLSTM", RF: "MPC + Random Forest", XGB: "MPC + XGBoost", NAIVE: "MPC + naif tahmin"}
MPC_PERFECT = "MPC + kusursuz tahmin [referans]"
ORDER = [REF_NONE, RANDOM, RULE_OLD, RULE, *MPC.values(), MPC_PERFECT, SAC_PLAIN, RF_SAC, XGB_SAC, HYBRID, REF_LP]
OBJ, BILL, VIOL = "Birleşik amaçta yakalanan (%)", "Fatura tasarrufu / teorik optimum (%)", "İhlal düşüşü (%)"
METRICS = [OBJ, BILL, VIOL, "Tasarruf (%)"]
SOC_END = 0.5


# 1) LP: teorik optimum ve MPC için ortak çözücü
def plan_lp(env, L, pr, soc0, soc_end=SOC_END):
    """Verilen yük ve fiyat dizisi için batarya planı. Amaç: fatura + bozunma + ceza × ihlal.
    c, d: şarj/deşarj gücü; g: şebekeden çekilen güç (≥ 0); s: ihlal; z: SoC (T+1 değer)."""
    T = len(L)
    cost = np.concatenate([np.full(T, env.deg), np.full(T, env.deg), pr, np.full(T, env.viol_pen), np.zeros(T + 1)])
    I, Z, Zs = sparse.identity(T, format="csr"), sparse.csr_matrix((T, T)), sparse.csr_matrix((T, T + 1))
    grid_rows = sparse.hstack([I, -I, -I, Z, Zs]).tocsr()            # L + c - d - g ≤ 0
    neg = pr < 0                                                     # negatif fiyatta g = L + c - d
    A_ub = sparse.vstack([grid_rows[~neg], sparse.hstack([I, -I, Z, -I, Zs])])
    b_ub = np.concatenate([-L[~neg], env.P_lim - L])
    col0 = sparse.csr_matrix((T, 1))
    soc_diff = (sparse.hstack([col0, I]) - sparse.hstack([I, col0])) * env.E
    A_eq = sparse.vstack([sparse.hstack([-BATT_ETA * I, I / BATT_ETA, Z, Z, soc_diff]),
                          sparse.csr_matrix(([1.0], ([0], [4 * T])), shape=(1, 5 * T + 1)),
                          grid_rows[neg]])
    b_eq = np.concatenate([np.zeros(T), [soc0], -L[neg]])
    bounds = ([(0, env.Pb)] * (2 * T) + [(0, None)] * (2 * T) + [(SOC_MIN, SOC_MAX)] * T
              + [(max(SOC_MIN, min(soc_end, SOC_MAX)), SOC_MAX)])
    res = linprog(cost, A_ub=A_ub.tocsr(), b_ub=b_ub, A_eq=A_eq.tocsr(), b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"LP çözülemedi: {res.message}")
    return (res.x[:T] - res.x[T:2 * T]) / env.Pb


def mpc_policy(env, forecast):
    """Her saatte forecast[t] (24 saatlik yük tahmini) ve bilinen 24 saatlik fiyatla LP; ilk eylem uygulanır."""
    def fn(t, soc):
        return plan_lp(env, np.clip(forecast[t], 0, None), env.price_h[t], soc)[0]
    return fn


def daily_rank_policy(env, index):
    """Her gün fiyatı en düşük k saatte şarj (hat sınırını aşmadan), en yüksek k saatte deşarj."""
    k = math.ceil((SOC_MAX - SOC_MIN) * env.E / env.Pb - 1e-9)
    day = pd.Series(env.price, index=index).groupby(index.normalize())
    low = day.rank(method="first").values
    high = day.rank(method="first", ascending=False).values

    def fn(t, soc):
        if low[t] <= k:
            return min(1.0, max(env.P_lim - env.load_now[t], 0.0) / env.Pb)
        return -1.0 if high[t] <= k else 0.0
    return fn, k


def rollout(env, eval_idx, policy_fn):
    soc, out = 0.5, dict(cost=[], violation=[])
    for t in eval_idx:
        soc, _, info = env.step(t, soc, policy_fn(t, soc))
        out["cost"].append(info["cost"]), out["violation"].append(info["violation"])
    return dict(cost=np.array(out["cost"]), violation=np.array(out["violation"]), soc_end=soc)


# 2) Operasyon: seçilen test katında bütün kontrolcüler
def run_operation(df, oof, folds, spec, test_fold):
    load, price = df["load"].values, df["price"].values
    H = CFG["horizon"]
    k = test_fold - 1
    train_idx, eval_idx = np.concatenate(folds[:k]), folds[k]
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
    env0 = envs["none"]

    t0 = time.time()
    fixed = {REF_NONE: rollout(env0, eval_idx, lambda t, soc: 0.0)}
    lo, hi = np.quantile(price[train_idx], [0.30, 0.70])
    fixed[RULE_OLD] = rollout(env0, eval_idx, v4.rule_based_policy(env0, lo, hi))
    rule_fn, k_hours = daily_rank_policy(env0, df.index)
    fixed[RULE] = rollout(env0, eval_idx, rule_fn)
    lp_actions = dict(zip(eval_idx, plan_lp(env0, load[eval_idx], price[eval_idx], 0.5)))
    fixed[REF_LP] = rollout(env0, eval_idx, lambda t, soc: lp_actions[t])
    truth = np.full((len(load), H), np.nan)
    truth[: len(load) - H + 1] = np.lib.stride_tricks.sliding_window_view(load, H)
    for m, name in MPC.items():
        fixed[name] = rollout(env0, eval_idx, mpc_policy(env0, oof[m]))
    fixed[MPC_PERFECT] = rollout(env0, eval_idx, mpc_policy(env0, truth))
    print(f"    kural, LP ve MPC: {time.time() - t0:.0f}s", flush=True)

    base = fixed[REF_NONE]
    pen = env0.viol_pen
    J = lambda tr: tr["cost"].sum() + pen * tr["violation"].sum()
    base_cost, base_viol, base_J = base["cost"].sum(), base["violation"].sum(), J(base)
    lp_bill, lp_J = base_cost - fixed[REF_LP]["cost"].sum(), base_J - J(fixed[REF_LP])

    def metrics(name, tr):
        saving = base_cost - tr["cost"].sum()
        return {"Kontrolcü": name, OBJ: (base_J - J(tr)) / lp_J * 100, BILL: saving / lp_bill * 100,
                VIOL: (1 - tr["violation"].sum() / base_viol) * 100 if base_viol > 0 else np.nan,
                "Tasarruf (%)": saving / base_cost * 100, "Ort. saatlik maliyet": tr["cost"].mean(),
                "Toplam ihlal": tr["violation"].sum(), "İhlalli saat (%)": (tr["violation"] > 0).mean() * 100,
                "Bitiş SoC": tr["soc_end"]}

    learners = [(SAC_PLAIN, "none"), (RF_SAC, RF), (XGB_SAC, XGB), (HYBRID, FULL)]
    seeds = [CFG["seed"] + s for s in range(CFG["n_seeds"])]
    jobs = [dict(name=name, seed=seed, env=env_kw[key], train_idx=train_idx) for seed in seeds for name, key in learners]
    sac = run_sac_jobs(jobs)

    rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        traces = dict(fixed)
        traces[RANDOM] = rollout(env0, eval_idx, lambda t, soc: rng.uniform(-1.0, 1.0))
        for name, key in learners:
            traces[name] = rollout(envs[key], eval_idx, actor_policy(sac[(name, seed)]["actor"], envs[key]))
        rows += [dict(metrics(name, tr), Tohum=seed) for name, tr in traces.items()]
    idx = df.index
    params = dict(E=E, Pb=Pb, P_lim=P_lim, k_hours=k_hours, eval_hours=len(eval_idx), train_hours=len(train_idx),
                  seeds=seeds, test_fold=test_fold,
                  train_period=[str(idx[train_idx[0]]), str(idx[train_idx[-1]])],
                  test_period=[str(idx[eval_idx[0]]), str(idx[eval_idx[-1]])],
                  over_limit_train_pct=float((load[train_idx] > P_lim).mean() * 100),
                  over_limit_test_pct=float((load[eval_idx] > P_lim).mean() * 100),
                  price_q30_q70=[float(lo), float(hi)])
    return pd.DataFrame(rows), params


def summarize(long):
    num = [c for c in long.columns if c not in ("Kontrolcü", "Tohum")]
    g = long.groupby("Kontrolcü", sort=False)[num]
    table = g.mean().join(g.std(ddof=1).add_suffix(" std")).join(g.min()[[OBJ]].add_suffix(" min"))
    return table.reindex([c for c in ORDER if c in table.index]).reset_index()


PAIRS = [(HYBRID, RULE), (HYBRID, MPC[FULL]), (HYBRID, MPC[XGB]), (HYBRID, RANDOM), (HYBRID, SAC_PLAIN),
         (HYBRID, RF_SAC), (HYBRID, XGB_SAC), (SAC_PLAIN, RULE), (RULE, RULE_OLD)]


def control_tests(long):
    rows = []
    for metric in METRICS:
        piv = long.pivot(index="Tohum", columns="Kontrolcü", values=metric)
        for a, b in PAIRS:
            rows.append(dict(Karşılaştırma=f"{a} − {b}", Metrik=metric, **paired_test(piv[a], piv[b])))
    return rows


def holm(p):
    p = np.asarray(p, float)
    order, m, run, out = np.argsort(p), len(p), 0.0, np.empty(len(p))
    for i, j in enumerate(order):
        run = max(run, (m - i) * p[j])
        out[j] = min(1.0, run)
    return out


def add_holm(results):
    """Aynı test katında, her ölçüt için üç veri kümesindeki bütün karşılaştırmalar bir aile sayılır.
    İki deterministik kontrolcünün farkı (varyans sıfır, p tanımsız) test değildir, aileye girmez."""
    for fold in {r["params"]["test_fold"] for r in results}:
        for metric in METRICS:
            cells = [t for r in results if r["params"]["test_fold"] == fold for t in r["tests"]
                     if t["Metrik"] == metric and t["p_wilcoxon"] == t["p_wilcoxon"] and t["std"] > 1e-9]
            for t, ph in zip(cells, holm([t["p_wilcoxon"] for t in cells])):
                t["p_holm"] = float(ph)


def report(results):
    lines = ["# v6: düzeltilmiş kural, MPC, bitiş koşulu, birleşik amaç, Holm düzeltmesi", ""]
    show = ["Kontrolcü", OBJ, OBJ + " std", OBJ + " min", BILL, VIOL, "Tasarruf (%)", "Bitiş SoC"]
    for r in results:
        p = r["params"]
        lines += [f"## {r['name']} · test katı {p['test_fold']}", "",
                  f"Kaynak: {r['source']} · test {p['test_period'][0]} – {p['test_period'][1]} ({p['eval_hours']} saat)",
                  f"Hat sınırı aşımı (bataryasız): eğitim %{p['over_limit_train_pct']:.1f}, test %{p['over_limit_test_pct']:.1f}"
                  f" · kural k = {p['k_hours']} saat", "", "```",
                  pd.DataFrame(r["controllers"])[show].round(2).to_string(index=False), "```", "", "```",
                  pd.DataFrame(r["tests"])[["Karşılaştırma", "Metrik", "fark", "ci_alt", "ci_ust", "p_wilcoxon",
                                            "p_holm", "pozitif"]].round(4).to_string(index=False), "```", ""]
    return "\n".join(lines)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(CFG["data_dir"], exist_ok=True)
    v4.set_seed(CFG["seed"])
    rng = np.random.default_rng(CFG["seed"])
    folds_to_run = [int(f) for f in os.environ.get("FOLDS", "5,4").split(",")]
    only = os.environ.get("ONLY")
    raw_path = os.path.join(OUT_DIR, "stats_v6_raw.json")
    results = json.load(open(raw_path, encoding="utf-8")) if os.path.exists(raw_path) else []
    print(f"cihaz={v4.DEVICE} · QUICK={v4.QUICK} · tohum={CFG['n_seeds']} · katlar={folds_to_run}", flush=True)
    for spec in v4.DATASETS:
        if only and spec["key"] != only:
            continue
        print(f"\n=== {spec['name']} ===", flush=True)
        df, source, synthetic, spec = v4.prepare_dataset(spec, rng)
        if synthetic and not v4.QUICK:
            raise RuntimeError("Gerçek veri yüklenemedi; ana koşu sentetik veriyle yapılmaz.")
        print(f"    kaynak: {source} · {len(df):,} saat · {df.index[0]} – {df.index[-1]}", flush=True)
        if os.environ.get("OOF_FROM"):                      # tahminleri yeniden eğitmeden önceki koşudan al
            src = os.environ["OOF_FROM"]
            oof = dict(np.load(os.path.join(src, f"oof_{spec['key']}.npz")))
            L, H = CFG["seq_len"], CFG["horizon"]
            tss = TimeSeriesSplit(n_splits=CFG["n_splits"], gap=H - 1)
            folds = [te + L for _, te in tss.split(np.arange(len(df) - L - H + 1))]
            eski = next(r for r in json.load(open(os.path.join(src, "stats_v6_raw.json"))) if r["key"] == spec["key"])
            fm_dl, fm_tree, dm = eski["fold_metrics"], {}, pd.DataFrame(eski["dm"])
        else:
            fm_dl, _, oof, folds = v4.run_forecasting_isolated(df)
            fm_tree, oof_tree = v5.run_tree_forecasting(df)
            oof = dict(oof, **oof_tree)
            dm = v5.forecast_tests(df, oof, folds)
        np.savez_compressed(os.path.join(OUT_DIR, f"oof_{spec['key']}.npz"), **{m: oof[m] for m in oof})
        for fold in folds_to_run:
            t0 = time.time()
            print(f"  [test katı {fold}]", flush=True)
            long, params = run_operation(df, oof, folds, spec, fold)
            long.to_csv(os.path.join(OUT_DIR, f"controllers_{spec['key']}_kat{fold}.csv"), index=False)
            results = [r for r in results if not (r["key"] == spec["key"] and r["params"]["test_fold"] == fold)]
            results.append(dict(name=spec["name"], key=spec["key"], source=source, synthetic=synthetic,
                                params=params, fold_metrics=dict(fm_dl, **fm_tree), dm=dm.to_dict("records"),
                                controllers=summarize(long).to_dict("records"), per_seed=long.to_dict("records"),
                                tests=control_tests(long)))
            add_holm(results)
            with open(raw_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=float)
            with open(os.path.join(OUT_DIR, "stats_v6_tables.md"), "w", encoding="utf-8") as f:
                f.write(report(results))
            print(summarize(long)[["Kontrolcü", OBJ, OBJ + " std", BILL, VIOL]].round(1).to_string(index=False))
            print(f"    süre: {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
