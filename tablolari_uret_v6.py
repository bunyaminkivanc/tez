"""stats_v6_raw.json -> Bölüm 7 tabloları (Markdown). Çalıştırma: python tablolari_uret_v6.py > tablolar_v7.md
Tablolarda elle girilmiş sayı yoktur; her değer ham sonuç dosyasından okunur.

FACS (Tahmin İsabetine Dayalı Kontrolcü Seçimi): Random Forest tahmini; test katından önceki katlarda Random Forest'ın
örneklem dışı R² ortalaması eşiğin altındaysa SAC + Random Forest, üstündeyse MPC + Random Forest. Seçimde test katının
hiçbir bilgisi kullanılmaz. Ham dosyadaki "CNN-BiLSTM + SAC [önerilen]" etiketi ilk sürümden kalmıştır."""
import json
import os
import statistics as st
from pathlib import Path

import numpy as np
from scipy import stats

RAW = Path(os.environ.get("V6_RAW", Path(__file__).resolve().parent / "thesis_benchmark_v7_outputs" / "stats_v6_raw.json"))
ESIK = 0.5
OLCEK = {"micro": ("Konut", "kW"), "meso": ("Bina", "kW"), "macro": ("Şebeke", "MW")}
SIRA = ["micro", "meso", "macro"]
OBJ, BILL, VIOL, TAS = ("Birleşik amaçta yakalanan (%)", "Fatura tasarrufu / teorik optimum (%)", "İhlal düşüşü (%)",
                        "Tasarruf (%)")
NONE, RND, RULE = "Bataryasız [referans]", "Rastgele politika", "Kural tabanlı yönetim"
M_NV, M_XGB, M_RF, M_DL, M_PF = ("MPC + naif tahmin", "MPC + XGBoost", "MPC + Random Forest", "MPC + CNN-BiLSTM",
                                 "MPC + kusursuz tahmin [referans]")
SAC, DL, XGB, RF, LP = ("SAC (tahminsiz)", "CNN-BiLSTM + SAC [önerilen]", "XGBoost + SAC", "Random Forest + SAC",
                        "Teorik optimum [referans]")
HYB = "Önerilen çerçeve"
SATIR = [RND, RULE, M_NV, M_DL, M_XGB, M_RF, SAC, DL, XGB, RF, HYB, M_PF, LP]
AD = {NONE: "Bataryasız işletme (alt referans)", RND: "Rastgele yönetim", RULE: "Kural tabanlı yönetim",
      M_NV: "MPC + naif tahmin", M_RF: "MPC + Random Forest (tek başına MPC)", M_XGB: "MPC + XGBoost",
      M_DL: "MPC + CNN-BiLSTM", SAC: "SAC (tahminsiz)", DL: "SAC + CNN-BiLSTM", XGB: "SAC + XGBoost",
      RF: "SAC + Random Forest (tek başına SAC)", HYB: "**FACS**",
      M_PF: "MPC + gerçek yük (referans)", LP: "Teorik optimum (üst referans)"}
KISA = {HYB: "FACS", RULE: "Kural", RF: "Tek başına SAC", M_RF: "Tek başına MPC", M_NV: "MPC + naif",
        SAC: "SAC (tahminsiz)", RND: "Rastgele"}
# Önceden belirlenmiş karşılaştırmalar. Holm ailesi: her katta ve ölçütte bu çiftlerin tanımlı testleri.
PAIRS = [(HYB, RULE), (HYB, RF), (HYB, M_RF), (HYB, M_NV), (HYB, SAC), (HYB, RND), (RF, SAC)]


def tr(x, d=1):
    s = f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return s.replace("-", "−")


def pv(p):
    return "< 0,001" if p < 0.001 else tr(p, 3)


def r2_once(r, fold):
    """Test katından önceki katlarda Random Forest'ın örneklem dışı R² ortalaması (seçimin tek girdisi)."""
    return st.mean(f["R2"] for f in r["fold_metrics"]["Random Forest"][: fold - 1])


def secim(r, fold):
    return RF if r2_once(r, fold) < ESIK else M_RF


def paired_test(a, b):
    """thesis_benchmark_stats_v5.paired_test ile aynı: aynı tohumdaki a − b farkı, t güven aralığı, Wilcoxon."""
    d = np.asarray(a, float) - np.asarray(b, float)
    n, mean, sd = len(d), d.mean(), d.std(ddof=1)
    out = dict(n=int(n), fark=float(mean), std=float(sd), ci_alt=np.nan, ci_ust=np.nan, p_wilcoxon=np.nan, p_t=np.nan,
               pozitif=int((d > 0).sum()))
    if sd > 1e-9:
        half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
        out.update(ci_alt=float(mean - half), ci_ust=float(mean + half),
                   p_t=float(stats.ttest_1samp(d, 0.0).pvalue),
                   p_wilcoxon=float(stats.wilcoxon(d, alternative="two-sided").pvalue))
    return out


def holm(p):
    order, m, run, out = sorted(range(len(p)), key=lambda i: p[i]), len(p), 0.0, [0.0] * len(p)
    for i, j in enumerate(order):
        run = max(run, (m - i) * p[j])
        out[j] = min(1.0, run)
    return out


def load(fold):
    data = {r["key"]: r for r in json.load(open(RAW)) if r["params"]["test_fold"] == fold}
    for r in data.values():
        r["secim"] = secim(r, fold)
        r["r2_once"] = r2_once(r, fold)
        piv = {}
        for row in r["per_seed"]:
            piv.setdefault(row["Kontrolcü"], {})[row["Tohum"]] = row
        seeds = sorted(piv[RF])
        piv[HYB] = piv[r["secim"]]
        # FACS'ın özet satırı (seçilen kontrolcünün satırı, adı değiştirilerek)
        c = {x["Kontrolcü"]: x for x in r["controllers"]}
        c[HYB] = dict(c[r["secim"]], Kontrolcü=HYB)
        r["ctrl"] = c
        tests = []
        for metric in (OBJ, VIOL, TAS):
            for a, b in PAIRS:
                t = paired_test([piv[a][s][metric] for s in seeds], [piv[b][s][metric] for s in seeds])
                t.update(Karşılaştırma=f"{a} − {b}", Metrik=metric, ayni=(a == HYB and b == r["secim"]))
                tests.append(t)
        r["tests"] = tests
    # Holm: bu katta, her ölçüt için tanımlı testler. İhlal düşüşünde şebeke ölçeği aileye girmez (Bölüm 7.10).
    for metric in (OBJ, VIOL, TAS):
        keys = ["micro", "meso"] if metric == VIOL else SIRA
        cells = [t for k in keys for t in data[k]["tests"] if t["Metrik"] == metric and t["p_wilcoxon"] == t["p_wilcoxon"]]
        for t, ph in zip(cells, holm([t["p_wilcoxon"] for t in cells])):
            t["p_holm"] = ph
        data[keys[0]].setdefault("aile", {})[metric] = len(cells)
    return data


def cell(c, name, key, d=1):
    x = c[name]
    std = x.get(key + " std")
    return f"{tr(x[key], d)} ± {tr(std, d)}" if std and std == std and std > 1e-9 else tr(x[key], d)


def test_satirlari(f, ciftler, metric, keys):
    out = ["| Karşılaştırma | Ölçek | Fark (puan) | %95 güven aralığı | Önde olduğu tohum | p (Wilcoxon) | p (Holm) |",
           "|---|---|---:|---:|---:|---:|---:|"]
    for a, b in ciftler:
        for i, k in enumerate(keys):
            t = next(x for x in f[k]["tests"] if x["Karşılaştırma"] == f"{a} − {b}" and x["Metrik"] == metric)
            ad = KISA[a] + " − " + KISA[b] if i == 0 else ""
            if t["ayni"]:
                out.append(f"| {ad} | {OLCEK[k][0]} | 0,0 | seçilen yöntemin kendisi | - | - | - |")
                continue
            ok = t["p_wilcoxon"] == t["p_wilcoxon"]
            ci = f"[{tr(t['ci_alt'])}; {tr(t['ci_ust'])}]" if ok else "deterministik"
            out.append(f"| {ad} | {OLCEK[k][0]} | {tr(t['fark'])} | {ci} | {t['pozitif']}/{t['n']} | "
                       f"{pv(t['p_wilcoxon']) if ok else '-'} | {pv(t['p_holm']) if ok else '-'} |")
    return out


def main():
    out = []
    f5, f4 = load(5), load(4)
    with open(RAW.parent / "testler_onerilen.json", "w", encoding="utf-8") as fh:
        json.dump({str(k): {key: dict(tests=r["tests"], secim=r["secim"], r2_once=r["r2_once"], ctrl=r["ctrl"][HYB])
                            for key, r in d.items()} for k, d in ((5, f5), (4, f4))}, fh, ensure_ascii=False, default=float)

    out += ["<<DONEM>>", "| Ölçek | Eğitim dönemi (SAC) | Test dönemi (5. kat) | Test dönemi (4. kat) | Sınır aşımı: eğitim / 5. kat test (%) |",
            "|---|---|---|---|---:|"]
    for k in SIRA:
        p, p4 = f5[k]["params"], f4[k]["params"]
        g = lambda s: s[:10]
        out.append(f"| {OLCEK[k][0]} | {g(p['train_period'][0])} – {g(p['train_period'][1])} | "
                   f"{g(p['test_period'][0])} – {g(p['test_period'][1])} | {g(p4['test_period'][0])} – {g(p4['test_period'][1])} | "
                   f"{tr(p['over_limit_train_pct'], 1)} / {tr(p['over_limit_test_pct'], 1)} |")

    out += ["", "<<TAHMIN>>", "| Model | RMSE | MAE | R² | MAPE (%) | DM | p |", "|---|---:|---:|---:|---:|---:|---:|"]
    for k in SIRA:
        r = f5[k]
        d = 0 if k == "macro" else 3
        dm = {x["Karşılaştırma"].split(" − ")[1]: x for x in r["dm"]}
        out.append(f"| **{OLCEK[k][0]} ({OLCEK[k][1]})** | | | | | | |")
        for model in ["Random Forest", "XGBoost", "CNN-BiLSTM", "Son 24 saat (naif)"]:
            rows = r["fold_metrics"][model]
            m = {q: (st.mean(x[q] for x in rows), st.pstdev(x[q] for x in rows)) for q in ("RMSE", "MAE", "R2", "MAPE_filtreli")}
            t = dm.get(model)
            ad = "**Random Forest**" if model == "Random Forest" else model
            out.append(f"| {ad} | {tr(m['RMSE'][0], d)} ± {tr(m['RMSE'][1], d)} | {tr(m['MAE'][0], d)} ± {tr(m['MAE'][1], d)} | "
                       f"{tr(m['R2'][0], 3)} ± {tr(m['R2'][1], 3)} | {tr(m['MAPE_filtreli'][0], 2)} ± {tr(m['MAPE_filtreli'][1], 2)} | "
                       f"{tr(t['DM'], 2) if t else '-'} | {pv(t['p']) if t else '-'} |")

    out += ["", "<<SECIM>>", "| Ölçek | Kat başına R² (Random Forest, 1.-5. kat) | Seçim girdisi: 5. kat için (1.-4. kat ort.) | Seçilen | Seçim girdisi: 4. kat için (1.-3. kat ort.) | Seçilen |",
            "|---|---|---:|---|---:|---|"]
    for k in SIRA:
        katlar = " / ".join(tr(x["R2"], 2) for x in f5[k]["fold_metrics"]["Random Forest"])
        ad = lambda s: "SAC" if s == RF else "MPC"
        out.append(f"| {OLCEK[k][0]} | {katlar} | {tr(f5[k]['r2_once'], 2)} | {ad(f5[k]['secim'])} | "
                   f"{tr(f4[k]['r2_once'], 2)} | {ad(f4[k]['secim'])} |")

    out += ["", "<<POTANSIYEL_KAT5>>", "| Yöntem | Konut | Bina | Şebeke |", "|---|---:|---:|---:|"]
    for name in SATIR:
        row = " | ".join(cell(f5[k]["ctrl"], name, OBJ) for k in SIRA)
        if name == HYB:
            row = " | ".join(f"**{v}**" for v in row.split(" | "))
        out.append(f"| {AD[name]} | {row} |")

    out += ["", "<<FATURA_IHLAL>>", "| Yöntem | Fatura düşüşü: konut | Fatura düşüşü: bina | Fatura düşüşü: şebeke | İhlal düşüşü: konut | İhlal düşüşü: bina | Bitiş doluluğu: konut / bina / şebeke |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for name in SATIR:
        cs = [f5[k]["ctrl"] for k in SIRA]
        fat = " | ".join(cell(c, name, TAS, 2) for c in cs)
        ihl = " | ".join(cell(c, name, VIOL) for c in cs[:2])
        soc = " / ".join(tr(c[name]["Bitiş SoC"], 2) for c in cs)
        out.append(f"| {AD[name]} | {fat} | {ihl} | {soc} |")

    ciftler = [(HYB, RULE), (HYB, RF), (HYB, M_RF), (HYB, M_NV), (HYB, SAC)]
    out += ["", "<<TESTLER>>"] + test_satirlari(f5, ciftler + [(RF, SAC)], OBJ, SIRA)
    out += ["", "<<IHLAL_TEST>>"] + test_satirlari(f5, ciftler, VIOL, ["micro", "meso"])
    out += ["", "<<TESTLER_KAT4>>"] + test_satirlari(f4, ciftler + [(RF, SAC)], OBJ, SIRA)

    out += ["", "<<KAT4>>", "| Yöntem | Konut: 5. kat | Konut: 4. kat | Bina: 5. kat | Bina: 4. kat | Şebeke: 5. kat | Şebeke: 4. kat |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for name in [RULE, M_NV, M_DL, M_XGB, M_RF, SAC, DL, XGB, RF, HYB, M_PF]:
        vals = [cell(f[k]["ctrl"], name, OBJ) for k in SIRA for f in (f5, f4)]
        row = " | ".join(vals)
        if name == HYB:
            row = " | ".join(f"**{v}**" for v in row.split(" | "))
        out.append(f"| {AD[name]} | {row} |")

    out += ["", "<<OZET>>", "| Yöntem | Altı durumun ortalaması | Kuralı geçtiği durum |", "|---|---:|---:|"]
    for name in [RULE, M_NV, M_RF, SAC, RF, HYB]:
        v = [f[k]["ctrl"][name][OBJ] for f in (f5, f4) for k in SIRA]
        kural = [f[k]["ctrl"][RULE][OBJ] for f in (f5, f4) for k in SIRA]
        gec = "-" if name == RULE else f"{sum(a > b for a, b in zip(v, kural))}/6"
        ad = AD[name].replace("**", "")
        row = f"| {ad} | {tr(st.mean(v))} | {gec} |"
        out.append(row.replace(f"| {ad} |", f"| **{ad}** |") if name == HYB else row)

    out += ["", "<<AILE>>", f"Holm ailesi büyüklüğü: 5. kat amaç {f5['micro']['aile'][OBJ]}, ihlal {f5['micro']['aile'][VIOL]}; "
            f"4. kat amaç {f4['micro']['aile'][OBJ]}"]
    print("\n".join(out))


if __name__ == "__main__":
    main()
