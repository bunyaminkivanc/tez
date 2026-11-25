"""Tablo 17: ceza katsayısı λ'ya duyarlılık (beşinci kat, on tohum).
λ = 5 değerleri v7 ana koşusundan (stats_v6_raw.json), λ = 2 ve 10 thesis_benchmark_lambda_v6.py çıktısından okunur.
Çalıştırma: python3 lambda_tablosu.py  ->  Markdown tablo (stdout)"""
import json
import statistics as st
from pathlib import Path

KOK = Path(__file__).resolve().parent
OBJ, TAS, VIOL = "Birleşik amaçta yakalanan (%)", "Tasarruf (%)", "İhlal düşüşü (%)"
SAC, MPC, KURAL = "Random Forest + SAC", "MPC + Random Forest", "Kural tabanlı yönetim"
OLCEK = {"micro": "Konut", "meso": "Bina", "macro": "Şebeke"}
SECIM = {"micro": SAC, "meso": MPC, "macro": MPC}          # Tablo 9: seçim R²'ye bağlı, λ'dan bağımsız

veri = {5.0: {r["key"]: r["per_seed"] for r in json.load(open(KOK / "thesis_benchmark_v7_outputs" / "stats_v6_raw.json"))
              if r["params"]["test_fold"] == 5}}
for lam in (2, 10):
    p = KOK / "thesis_benchmark_v7_lambda" / f"lambda{lam}_s10.json"
    veri[float(lam)] = {r["key"]: r["per_seed"] for r in json.load(open(p))}


def tr(x, d=1):
    return f"{x:.{d}f}".replace(".", ",").replace("-", "−")


def ozet(rows, ad, m):
    x = [r[m] for r in rows if r["Kontrolcü"] == ad]
    return st.mean(x), (st.stdev(x) if len(x) > 1 else 0.0)


if __name__ == "__main__":
    print("| Ölçek | λ | Kural | Tek başına MPC | Tek başına SAC | FACS | MPC fatura / ihlal düşüşü | SAC fatura / ihlal düşüşü |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for k, ad in OLCEK.items():
        for lam in sorted(veri):
            if k not in veri[lam]:
                continue
            rows = veri[lam][k]
            kural, mpc, (sac, sac_sd) = ozet(rows, KURAL, OBJ)[0], ozet(rows, MPC, OBJ)[0], ozet(rows, SAC, OBJ)
            oner = sac if SECIM[k] == SAC else mpc
            iyi = "" if oner >= max(sac, mpc) - 1e-9 else " *"
            print(f"| {ad} | {lam:g} | {tr(kural)} | {tr(mpc)} | {tr(sac)} ± {tr(sac_sd)} | **{tr(oner)}**{iyi} | "
                  f"%{tr(ozet(rows, MPC, TAS)[0])} / %{tr(ozet(rows, MPC, VIOL)[0])} | "
                  f"%{tr(ozet(rows, SAC, TAS)[0])} / %{tr(ozet(rows, SAC, VIOL)[0])} |")
    print("\n* FACS'ın seçtiği karar yöntemi, o λ değerinde iki yöntemden düşük olanıdır.")
