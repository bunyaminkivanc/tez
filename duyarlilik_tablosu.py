"""Tablo 18 (tarife) ve Tablo 19 (batarya boyutu): beşinci kat, λ = 5, SAC'ta on tohum.
1× / tezdeki tarife değerleri v7 ana koşusundan, diğerleri thesis_benchmark_lambda_v6.py çıktısından
(FIYAT=tou|piyasa ve BATARYA_KAT=0.5|2) okunur.
Çalıştırma: python3 duyarlilik_tablosu.py  ->  Markdown tablolar (stdout)"""
import json
from pathlib import Path

from lambda_tablosu import KURAL, MPC, OBJ, OLCEK, SAC, SECIM, VIOL, ozet, tr

IST = Path(__file__).resolve().parent
ana = {r["key"]: r["per_seed"] for r in json.load(open(IST / "thesis_benchmark_v7_outputs" / "stats_v6_raw.json"))
       if r["params"]["test_fold"] == 5}


def oku(p):
    return {r["key"]: r["per_seed"] for r in json.load(open(p))}


def satir(ad, kosul, rows):
    kural, mpc, (sac, sd) = ozet(rows, KURAL, OBJ)[0], ozet(rows, MPC, OBJ)[0], ozet(rows, SAC, OBJ)
    oner = sac if SECIM[k] == SAC else mpc
    iyi = "" if oner >= max(sac, mpc) - 1e-9 else " *"
    print(f"| {ad} | {kosul} | {tr(kural)} | {tr(mpc)} | {tr(sac)} ± {tr(sd)} | **{tr(oner)}**{iyi} | "
          f"%{tr(ozet(rows, MPC, VIOL)[0])} |")


BASLIK = "| Kural | Tek başına MPC | Tek başına SAC | FACS | MPC ihlal düşüşü |\n|---|---|---:|---:|---:|---:|---:|"
TEZDEKI = {"micro": "TOU", "meso": "TOU", "macro": "Piyasa"}
DIGER = {"micro": ("Piyasa", "lambda5_s10_micro.json"), "meso": ("Piyasa", "lambda5_s10_meso.json"),
         "macro": ("TOU", "lambda5_s10_macro.json")}
print("<<TARIFE>>\n| Ölçek | Tarife " + BASLIK)
for k, ad in OLCEK.items():
    satir(ad, TEZDEKI[k] + " (ana koşu)", ana[k])
    t, f = DIGER[k]
    satir(ad, t, oku(IST / "thesis_benchmark_v7_tarife" / f)[k])

print("\n<<BATARYA>>\n| Ölçek | Batarya " + BASLIK)
bat = {K: oku(IST / f"thesis_benchmark_v7_batarya_x{K}" / "lambda5_s10.json") for K in ("0.5", "2")}
for k, ad in OLCEK.items():
    satir(ad, "0,5×", bat["0.5"][k])
    satir(ad, "1× (ana koşu)", ana[k])
    satir(ad, "2×", bat["2"][k])
print("\n* FACS'ın seçtiği karar yöntemi, o koşulda iki yöntemden düşük olanıdır.")
