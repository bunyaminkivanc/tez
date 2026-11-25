"""Tezdeki şekilleri Bölüm 7 tablolarındaki sayılardan üretir (yeni deney koşmaz).
Çalıştırma: python3 sekilleri_uret.py  ->  sekiller/*.png"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FuncFormatter

import os
OUT = Path(os.environ.get("SEKIL_OUT") or Path(__file__).resolve().parent / "sekiller")
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": os.environ.get("SEKIL_FONT", "Times New Roman"), "mathtext.fontset": "stix", "font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True})

GRI, TURUNCU, YESIL, MAVI, MOR, KIRMIZI, ACIKMAVI = ("#7F7F7F", "#E69F00", "#009E73", "#0072B2", "#CC79A7",
                                                      "#D55E00", "#56B4E9")
OLCEK = ["Konut", "Bina", "Şebeke"]
MODEL = ["CNN", "BiLSTM", "Attention", "CNN-BiLSTM-\nAttention", "CNN-BiLSTM"]
RENK_M = [ACIKMAVI, YESIL, TURUNCU, MOR, MAVI]

# Tablo 6: RMSE (ort, std)
RMSE = {"Konut": [(0.660, 0.064), (0.654, 0.061), (0.713, 0.085), (0.678, 0.069), (0.667, 0.075)],
        "Bina": [(11.632, 3.327), (10.773, 2.519), (13.458, 3.205), (12.155, 3.873), (10.988, 2.781)],
        "Şebeke": [(3631, 1702), (3188, 728), (4365, 1583), (3287, 823), (3146, 753)]}
# Tablo 6: MAPE (ort, std)
MAPE = {"Konut": [(76.51, 19.56), (71.59, 17.26), (83.89, 15.88), (77.13, 20.82), (72.39, 16.58)],
        "Bina": [(14.01, 3.41), (12.73, 1.98), (17.17, 3.09), (13.64, 2.78), (12.92, 2.13)],
        "Şebeke": [(5.14, 2.27), (4.45, 1.00), (6.44, 1.97), (4.54, 0.90), (4.35, 0.95)]}
# Tablo 10: kontrol yöntemleri: 10 tohumlu koşunun çıktısından okunur
import json
_V6 = os.environ.get("V6_RAW") or Path(__file__).resolve().parent / "thesis_benchmark_v7_outputs" / "stats_v6_raw.json"
_RAW = {r["key"]: {c["Kontrolcü"]: c for c in r["controllers"]} for r in json.load(open(_V6))
        if r["params"]["test_fold"] == 5}
# FACS ham dosyada yoktur (anahtar adı "Önerilen çerçeve"); tablolari_uret_v6.py seçimi yapar ve testler_onerilen.json'a yazar.
_SEC = json.load(open(Path(_V6).parent / "testler_onerilen.json"))["5"]
for _k, _v in _SEC.items():
    _RAW[_k]["Önerilen çerçeve"] = _v["ctrl"]
_AD = ["Kural tabanlı yönetim", "MPC + naif tahmin", "MPC + Random Forest", "SAC (tahminsiz)", "Random Forest + SAC",
       "Önerilen çerçeve"]
KONTROL = ["Kural tabanlı", "MPC + naif", "Tek başına MPC", "SAC (tahminsiz)", "Tek başına SAC", "FACS"]
SADE = os.environ.get("SADE") == "1"     # SADE=1: teknik terimsiz etiketler
if SADE:
    KONTROL = ["Sabit kural", "Planlayıcı + dünkü yük", "Tek başına planlayıcı", "Ajan (tahminsiz)", "Tek başına ajan",
               "FACS"]
RENK_K = [TURUNCU, GRI, ACIKMAVI, YESIL, MOR, MAVI]
_ANAHTAR = {"Konut": "micro", "Bina": "meso", "Şebeke": "macro"}


def _seri(metrik):
    return {o: [(_RAW[k][a][metrik], _RAW[k][a].get(metrik + " std") or 0) for a in _AD] for o, k in _ANAHTAR.items()}


POT_K, IHL_K = _seri("Birleşik amaçta yakalanan (%)"), _seri("İhlal düşüşü (%)")


def tr(x, d=1):
    return f"{x:.{d}f}".replace(".", ",").replace("-", "−")


VIRGUL = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",").replace("-", "−"))


def kaydet(fig, ad):
    fig.savefig(OUT / ad, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("yazıldı:", ad)


def sekil_5_1():
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    ax.set_xlim(0, 9.2), ax.set_ylim(0, 4.6), ax.axis("off")
    kutular = [(0.3, "K1: Tahmin\nRandom Forest\n24 sa yük öngörüsü", "#EEF3FA"),
               (3.45, "K2: Karar\nR² < 0,5 ise SAC\nR² ≥ 0,5 ise MPC", "#EEF3FA"),
               (6.6, "Ortam\nGüç dengesi + batarya\nFatura, ihlal, yeni SoC", "#F5F1E8")]
    for x, metin, renk in kutular:
        ax.add_patch(FancyBboxPatch((x, 1.6), 2.3, 1.5, boxstyle="round,pad=0.03,rounding_size=0.1",
                                    fc=renk, ec="#3F5163", lw=1.6))
        ax.text(x + 1.15, 2.35, metin, ha="center", va="center", fontsize=12.5, linespacing=1.6)
    ok = dict(arrowstyle="-|>", color="#3F5163", lw=1.6, mutation_scale=16)
    for x in (2.6, 5.75):
        ax.annotate("", xy=(x + 0.85, 2.35), xytext=(x, 2.35), arrowprops=ok)
    ax.text(1.45, 4.15, "Son 24 saatin yükü + takvim", ha="center", fontsize=12.5)
    ax.annotate("", xy=(1.45, 3.12), xytext=(1.45, 3.95), arrowprops=ok)
    ax.text(4.6, 4.15, "Fiyat + SoC + önceki R²", ha="center", fontsize=12.5)
    ax.annotate("", xy=(4.6, 3.12), xytext=(4.6, 3.95), arrowprops=ok)
    geri = dict(arrowstyle="-|>", color="#55705A", lw=1.6, mutation_scale=16)
    ax.plot([7.75, 7.75, 4.6], [1.6, 0.75, 0.75], color="#55705A", lw=1.6)
    ax.annotate("", xy=(4.6, 1.58), xytext=(4.6, 0.75), arrowprops=geri)
    ax.text(6.2, 0.35, "Uygulanan güçten hesaplanan SoC(t+1), fatura ve ihlal", ha="center", color="#3D5241",
            fontsize=12.5)
    kaydet(fig, "hibrit_kapali_cevrim.png")


def cubuk_paneli(veri, etiketler, renkler, ylabel, ad, vurgu=None, ylim=None, olcekler=None):
    olcekler = olcekler or OLCEK
    fig, axes = plt.subplots(1, len(olcekler), figsize=(3.07 * len(olcekler), 3.7))
    for ax, olcek in zip(axes, olcekler):
        ort = [v[0] for v in veri[olcek]]
        std = [v[1] for v in veri[olcek]]
        cub = ax.bar(range(len(ort)), ort, yerr=[s if s else 0 for s in std], color=renkler, capsize=3,
                     error_kw=dict(lw=1, ecolor="#333333"), width=0.68)
        if vurgu is not None:
            cub[vurgu].set_edgecolor("black"), cub[vurgu].set_linewidth(1.6)
        for i, (o, s) in enumerate(zip(ort, std)):
            ax.text(i, max(o + s, 0) + (max(ort) * 0.02), tr(o, 1 if abs(o) >= 10 else 2), ha="center", va="bottom", fontsize=8)
        if False:
            pass
        else:
            ax.set_xticks(range(len(ort)), [e.replace("\n", "") for e in etiketler], fontsize=8.5, rotation=40, ha="right",
                          rotation_mode="anchor")
        ax.set_title(olcek, fontsize=12, fontweight="bold")
        ax.grid(axis="x", visible=False)
        ax.yaxis.set_major_formatter(VIRGUL)
        if ylim:
            ax.set_ylim(*ylim)
    axes[0].set_ylabel(ylabel)
    fig.tight_layout()
    kaydet(fig, ad)


_TAM = {r["key"]: r for r in json.load(open(_V6)) if r["params"]["test_fold"] == 5}
# Aynı yük bilgisini kullanan MPC ve SAC çiftleri; tahminsiz SAC'ın gördüğü bilgi son 24 saattir (naif tahmin).
_KAYNAK = [("Son 24 saat (naif)", "Naif (son 24 saat)", "MPC + naif tahmin", "SAC (tahminsiz)", GRI),
           ("CNN-BiLSTM", "CNN-BiLSTM", "MPC + CNN-BiLSTM", "CNN-BiLSTM + SAC [önerilen]", YESIL),
           ("Random Forest", "Random Forest (seçilen)", "MPC + Random Forest", "Random Forest + SAC", MAVI),
           ("XGBoost", "XGBoost", "MPC + XGBoost", "XGBoost + SAC", MOR)]


def sekil_6_4():
    """Tahmin hatası (RMSE, beş kat ortalaması) ile aynı tahmini kullanan MPC ve SAC'ın birleşik amaçta yakaladığı potansiyel."""
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.8))
    for ax, olcek in zip(axes, OLCEK):
        r = _TAM[_ANAHTAR[olcek]]
        c = {x["Kontrolcü"]: x for x in r["controllers"]}
        for model, etiket, mpc, sac, renk in _KAYNAK:
            rm = sum(x["RMSE"] for x in r["fold_metrics"][model]) / len(r["fold_metrics"][model])
            ax.plot(rm, c[mpc]["Birleşik amaçta yakalanan (%)"], "s", ms=7, color=renk, mec="black", mew=0.6)
            ax.errorbar(rm, c[sac]["Birleşik amaçta yakalanan (%)"], yerr=c[sac]["Birleşik amaçta yakalanan (%) std"],
                        fmt="o", ms=7, color=renk, ecolor="#888888", elinewidth=1, capsize=3, label=etiket)
        ax.set_title(olcek, fontsize=12, fontweight="bold")
        birim = "MW" if olcek == "Şebeke" else "kW"
        ax.set_xlabel(f"Ortalama tahmin hatası ({birim})\n(sola gittikçe iyi)" if SADE else f"RMSE ({birim})\n(düşük daha iyi)")
        ax.xaxis.set_major_formatter(VIRGUL)
        ax.yaxis.set_major_formatter(VIRGUL)
    axes[0].set_ylabel("Başarı puanı (%)" if SADE else "Birleşik amaçta yakalanan\npotansiyel (%)")
    h, l = axes[0].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    h += [Line2D([], [], marker="s", ls="", color="white", mec="black", ms=7), Line2D([], [], marker="o", ls="", color="white", mec="black", ms=7)]
    l = [("Dünü kopyalayan" if x.startswith("Naif") else x.replace(" (seçilen)", "")) for x in l] if SADE else l
    l += ["Planlayıcı (MPC)", "Ajan (SAC)"] if SADE else ["MPC", "SAC"]
    fig.legend(h, l, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.08), fontsize=9.5)
    fig.tight_layout()
    kaydet(fig, "sekil_6_4_iki_yonlu.png")


if __name__ == "__main__":
    sekil_5_1()
    cubuk_paneli(MAPE, MODEL, RENK_M, "MAPE (%)", "sekil_6_1_tahmin_mimarileri.png", vurgu=4)
    cubuk_paneli(POT_K, KONTROL, RENK_K, "Başarı puanı (%)" if SADE else "Yakalanan potansiyel (%)", "sekil_6_2_yakalanan_potansiyel.png",
                 vurgu=5, ylim=(0, 112))
    cubuk_paneli(IHL_K, KONTROL, RENK_K, "Sınır aşımındaki düşüş (%)" if SADE else "İhlal düşüşü (%)", "sekil_6_3_ihlal_dususu.png", vurgu=5,
                 olcekler=["Konut", "Bina"])
    sekil_6_4()
