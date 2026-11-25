"""Şekil 6: konut, beşinci kat, tek bir günde MPC + Random Forest ile SAC + Random Forest (tohum 42).
Veri: thesis_benchmark_v7_lambda/iz_lambda5_s1_micro_micro.npz (thesis_benchmark_lambda_v6.py).

Gün seçim kuralı (sonuçlara bakmadan önce belirlendi): test dönemindeki tam günler arasında, Random Forest'ın o günün
00.00'ında yaptığı 24 saatlik tahminin mutlak hatası toplamı en büyük olan gün. Kontrolcülerin sonucu seçime girmez.
Çalıştırma: python3 tek_gun_sekli.py  ->  sekiller/sekil_6_5_tek_gun.png ve seçilen günün özeti (stdout)"""
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
IZ = HERE / "thesis_benchmark_v7_lambda" / "iz_lambda5_s1_micro_micro.npz"
plt.rcParams.update({"font.family": os.environ.get("SEKIL_FONT", "Times New Roman"), "mathtext.fontset": "stix",
                     "font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.axisbelow": True})
GRI, MAVI, KIRMIZI, TURUNCU = "#7F7F7F", "#0072B2", "#D55E00", "#E69F00"

z = np.load(IZ, allow_pickle=True)
zaman = z["time"].astype("datetime64[h]")
gun_basi = [i for i in range(len(zaman) - 23) if int(zaman[i].astype(int)) % 24 == 0]
hata = [np.abs(z["load"][i:i + 24] - z["rf_dayahead"][i]).sum() for i in gun_basi]
i0 = gun_basi[int(np.argmax(hata))]
s = slice(i0, i0 + 24)
saat = np.arange(25)
uz = lambda y: np.append(y, y[-1])                  # step çizimi için son saati tamamla
P_lim = float(z["P_lim"])

fig, axs = plt.subplots(3, 1, figsize=(7.2, 7.4), sharex=True)
for ax in axs:
    ax.axvspan(17, 22, color=TURUNCU, alpha=0.12, lw=0)
axs[0].step(saat, uz(z["load"][s]), where="post", color="black", lw=1.6, label="Gerçek yük")
axs[0].step(saat, uz(z["rf_dayahead"][i0]), where="post", color=GRI, lw=1.4, ls="--",
            label="Random Forest tahmini (00.00)")
axs[0].axhline(P_lim, color=KIRMIZI, lw=1, ls=":", label="Bağlantı sınırı")
axs[0].set_ylabel("Yük (kW)")
axs[1].step(saat, uz(z["mpc_p_grid"][s]), where="post", color=MAVI, lw=1.6, label="MPC + Random Forest")
axs[1].step(saat, uz(z["sac_p_grid"][s]), where="post", color=KIRMIZI, lw=1.6, label="SAC + Random Forest")
axs[1].axhline(P_lim, color=KIRMIZI, lw=1, ls=":")
axs[1].set_ylabel("Şebekeden çekilen\ngüç (kW)")
axs[2].plot(saat, np.append(z["mpc_soc"][s], z["mpc_soc"][i0 + 24]), "-o", ms=3, color=MAVI, lw=1.4)
axs[2].plot(saat, np.append(z["sac_soc"][s], z["sac_soc"][i0 + 24]), "-o", ms=3, color=KIRMIZI, lw=1.4)
axs[2].set_ylabel("Batarya doluluğu")
axs[2].set_ylim(0, 1)
axs[2].set_xlabel("Saat")
axs[2].set_xticks(range(0, 25, 3))
axs[2].set_xlim(0, 24)
for ax in axs[:2]:
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=9, frameon=False)
fig.align_ylabels(axs)
fig.tight_layout()
(HERE / "sekiller").mkdir(exist_ok=True)
fig.savefig(HERE / "sekiller" / "sekil_6_5_tek_gun.png", dpi=200, bbox_inches="tight", facecolor="white")

g = lambda k: z[k][s]
print(f"gün: {zaman[i0].astype('datetime64[D]')} · RF günlük mutlak hata toplamı {max(hata):.2f} kWh "
      f"(test günlerinin ortancası {np.median(hata):.2f})")
for ad, on in (("MPC", "mpc"), ("SAC", "sac")):
    print(f"{ad}: fatura+yıpranma {g(on + '_cost').sum():.3f} · ihlal {g(on + '_violation').sum():.3f} kWh · "
          f"ihlalli saat {(g(on + '_violation') > 0).sum()} · puantta (17-22) deşarj "
          f"{-g(on + '_p')[17:22].clip(max=0).sum():.3f} kWh")
print(f"bataryasız ihlal {np.clip(g('load') - P_lim, 0, None).sum():.3f} kWh · sınır {P_lim:.3f} kW")
