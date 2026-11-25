"""thesis_benchmark_experiment_v3.py: tahminsiz SAC ile CNN-BiLSTM + SAC'ın üç tohumla karşılaştırması (ara sürüm).

v2 ile aynı veri, aynı düzen ve aynı tohumlar; Katman 1 CNN-BiLSTM'dir (v2'de CNN-BiLSTM-Attention).
Her veri kümesinde 3 tohum × 4 SAC eğitilir:
  SAC (K2) | SAC + PINN (K2+K3) | CNN-BiLSTM + SAC (K1+K2) | CNN-BiLSTM + SAC + PINN (K1+K2+K3)
K1 = tahmin, K2 = SAC, K3 = PINN. Bataryasız işletme ve teorik optimum yalnızca ölçüm referansıdır.

Çalıştırma (Google Colab): !python thesis_benchmark_experiment_v3.py
Seçenekler: QUICK=1, FORCE_SYNTHETIC=1, SEEDS=3, SAC_WORKERS=8
Yerel CSV: thesis_data/{micro,macro,meso}.csv (timestamp,load[,price])
Çıktılar: thesis_benchmark_v3_outputs/ (benchmark_v3_tables.md, benchmark_v3_raw.json, *.csv, *.png)
İndirme başarısız olursa sentetik veri kullanılır ve tablolarda "SENTETİK" olarak işaretlenir.
"""

import os
import sys
import json
import time
import random
import zipfile
import warnings
import faulthandler
import urllib.request
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib
if "google.colab" not in sys.modules and not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
if sys.platform == "darwin":   # macOS'ta TF + PyTorch OpenMP çakışması segfault üretir (Colab/Linux etkilenmez)
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

# TensorFlow yalnızca tahmin alt sürecinde yüklenir (bkz. _load_tf). TF ve PyTorch aynı süreçte
# GPU kullanınca (farklı CUDA sürümleri) PyTorch'un ilk CUDA çağrısı süreci sessizce öldürüyor.
tf = keras = layers = None

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
faulthandler.enable()   # segfault gibi sessiz çökmelerde nerede öldüğünü yazdır

# Yapılandırma
QUICK = os.environ.get("QUICK", "0") == "1"
FORCE_SYNTHETIC = os.environ.get("FORCE_SYNTHETIC", "0") == "1"

CFG = dict(
    seed=42,
    out_dir="thesis_benchmark_v3_outputs",
    data_dir="thesis_data",
    max_hours=24 * 365 * 2,
    seq_len=24, horizon=24, n_splits=5,
    dl_epochs=150, dl_batch=128, dl_patience=15,
    sac_steps=40000, sac_warmup=4000, sac_batch=256, sac_episode_len=168,
    n_seeds=int(os.environ.get("SEEDS", "3")),
    sac_workers=int(os.environ.get("SAC_WORKERS", "0")) or min(os.cpu_count() or 2, 8),
    pinn_epochs=400,
    near_zero_alpha=0.10,
    batt_energy_ratio=2.0, batt_power_ratio=0.5,
    line_limit_q=0.85, viol_penalty=5.0, pinn_margin_pu=0.03,
)
if QUICK:
    CFG.update(max_hours=24 * 120, dl_epochs=3, sac_steps=1500, sac_warmup=300, pinn_epochs=60, n_seeds=1)

TOU_TARIFF = {"gece (22-06)": 1.55, "gündüz (06-17)": 2.60, "puant (17-22)": 4.10}

V0, V_MIN, V_MAX = 1.00, 0.95, 1.05
R_PU, X_PU, TAN_PHI = 0.03, 0.04, 0.50   # R + X*tanφ = 0.05 -> P = P_lim'de V = 0.95 pu
BATT_ETA, SOC_MIN, SOC_MAX, DEG_COST_FRAC = 0.95, 0.10, 0.90, 0.02

# Katman 1 bileşenleri (son eleman hibritin kendi tahmincisi). İç anahtarlar kısa; raporda FC_LABEL kullanılır.
FULL = "CNN-BiLSTM"
FORECASTERS = [FULL]   # Yalnızca CNN-BiLSTM eğitilir (diğer mimariler v2'de karşılaştırıldı)
FC_LABEL = {"CNN": "CNN [bileşen]", "BiLSTM": "BiLSTM [bileşen]", "Attention": "Attention [bileşen]",
            "CNN-BiLSTM": "CNN-BiLSTM [bileşen]", FULL: "CNN-BiLSTM [HİBRİT K1]"}

# Katman kombinasyonları (enerji kıyası). K1 = tahmin, K2 = SAC, K3 = PINN.
REF_NONE, REF_LP = "Bataryasız [referans]", "Teorik optimum [referans]"
K2, K3, K23, K12, K123 = ("Yalnız SAC (K2)", "Yalnız PINN (K3)", "SAC + PINN (K2+K3)",
                          "Tahmin + SAC (K1+K2)", "HİBRİT [ÖNERİLEN] (K1+K2+K3)")
LAYER_ORDER = [REF_NONE, K3, K2, K23, K12, K123, REF_LP]
COMPONENTS = [K3, K2, K23, K12, K123]


def role(name):
    """Her satırın rapordaki rolü: önerilen model, onun bir parçası ya da yalnızca ölçüm referansı."""
    if name in (K123, FULL, FC_LABEL[FULL]):
        return "Önerilen hibrit"
    if name in (REF_NONE, REF_LP):
        return "Referans (rakip değil)"
    return "Hibrit bileşeni"


GLOSSARY = [
    ("Önerilen hibrit", "3 katmanlı model: K1 tahmin → K2 karar → K3 fiziksel güvenlik. Tablolarda **kalın** ve "
                        "[ÖNERİLEN] / [HİBRİT K1] etiketli."),
    ("K1 — CNN-BiLSTM", "Tahmin katmanı. Geçmiş 24 saatin yükünden önümüzdeki 24 saatin yükünü tahmin eder. "
                           "CNN (kısa vadeli örüntüler) + BiLSTM (zaman bağımlılığı, iki yönlü) + Attention "
                           "(önemli saatlere ağırlık verme)."),
    ("K2 — SAC (Soft Actor-Critic)", "Karar katmanı. Pekiştirmeli öğrenme ile bataryayı yöneten ajan: her saat "
                                     "tahmin, fiyat ve batarya doluluğuna bakıp şarj/deşarj kararı verir; faturayı "
                                     "düşürüp hat limitini aşmamayı deneme-yanılma ile öğrenir (Haarnoja vd., 2018)."),
    ("K3 — PINN (Physics-Informed Neural Network)", "Fiziksel güvenlik katmanı. Kirchhoff güç dengesi ve "
                                                    "gerilim düşümü (LinDistFlow) denklemlerini kayıp fonksiyonunda "
                                                    "kullanan sinir ağı; SAC kararı gerilimi 0,95 pu altına "
                                                    "düşürecekse kararı düzeltir."),
    ("Hibrit bileşeni", "Önerilen hibritin bir katmanı tek başına ya da eksik kombinasyonu. Amaç "
                                   "her katmanın gerçekten katkı sağlayıp sağlamadığını ölçmek."),
    ("[bileşen] tahminciler", "K1'in parçaları tek başına: CNN, BiLSTM, Attention, CNN-BiLSTM. Aynı veri ve "
                              "eğitim koşullarında tam K1 ile kıyaslanır."),
    ("Referans (rakip değil)", "Bataryasız = hiç batarya kullanılmaz (tasarruf %0 noktası). Teorik optimum = "
                               "geleceği tam bilen doğrusal programlama (LP); gerçekte ulaşılamaz, %100 potansiyel "
                               "noktası."),
    ("Yakalanan potansiyel", "Kontrolcü tasarrufu / teorik optimum tasarrufu × 100. Ulaşılabilir en iyi sonucun "
                             "yüzde kaçının yakalandığı."),
    ("Filtrelenmiş MAPE", "Ortalama mutlak yüzde hata; sıfıra yakın yük saatleri (payda patlaması) dışlanarak."),
]
FIG_NOTE = ("K1 = CNN-BiLSTM tahmin katmanı · K2 = SAC (Soft Actor-Critic) pekiştirmeli öğrenme ile batarya "
            "kontrolü · K3 = PINN fizik-bilgili sinir ağı ile gerilim/hat güvenliği · [ÖNERİLEN]/[HİBRİT K1] = "
            "önerilen model, [bileşen] = onun parçası, [referans] = rakip değil")

# Okabe-Ito sabit sıra; her varlık her grafikte aynı rengi taşır
PALETTE = {
    "CNN": "#E69F00", "BiLSTM": "#009E73", "Attention": "#CC79A7", "CNN-BiLSTM": "#56B4E9", FULL: "#0072B2",
    REF_NONE: "#7F7F7F", K3: "#CC79A7", K2: "#E69F00", K23: "#56B4E9", K12: "#009E73", K123: "#0072B2",
    REF_LP: "#D55E00",
}
PALETTE.update({FC_LABEL[m]: PALETTE[m] for m in FORECASTERS})


def _load_tf():
    global tf, keras, layers
    import tensorflow as _tf
    from tensorflow import keras as _keras
    from tensorflow.keras import layers as _layers
    tf, keras, layers = _tf, _keras, _layers
    tf.get_logger().setLevel("ERROR")
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if tf is not None:      # yalnızca tahmin alt sürecinde yüklü
        tf.random.set_seed(seed)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 1) Veri yükleme (v1 ile aynı)
def _download(url, dest):
    if not os.path.exists(dest):
        print(f"    indiriliyor: {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as r, open(dest + ".part", "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
        os.replace(dest + ".part", dest)
    return dest


def tou_price(index):
    h = index.hour
    return np.where((h >= 6) & (h < 17), TOU_TARIFF["gündüz (06-17)"],
                    np.where((h >= 17) & (h < 22), TOU_TARIFF["puant (17-22)"], TOU_TARIFF["gece (22-06)"]))


def _local_csv(key):
    path = os.path.join(CFG["data_dir"], f"{key}.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp").sort_index()


def load_micro():
    local = _local_csv("micro")
    if local is not None:
        return local, "Yerel CSV (micro.csv)"
    url = "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip"
    zpath = _download(url, os.path.join(CFG["data_dir"], "uci_household.zip"))
    with zipfile.ZipFile(zpath) as z:
        name = [n for n in z.namelist() if n.endswith(".txt")][0]
        with z.open(name) as f:
            raw = pd.read_csv(f, sep=";", usecols=["Date", "Time", "Global_active_power"],
                              na_values=["?"], low_memory=False)
    ts = pd.to_datetime(raw["Date"] + " " + raw["Time"], format="%d/%m/%Y %H:%M:%S")
    load = pd.Series(raw["Global_active_power"].astype(float).values, index=ts).resample("h").mean()
    return pd.DataFrame({"load": load}), "UCI Household Power (gerçek)"


def load_macro():
    local = _local_csv("macro")
    if local is not None:
        return local, "Yerel CSV (macro.csv, ör. EPİAŞ)"
    url = "https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv"
    path = _download(url, os.path.join(CFG["data_dir"], "opsd_60min.csv"))
    cols = ["utc_timestamp", "DE_load_actual_entsoe_transparency", "DE_LU_price_day_ahead"]
    raw = pd.read_csv(path, usecols=cols, parse_dates=["utc_timestamp"]).set_index("utc_timestamp").dropna()
    raw.index = raw.index.tz_localize(None)
    return raw.rename(columns={cols[1]: "load", cols[2]: "price"}), "OPSD / ENTSO-E DE-LU (gerçek)"


def load_meso():
    local = _local_csv("meso")
    if local is not None:
        return local, "Yerel CSV (meso.csv, ör. ASHRAE)"
    url = ("https://media.githubusercontent.com/media/buds-lab/building-data-genome-project-2/"
           "master/data/meters/cleaned/electricity_cleaned.csv")
    path = _download(url, os.path.join(CFG["data_dir"], "bdg2_electricity.csv"))
    header = pd.read_csv(path, nrows=0).columns
    offices = [c for c in header if "office" in c.lower()][:40]
    if not offices:
        raise ValueError("BDG2 başlığında ofis sayacı bulunamadı")
    raw = pd.read_csv(path, usecols=["timestamp"] + offices, parse_dates=["timestamp"]).set_index("timestamp")
    stats = pd.DataFrame({"nan": raw.isna().mean(), "mean": raw.mean()})
    best = stats[(stats["mean"] > 10)].sort_values("nan").index[0]
    return pd.DataFrame({"load": raw[best]}), f"BDG2 / ASHRAE ofis sayacı '{best}' (gerçek)"


def _hours(n):
    return pd.date_range("2022-01-01", periods=n, freq="h")


def synth_micro(n, rng):
    idx = _hours(n)
    h, doy = idx.hour.values, idx.dayofyear.values
    season = 1 + 0.35 * np.cos(2 * np.pi * (doy - 15) / 365)
    profile = (0.25 + 0.9 * np.exp(-((h - 8) ** 2) / 3) + 1.6 * np.exp(-((h - 20) ** 2) / 5))
    night = (h >= 1) & (h <= 5)
    base = profile * season * rng.lognormal(0, 0.25, n)
    base[night] *= rng.uniform(0.02, 0.4, night.sum())
    spikes = rng.binomial(1, 0.04, n) * rng.uniform(1, 3, n)
    return pd.DataFrame({"load": np.clip(base + spikes, 0.005, None)}, index=idx)


def synth_macro(n, rng):
    idx = _hours(n)
    h, dow, doy = idx.hour.values, idx.dayofweek.values, idx.dayofyear.values
    daily = 0.82 + 0.18 * np.sin(2 * np.pi * (h - 6) / 24) + 0.06 * np.exp(-((h - 19) ** 2) / 4)
    weekly = np.where(dow >= 5, 0.86, 1.0)
    season = 1 + 0.12 * np.cos(2 * np.pi * (doy - 20) / 365) + 0.06 * np.cos(4 * np.pi * (doy - 200) / 365)
    ar = np.zeros(n)
    for t in range(1, n):
        ar[t] = 0.95 * ar[t - 1] + rng.normal(0, 0.008)
    load = 40000 * daily * weekly * season * (1 + ar)
    price = 18 + 0.0016 * load + 25 * (load / load.mean() - 1) ** 2 * 10 + rng.normal(0, 6, n)
    return pd.DataFrame({"load": load, "price": np.clip(price, -5, None)}, index=idx)


def synth_meso(n, rng):
    idx = _hours(n)
    h, dow, doy = idx.hour.values, idx.dayofweek.values, idx.dayofyear.values
    occupied = ((h >= 8) & (h < 18) & (dow < 5)).astype(float)
    hvac = 1 + 0.4 * np.abs(np.cos(2 * np.pi * (doy - 200) / 365))
    load = (60 + 140 * occupied * hvac + 25 * ((h >= 6) & (h < 8) & (dow < 5))) * rng.lognormal(0, 0.07, n)
    return pd.DataFrame({"load": load}, index=idx)


DATASETS = [
    dict(key="micro", name="Mikro-Konut (UCI)", loader=load_micro, synth=synth_micro,
         unit="kW", energy="kWh", currency="TL"),
    # Şebeke ölçeğinde batarya = ort. yükün %2'si güç, %8'i enerji (4 saatlik depolama)
    dict(key="macro", name="Makro-Şebeke (ENTSO-E/EPİAŞ)", loader=load_macro, synth=synth_macro,
         unit="MW", energy="MWh", currency="EUR", batt_energy_ratio=0.08, batt_power_ratio=0.02),
    dict(key="meso", name="Mezo-Bina (ASHRAE/BDG2)", loader=load_meso, synth=synth_meso,
         unit="kW", energy="kWh", currency="TL"),
]


def prepare_dataset(spec, rng):
    df, source, synthetic = None, None, False
    if not FORCE_SYNTHETIC:
        try:
            df, source = spec["loader"]()
        except Exception as exc:  # ağ / format hatası -> sentetik
            print(f"    [UYARI] {spec['name']} indirilemedi ({type(exc).__name__}: {exc}). Sentetik veriye geçiliyor.")
    if df is None:
        df, source, synthetic = spec["synth"](CFG["max_hours"], rng), "SENTETİK üreteç", True

    df = df[~df.index.duplicated()].sort_index().asfreq("h")
    df = df.interpolate(limit=6).dropna()
    df = df.asfreq("h").ffill()
    df = df.iloc[-CFG["max_hours"]:].copy()
    df["load"] = df["load"].clip(lower=0)
    if "price" not in df.columns:
        df["price"] = tou_price(df.index)
        source += " + örnekleyici 3 zamanlı tarife"
        if spec["currency"] == "EUR":
            spec = dict(spec, currency="TL")
    if spec["key"] == "macro" and spec["currency"] == "EUR" and "EPİAŞ" in source:
        spec = dict(spec, currency="TL")
    return df, source, synthetic, spec


# 2) Ön işleme
def calendar_features(index):
    h, dow, m = index.hour.values, index.dayofweek.values, index.month.values
    return np.stack([np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
                     np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
                     np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12)], axis=1).astype(np.float32)


def build_windows(load, index, L, H):
    """t başlangıç saati için girdi: load[t-L:t]; hedef: load[t:t+H] (doğrudan çok adımlı)."""
    n = len(load) - L - H + 1
    win = np.lib.stride_tricks.sliding_window_view(load, L)[:n].astype(np.float32)
    target = np.lib.stride_tricks.sliding_window_view(load[L:], H)[:n].astype(np.float32)
    cal = calendar_features(index)
    cal_seq = np.lib.stride_tricks.sliding_window_view(cal, (L, cal.shape[1]))[:n, 0]
    return win, target, cal_seq


def near_zero_threshold(y_train):
    """Gece ~0 tüketimde MAPE paydası patlar; eşik altı noktalar MAPE'den dışlanır (diğer metrikler tüm noktalarda)."""
    return max(CFG["near_zero_alpha"] * float(np.mean(y_train)), 1e-6)


def forecast_metrics(y, p, tau):
    y, p = np.ravel(y), np.ravel(p)
    mask = y > tau
    return dict(RMSE=float(np.sqrt(mean_squared_error(y, p))), MAE=float(mean_absolute_error(y, p)),
                R2=float(r2_score(y, p)),
                MAPE_ham=float(np.mean(np.abs(y - p) / np.maximum(np.abs(y), 1e-6)) * 100),
                MAPE_filtreli=float(np.mean(np.abs(y[mask] - p[mask]) / y[mask]) * 100),
                dislanan_pct=float((~mask).mean() * 100))


# 3) Katman 1 ve bileşenleri (Keras): aynı girdi, aynı çıkış başlığı, aynı eğitim protokolü
def build_forecaster(name, L, n_feat, H):
    inp = keras.Input(shape=(L, n_feat))

    def conv_block(x):
        x = layers.Conv1D(64, 3, padding="causal", activation="relu")(x)
        return layers.Conv1D(64, 3, padding="causal", activation="relu")(x)

    def attention_pool(x):                                   # hibritteki Bahdanau tipi zamansal dikkat
        score = layers.Dense(1)(layers.Dense(64, activation="tanh")(x))
        weights = layers.Softmax(axis=1)(score)
        return layers.Flatten()(layers.Dot(axes=1)([weights, x]))   # Σ_t α_t h_t

    if name == "CNN":
        z = layers.Flatten()(conv_block(inp))
    elif name == "BiLSTM":
        z = layers.Bidirectional(layers.LSTM(64))(inp)
    elif name == "Attention":
        # Hibritin dikkat bloğu tek başına: girdinin doğrusal izdüşümü (BiLSTM çıkış genişliği 128) + zamansal dikkat.
        # 24 sa pencerede saat sin/cos girdisi her adımın konumunu tekil olarak kodlar.
        z = attention_pool(layers.Dense(128)(inp))
    elif name == "CNN-BiLSTM":
        z = layers.Bidirectional(layers.LSTM(64))(conv_block(inp))
    elif name == FULL:
        z = attention_pool(layers.Bidirectional(layers.LSTM(64, return_sequences=True))(conv_block(inp)))
    else:
        raise ValueError(name)
    z = layers.Dropout(0.2)(layers.Dense(64, activation="relu")(z))
    model = keras.Model(inp, layers.Dense(H)(z), name=name.replace("-", "_"))
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return model


def run_forecasting(df):
    """5 katlı TimeSeriesSplit; her katta tüm bileşenler aynı veri, aynı tohum ve aynı erken durdurma ile eğitilir."""
    L, H = CFG["seq_len"], CFG["horizon"]
    load = df["load"].values.astype(np.float64)
    win, target, cal_seq = build_windows(load, df.index, L, H)
    n = len(target)
    tss = TimeSeriesSplit(n_splits=CFG["n_splits"], gap=H - 1)
    fold_metrics = {m: [] for m in FORECASTERS}
    histories = {m: [] for m in FORECASTERS}
    oof = {m: np.full((len(load), H), np.nan) for m in FORECASTERS}   # oof[m][t] = t saatinde yapılan 24 sa tahmin
    folds = []

    for k, (tr, te) in enumerate(tss.split(np.arange(n))):
        scaler = StandardScaler().fit(load[: tr[-1] + L + H].reshape(-1, 1))
        mu, sd = scaler.mean_[0], scaler.scale_[0]
        X_seq = np.concatenate([((win - mu) / sd)[..., None], cal_seq], axis=2).astype(np.float32)
        yz = (target - mu) / sd
        n_val = max(int(0.1 * len(tr)), L)
        tr_fit, va = tr[:-(n_val + H - 1)], tr[-n_val:]
        tau = near_zero_threshold(target[tr])
        line = []
        for m in FORECASTERS:
            set_seed(CFG["seed"] + k)
            model = build_forecaster(m, L, X_seq.shape[2], H)
            t0 = time.time()
            hist = model.fit(X_seq[tr_fit], yz[tr_fit], validation_data=(X_seq[va], yz[va]),
                             epochs=CFG["dl_epochs"], batch_size=CFG["dl_batch"], verbose=0,
                             callbacks=[keras.callbacks.EarlyStopping(patience=CFG["dl_patience"],
                                                                      restore_best_weights=True),
                                        keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5)])
            fit_s = time.time() - t0
            pred = np.clip(model.predict(X_seq[te], verbose=0, batch_size=1024) * sd + mu, 0, None)
            met = forecast_metrics(target[te], pred, tau)
            met.update(parametre=int(model.count_params()), egitim_s=float(fit_s), epoch=len(hist.history["loss"]))
            fold_metrics[m].append(met)
            histories[m].append({kk: [float(v) for v in vals] for kk, vals in hist.history.items()})
            oof[m][te + L] = pred
            line.append(f"{m}={met['MAPE_filtreli']:.2f}")
            keras.backend.clear_session()
        folds.append(te + L)
        print(f"    kat {k + 1}/{CFG['n_splits']}  MAPE_f  " + "  ".join(line), flush=True)
    return fold_metrics, histories, oof, folds


def _forecast_worker(df):
    """Ayrı (spawn) süreç: TF burada GPU'yu kullanır, ana süreçteki PyTorch CUDA'sına dokunmaz."""
    _load_tf()
    print(f"    TensorFlow {tf.__version__} · GPU: {len(tf.config.list_physical_devices('GPU'))}", flush=True)
    return run_forecasting(df)


def run_forecasting_isolated(df):
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as ex:
        return ex.submit(_forecast_worker, df).result()


# 4) Fiziksel model ve ortam (v1 ile aynı)
def feeder_voltage_pu(p_grid_pu):
    """Tek dallı radyal besleyici, LinDistFlow: V ≈ V0 - (R·P + X·Q)/V0, Q = P·tanφ (pu)."""
    return V0 - (R_PU * p_grid_pu + X_PU * p_grid_pu * TAN_PHI) / V0


class BatteryFeederEnv:
    """Saatlik batarya dispeçi. Eylem a∈[-1,1] -> p_batt = a·P_b (+ şarj, - deşarj). KCL: P_grid = P_load + P_batt.
    Ödül = bataryasız duruma göre −(Δfatura + ceza·Δihlal). Gözlem: 24 sa yük bilgisi, 24 sa fiyat, SOC, takvim.
    load_now: kontrolcünün içinde bulunulan saatin yükü için kestirimi (PINN kullanır)."""

    def __init__(self, load, load_info, load_now, price, index, E, Pb, P_lim, price_scale):
        self.load, self.load_info, self.load_now, self.price = load, load_info, load_now, price
        H = load_info.shape[1]
        padded = np.concatenate([price, np.full(H, price[-1])])
        self.price_h = np.lib.stride_tricks.sliding_window_view(padded, H)[:len(price)]
        self.cal = calendar_features(index)[:, :4]
        self.E, self.Pb, self.P_lim, self.price_scale = E, Pb, P_lim, price_scale
        self.deg = DEG_COST_FRAC * price_scale
        self.viol_pen = CFG["viol_penalty"] * price_scale
        self.cost_scale = price_scale * Pb

    def obs(self, t, soc):
        return np.concatenate([self.load_info[t] / self.P_lim, self.price_h[t] / self.price_scale,
                               [soc], self.cal[t]]).astype(np.float32)

    def feasible_power(self, p, soc):
        p_max_ch = (SOC_MAX - soc) * self.E / BATT_ETA
        p_max_dis = (soc - SOC_MIN) * self.E * BATT_ETA
        return float(np.clip(p, -min(self.Pb, p_max_dis), min(self.Pb, p_max_ch)))

    def step(self, t, soc, a):
        p = self.feasible_power(float(a) * self.Pb, soc)
        soc_next = soc + (p * BATT_ETA if p >= 0 else p / BATT_ETA) / self.E
        p_grid = self.load[t] + p
        cost = self.price[t] * max(p_grid, 0.0) + self.deg * abs(p)
        violation = max(0.0, abs(p_grid) - self.P_lim)
        v = feeder_voltage_pu(p_grid / self.P_lim)
        base_cost = self.price[t] * max(self.load[t], 0.0)
        base_viol = max(0.0, abs(self.load[t]) - self.P_lim)
        reward = -((cost - base_cost) + self.viol_pen * (violation - base_viol)) / self.cost_scale
        return soc_next, reward, dict(cost=cost, violation=violation, v=v, p=p, p_grid=p_grid)


def perfect_foresight_lp(env, idx):
    """Gelecekteki yük ve fiyatı tam bilen LP (HiGHS), SAC ile aynı amaç. Erişilemez üst sınır referansı."""
    from scipy import sparse
    from scipy.optimize import linprog
    T, L, pr = len(idx), env.load[idx], env.price[idx]
    cost = np.concatenate([np.full(T, env.deg), np.full(T, env.deg), pr, np.full(T, env.viol_pen), np.zeros(T + 1)])
    I, Z, Zs = sparse.identity(T, format="csr"), sparse.csr_matrix((T, T)), sparse.csr_matrix((T, T + 1))
    grid_rows = sparse.hstack([I, -I, -I, Z, Zs]).tocsr()            # L + c - d - g
    neg = pr < 0   # negatif fiyatta g = L + c - d eşitliği (aksi halde "sonsuz çek" ile sınırsız)
    A_ub = sparse.vstack([grid_rows[~neg], sparse.hstack([I, -I, Z, -I, Zs])])
    b_ub = np.concatenate([-L[~neg], env.P_lim - L])
    col0 = sparse.csr_matrix((T, 1))
    soc_diff = (sparse.hstack([col0, I]) - sparse.hstack([I, col0])) * env.E
    A_eq = sparse.vstack([sparse.hstack([-BATT_ETA * I, I / BATT_ETA, Z, Z, soc_diff]),
                          sparse.hstack([sparse.csr_matrix((1, 4 * T)),
                                         sparse.csr_matrix(([1.0], ([0], [0])), shape=(1, T + 1))]),
                          grid_rows[neg]])
    b_eq = np.concatenate([np.zeros(T), [0.5], -L[neg]])
    bounds = [(0, env.Pb)] * (2 * T) + [(0, None)] * (2 * T) + [(SOC_MIN, SOC_MAX)] * (T + 1)
    res = linprog(cost, A_ub=A_ub.tocsr(), b_ub=b_ub, A_eq=A_eq.tocsr(), b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"Teorik optimum LP çözülemedi: {res.message}")
    return (res.x[:T] - res.x[T:2 * T]) / env.Pb


# 5) Katman 2: SAC
class Actor(nn.Module):
    def __init__(self, s_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(s_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.mu, self.log_std = nn.Linear(hidden, 1), nn.Linear(hidden, 1)

    def forward(self, s):
        h = self.net(s)
        return self.mu(h), self.log_std(h).clamp(-5, 2)

    def sample(self, s):
        mu, log_std = self(s)
        std = log_std.exp()
        u = mu + std * torch.randn_like(mu)
        a = torch.tanh(u)
        logp = (-0.5 * ((u - mu) / std) ** 2 - log_std - 0.5 * np.log(2 * np.pi)).sum(1, keepdim=True)
        logp -= torch.log(1 - a.pow(2) + 1e-6).sum(1, keepdim=True)
        return a, logp

    def act(self, s):
        return torch.tanh(self(s)[0])


class Critic(nn.Module):
    def __init__(self, s_dim, hidden=256):
        super().__init__()
        def q():
            return nn.Sequential(nn.Linear(s_dim + 1, hidden), nn.ReLU(), nn.Linear(hidden, hidden),
                                 nn.ReLU(), nn.Linear(hidden, 1))
        self.q1, self.q2 = q(), q()

    def forward(self, s, a):
        sa = torch.cat([s, a], 1)
        return self.q1(sa), self.q2(sa)


class SAC:
    def __init__(self, s_dim, gamma=0.99, tau=0.005, lr=3e-4):
        self.actor = Actor(s_dim).to(DEVICE)
        self.critic, self.target = Critic(s_dim).to(DEVICE), Critic(s_dim).to(DEVICE)
        self.target.load_state_dict(self.critic.state_dict())
        self.log_alpha = torch.zeros(1, requires_grad=True, device=DEVICE)
        self.target_entropy = -1.0
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.opt_al = torch.optim.Adam([self.log_alpha], lr=lr)
        self.gamma, self.tau = gamma, tau

    def update(self, s, a, r, s2, d):
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            a2, logp2 = self.actor.sample(s2)
            q1t, q2t = self.target(s2, a2)
            y = r + self.gamma * (1 - d) * (torch.min(q1t, q2t) - alpha * logp2)
        q1, q2 = self.critic(s, a)
        loss_c = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.opt_c.zero_grad(); loss_c.backward(); self.opt_c.step()

        a_new, logp = self.actor.sample(s)
        q1n, q2n = self.critic(s, a_new)
        loss_a = (alpha * logp - torch.min(q1n, q2n)).mean()
        self.opt_a.zero_grad(); loss_a.backward(); self.opt_a.step()

        loss_al = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.opt_al.zero_grad(); loss_al.backward(); self.opt_al.step()

        with torch.no_grad():
            for p, pt in zip(self.critic.parameters(), self.target.parameters()):
                pt.mul_(1 - self.tau).add_(self.tau * p)
        return loss_c.item(), loss_a.item()


def train_sac(env, train_idx, seed, safety=None):
    """safety verilirse PINN eğitim döngüsündedir: ortamın parçası sayılır, ajan eyleminin düzeltileceğini
    bilerek öğrenir (tamponda ajanın kendi eylemi saklanır)."""
    set_seed(seed)
    s_dim = len(env.obs(train_idx[0], 0.5))
    agent = SAC(s_dim)
    cap = CFG["sac_steps"]
    buf = {k: np.zeros((cap, d), np.float32) for k, d in [("s", s_dim), ("a", 1), ("r", 1), ("s2", s_dim), ("d", 1)]}
    ep_len = CFG["sac_episode_len"]
    starts = train_idx[: len(train_idx) - ep_len - 1]
    log = dict(critic=[], actor=[])
    rng = np.random.default_rng(seed)
    t_ep, soc, step_in_ep = int(rng.choice(starts)), rng.uniform(SOC_MIN, SOC_MAX), 0
    ep_ret, returns = 0.0, []
    for i in range(cap):
        t = t_ep + step_in_ep
        s = env.obs(t, soc)
        a = rng.uniform(-1, 1) if i < CFG["sac_warmup"] else float(
            agent.actor.sample(torch.as_tensor(s, device=DEVICE)[None])[0].item())
        soc2, r, _ = env.step(t, soc, safety(t, soc, a) if safety is not None else a)
        done = step_in_ep == ep_len - 1
        s2 = env.obs(t + 1, soc2)
        for k, v in zip(["s", "a", "r", "s2", "d"], [s, [a], [r], s2, [float(done)]]):
            buf[k][i] = v
        soc, step_in_ep, ep_ret = soc2, step_in_ep + 1, ep_ret + r
        if done:
            returns.append(ep_ret)
            t_ep, soc, step_in_ep, ep_ret = int(rng.choice(starts)), rng.uniform(SOC_MIN, SOC_MAX), 0, 0.0
        if i >= CFG["sac_warmup"]:
            idx = rng.integers(0, i + 1, CFG["sac_batch"])
            batch = [torch.as_tensor(buf[k][idx], device=DEVICE) for k in ["s", "a", "r", "s2", "d"]]
            lc, la = agent.update(*batch)
            log["critic"].append(lc); log["actor"].append(la)
    return agent, log, returns


# 6) Katman 3: PINN güvenlik katmanı
class SafetyPINN(nn.Module):
    """Girdi: (L̂_pu, a, SOC, Pb_pu, E_pu). Çıktı: düzeltilmiş eylem a*, P̂_grid, V̂.
    Kayıp = sadakat + KCL artığı + KVL/LinDistFlow artığı + eşitsizlik (V_min, SOC) cezaları."""

    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(5, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, 3))

    def forward(self, x):
        o = self.net(x)
        a_corr = torch.tanh(torch.atanh(x[:, 1:2].clamp(-0.999, 0.999)) + o[:, 0:1])
        return a_corr, o[:, 1:2], o[:, 2:3] + V0


def pinn_loss(model, x, margin):
    a_in, l_pu, soc, pb_pu, e_pu = x[:, 1:2], x[:, 0:1], x[:, 2:3], x[:, 3:4], x[:, 4:5]
    a, p_hat, v_hat = model(x)
    k = R_PU + X_PU * TAN_PHI
    l_fid = ((a - a_in) ** 2).mean()
    l_kcl = ((p_hat - (l_pu + a * pb_pu)) ** 2).mean()                 # Kirchhoff akım/güç dengesi
    l_kvl = ((v_hat - (V0 - k * p_hat / V0)) ** 2).mean()             # Kirchhoff gerilim (LinDistFlow)
    soc_next = soc + a * pb_pu / e_pu
    v_floor = V_MIN + k * margin
    # Gerilim ihlali k'ya bölünerek güç (pu) ölçeğine taşınır (aksi halde kare ceza ~1e-5 kalıp etkisiz olur)
    l_ineq = ((F.relu(v_floor - v_hat) / k) ** 2 + (F.relu(v_hat - V_MAX) / k) ** 2).mean() \
        + (F.relu(soc_next - SOC_MAX) ** 2 + F.relu(SOC_MIN - soc_next) ** 2).mean()
    total = l_fid + 10 * l_kcl + 10 * l_kvl + 1000 * l_ineq
    return total, dict(toplam=total.item(), sadakat=l_fid.item(), KCL=l_kcl.item(),
                       KVL=l_kvl.item(), esitsizlik=l_ineq.item())


def train_pinn(env, train_idx, seed):
    """Politikadan bağımsız: eğitim dönemindeki her saat için rastgele SOC ve rastgele eylem."""
    set_seed(seed)
    rng = np.random.default_rng(seed)
    pb_pu, e_pu = env.Pb / env.P_lim, env.E / env.P_lim
    rows = []
    for t in train_idx:
        for soc, a in zip(rng.uniform(SOC_MIN, SOC_MAX, 3), rng.uniform(-1, 1, 3)):
            rows.append([env.load_now[t] / env.P_lim, a, soc, pb_pu, e_pu])
    X = torch.as_tensor(np.array(rows, np.float32), device=DEVICE)
    model = SafetyPINN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, CFG["pinn_epochs"])
    log = []
    for _ in range(CFG["pinn_epochs"]):
        perm = torch.randperm(len(X), device=DEVICE)
        for b in range(0, len(X), 2048):
            loss, _ = pinn_loss(model, X[perm[b:b + 2048]], CFG["pinn_margin_pu"])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        with torch.no_grad():
            log.append(pinn_loss(model, X, CFG["pinn_margin_pu"])[1])
    model.eval()
    return model, log


def make_safety(pinn, env):
    pb_pu, e_pu = env.Pb / env.P_lim, env.E / env.P_lim

    @torch.no_grad()
    def safety(t, soc, a):
        x = torch.tensor([[env.load_now[t] / env.P_lim, a, soc, pb_pu, e_pu]], dtype=torch.float32, device=DEVICE)
        return pinn(x)[0].item()
    return safety


# 7) Paralel SAC eğitimi (her iş ayrı spawn süreç; GPU paylaşılır)
def _to_numpy_state(module):
    return {k: v.detach().cpu().numpy() for k, v in module.state_dict().items()}


def _load_state(module, state):
    module.load_state_dict({k: torch.as_tensor(v) for k, v in state.items()})
    return module.to(DEVICE).eval()


def _sac_job(job):
    torch.set_num_threads(1)
    env = BatteryFeederEnv(**job["env"])
    safety = None
    if job["pinn_state"] is not None:
        safety = make_safety(_load_state(SafetyPINN(), job["pinn_state"]), env)
    agent, log, returns = train_sac(env, job["train_idx"], job["seed"], safety)
    return dict(name=job["name"], seed=job["seed"], actor=_to_numpy_state(agent.actor), log=log, returns=returns)


def run_sac_jobs(jobs):
    results, t0 = {}, time.time()
    workers = max(1, min(CFG["sac_workers"], len(jobs)))
    print(f"    {len(jobs)} SAC eğitimi, {workers} paralel süreç ({CFG['sac_steps']:,} adım/iş)", flush=True)
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as ex:
        futures = [ex.submit(_sac_job, j) for j in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            results[(res["name"], res["seed"])] = res
            last = np.mean(res["returns"][-10:]) if res["returns"] else float("nan")
            print(f"      [{i}/{len(jobs)}] {res['name']} · tohum {res['seed']} · son 10 bölüm ödülü {last:.1f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    return results


def actor_policy(state, env):
    actor = _load_state(Actor(len(env.obs(0, 0.5))), state)

    @torch.no_grad()
    def fn(t, soc):
        return actor.act(torch.as_tensor(env.obs(t, soc), device=DEVICE)[None]).item()
    return fn


# 8) Operasyon: katman kombinasyonları + Katman-1 bileşeni × SAC
def rollout(env, eval_idx, policy_fn):
    soc, out = 0.5, dict(cost=[], violation=[], v=[])
    for t in eval_idx:
        soc, _, info = env.step(t, soc, policy_fn(t, soc))
        for k in out:
            out[k].append(info[k])
    return {k: np.array(v) for k, v in out.items()}


def aggregate(op_long, order):
    grouped = op_long.drop(columns="Tohum").groupby("Kontrolcü")
    order = [c for c in order if c in grouped.groups]
    table = grouped.mean().reindex(order)
    std = grouped.std(ddof=0).reindex(order)
    for c in ["Tasarruf (%)", "Yakalanan potansiyel (%)", "Toplam ihlal", "İhlal düşüşü (%)"]:
        table[c + " std"] = std[c]
    return table.reset_index()


def run_operation(df, oof, folds, spec):
    load, price = df["load"].values, df["price"].values
    H = CFG["horizon"]
    train_idx = np.concatenate(folds[:-1])            # kat 1-4 test bölgeleri (örneklem dışı tahminli)
    eval_idx = folds[-1]                               # kat 5 test bölgesi (görülmemiş)
    ref = load[: train_idx[-1] + 1]
    mean_load = float(ref.mean())
    E = spec.get("batt_energy_ratio", CFG["batt_energy_ratio"]) * mean_load
    Pb = spec.get("batt_power_ratio", CFG["batt_power_ratio"]) * mean_load
    P_lim = float(np.quantile(ref, CFG["line_limit_q"]))
    price_scale = float(np.mean(np.abs(price[train_idx]))) + 1e-9

    # Tahminsiz kontrolcüler tahmincinin gördüğü ham bilgiyi (son 24 sa ölçüm) görür -> tek fark tahmin katmanıdır
    history = np.full((len(load), H), np.nan)
    history[H:] = np.lib.stride_tricks.sliding_window_view(load, H)[: len(load) - H]
    last_meas = np.concatenate([[load[0]], load[:-1]])
    infos = {"none": (history, last_meas)}
    infos.update({m: (oof[m], oof[m][:, 0]) for m in FORECASTERS})
    env_kw = {key: dict(load=load, load_info=info, load_now=now, price=price, index=df.index,
                        E=E, Pb=Pb, P_lim=P_lim, price_scale=price_scale) for key, (info, now) in infos.items()}
    envs = {key: BatteryFeederEnv(**kw) for key, kw in env_kw.items()}
    env_none, env_full = envs["none"], envs[FULL]

    fixed_traces = {REF_NONE: rollout(env_none, eval_idx, lambda t, soc: 0.0)}
    try:
        lp_actions = dict(zip(eval_idx, perfect_foresight_lp(env_none, eval_idx)))
        fixed_traces[REF_LP] = rollout(env_none, eval_idx, lambda t, soc: lp_actions[t])
        print("    Teorik optimum (LP) çözüldü")
    except RuntimeError as exc:
        print(f"    [UYARI] {exc} — 'Yakalanan potansiyel' bu veri setinde boş kalacak.")
    base = fixed_traces[REF_NONE]
    base_viol = base["violation"].sum()
    lp_saving = base["cost"].sum() - fixed_traces[REF_LP]["cost"].sum() if REF_LP in fixed_traces else np.nan

    def metrics(name, tr):
        saving = base["cost"].sum() - tr["cost"].sum()
        return {
            "Kontrolcü": name,
            "Ort. saatlik maliyet": tr["cost"].mean(),
            "Tasarruf (%)": saving / base["cost"].sum() * 100,
            "Yakalanan potansiyel (%)": saving / lp_saving * 100 if lp_saving > 0 else np.nan,
            "Toplam ihlal": tr["violation"].sum(),
            "İhlalli saat (%)": (tr["violation"] > 0).mean() * 100,
            "İhlal düşüşü (%)": (1 - tr["violation"].sum() / base_viol) * 100 if base_viol > 0 else np.nan,
            "Min. gerilim (pu)": tr["v"].min(),
        }

    seeds = [CFG["seed"] + k for k in range(CFG["n_seeds"])]
    pinns, pinn_logs, jobs = {}, {}, []
    for seed in seeds:
        # İki PINN: tahminsiz (son ölçüm) ve tahminli (Katman-1 kestirimi) yük bilgisine göre
        pinns[("none", seed)], pinn_logs[("none", seed)] = train_pinn(env_none, train_idx, seed)
        pinns[(FULL, seed)], pinn_logs[(FULL, seed)] = train_pinn(env_full, train_idx, seed)
        for name, env_key, with_pinn in ([(K2, "none", False), (K23, "none", True), (K123, FULL, True)]
                                          + [(f"K1[{m}]+K2", m, False) for m in FORECASTERS]):
            jobs.append(dict(name=name, seed=seed, env=env_kw[env_key], train_idx=train_idx,
                             pinn_state=_to_numpy_state(pinns[(env_key, seed)]) if with_pinn else None))
    print(f"    PINN'ler eğitildi ({2 * len(seeds)} adet)")
    sac = run_sac_jobs(jobs)

    layer_rows, fc_rows, traces = [], [], None
    for seed in seeds:
        safe_none = make_safety(pinns[("none", seed)], env_none)
        safe_full = make_safety(pinns[(FULL, seed)], env_full)
        pol = {name: actor_policy(sac[(name, seed)]["actor"], envs[key])
               for name, key in [(K2, "none"), (K23, "none"), (K123, FULL)]
               + [(f"K1[{m}]+K2", m) for m in FORECASTERS]}
        fc_traces = {m: rollout(envs[m], eval_idx, pol[f"K1[{m}]+K2"]) for m in FORECASTERS}
        seed_traces = dict(fixed_traces)
        seed_traces.update({
            K3: rollout(env_none, eval_idx, lambda t, soc: safe_none(t, soc, 0.0)),    # planlama yok, yalnız güvenlik
            K2: rollout(env_none, eval_idx, pol[K2]),
            K23: rollout(env_none, eval_idx, lambda t, soc: safe_none(t, soc, pol[K23](t, soc))),
            K12: fc_traces[FULL],
            K123: rollout(env_full, eval_idx, lambda t, soc: safe_full(t, soc, pol[K123](t, soc))),
        })
        layer_rows += [dict(metrics(n, tr), Tohum=seed) for n, tr in seed_traces.items()]
        fc_rows += [dict(metrics(m, tr), Tohum=seed) for m, tr in fc_traces.items()]
        if traces is None:   # grafikler 1. tohumun izlerini kullanır
            traces = seed_traces

    layer_long, fc_long = pd.DataFrame(layer_rows), pd.DataFrame(fc_rows)
    s0 = seeds[0]
    rl_logs = {name: sac[(name, s0)]["log"] for name in [K2, K23, f"K1[{FULL}]+K2", K123]}
    rl_logs["pinn"] = pinn_logs[(FULL, s0)]
    params = dict(E=E, Pb=Pb, P_lim=P_lim, eval_hours=len(eval_idx), train_hours=len(train_idx), seeds=seeds)
    return (aggregate(layer_long, LAYER_ORDER), layer_long, aggregate(fc_long, FORECASTERS), fc_long,
            traces, eval_idx, rl_logs, params)


def contributions(layer_long):
    """Her katmanın marjinal katkısı: aynı tohumdaki iki kombinasyon farkı (eşleştirilmiş), tohumlar üzerinden ort ± std."""
    pairs = [("K1 (tahmin) katkısı — PINN'siz", K12, K2), ("K1 (tahmin) katkısı — PINN'li", K123, K23),
             ("K3 (PINN) katkısı — tahminsiz", K23, K2), ("K3 (PINN) katkısı — tahminli", K123, K12),
             ("K2 (SAC planlama) katkısı — PINN'li", K23, K3)]
    rows = []
    for metric in ["Yakalanan potansiyel (%)", "İhlal düşüşü (%)", "Tasarruf (%)"]:
        piv = layer_long.pivot(index="Tohum", columns="Kontrolcü", values=metric)
        for label, a, b in pairs:
            d = piv[a] - piv[b]
            rows.append(dict(Katkı=label, Metrik=metric, Ort=d.mean(), Std=d.std(ddof=0)))
    return pd.DataFrame(rows)


# 9) Grafikler
def _style():
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.05,
                  rc={"axes.spines.top": False, "axes.spines.right": False, "grid.color": "#E6E6E6",
                      "lines.linewidth": 2, "savefig.dpi": 300, "figure.dpi": 110})


def _smooth(x, w=200):
    x = np.asarray(x, float)
    return x if len(x) < w else np.convolve(x, np.ones(w) / w, mode="valid")


def _label(res):
    return res["name"].replace(" (", "\n(") + ("\n[SENTETİK]" if res["synthetic"] else "")


def _finish(fig, path):
    fig.tight_layout()
    fig.text(0.5, -0.01, FIG_NOTE, ha="center", va="top", fontsize=8, color="#555555", wrap=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curves(results, path):
    fig, axes = plt.subplots(len(results), 3, figsize=(17, 4.3 * len(results)), squeeze=False)
    for r, res in enumerate(results):
        ax = axes[r, 0]
        for m in FORECASTERS:
            h = res["histories"][m][-1]
            ax.plot(h["val_loss"], color=PALETTE[m], label=FC_LABEL[m], lw=2.5 if m == FULL else 1.6)
        ax.set(title=f"{res['name']}\nKatman-1 bileşenleri: doğrulama kaybı (son kat)", xlabel="Epoch",
               ylabel="MSE (z-ölçek)")
        ax.legend(frameon=False, fontsize=8)

        ax = axes[r, 1]
        for name, color in [(K2, PALETTE[K2]), (K23, PALETTE[K23]), (f"K1[{FULL}]+K2", PALETTE[K12]),
                            (K123, PALETTE[K123])]:
            ax.plot(_smooth(res["rl_logs"][name]["critic"]), color=color,
                    label=K12 if name.startswith("K1[") else name)
        ax.set(title="SAC kritik (Q) kaybı, katman kombinasyonları (1. tohum)", xlabel="Güncelleme adımı",
               ylabel="Kayıp")
        ax.set_yscale("log"); ax.legend(frameon=False, fontsize=8)

        ax = axes[r, 2]
        pl = pd.DataFrame(res["rl_logs"]["pinn"])
        for col, c in [("toplam", "#000000"), ("sadakat", "#0072B2"), ("KCL", "#E69F00"),
                       ("KVL", "#009E73"), ("esitsizlik", "#CC79A7")]:
            ax.plot(pl[col].clip(lower=1e-12), color=c, label=col)
        ax.set(title="Katman-3 PINN kayıp bileşenleri (1. tohum)", xlabel="Epoch", ylabel="Kayıp (log)")
        ax.set_yscale("log"); ax.legend(frameon=False, ncol=2, fontsize=8)
    _finish(fig, path)


def _bar_labels(ax):
    for c in ax.containers:
        if hasattr(c, "patches"):
            ax.bar_label(c, fmt="%.1f", fontsize=7, padding=2, color="#333333")


def plot_forecast_components(results, path):
    rows = [dict(Veri=_label(res), Model=FC_LABEL[m], MAPE=f["MAPE_filtreli"], R2=f["R2"])
            for res in results for m in FORECASTERS for f in res["fold_metrics"][m]]
    d = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(18, 5.6))
    for ax, col, title in [(axes[0], "MAPE", "(a) Filtrelenmiş MAPE (%), 24 sa ufuk — düşük iyi"),
                           (axes[1], "R2", "(b) R² — yüksek iyi")]:
        sns.barplot(data=d, x="Veri", y=col, hue="Model", hue_order=[FC_LABEL[m] for m in FORECASTERS],
                    palette=PALETTE, errorbar="sd",
                    capsize=0.06, ax=ax, saturation=1, edgecolor="white", linewidth=2)
        ax.set(title=title + " · 5 kat ort. ± ss", xlabel="", ylabel=col)
        ax.legend(frameon=False, fontsize=8, title="Katman-1 bileşeni")
        _bar_labels(ax)
    _finish(fig, path)


def plot_energy_components(results, path):
    rows = []
    for res in results:
        for _, row in res["layer_long"].iterrows():
            if row["Kontrolcü"] in COMPONENTS:
                rows.append(dict(Veri=_label(res), Kontrolcü=row["Kontrolcü"],
                                 Potansiyel=row["Yakalanan potansiyel (%)"], Ihlal=row["İhlal düşüşü (%)"]))
    d = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(18, 5.6))
    for ax, col, title in [(axes[0], "Potansiyel", "(a) Yakalanan tasarruf potansiyeli (%) — LP optimumu = 100"),
                           (axes[1], "Ihlal", "(b) Fiziksel ihlal düşüşü (%) — bataryasıza göre")]:
        sns.barplot(data=d, x="Veri", y=col, hue="Kontrolcü", hue_order=COMPONENTS, palette=PALETTE, errorbar="sd",
                    capsize=0.06, ax=ax, saturation=1, edgecolor="white", linewidth=2)
        ax.axhline(0, color="#555555", lw=0.8)
        ax.set(title=title + " · tohum ort. ± ss", xlabel="", ylabel="%")
        ax.legend(frameon=False, fontsize=8, title="Katman kombinasyonu")
        _bar_labels(ax)
    _finish(fig, path)


def plot_two_way(results, path):
    """İki yönlü kıyas: her Katman-1 bileşeni için x = tahmin hatası, y = SAC'a bağlandığında enerji potansiyeli."""
    fig, axes = plt.subplots(1, len(results), figsize=(6.2 * len(results), 5.4), squeeze=False)
    for ax, res in zip(axes[0], results):
        fc = pd.DataFrame({m: pd.DataFrame(res["fold_metrics"][m])["MAPE_filtreli"] for m in FORECASTERS})
        en = res["fc_table"].set_index("Kontrolcü")
        for m in FORECASTERS:
            x, xe = fc[m].mean(), fc[m].std(ddof=0)
            y, ye = en.loc[m, "Yakalanan potansiyel (%)"], en.loc[m, "Yakalanan potansiyel (%) std"]
            ax.errorbar(x, y, xerr=xe, yerr=ye, fmt="o", ms=9 if m == FULL else 8, color=PALETTE[m],
                        ecolor=PALETTE[m], elinewidth=1.2, capsize=3, mec="white", mew=2, label=FC_LABEL[m])
            ax.annotate(FC_LABEL[m], (x, y), xytext=(6, 6), textcoords="offset points", fontsize=8,
                        color="#333333", fontweight="bold" if m == FULL else "normal")
        ax.set(title=f"{res['name']}", xlabel="Filtrelenmiş MAPE (%) ← daha iyi",
               ylabel="Yakalanan potansiyel (%) → daha iyi")
    axes[0, 0].legend(frameon=False, fontsize=8, title="Katman-1 bileşeni + SAC")
    fig.suptitle("İki yönlü kıyas: tahmin başarısı × enerji tasarrufu (5 kat / tohum ort. ± ss)", y=1.02)
    _finish(fig, path)


def plot_pinn_violations(results, path):
    fig, axes = plt.subplots(len(results), 2, figsize=(16, 4.2 * len(results)), squeeze=False,
                             gridspec_kw={"width_ratios": [2.2, 1]})
    for r, res in enumerate(results):
        pre, post = res["traces"][K12]["violation"], res["traces"][K123]["violation"]
        w = 168
        roll = np.convolve(pre, np.ones(w), mode="valid") if len(pre) > w else pre
        sl = slice(int(np.argmax(roll)) if len(roll) else 0, None)
        sl = slice(sl.start, sl.start + w)
        ts = res["eval_index"][sl]
        ax = axes[r, 0]
        ax.fill_between(ts, pre[sl], step="mid", color=PALETTE[K12], alpha=0.45, label=K12)
        ax.fill_between(ts, post[sl], step="mid", color=PALETTE[K123], alpha=0.85, label=K123)
        ax.set(title=f"{res['name']} — en yoğun ihlal haftası (1. tohum)",
               ylabel=f"Hat/gerilim ihlali ({res['spec']['unit']})")
        if pre.sum() == 0 and post.sum() == 0:
            ax.text(0.5, 0.5, "Test döneminde ihlal oluşmadı", transform=ax.transAxes, ha="center", color="#555555")
        ax.legend(frameon=False, loc="upper right")
        ax.tick_params(axis="x", rotation=20)

        ax = axes[r, 1]
        op = res["layer_table"].set_index("Kontrolcü")
        names = list(op.index)
        vals, errs = op["Toplam ihlal"].values, op["Toplam ihlal std"].fillna(0).values
        ax.barh(names, vals, xerr=errs, color=[PALETTE[n] for n in names], height=0.6,
                error_kw=dict(ecolor="#333333", capsize=3))
        for i, v in enumerate(vals):
            ax.text(v + errs[i], i, f" {v:,.1f}", va="center", fontsize=8, color="#333333")
        ax.invert_yaxis()
        ax.set(title=f"Toplam ihlal ({res['spec']['energy']}), tohum ort. ± ss", xlabel="")
        ax.margins(x=0.25)
    _finish(fig, path)


# 10) Markdown raporu
def _fmt(x, d=3):
    return "—" if pd.isna(x) else f"{x:,.{d}f}"


def _pm(row, col, d):
    s = row.get(col + " std", np.nan)
    return _fmt(row[col], d) + ("" if pd.isna(s) or s == 0 else f" ± {_fmt(s, d)}")


def _ms(series, d):
    return f"{_fmt(series.mean(), d)} ± {_fmt(series.std(ddof=0), d)}"


def markdown_report(results):
    out = ["# Önerilen hibrit: CNN-BiLSTM + SAC + PINN — v3", "",
           f"_Üretim zamanı: {time.strftime('%Y-%m-%d %H:%M')} · QUICK={QUICK} · RL tohumları="
           f"{CFG['seed']}…{CFG['seed'] + CFG['n_seeds'] - 1} (n={CFG['n_seeds']}) · TimeSeriesSplit k={CFG['n_splits']} · "
           f"pencere={CFG['seq_len']} sa · ufuk={CFG['horizon']} sa_", "",
           "Bu raporda dış rakip yoktur: her satır önerilen hibritin bir bileşeni veya kısmi kombinasyonudur. "
           "Bataryasız ve Teorik Optimum yalnızca ölçüm referansıdır.", "",
           "## Modeller, roller ve kısaltmalar", "",
           "| Terim | Açıklama |", "|---|---|"]
    out += [f"| **{term}** | {desc} |" for term, desc in GLOSSARY]
    out += ["", "Tablolardaki **Rol** sütunu: *Önerilen hibrit* = tezde önerilen model · *Hibrit bileşeni* "
            "= önerilen modelin bir parçası, katkısını ölçmek için tek başına çalıştırıldı · *Referans (rakip değil)* "
            "= yalnızca ölçek noktası.", ""]
    if any(r["synthetic"] for r in results):
        out += ["> **UYARI:** En az bir veri seti SENTETİK üreteçten gelmektedir; bu satırlar teze gerçek deney "
                "bulgusu olarak yazılmamalıdır.", ""]

    out += ["## Tablo 1 — Veri setleri", "",
            "| Ölçek | Kaynak | Saat sayısı | Ort. yük | Fiyat |", "|---|---|---:|---:|---|"]
    for r in results:
        out.append(f"| {r['name']}{' ⚠SENTETİK' if r['synthetic'] else ''} | {r['source']} | {r['n_hours']:,} | "
                   f"{_fmt(r['mean_load'], 2)} {r['spec']['unit']} | {r['spec']['currency']}/{r['spec']['energy']} |")

    out += ["", "## Tablo 2 — Model performansı: Katman-1 bileşenleri (24 sa ufuk, 5 kat ort. ± std)", "",
            "| Veri seti | Model | Rol | RMSE | MAE | R² | Ham MAPE (%) | Filtrelenmiş MAPE (%) | Parametre | "
            "Eğitim süresi (s/kat) | Epoch |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        for m in FORECASTERS:
            f = pd.DataFrame(r["fold_metrics"][m])
            b = "**" if m == FULL else ""
            out.append(f"| {r['name']} | {b}{FC_LABEL[m]}{b} | {role(m)} | {_ms(f['RMSE'], 3)} | {_ms(f['MAE'], 3)} | "
                       f"{_ms(f['R2'], 3)} | "
                       f"{_ms(f['MAPE_ham'], 2)} | {_ms(f['MAPE_filtreli'], 2)} | {int(f['parametre'].iloc[0]):,} | "
                       f"{_fmt(f['egitim_s'].mean(), 1)} | {_fmt(f['epoch'].mean(), 0)} |")

    out += ["", f"## Tablo 3 — Enerji: katman kombinasyonları (kat 5 test dönemi, {CFG['n_seeds']} tohum ort. ± std)", "",
            "| Veri seti | Kombinasyon | Rol | Ort. saatlik maliyet | Tasarruf (%) | Yakalanan potansiyel (%) | "
            "Toplam ihlal | İhlalli saat (%) | İhlal düşüşü (%) | Min. V (pu) |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        cur, en = r["spec"]["currency"], r["spec"]["energy"]
        for _, row in r["layer_table"].iterrows():
            b = "**" if row["Kontrolcü"] == K123 else ""
            out.append(f"| {r['name']} | {b}{row['Kontrolcü']}{b} | {role(row['Kontrolcü'])} | "
                       f"{_fmt(row['Ort. saatlik maliyet'], 2)} {cur}/sa | "
                       f"{_pm(row, 'Tasarruf (%)', 2)} | {_pm(row, 'Yakalanan potansiyel (%)', 1)} | "
                       f"{_pm(row, 'Toplam ihlal', 2)} {en} | {_fmt(row['İhlalli saat (%)'], 2)} | "
                       f"{_pm(row, 'İhlal düşüşü (%)', 2)} | {_fmt(row['Min. gerilim (pu)'], 4)} |")

    out += ["", "## Tablo 4 — İki yönlü kıyas: her Katman-1 bileşeni, tahmin başarısı × SAC'a bağlandığında enerji", "",
            "| Veri seti | Tahmin modeli (+ aynı SAC) | Rol | Filtrelenmiş MAPE (%) | R² | Tasarruf (%) | "
            "Yakalanan potansiyel (%) | İhlal düşüşü (%) |",
            "|---|---|---|---:|---:|---:|---:|---:|"]
    for r in results:
        en = r["fc_table"].set_index("Kontrolcü")
        for m in FORECASTERS:
            f, row = pd.DataFrame(r["fold_metrics"][m]), en.loc[m]
            b = "**" if m == FULL else ""
            out.append(f"| {r['name']} | {b}{FC_LABEL[m]}{b} | {role(m)} | {_ms(f['MAPE_filtreli'], 2)} | "
                       f"{_ms(f['R2'], 3)} | "
                       f"{_pm(row, 'Tasarruf (%)', 2)} | {_pm(row, 'Yakalanan potansiyel (%)', 1)} | "
                       f"{_pm(row, 'İhlal düşüşü (%)', 2)} |")

    out += ["", "## Tablo 5 — Katmanların marjinal katkısı (aynı tohumda eşleştirilmiş fark, ort. ± std; pozitif = katman "
            "yararlı)", "",
            "| Veri seti | Katkı | Δ Yakalanan potansiyel (puan) | Δ İhlal düşüşü (puan) | Δ Tasarruf (puan) |",
            "|---|---|---:|---:|---:|"]
    for r in results:
        c = r["contrib"]
        for label in c["Katkı"].unique():
            cells = []
            for metric, d in [("Yakalanan potansiyel (%)", 1), ("İhlal düşüşü (%)", 2), ("Tasarruf (%)", 2)]:
                row = c[(c["Katkı"] == label) & (c["Metrik"] == metric)].iloc[0]
                cells.append(f"{row['Ort']:+.{d}f} ± {_fmt(row['Std'], d)}" if not pd.isna(row["Ort"]) else "—")
            out.append(f"| {r['name']} | {label} | " + " | ".join(cells) + " |")

    out += ["", "### Deney kurulumu notları", "",
            "- Katman-1 bileşenleri aynı girdi (24 sa yük + takvim), aynı çıkış başlığı (Dense 64 → 24), aynı optimizasyon, "
            "aynı erken durdurma ve aynı tohumla eğitilir. 'Attention' = hibritin zamansal dikkat bloğunun girdinin "
            "doğrusal izdüşümüne uygulanmış hâli.",
            f"- Tahminler {CFG['horizon']} saatlik doğrudan çok adımlıdır; TimeSeriesSplit gap={CFG['horizon'] - 1}; "
            "Z-score parametreleri her katta yalnızca eğitim bölümünden öğrenilir.",
            "- Filtrelenmiş MAPE yalnızca y > %.0f%% × ort. yük noktalarında hesaplanır." % (CFG["near_zero_alpha"] * 100),
            "- K1 = tahmin katmanı, K2 = SAC, K3 = PINN güvenlik katmanı. Tahminsiz kombinasyonlar son 24 sa ölçümü, "
            "tahminli olanlar 24 sa tahmini gözlemler; hepsi 24 sa gün öncesi fiyatı görür ve aynı ödülle eğitilir.",
            "- K3 tek başına: planlama yok (eylem 0), PINN yalnızca ihlal öngördüğünde bataryayı kullanır. "
            "PINN'li kombinasyonlarda PINN SAC'ın eğitim döngüsündedir.",
            "- SAC'lar kat 1–4 test bölgelerinde (Katman-1 için örneklem dışı tahminler) eğitilir, kat 5'te değerlendirilir.",
            f"- SAC ödülü bataryasız duruma göre farktır; ihlal cezası = {CFG['viol_penalty']} × ort. fiyat / birim enerji. "
            "Tablolardaki maliyet cezasız faturadır.",
            "- Teorik Optimum: gelecekteki yük ve fiyatı tam bilen LP (HiGHS). Yakalanan potansiyel = tasarruf / LP tasarrufu.",
            f"- Fiziksel model: tek dallı radyal besleyici, LinDistFlow; hat limiti eğitim yükünün "
            f"%{CFG['line_limit_q'] * 100:.0f} yüzdeliği (bu limitte V={V_MIN} pu).",
            f"- Batarya: E = {CFG['batt_energy_ratio']}× ort. yük, P = {CFG['batt_power_ratio']}× ort. yük "
            f"(Makro-Şebeke: E = 0.08×, P = 0.02×), η={BATT_ETA}, SOC∈[{SOC_MIN},{SOC_MAX}].",
            f"- Öğrenen kontrolcüler {CFG['n_seeds']} tohumla eğitilir; Katman-1 tahminleri tohumlar arasında ortaktır. "
            "Zaman serisi ve kayıp grafikleri 1. tohuma aittir."]
    return "\n".join(out)


# 11) Ana akış
def main():
    os.makedirs(CFG["out_dir"], exist_ok=True)
    os.makedirs(CFG["data_dir"], exist_ok=True)
    set_seed(CFG["seed"])
    _style()
    rng = np.random.default_rng(CFG["seed"])
    gpu = torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "yok"
    print(f"PyTorch {torch.__version__} · cihaz={DEVICE} ({gpu}) · QUICK={QUICK} · tohum={CFG['n_seeds']} · "
          f"SAC süreçleri={CFG['sac_workers']}")
    print("Önerilen hibrit = K1 CNN-BiLSTM (tahmin) + K2 SAC (pekiştirmeli öğrenme ile batarya kararı) + "
          "K3 PINN (fiziksel güvenlik). Diğer satırlar bu modelin bileşenleri veya referanslardır.")

    results = []
    for spec in DATASETS:
        t0 = time.time()
        print(f"\n=== {spec['name']} ===")
        df, source, synthetic, spec = prepare_dataset(spec, rng)
        print(f"    kaynak: {source} · {len(df):,} saat · {df.index[0]} → {df.index[-1]}")

        print(f"  [A] Katman-1 bileşenleri: {', '.join(FORECASTERS)}")
        fold_metrics, histories, oof, folds = run_forecasting_isolated(df)
        with open(os.path.join(CFG["out_dir"], f"forecast_{spec['key']}.json"), "w", encoding="utf-8") as f:
            json.dump(dict(name=spec["name"], source=source, synthetic=synthetic, fold_metrics=fold_metrics),
                      f, ensure_ascii=False, indent=2)

        print("  [B] Katman kombinasyonları + [A×B] Katman-1 bileşeni × SAC")
        layer_table, layer_long, fc_table, fc_long, traces, eval_idx, rl_logs, params = \
            run_operation(df, oof, folds, spec)
        contrib = contributions(layer_long)
        print(layer_table[["Kontrolcü", "Tasarruf (%)", "Yakalanan potansiyel (%)", "Toplam ihlal",
                           "İhlal düşüşü (%)"]].round(2).to_string(index=False))
        print(fc_table[["Kontrolcü", "Tasarruf (%)", "Yakalanan potansiyel (%)", "İhlal düşüşü (%)"]]
              .round(2).to_string(index=False))
        for name, frame in [("layers", layer_long), ("forecaster_sac", fc_long), ("contributions", contrib)]:
            frame.to_csv(os.path.join(CFG["out_dir"], f"{name}_{spec['key']}.csv"), index=False)

        results.append(dict(name=spec["name"], spec=spec, source=source, synthetic=synthetic, n_hours=len(df),
                            mean_load=float(df["load"].mean()), fold_metrics=fold_metrics, histories=histories,
                            layer_table=layer_table, layer_long=layer_long, fc_table=fc_table, contrib=contrib,
                            traces=traces, eval_index=df.index[eval_idx], rl_logs=rl_logs, params=params))
        print(f"    süre: {time.time() - t0:.0f}s")

    out = CFG["out_dir"]
    figures = {"loss_curves.png": plot_loss_curves, "forecast_components.png": plot_forecast_components,
               "energy_components.png": plot_energy_components, "two_way_comparison.png": plot_two_way,
               "pinn_violations.png": plot_pinn_violations}
    report = markdown_report(results)
    with open(os.path.join(out, "benchmark_v3_tables.md"), "w", encoding="utf-8") as f:
        f.write(report)
    with open(os.path.join(out, "benchmark_v3_raw.json"), "w", encoding="utf-8") as f:
        json.dump([dict(name=r["name"], source=r["source"], synthetic=r["synthetic"], params=r["params"],
                        fold_metrics=r["fold_metrics"], layers=r["layer_table"].to_dict("records"),
                        forecaster_sac=r["fc_table"].to_dict("records"), contributions=r["contrib"].to_dict("records"))
                   for r in results], f, ensure_ascii=False, indent=2)

    print("\n" + report, flush=True)
    done = []                      # tablolar yazıldıktan sonra çizilir; grafik hatası koşuyu düşürmez
    for name, fn in figures.items():
        try:
            fn(results, os.path.join(out, name))
            done.append(name)
        except Exception as exc:
            print(f"    [UYARI] {name} çizilemedi ({type(exc).__name__}: {exc})")
    figures = {name: figures[name] for name in done}
    print(f"\nÇıktılar: {os.path.abspath(out)}/  ({', '.join(figures)}, benchmark_v3_tables.md, benchmark_v3_raw.json)")
    if "google.colab" in sys.modules:
        from IPython.display import Image, display
        for name in figures:
            display(Image(os.path.join(out, name)))


if __name__ == "__main__":
    main()
