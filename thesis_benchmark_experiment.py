"""thesis_benchmark_experiment.py: ağaç tabanlı referanslarla tahmin karşılaştırmasının ilk uygulaması.

Üç katmanlı hibrit model:
  Katman 1: CNN-BiLSTM ve zamansal dikkat (TensorFlow/Keras), 24 saatlik doğrudan çok adımlı yük tahmini
  Katman 2: Soft Actor-Critic (PyTorch), 24 saatlik tahmin ve 24 saatlik fiyatla batarya kararı
  Katman 3: PINN güvenlik katmanı (PyTorch), KCL güç dengesi ve LinDistFlow (KVL) gerilim düşümü kısıtlarıyla
            eylem düzeltme
Karşılaştırılanlar: Random Forest ve XGBoost (aynı 24 saatlik ufuk); bataryasız işletme, kural tabanlı yönetim,
tahminsiz SAC (aynı ham bilgi: son 24 saatin ölçümü, aynı ödül) ve teorik optimum (geleceği tam bilen doğrusal
program, üst sınır).

Veri kümeleri (öncelik sırası: yerel CSV, açık kaynak, sentetik üreteç):
  micro: UCI Individual Household Electric Power Consumption (kW)
  macro: Open Power System Data (ENTSO-E yükü ve gün öncesi fiyat, DE-LU, MW, EUR/MWh); yerel dosya thesis_data/macro.csv
  meso:  Building Data Genome Project 2 ofis binası elektrik sayacı (kWh); yerel dosya thesis_data/meso.csv
Fiyat sütunu olmayan veri kümelerinde örnek üç zamanlı tarife (gündüz/puant/gece, TOU_TARIFF) kullanılır.
İndirme başarısız olursa sentetik veri kullanılır ve tablolarda "SENTETİK" olarak işaretlenir.

Çalıştırma: python thesis_benchmark_experiment.py
Seçenekler: QUICK=1 (kısa deneme), FORCE_SYNTHETIC=1 (yalnız sentetik veri)
"""

import os
import sys
import json
import time
import faulthandler
import random
import zipfile
import warnings
import urllib.request

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
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

# TensorFlow yalnızca tahmin alt sürecinde yüklenir (bkz. _load_tf). TF ve PyTorch aynı süreçte
# GPU kullanınca (farklı CUDA sürümleri) PyTorch'un ilk CUDA çağrısı süreci sessizce öldürüyor.
tf = keras = layers = None

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

warnings.filterwarnings("ignore")
faulthandler.enable()   # segfault gibi sessiz çökmelerde nerede öldüğünü yazdır

# Yapılandırma
QUICK = os.environ.get("QUICK", "0") == "1"
FORCE_SYNTHETIC = os.environ.get("FORCE_SYNTHETIC", "0") == "1"

CFG = dict(
    seed=42,
    out_dir="thesis_benchmark_outputs",
    data_dir="thesis_data",
    max_hours=24 * 365 * 2,          # her setten son 2 yıl (saatlik)
    seq_len=24,                      # 24 saatlik geçmiş pencere
    horizon=24,                      # 24 saatlik tahmin/planlama ufku
    n_splits=5,                      # TimeSeriesSplit kat sayısı
    dl_epochs=150, dl_batch=128, dl_patience=15,
    rf_trees=200, xgb_rounds=400,
    sac_steps=40000, sac_warmup=4000, sac_batch=256, sac_episode_len=168,
    n_seeds=int(os.environ.get("SEEDS", "3")),   # RL tohum tekrarı: sonuçlar ortalama ± std
    pinn_epochs=400,
    near_zero_alpha=0.10,            # MAPE filtresi: y <= alpha * ortalama(y_train) dışlanır
    batt_energy_ratio=2.0,           # batarya kapasitesi = 2 x ortalama saatlik yük (kWh)
    batt_power_ratio=0.5,            # batarya gücü = 0.5 x ortalama yük (kW)
    line_limit_q=0.85,               # besleyici limiti = eğitim yükünün %85 yüzdeliği (kısıtın bağlayıcı olduğu senaryo)
    viol_penalty=5.0,                # ihlal cezası = 5 x ortalama fiyat / birim enerji (tüm öğrenen kontrolcülerde aynı)
    pinn_margin_pu=0.03,             # PINN'in tahmin hatasına karşı bıraktığı pay (pu)
)
if QUICK:
    CFG.update(max_hours=24 * 120, dl_epochs=3, rf_trees=40, xgb_rounds=80,
               sac_steps=1500, sac_warmup=300, pinn_epochs=60, n_seeds=1)

# Örnek üç zamanlı tarife (TL/kWh)
TOU_TARIFF = {"gece (22-06)": 1.55, "gündüz (06-17)": 2.60, "puant (17-22)": 4.10}

# LinDistFlow tek hatlı besleyici parametreleri (pu, baz güç = hat limiti P_lim)
V0, V_MIN, V_MAX = 1.00, 0.95, 1.05
R_PU, X_PU, TAN_PHI = 0.03, 0.04, 0.50   # R + X*tanφ = 0.05 -> P = P_lim'de V = 0.95 pu
BATT_ETA, SOC_MIN, SOC_MAX, DEG_COST_FRAC = 0.95, 0.10, 0.90, 0.02

# Renk körü dostu sabit kategorik sıra (Okabe-Ito); her model her grafikte aynı rengi taşır.
PALETTE = {
    "CNN-BiLSTM-Att (Önerilen)": "#0072B2", "Random Forest": "#E69F00", "XGBoost": "#009E73",
    "Bataryasız": "#7F7F7F", "Kural Tabanlı": "#E69F00", "Saf SAC": "#009E73",
    "Hibrit (PINN'siz, ablasyon)": "#CC79A7", "Hibrit 3-Katman (Önerilen)": "#0072B2",
    "Teorik Optimum (tam öngörü)": "#D55E00",
}
CONTROLLER_ORDER = ["Bataryasız", "Kural Tabanlı", "Saf SAC", "Hibrit (PINN'siz, ablasyon)",
                    "Hibrit 3-Katman (Önerilen)", "Teorik Optimum (tam öngörü)"]


def _load_tf():
    global tf, keras, layers
    import tensorflow as _tf
    from tensorflow import keras as _keras
    from tensorflow.keras import layers as _layers
    tf, keras, layers = _tf, _keras, _layers
    tf.get_logger().setLevel("ERROR")   # her katta yeni model kurulduğu için oluşan zararsız retracing uyarılarını gizle
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if tf is not None:      # yalnızca tahmin alt sürecinde yüklü
        tf.random.set_seed(seed)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 1) Veri yükleme
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
    df = pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    return df


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
    raw = pd.read_csv(path, usecols=cols, parse_dates=["utc_timestamp"])
    raw = raw.set_index("utc_timestamp").dropna()
    raw.index = raw.index.tz_localize(None)
    df = raw.rename(columns={cols[1]: "load", cols[2]: "price"})
    return df, "OPSD / ENTSO-E DE-LU (gerçek)"


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


# Sentetik üreteçler (yalnızca indirme başarısız olursa veya FORCE_SYNTHETIC=1)
def _hours(n):
    return pd.date_range("2022-01-01", periods=n, freq="h")


def synth_micro(n, rng):
    idx = _hours(n)
    h, doy = idx.hour.values, idx.dayofyear.values
    season = 1 + 0.35 * np.cos(2 * np.pi * (doy - 15) / 365)
    profile = (0.25 + 0.9 * np.exp(-((h - 8) ** 2) / 3) + 1.6 * np.exp(-((h - 20) ** 2) / 5))
    night = (h >= 1) & (h <= 5)
    base = profile * season * rng.lognormal(0, 0.25, n)
    base[night] *= rng.uniform(0.02, 0.4, night.sum())          # sıfıra yakın gece tüketimi
    spikes = rng.binomial(1, 0.04, n) * rng.uniform(1, 3, n)    # fırın/çamaşır makinesi
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
    df = df.asfreq("h").ffill()                       # uzun boşluklar: ileri doldurma
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


# 2) Ön işleme: pencereleme, Z-score (kat içi), sıfıra yakın yük filtresi
def calendar_features(index):
    h, dow, m = index.hour.values, index.dayofweek.values, index.month.values
    return np.stack([np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
                     np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
                     np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12)], axis=1).astype(np.float32)


def build_windows(load, index, L, H):
    """t başlangıç saati için girdi: load[t-L:t]; hedef: load[t:t+H] (doğrudan çok adımlı). Ham pencereler döner."""
    n = len(load) - L - H + 1
    win = np.lib.stride_tricks.sliding_window_view(load, L)[:n].astype(np.float32)
    target = np.lib.stride_tricks.sliding_window_view(load[L:], H)[:n].astype(np.float32)
    cal = calendar_features(index)
    cal_seq = np.lib.stride_tricks.sliding_window_view(cal, (L, cal.shape[1]))[:n, 0]
    cal_target = cal[L:L + n]
    return win, target, cal_seq, cal_target


def near_zero_threshold(y_train):
    """Gece ~0 tüketimde |y-ŷ|/|y| patlar. Eşik altı noktalar MAPE'den dışlanır
    (RMSE/MAE/R² tüm noktalarda hesaplanır; eğitim verisinden nokta silinmez -> dizi bütünlüğü korunur)."""
    return max(CFG["near_zero_alpha"] * float(np.mean(y_train)), 1e-6)


def forecast_metrics(y, p, tau):
    y, p = np.ravel(y), np.ravel(p)   # 24 sa ufkun tüm adımları birlikte
    raw_mape = float(np.mean(np.abs(y - p) / np.maximum(np.abs(y), 1e-6)) * 100)
    mask = y > tau
    filt_mape = float(np.mean(np.abs(y[mask] - p[mask]) / y[mask]) * 100)
    return dict(RMSE=float(np.sqrt(mean_squared_error(y, p))), MAE=float(mean_absolute_error(y, p)),
                R2=float(r2_score(y, p)), MAPE_ham=raw_mape, MAPE_filtreli=filt_mape,
                dislanan_pct=float((~mask).mean() * 100))


# 3) Katman 1: CNN-BiLSTM + Attention (Keras)
def build_cnn_bilstm_att(L, n_feat, H):
    inp = keras.Input(shape=(L, n_feat))
    x = layers.Conv1D(64, 3, padding="causal", activation="relu")(inp)
    x = layers.Conv1D(64, 3, padding="causal", activation="relu")(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    score = layers.Dense(1)(layers.Dense(64, activation="tanh")(x))       # Bahdanau tipi skor
    weights = layers.Softmax(axis=1)(score)
    context = layers.Flatten()(layers.Dot(axes=1)([weights, x]))          # Σ_t α_t h_t
    z = layers.Dropout(0.2)(layers.Dense(64, activation="relu")(context))
    out = layers.Dense(H)(z)                                               # 24 saatin tamamı tek seferde
    model = keras.Model(inp, out, name="CNN_BiLSTM_Attention")
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return model


def run_forecasting(df):
    """5 katlı TimeSeriesSplit. Her katta Z-score yalnızca eğitim bölümünden öğrenilir."""
    L, H = CFG["seq_len"], CFG["horizon"]
    load = df["load"].values.astype(np.float64)
    win, target, cal_seq, cal_target = build_windows(load, df.index, L, H)
    n = len(target)
    tss = TimeSeriesSplit(n_splits=CFG["n_splits"], gap=H - 1)   # eğitim hedefleri test dönemine taşmasın
    fold_metrics = {m: [] for m in ["CNN-BiLSTM-Att (Önerilen)", "Random Forest", "XGBoost"]}
    histories, folds = [], []
    oof_pred = np.full((len(load), H), np.nan)            # oof_pred[t] = t saatinde yapılan t..t+H-1 tahmini

    for k, (tr, te) in enumerate(tss.split(np.arange(n))):
        t0 = time.time()
        scaler = StandardScaler().fit(load[: tr[-1] + L + H].reshape(-1, 1))
        mu, sd = scaler.mean_[0], scaler.scale_[0]
        Wz, yz = (win - mu) / sd, (target - mu) / sd
        X_seq = np.concatenate([Wz[..., None], cal_seq], axis=2).astype(np.float32)
        X_tab = np.concatenate([Wz, cal_target], axis=1)

        n_val = max(int(0.1 * len(tr)), L)
        tr_fit, va = tr[:-(n_val + H - 1)], tr[-n_val:]
        tau = near_zero_threshold(target[tr])
        y_true = target[te]

        # CNN-BiLSTM ve zamansal dikkat
        set_seed(CFG["seed"] + k)
        model = build_cnn_bilstm_att(L, X_seq.shape[2], H)
        hist = model.fit(X_seq[tr_fit], yz[tr_fit], validation_data=(X_seq[va], yz[va]),
                         epochs=CFG["dl_epochs"], batch_size=CFG["dl_batch"], verbose=0,
                         callbacks=[keras.callbacks.EarlyStopping(patience=CFG["dl_patience"],
                                                                  restore_best_weights=True),
                                    keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5)])
        histories.append(hist.history)
        p_dl = model.predict(X_seq[te], verbose=0, batch_size=1024) * sd + mu
        p_dl = np.clip(p_dl, 0, None)
        fold_metrics["CNN-BiLSTM-Att (Önerilen)"].append(forecast_metrics(y_true, p_dl, tau))
        oof_pred[te + L] = p_dl

        # Random Forest
        rf = RandomForestRegressor(n_estimators=CFG["rf_trees"], min_samples_leaf=5, max_features=0.5,
                                   n_jobs=-1, random_state=CFG["seed"])
        rf.fit(X_tab[tr], yz[tr])
        p_rf = np.clip(rf.predict(X_tab[te]) * sd + mu, 0, None)
        fold_metrics["Random Forest"].append(forecast_metrics(y_true, p_rf, tau))

        # XGBoost
        xg = xgb.XGBRegressor(n_estimators=CFG["xgb_rounds"], learning_rate=0.05, max_depth=6,
                              subsample=0.8, colsample_bytree=0.8, early_stopping_rounds=30,
                              random_state=CFG["seed"], n_jobs=-1, tree_method="hist")
        xg.fit(X_tab[tr_fit], yz[tr_fit], eval_set=[(X_tab[va], yz[va])], verbose=False)
        p_xg = np.clip(xg.predict(X_tab[te]) * sd + mu, 0, None)
        fold_metrics["XGBoost"].append(forecast_metrics(y_true, p_xg, tau))

        folds.append(te + L)
        print(f"    kat {k + 1}/{CFG['n_splits']}  "
              f"MAPE_f DL={fold_metrics['CNN-BiLSTM-Att (Önerilen)'][-1]['MAPE_filtreli']:.2f}  "
              f"RF={fold_metrics['Random Forest'][-1]['MAPE_filtreli']:.2f}  "
              f"XGB={fold_metrics['XGBoost'][-1]['MAPE_filtreli']:.2f}  "
              f"(DL {len(hist.history['loss'])} epoch, {time.time() - t0:.0f}s)")
        keras.backend.clear_session()

    return fold_metrics, histories, oof_pred, folds


def _forecast_worker(df):
    """Ayrı (spawn) süreçte çalışır: TF burada GPU'yu kullanır, ana süreçteki PyTorch CUDA'sına dokunmaz."""
    _load_tf()
    print(f"    TensorFlow {tf.__version__} · GPU: {len(tf.config.list_physical_devices('GPU'))}", flush=True)
    fold_metrics, histories, oof_pred, folds = run_forecasting(df)
    histories = [{k: [float(v) for v in vals] for k, vals in h.items()} for h in histories]
    return fold_metrics, histories, oof_pred, folds


def run_forecasting_isolated(df):
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as ex:
        return ex.submit(_forecast_worker, df).result()


# 4) Fiziksel model: KCL güç dengesi + LinDistFlow (KVL) gerilim düşümü
def feeder_voltage_pu(p_grid_pu):
    """Tek dallı radyal besleyici, LinDistFlow: V ≈ V0 - (R·P + X·Q)/V0, Q = P·tanφ (pu)."""
    return V0 - (R_PU * p_grid_pu + X_PU * p_grid_pu * TAN_PHI) / V0


class BatteryFeederEnv:
    """Saatlik batarya dispeçi. Eylem a∈[-1,1] -> p_batt = a·P_b (+ şarj, - deşarj).
    KCL: P_grid = P_load + P_batt.  Fatura = fiyat·max(P_grid,0) + degradasyon.
    Ödül = -(fatura + ihlal cezası). Gözlem: 24 sa yük bilgisi (load_info), 24 sa fiyat, SOC, takvim.
    load_now: kontrolcünün içinde bulunulan saatin yükü için kestirimi (kural tabanlı ve PINN kullanır)."""

    def __init__(self, load, load_info, load_now, price, index, E, Pb, P_lim, price_scale):
        self.load, self.load_info, self.load_now, self.price = load, load_info, load_now, price
        H = load_info.shape[1]
        padded = np.concatenate([price, np.full(H, price[-1])])
        self.price_h = np.lib.stride_tricks.sliding_window_view(padded, H)[:len(price)]  # gün öncesi fiyatlar bilinir
        self.cal = calendar_features(index)[:, :4]
        self.E, self.Pb, self.P_lim, self.price_scale = E, Pb, P_lim, price_scale
        self.deg = DEG_COST_FRAC * price_scale
        self.viol_pen = CFG["viol_penalty"] * price_scale
        self.cost_scale = price_scale * Pb      # ödül batarya etkisi ölçeğinde (~O(1))

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
        # Ödül bataryasız duruma göre farktır: faturanın bataryanın etkileyemediği kısmı (yükün kendi maliyeti ve
        # kendi ihlali) eylemden bağımsız olduğu için çıkarılır -> optimum politika değişmez, öğrenme sinyali güçlenir.
        base_cost = self.price[t] * max(self.load[t], 0.0)
        base_viol = max(0.0, abs(self.load[t]) - self.P_lim)
        reward = -((cost - base_cost) + self.viol_pen * (violation - base_viol)) / self.cost_scale
        return soc_next, reward, dict(cost=cost, violation=violation, v=v, p=p, p_grid=p_grid)


def perfect_foresight_lp(env, idx):
    """Gelecekteki yük ve fiyatı tam bilen doğrusal program (HiGHS). SAC ödülüyle aynı amaç:
    fatura + degradasyon + ihlal cezası. Erişilebilir teorik en iyi -> 'yakalanan potansiyel' referansı."""
    from scipy import sparse
    from scipy.optimize import linprog
    T, L, pr = len(idx), env.load[idx], env.price[idx]
    # değişkenler: şarj c(T) | deşarj d(T) | şebeke g(T) | ihlal v(T) | SOC s(T+1)
    cost = np.concatenate([np.full(T, env.deg), np.full(T, env.deg), pr, np.full(T, env.viol_pen), np.zeros(T + 1)])
    I, Z, Zs = sparse.identity(T, format="csr"), sparse.csr_matrix((T, T)), sparse.csr_matrix((T, T + 1))
    grid_rows = sparse.hstack([I, -I, -I, Z, Zs]).tocsr()            # L + c - d - g
    # Fiyat >= 0: g >= L + c - d (minimizasyon g'yi pozitif kısma oturtur).
    # Fiyat < 0 (ör. DE-LU 2020): eşitsizlik "sonsuz çek" ile sınırsız olur; g = L + c - d eşitliği kullanılır
    # (negatif fiyatta dışarı vermenin faydası yoktur, g >= 0 sınırı korunur).
    neg = pr < 0
    A_ub = sparse.vstack([grid_rows[~neg],
                          sparse.hstack([I, -I, Z, -I, Zs])])      # L + c - d - P_lim <= v
    b_ub = np.concatenate([-L[~neg], env.P_lim - L])
    col0 = sparse.csr_matrix((T, 1))
    soc_diff = (sparse.hstack([col0, I]) - sparse.hstack([I, col0])) * env.E   # E·(s_{t+1} - s_t)
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


# 5) Katman 2: Soft Actor-Critic (PyTorch)
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

    @torch.no_grad()
    def policy(self, s):
        return self.actor.act(torch.as_tensor(s, device=DEVICE)[None]).item()


def train_sac(env, train_idx, seed, label, safety=None):
    """safety: (t, soc, a) -> uygulanan eylem. Verilirse güvenlik katmanı eğitim döngüsündedir: katman ortamın
    parçası sayılır, ajan eyleminin düzeltileceğini bilerek öğrenir (tamponda ajanın kendi eylemi saklanır)."""
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
    ep_ret, returns, t0 = 0.0, [], time.time()
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
        if (i + 1) % max(cap // 10, 1) == 0:
            print(f"      SAC [{label}] adım {i + 1:,}/{cap:,}  son 10 bölüm ort. ödül "
                  f"{np.mean(returns[-10:]) if returns else float('nan'):.2f}  ({time.time() - t0:.0f}s)", flush=True)
        if i >= CFG["sac_warmup"]:
            idx = rng.integers(0, i + 1, CFG["sac_batch"])
            batch = [torch.as_tensor(buf[k][idx], device=DEVICE) for k in ["s", "a", "r", "s2", "d"]]
            lc, la = agent.update(*batch)
            log["critic"].append(lc); log["actor"].append(la)
    print(f"    SAC [{label}] eğitildi ({cap} adım)")
    return agent, log


# 6) Katman 3: PINN güvenlik katmanı
class SafetyPINN(nn.Module):
    """Girdi: (L̂_pu, a_SAC, SOC, Pb_pu, E_pu). Çıktı: düzeltilmiş eylem a*, P̂_grid, V̂.
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
    a_sac, l_pu, soc, pb_pu, e_pu = x[:, 1:2], x[:, 0:1], x[:, 2:3], x[:, 3:4], x[:, 4:5]
    a, p_hat, v_hat = model(x)
    l_fid = ((a - a_sac) ** 2).mean()
    l_kcl = ((p_hat - (l_pu + a * pb_pu)) ** 2).mean()                               # Kirchhoff akım/güç dengesi
    l_kvl = ((v_hat - (V0 - (R_PU + X_PU * TAN_PHI) * p_hat / V0)) ** 2).mean()        # Kirchhoff gerilim (LinDistFlow)
    k = R_PU + X_PU * TAN_PHI
    v_floor = V_MIN + k * margin
    soc_next = soc + a * pb_pu / e_pu
    # Gerilim ihlali k'ya bölünerek güç (pu) ölçeğine taşınır: 0.005 pu gerilim farkı = %10 hat aşımı.
    # Bölmeden kare ceza ~1e-5 kalıyor ve sadakat terimi karşısında etkisiz oluyordu.
    l_ineq = ((F.relu(v_floor - v_hat) / k) ** 2 + (F.relu(v_hat - V_MAX) / k) ** 2).mean() \
        + (F.relu(soc_next - SOC_MAX) ** 2 + F.relu(SOC_MIN - soc_next) ** 2).mean()
    total = l_fid + 10 * l_kcl + 10 * l_kvl + 1000 * l_ineq
    return total, dict(toplam=total.item(), sadakat=l_fid.item(), KCL=l_kcl.item(),
                       KVL=l_kvl.item(), esitsizlik=l_ineq.item())


def train_pinn(env, train_idx, seed):
    set_seed(seed)
    rng = np.random.default_rng(seed)
    pb_pu, e_pu = env.Pb / env.P_lim, env.E / env.P_lim
    # Politikadan bağımsız güvenlik katmanı: eğitim dönemindeki her saat için rastgele SOC ve rastgele eylem.
    # Böylece katman, onu eğitim döngüsünde kullanacak SAC'tan önce eğitilebilir.
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
            loss, parts = pinn_loss(model, X[perm[b:b + 2048]], CFG["pinn_margin_pu"])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        with torch.no_grad():
            log.append(pinn_loss(model, X, CFG["pinn_margin_pu"])[1])
    print(f"    PINN eğitildi: son kayıp {log[-1]['toplam']:.4g} (eşitsizlik {log[-1]['esitsizlik']:.2e})")
    return model, log


# 7) Operasyon simülasyonu ve kıyas kontrolcüleri
def rollout(env, eval_idx, policy_fn):
    soc, out = 0.5, dict(cost=[], violation=[], v=[], p_grid=[])
    for t in eval_idx:
        a = policy_fn(t, soc)
        soc, _, info = env.step(t, soc, a)
        for k in out:
            out[k].append(info[k])
    return {k: np.array(v) for k, v in out.items()}


def rule_based_policy(env, price_lo, price_hi):
    """Ucuz saatte şarj (hat limitini aşmayacak kadar), pahalı saatte deşarj."""
    def fn(t, soc):
        if env.price[t] <= price_lo:
            headroom = max(env.P_lim - env.load_now[t], 0.0)
            return min(1.0, headroom / env.Pb)
        if env.price[t] >= price_hi:
            return -1.0
        return 0.0
    return fn


def run_operation(df, oof_pred, folds, spec):
    load, price = df["load"].values, df["price"].values
    train_idx = np.concatenate(folds[:-1])            # kat 1-4 test bölgeleri (OOF tahminli)
    eval_idx = folds[-1]                               # kat 5 test bölgesi (görülmemiş)
    ref = load[: train_idx[-1] + 1]
    mean_load = float(ref.mean())
    E = spec.get("batt_energy_ratio", CFG["batt_energy_ratio"]) * mean_load
    Pb = spec.get("batt_power_ratio", CFG["batt_power_ratio"]) * mean_load
    P_lim = float(np.quantile(ref, CFG["line_limit_q"]))
    price_scale = float(np.mean(np.abs(price[train_idx]))) + 1e-9

    H = CFG["horizon"]
    # Saf SAC tahmin katmanı olmadan, tahmincinin gördüğü ham bilgiyi (son 24 sa ölçüm) görür:
    # iki SAC arasındaki tek fark açık tahmin katmanıdır.
    history = np.full((len(load), H), np.nan)
    history[H:] = np.lib.stride_tricks.sliding_window_view(load, H)[: len(load) - H]
    last_meas = np.concatenate([[load[0]], load[:-1]])
    env_pure = BatteryFeederEnv(load, history, last_meas, price, df.index, E, Pb, P_lim, price_scale)
    env_hyb = BatteryFeederEnv(load, oof_pred, oof_pred[:, 0], price, df.index, E, Pb, P_lim, price_scale)

    try:
        lp_actions = dict(zip(eval_idx, perfect_foresight_lp(env_pure, eval_idx)))
        print("    Teorik optimum (LP) çözüldü")
    except RuntimeError as exc:   # LP yalnızca referans: çözülemezse koşuyu kaybetme, satırı atla
        lp_actions = None
        print(f"    [UYARI] {exc} — 'Teorik Optimum' satırı ve 'Yakalanan potansiyel' bu veri setinde boş kalacak.")

    lo, hi = np.quantile(price[train_idx], [0.30, 0.70])
    fixed = {
        "Bataryasız": (env_pure, lambda t, soc: 0.0),
        "Kural Tabanlı": (env_pure, rule_based_policy(env_pure, lo, hi)),
    }
    if lp_actions is not None:
        fixed["Teorik Optimum (tam öngörü)"] = (env_pure, lambda t, soc: lp_actions[t])
    fixed_traces = {name: rollout(env, eval_idx, fn) for name, (env, fn) in fixed.items()}

    base = fixed_traces["Bataryasız"]
    base_viol = base["violation"].sum()
    lp_saving = (base["cost"].sum() - fixed_traces["Teorik Optimum (tam öngörü)"]["cost"].sum()
                 if lp_actions is not None else np.nan)

    def metrics(name, tr):
        saving = base["cost"].sum() - tr["cost"].sum()
        return {
            "Kontrolcü": name,
            "Ort. saatlik maliyet": tr["cost"].mean(),
            "Tasarruf (%)": saving / base["cost"].sum() * 100,
            # Tam öngörülü optimumun sağladığı tasarrufun ne kadarı yakalandı
            "Yakalanan potansiyel (%)": saving / lp_saving * 100 if lp_saving > 0 else np.nan,
            "Toplam ihlal": tr["violation"].sum(),
            "İhlalli saat (%)": (tr["violation"] > 0).mean() * 100,
            # Referansta ihlal yoksa oran tanımsızdır (NaN -> tabloda "—")
            "İhlal düşüşü (%)": (1 - tr["violation"].sum() / base_viol) * 100 if base_viol > 0 else np.nan,
            "Min. gerilim (pu)": tr["v"].min(),
        }

    pb_pu, e_pu = Pb / P_lim, E / P_lim
    rows, traces, rl_logs = [], None, None
    for k in range(CFG["n_seeds"]):
        seed = CFG["seed"] + k
        print(f"  --- tohum {k + 1}/{CFG['n_seeds']} (seed={seed}) ---")
        sac_pure, log_pure = train_sac(env_pure, train_idx, seed, "Saf SAC")
        sac_hyb, log_hyb = train_sac(env_hyb, train_idx, seed, "Hibrit SAC, PINN'siz")
        pinn, log_pinn = train_pinn(env_hyb, train_idx, seed)

        @torch.no_grad()
        def safety(t, soc, a, pinn=pinn):
            x = torch.tensor([[env_hyb.load_now[t] / P_lim, a, soc, pb_pu, e_pu]], dtype=torch.float32, device=DEVICE)
            return pinn(x)[0].item()

        # Katman 3 eğitim döngüsünde: SAC, PINN'in eylemlerini düzelteceğini bilerek öğrenir
        sac_safe, log_safe = train_sac(env_hyb, train_idx, seed, "Hibrit SAC + PINN döngüde", safety=safety)
        learned = {
            "Saf SAC": (env_pure, lambda t, soc, ag=sac_pure: ag.policy(env_pure.obs(t, soc))),
            "Hibrit (PINN'siz, ablasyon)": (env_hyb, lambda t, soc, ag=sac_hyb: ag.policy(env_hyb.obs(t, soc))),
            "Hibrit 3-Katman (Önerilen)": (env_hyb, lambda t, soc, ag=sac_safe, sf=safety:
                                           sf(t, soc, ag.policy(env_hyb.obs(t, soc)))),
        }
        seed_traces = {**fixed_traces, **{n: rollout(env, eval_idx, fn) for n, (env, fn) in learned.items()}}
        rows += [dict(metrics(n, tr), Tohum=seed) for n, tr in seed_traces.items()]
        if traces is None:   # grafikler ilk tohumun izlerini kullanır
            traces = seed_traces
            rl_logs = dict(sac_pure=log_pure, sac_hyb=log_hyb, sac_safe=log_safe, pinn=log_pinn)

    op_long = pd.DataFrame(rows)
    order = [c for c in CONTROLLER_ORDER if c in set(op_long["Kontrolcü"])]
    grouped = op_long.drop(columns="Tohum").groupby("Kontrolcü")
    op_table = grouped.mean().reindex(order)
    std = grouped.std(ddof=0).reindex(order)
    for c in ["Tasarruf (%)", "Yakalanan potansiyel (%)", "Toplam ihlal"]:
        op_table[c + " std"] = std[c]
    op_table = op_table.reset_index()
    params = dict(E=E, Pb=Pb, P_lim=P_lim, eval_hours=len(eval_idx), train_hours=len(train_idx),
                  seeds=[CFG["seed"] + k for k in range(CFG["n_seeds"])])
    return op_table, op_long, traces, eval_idx, rl_logs, params


# 8) Grafikler
def _smooth(x, w=200):
    x = np.asarray(x, float)
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


def _style():
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.05,
                  rc={"axes.spines.top": False, "axes.spines.right": False, "grid.color": "#E6E6E6",
                      "lines.linewidth": 2, "savefig.dpi": 300, "figure.dpi": 110})


def plot_loss_curves(results, path):
    n = len(results)
    fig, axes = plt.subplots(n, 3, figsize=(16, 4.2 * n), squeeze=False)
    for r, res in enumerate(results):
        ax = axes[r, 0]
        for k, h in enumerate(res["histories"]):
            ax.plot(h["loss"], color="#0072B2", alpha=0.35 + 0.13 * k, lw=1.5,
                    label="Eğitim" if k == len(res["histories"]) - 1 else None)
            ax.plot(h["val_loss"], color="#E69F00", ls="--", alpha=0.35 + 0.13 * k, lw=1.5,
                    label="Doğrulama" if k == len(res["histories"]) - 1 else None)
        ax.set(title=f"{res['name']}\nCNN-BiLSTM-Att (5 kat; koyu = son kat)", xlabel="Epoch", ylabel="MSE (z-ölçek)")
        ax.legend(frameon=False)

        ax = axes[r, 1]
        for lbl, key, c in [("Saf SAC", "sac_pure", "#009E73"), ("Hibrit, PINN'siz", "sac_hyb", "#CC79A7"),
                            ("Hibrit + PINN döngüde", "sac_safe", "#0072B2")]:
            ax.plot(_smooth(res["rl_logs"][key]["critic"]), color=c, label=f"{lbl} — kritik")
        ax.set(title="SAC kritik (Q) kaybı (hareketli ort., 1. tohum)", xlabel="Güncelleme adımı", ylabel="Kayıp")
        ax.set_yscale("log"); ax.legend(frameon=False)

        ax = axes[r, 2]
        pl = pd.DataFrame(res["rl_logs"]["pinn"])
        for col, c in [("toplam", "#000000"), ("sadakat", "#0072B2"), ("KCL", "#E69F00"),
                       ("KVL", "#009E73"), ("esitsizlik", "#CC79A7")]:
            ax.plot(pl[col].clip(lower=1e-12), color=c, label=col)
        ax.set(title="PINN kayıp bileşenleri", xlabel="Epoch", ylabel="Kayıp (log)")
        ax.set_yscale("log"); ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_pinn_violations(results, path):
    n = len(results)
    fig, axes = plt.subplots(n, 2, figsize=(16, 4.2 * n), squeeze=False, gridspec_kw={"width_ratios": [2.2, 1]})
    for r, res in enumerate(results):
        tr_pre, tr_post = res["traces"]["Hibrit (PINN'siz, ablasyon)"], res["traces"]["Hibrit 3-Katman (Önerilen)"]
        # İhlalin en yoğun olduğu 7 günlük pencere
        w = 168
        roll = np.convolve(tr_pre["violation"], np.ones(w), mode="valid") if len(tr_pre["violation"]) > w else tr_pre["violation"]
        s = int(np.argmax(roll)) if len(roll) else 0
        sl = slice(s, s + w)
        ts = res["eval_index"][sl]
        ax = axes[r, 0]
        ax.fill_between(ts, tr_pre["violation"][sl], step="mid", color="#CC79A7", alpha=0.45, label="Hibrit, PINN'siz")
        ax.fill_between(ts, tr_post["violation"][sl], step="mid", color="#0072B2", alpha=0.85,
                        label="Hibrit, PINN eğitim döngüsünde")
        ax.set(title=f"{res['name']} — en yoğun ihlal haftası (1. tohum)",
               ylabel=f"Hat/gerilim ihlali ({res['spec']['unit']})")
        if tr_pre["violation"].sum() == 0 and tr_post["violation"].sum() == 0:
            ax.text(0.5, 0.5, "Test döneminde hibrit kontrolcüde ihlal oluşmadı", transform=ax.transAxes,
                    ha="center", va="center", color="#555555")
        ax.legend(frameon=False, loc="upper right")
        ax.tick_params(axis="x", rotation=20)

        ax = axes[r, 1]
        op = res["op_table"].set_index("Kontrolcü")
        names = list(op.index)
        vals, errs = op["Toplam ihlal"].values, op["Toplam ihlal std"].values
        ax.barh(names, vals, xerr=errs, color=[PALETTE[nm] for nm in names], height=0.6,
                error_kw=dict(ecolor="#333333", capsize=3))
        for i, v in enumerate(vals):
            ax.text(v + errs[i], i, f" {v:,.1f}", va="center", fontsize=8, color="#333333")
        ax.invert_yaxis()
        ax.set(title=f"Toplam ihlal ({res['spec']['energy']}), tohum ort. ± ss", xlabel="")
        ax.margins(x=0.25)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_benchmark(results, path):
    fc_rows, op_rows = [], []
    for res in results:
        label = res["name"].replace(" (", "\n(") + ("\n[SENTETİK]" if res["synthetic"] else "")
        for model, folds in res["fold_metrics"].items():
            for f in folds:
                fc_rows.append(dict(Veri=label, Model=model, MAPE=f["MAPE_filtreli"], R2=f["R2"]))
        for _, row in res["op_long"].iterrows():      # tohum başına satırlar -> hata çubukları
            op_rows.append(dict(Veri=label, Kontrolcü=row["Kontrolcü"], Tasarruf=row["Tasarruf (%)"],
                                Ihlal=row["İhlal düşüşü (%)"]))
    fc, op = pd.DataFrame(fc_rows), pd.DataFrame(op_rows)
    op = op[op["Kontrolcü"] != "Bataryasız"]

    fig, axes = plt.subplots(1, 3, figsize=(20, 5.8))
    fc_order = ["CNN-BiLSTM-Att (Önerilen)", "Random Forest", "XGBoost"]
    sns.barplot(data=fc, x="Veri", y="MAPE", hue="Model", hue_order=fc_order, palette=PALETTE,
                errorbar="sd", capsize=0.08, ax=axes[0], saturation=1, edgecolor="white", linewidth=2)
    axes[0].set(title="(a) Filtrelenmiş MAPE (%), 24 sa ufuk — 5 kat ort. ± ss, düşük iyi", xlabel="", ylabel="MAPE (%)")

    op_order = ["Kural Tabanlı", "Saf SAC", "Hibrit (PINN'siz, ablasyon)", "Hibrit 3-Katman (Önerilen)",
                "Teorik Optimum (tam öngörü)"]
    sns.barplot(data=op, x="Veri", y="Tasarruf", hue="Kontrolcü", hue_order=op_order, palette=PALETTE,
                errorbar="sd", capsize=0.08, ax=axes[1], saturation=1, edgecolor="white", linewidth=2)
    axes[1].set(title="(b) Bataryasıza göre maliyet tasarrufu (%) — tohum ort. ± ss", xlabel="", ylabel="Tasarruf (%)")

    sns.barplot(data=op, x="Veri", y="Ihlal", hue="Kontrolcü", hue_order=op_order, palette=PALETTE,
                errorbar="sd", capsize=0.08, ax=axes[2], saturation=1, edgecolor="white", linewidth=2)
    axes[2].set(title="(c) Fiziksel ihlal düşüşü (%) — bataryasıza göre, tohum ort. ± ss", xlabel="",
                ylabel="İhlal düşüşü (%)")
    for ax in axes:
        ax.axhline(0, color="#555555", lw=0.8)
        ax.tick_params(axis="x", labelsize=9)
        ax.legend(frameon=False, fontsize=8)
        for c in ax.containers:
            ax.bar_label(c, fmt="%.1f", fontsize=7, padding=2, color="#333333")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# 9) Markdown tabloları
def _fmt(x, d=3):
    return "—" if pd.isna(x) else f"{x:,.{d}f}"


def _pm(row, col, d):
    """Tohumlar arası std > 0 ise 'ort ± std', deterministik kontrolcülerde yalnızca değer."""
    s = row.get(col + " std", np.nan)
    return _fmt(row[col], d) + ("" if pd.isna(s) or s == 0 else f" ± {_fmt(s, d)}")


def markdown_report(results):
    out = ["# Benchmark Sonuçları", "",
           f"_Üretim zamanı: {time.strftime('%Y-%m-%d %H:%M')} · QUICK={QUICK} · "
           f"RL tohumları={CFG['seed']}…{CFG['seed'] + CFG['n_seeds'] - 1} (n={CFG['n_seeds']}) · "
           f"TimeSeriesSplit k={CFG['n_splits']} · pencere={CFG['seq_len']} sa_", ""]
    if any(r["synthetic"] for r in results):
        out += ["> **UYARI:** En az bir veri seti SENTETİK üreteçten gelmektedir. Bu satırlar yöntemin "
                "çalıştığını gösterir; teze gerçek deney bulgusu olarak yazılmamalıdır.", ""]

    out += ["## Tablo 1 — Veri setleri", "",
            "| Ölçek | Kaynak | Saat sayısı | Ort. yük | MAPE filtre eşiği τ | Fiyat |",
            "|---|---|---:|---:|---:|---|"]
    for r in results:
        out.append(f"| {r['name']}{' ⚠SENTETİK' if r['synthetic'] else ''} | {r['source']} | {r['n_hours']:,} | "
                   f"{_fmt(r['mean_load'], 2)} {r['spec']['unit']} | {CFG['near_zero_alpha']:.0%} × ort. yük | "
                   f"{r['spec']['currency']}/{r['spec']['energy']} |")

    out += ["", f"## Tablo 2 — Tahmin başarısı ({CFG['horizon']} sa ufuk, 5 katlı TimeSeriesSplit, ortalama ± std)", "",
            "| Veri seti | Model | RMSE | MAE | R² | Ham MAPE (%) | Filtrelenmiş MAPE (%) | Dışlanan nokta (%) |",
            "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        for model, folds in r["fold_metrics"].items():
            m = pd.DataFrame(folds)
            cell = lambda c, d=3: f"{_fmt(m[c].mean(), d)} ± {_fmt(m[c].std(ddof=0), d)}"
            bold = "**" if "Önerilen" in model else ""
            out.append(f"| {r['name']} | {bold}{model}{bold} | {cell('RMSE')} | {cell('MAE')} | {cell('R2')} | "
                       f"{cell('MAPE_ham', 2)} | {cell('MAPE_filtreli', 2)} | {_fmt(m['dislanan_pct'].mean(), 2)} |")

    out += ["", f"## Tablo 3 — Ekonomik ve fiziksel başarı (kat 5 test dönemi, görülmemiş veri; öğrenen "
                f"kontrolcüler {CFG['n_seeds']} tohum ort. ± std)", "",
            "| Veri seti | Kontrolcü | Ort. saatlik maliyet | Tasarruf (%) | Yakalanan potansiyel (%) | Toplam ihlal | "
            "İhlalli saat (%) | İhlal düşüşü (%) | Min. V (pu) |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        cur, en = r["spec"]["currency"], r["spec"]["energy"]
        for _, row in r["op_table"].iterrows():
            bold = "**" if "Önerilen" in row["Kontrolcü"] else ""
            out.append(f"| {r['name']} | {bold}{row['Kontrolcü']}{bold} | {_fmt(row['Ort. saatlik maliyet'], 2)} {cur}/sa | "
                       f"{_pm(row, 'Tasarruf (%)', 2)} | {_pm(row, 'Yakalanan potansiyel (%)', 1)} | "
                       f"{_pm(row, 'Toplam ihlal', 2)} {en} | "
                       f"{_fmt(row['İhlalli saat (%)'], 2)} | {_fmt(row['İhlal düşüşü (%)'], 2)} | "
                       f"{_fmt(row['Min. gerilim (pu)'], 4)} |")

    out += ["", "## Tablo 4 — PINN katmanının etkisi (PINN'siz hibrit → PINN eğitim döngüsünde hibrit, tohum ort.)", "",
            "| Veri seti | İhlal öncesi | İhlal sonrası | İhlal azalımı (%) | Maliyet farkı (%) |",
            "|---|---:|---:|---:|---:|"]
    for r in results:
        op = r["op_table"].set_index("Kontrolcü")
        pre, post = op.loc["Hibrit (PINN'siz, ablasyon)"], op.loc["Hibrit 3-Katman (Önerilen)"]
        red = (1 - post["Toplam ihlal"] / pre["Toplam ihlal"]) * 100 if pre["Toplam ihlal"] > 0 else float("nan")
        dc = (post["Ort. saatlik maliyet"] / pre["Ort. saatlik maliyet"] - 1) * 100
        out.append(f"| {r['name']} | {_fmt(pre['Toplam ihlal'], 2)} | {_fmt(post['Toplam ihlal'], 2)} | "
                   f"{_fmt(red, 2)} | {dc:+.2f} |")

    out += ["", "### Deney kurulumu notları", "",
            "- Ham MAPE paydasında |y| kullanılır (ε=1e-6); filtrelenmiş MAPE yalnızca y > τ noktalarında hesaplanır.",
            "- Z-score parametreleri her katta yalnızca eğitim bölümünden öğrenilir (veri sızıntısı yok).",
            f"- Tahminler {CFG['horizon']} saatlik doğrudan çok adımlıdır; TimeSeriesSplit gap={CFG['horizon'] - 1} ile "
            "eğitim hedeflerinin test dönemine taşması engellenir.",
            "- SAC'lar kat 1–4 test bölgelerinde (katman-1 için örneklem dışı tahminler) eğitilir, kat 5'te değerlendirilir.",
            "- Saf SAC son 24 sa ölçülen yükü, hibrit SAC CNN-BiLSTM-Att'ın 24 sa tahminini gözlemler; ikisi de 24 sa "
            "gün öncesi fiyatı görür ve ödülleri aynıdır (fatura + ihlal cezası). Tek fark açık tahmin katmanıdır.",
            f"- İhlal cezası = {CFG['viol_penalty']} × ortalama fiyat / birim enerji; tablolardaki maliyet cezasız faturadır.",
            "- Teorik Optimum: gelecekteki yük ve fiyatı tam bilen LP (HiGHS); gerçekte erişilemez, üst sınır referansıdır. "
            "Yakalanan potansiyel = kontrolcü tasarrufu / LP tasarrufu.",
            "- Fiziksel model: tek dallı radyal besleyici, LinDistFlow V≈V0−(R·P+X·Q)/V0; hat limiti eğitim yükünün "
            f"%{CFG['line_limit_q'] * 100:.0f} yüzdeliği; bu limitte V={V_MIN} pu.",
            f"- Batarya: E = {CFG['batt_energy_ratio']}× ort. yük, P = {CFG['batt_power_ratio']}× ort. yük "
            "(Makro-Şebeke: E = 0.08×, P = 0.02×), "
            f"η={BATT_ETA}, SOC∈[{SOC_MIN},{SOC_MAX}].",
            "- SAC ödülü bataryasız duruma göre farktır (−Δfatura − ceza·Δihlal); eylemden bağımsız terim çıkarıldığı "
            "için optimum politika değişmez.",
            "- PINN güvenlik katmanı politikadan bağımsız eğitilir (rastgele eylem/SOC) ve önerilen modelde SAC "
            "eğitim döngüsündedir: SAC, eylemlerinin PINN tarafından düzeltileceğini bilerek öğrenir. "
            "Ablasyon satırı aynı ödülle PINN'siz eğitilmiş hibrit SAC'tır.",
            f"- Öğrenen kontrolcüler {CFG['n_seeds']} farklı tohumla eğitilip ortalama ± std raporlanır; tahmin "
            "katmanı tohumlar arasında ortaktır. Grafiklerdeki zaman serileri ve kayıp eğrileri 1. tohuma aittir."]
    return "\n".join(out)


# 10) Ana akış
def main():
    os.makedirs(CFG["out_dir"], exist_ok=True)
    os.makedirs(CFG["data_dir"], exist_ok=True)
    set_seed(CFG["seed"])
    _style()
    rng = np.random.default_rng(CFG["seed"])
    gpu = torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "yok"
    print(f"PyTorch {torch.__version__} · cihaz={DEVICE} ({gpu}) · QUICK={QUICK}")

    results = []
    for spec in DATASETS:
        t0 = time.time()
        print(f"\n=== {spec['name']} ===")
        df, source, synthetic, spec = prepare_dataset(spec, rng)
        print(f"    kaynak: {source} · {len(df):,} saat · {df.index[0]} → {df.index[-1]}")

        print("  [Katman 1] Tahmin modelleri (5 katlı TimeSeriesSplit)")
        fold_metrics, histories, oof_pred, folds = run_forecasting_isolated(df)
        with open(os.path.join(CFG["out_dir"], f"forecast_{spec['key']}.json"), "w", encoding="utf-8") as f:
            json.dump(dict(name=spec["name"], source=source, synthetic=synthetic, fold_metrics=fold_metrics),
                      f, ensure_ascii=False, indent=2)

        print("  [Katman 2+3] SAC + PINN operasyon optimizasyonu")
        op_table, op_long, traces, eval_idx, rl_logs, params = run_operation(df, oof_pred, folds, spec)
        print(op_table.round(3).to_string(index=False))
        op_table.to_csv(os.path.join(CFG["out_dir"], f"operation_{spec['key']}.csv"), index=False)
        op_long.to_csv(os.path.join(CFG["out_dir"], f"operation_{spec['key']}_per_seed.csv"), index=False)

        results.append(dict(name=spec["name"], spec=spec, source=source, synthetic=synthetic,
                            n_hours=len(df), mean_load=float(df["load"].mean()),
                            fold_metrics=fold_metrics, histories=histories, op_table=op_table, op_long=op_long,
                            traces=traces, eval_index=df.index[eval_idx], rl_logs=rl_logs, params=params))
        print(f"    süre: {time.time() - t0:.0f}s")

    out = CFG["out_dir"]
    plot_loss_curves(results, os.path.join(out, "loss_curves.png"))
    plot_pinn_violations(results, os.path.join(out, "pinn_violations.png"))
    plot_benchmark(results, os.path.join(out, "multi_dataset_benchmark.png"))

    report = markdown_report(results)
    with open(os.path.join(out, "benchmark_tables.md"), "w", encoding="utf-8") as f:
        f.write(report)
    with open(os.path.join(out, "benchmark_raw.json"), "w", encoding="utf-8") as f:
        json.dump([dict(name=r["name"], source=r["source"], synthetic=r["synthetic"], params=r["params"],
                        fold_metrics=r["fold_metrics"], operation=r["op_table"].to_dict("records"))
                   for r in results], f, ensure_ascii=False, indent=2)

    print("\n" + report)
    print(f"\nÇıktılar: {os.path.abspath(out)}/  (loss_curves.png, pinn_violations.png, "
          f"multi_dataset_benchmark.png, benchmark_tables.md, benchmark_raw.json)")
    if "google.colab" in sys.modules:
        from IPython.display import Image, display
        for name in ["loss_curves.png", "pinn_violations.png", "multi_dataset_benchmark.png"]:
            display(Image(os.path.join(out, name)))


if __name__ == "__main__":
    main()
