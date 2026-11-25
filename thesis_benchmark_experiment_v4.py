"""thesis_benchmark_experiment_v4.py: veri hazırlama, tahmin katmanı, ortam ve SAC.

Katman 1 (CNN-BiLSTM): son 24 saatin yükünden gelecek 24 saatin yükünü tahmin eder.
Katman 2 (SAC): yük tahmini, gün öncesi fiyat ve batarya doluluğundan saatlik şarj/deşarj kararını verir.
Ortam: tek dallı radyal besleyici (Kirchhoff güç dengesi ve LinDistFlow gerilim bağıntısı) ve batarya.
thesis_benchmark_stats_v5.py, thesis_benchmark_stats_v6.py ve thesis_benchmark_lambda_v6.py bu modülü kullanır.

Veri kümeleri: UCI Individual Household Electric Power Consumption (konut), Building Data Genome Project 2
Robin_office_Gayle sayacı (bina), Open Power System Data DE-LU yükü ve gün öncesi fiyatı (şebeke).

Tek başına çalıştırıldığında aynı test döneminde (5. kat) karşılaştırılan kontrolcüler:
  bataryasız işletme (referans), rastgele politika, kural tabanlı yönetim, SAC (tahminsiz),
  CNN-BiLSTM + SAC, teorik optimum (referans, geleceği tam bilen doğrusal program)

Çalıştırma: python thesis_benchmark_experiment_v4.py
Seçenekler: QUICK=1 (kısa deneme), FORCE_SYNTHETIC=1, SEEDS=3, SAC_WORKERS=8
Yerel CSV: thesis_data/{micro,macro,meso}.csv (timestamp,load[,price])
Çıktılar: thesis_benchmark_v4_outputs/ (benchmark_v4_tables.md, benchmark_v4_raw.json, *.csv)
İndirme başarısız olursa sentetik veri kullanılır ve çıktılarda "SENTETİK" olarak işaretlenir.
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
    out_dir="thesis_benchmark_v4_outputs",
    data_dir="thesis_data",
    max_hours=24 * 365 * 2,
    seq_len=24, horizon=24, n_splits=5,
    dl_epochs=150, dl_batch=128, dl_patience=15,
    sac_steps=40000, sac_warmup=4000, sac_batch=256, sac_episode_len=168,
    n_seeds=int(os.environ.get("SEEDS", "3")),
    sac_workers=int(os.environ.get("SAC_WORKERS", "0")) or min(os.cpu_count() or 2, 8),
    near_zero_alpha=0.10,
    batt_energy_ratio=2.0, batt_power_ratio=0.5,
    line_limit_q=0.85, viol_penalty=float(os.environ.get("VIOL_PENALTY", "5.0")),
)
if QUICK:
    CFG.update(max_hours=24 * 120, dl_epochs=3, sac_steps=1500, sac_warmup=300, n_seeds=1)

TOU_TARIFF = {"gece (22-06)": 1.55, "gündüz (06-17)": 2.60, "puant (17-22)": 4.10}

V0, V_MIN, V_MAX = 1.00, 0.95, 1.05
R_PU, X_PU, TAN_PHI = 0.03, 0.04, 0.50   # R + X*tanφ = 0.05 -> P = P_lim'de V = 0.95 pu
BATT_ETA, SOC_MIN, SOC_MAX, DEG_COST_FRAC = 0.95, 0.10, 0.90, 0.02

FULL = "CNN-BiLSTM"          # Katman 1
FORECASTERS = [FULL]

REF_NONE, REF_LP = "Bataryasız [referans]", "Teorik optimum [referans]"
RANDOM, RULE, SAC_PLAIN, HYBRID = ("Rastgele politika", "Kural tabanlı yönetim", "SAC (tahminsiz)",
                                   "CNN-BiLSTM + SAC [önerilen]")
ORDER = [REF_NONE, RANDOM, RULE, SAC_PLAIN, HYBRID, REF_LP]
STD_COLS = ["Tasarruf (%)", "Yakalanan potansiyel (%)", "Toplam ihlal", "İhlal düşüşü (%)"]


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
    best = "Robin_office_Gayle"                              # tezde kullanılan ofis sayacı
    raw = pd.read_csv(path, usecols=["timestamp", best], parse_dates=["timestamp"]).set_index("timestamp")
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
    fiyat = os.environ.get("FIYAT")                         # tarife ile tahmin isabetinin etkisini ayırma deneyi
    if fiyat == "tou":                                       # bütün ölçeklerde üç zamanlı tarife
        df["price"] = tou_price(df.index)
        source += " + FIYAT=tou"
    elif fiyat == "piyasa" and spec["key"] != "macro":       # şebekenin gerçek gün öncesi fiyatı, saat sırasıyla
        sebeke = next(s for s in DATASETS if s["key"] == "macro")
        pdf, _ = sebeke["loader"]()
        pdf = pdf[~pdf.index.duplicated()].sort_index().asfreq("h").interpolate(limit=6).dropna().asfreq("h").ffill()
        df["price"] = pdf["price"].values[-len(df):]
        source += " + FIYAT=piyasa (DE-LU gün öncesi, saat sırasıyla)"
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
    """Tahmin katmanı: iki nedensel evrişim katmanı + çift yönlü LSTM (CNN-BiLSTM).
    Beş mimarinin karşılaştırması (CNN, BiLSTM, Attention, CNN-BiLSTM-Attention) thesis_benchmark_experiment_v2.py'dedir."""
    if name != FULL:
        raise ValueError(name)
    inp = keras.Input(shape=(L, n_feat))
    x = layers.Conv1D(64, 3, padding="causal", activation="relu")(inp)
    x = layers.Conv1D(64, 3, padding="causal", activation="relu")(x)
    z = layers.Bidirectional(layers.LSTM(64))(x)
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


# 4) Fiziksel model ve ortam
def feeder_voltage_pu(p_grid_pu):
    """Tek dallı radyal besleyici, LinDistFlow: V ≈ V0 - (R·P + X·Q)/V0, Q = P·tanφ (pu)."""
    return V0 - (R_PU * p_grid_pu + X_PU * p_grid_pu * TAN_PHI) / V0


class BatteryFeederEnv:
    """Saatlik batarya dispeçi. Eylem a∈[-1,1] -> p_batt = a·P_b (+ şarj, - deşarj). KCL: P_grid = P_load + P_batt.
    Ödül = bataryasız duruma göre −(Δfatura + ceza·Δihlal). Gözlem: 24 sa yük bilgisi, 24 sa fiyat, SOC, takvim.
    load_now: kontrolcünün içinde bulunulan saatin yükü için kestirimi (kural tabanlı yönetim kullanır)."""

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


def train_sac(env, train_idx, seed):
    """SAC eğitimi: ilk sac_warmup adım rastgele eylemle deneyim toplanır, sonra her adımda bir güncelleme yapılır."""
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
        soc2, r, _ = env.step(t, soc, a)
        done = step_in_ep == ep_len - 1
        s2 = env.obs(t + 1, soc2)
        # Bölüm sonu bir zaman sınırıdır, gerçek bitiş değildir: hedef son durumun değeriyle sürdürülür
        # (Pardo ve diğ., 2018). ZAMAN_SINIRI_KES=1 v6'daki eski davranışı (hedefi kesmeyi) üretir.
        kes = float(done and os.environ.get("ZAMAN_SINIRI_KES") == "1")
        for k, v in zip(["s", "a", "r", "s2", "d"], [s, [a], [r], s2, [kes]]):
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


# 6) Paralel SAC eğitimi (her iş ayrı süreç; GPU paylaşılır)
def _to_numpy_state(module):
    return {k: v.detach().cpu().numpy() for k, v in module.state_dict().items()}


def _load_state(module, state):
    module.load_state_dict({k: torch.as_tensor(v) for k, v in state.items()})
    return module.to(DEVICE).eval()


def _sac_job(job):
    torch.set_num_threads(1)
    env = BatteryFeederEnv(**job["env"])
    agent, log, returns = train_sac(env, job["train_idx"], job["seed"])
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


def rollout(env, eval_idx, policy_fn):
    soc, out = 0.5, dict(cost=[], violation=[], v=[])
    for t in eval_idx:
        soc, _, info = env.step(t, soc, policy_fn(t, soc))
        for k in out:
            out[k].append(info[k])
    return {k: np.array(v) for k, v in out.items()}


# 7) Operasyon: aynı test döneminde bütün kontrolcüler
def rule_based_policy(env, price_lo, price_hi):
    """Ucuz saatte şarj (hat sınırını aşmayacak kadar), pahalı saatte tam güçle deşarj."""
    def fn(t, soc):
        if env.price[t] <= price_lo:
            headroom = max(env.P_lim - env.load_now[t], 0.0)
            return min(1.0, headroom / env.Pb)
        if env.price[t] >= price_hi:
            return -1.0
        return 0.0
    return fn


def aggregate(long):
    num = [c for c in long.columns if c not in ("Kontrolcü", "Tohum")]
    grouped = long.groupby("Kontrolcü", sort=False)[num]
    table, std = grouped.mean(), grouped.std(ddof=0)
    for c in STD_COLS:
        table[c + " std"] = std[c]
    return table.reindex([c for c in ORDER if c in table.index]).reset_index()


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
    infos = {"none": (history, last_meas), FULL: (oof[FULL], oof[FULL][:, 0])}
    env_kw = {key: dict(load=load, load_info=info, load_now=now, price=price, index=df.index,
                        E=E, Pb=Pb, P_lim=P_lim, price_scale=price_scale) for key, (info, now) in infos.items()}
    envs = {key: BatteryFeederEnv(**kw) for key, kw in env_kw.items()}
    env_none, env_full = envs["none"], envs[FULL]

    fixed = {REF_NONE: rollout(env_none, eval_idx, lambda t, soc: 0.0)}
    lo, hi = np.quantile(price[train_idx], [0.30, 0.70])
    fixed[RULE] = rollout(env_none, eval_idx, rule_based_policy(env_none, lo, hi))
    try:
        lp_actions = dict(zip(eval_idx, perfect_foresight_lp(env_none, eval_idx)))
        fixed[REF_LP] = rollout(env_none, eval_idx, lambda t, soc: lp_actions[t])
        print("    Teorik optimum (LP) çözüldü")
    except RuntimeError as exc:
        print(f"    [UYARI] {exc} — 'Yakalanan potansiyel' bu veri setinde boş kalacak.")
    base = fixed[REF_NONE]
    base_viol = base["violation"].sum()
    lp_saving = base["cost"].sum() - fixed[REF_LP]["cost"].sum() if REF_LP in fixed else np.nan

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
    jobs = [dict(name=name, seed=seed, env=env_kw[key], train_idx=train_idx)
            for seed in seeds for name, key in [(SAC_PLAIN, "none"), (HYBRID, FULL)]]
    sac = run_sac_jobs(jobs)

    rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        traces = dict(fixed)
        traces[RANDOM] = rollout(env_none, eval_idx, lambda t, soc: rng.uniform(-1.0, 1.0))
        traces[SAC_PLAIN] = rollout(env_none, eval_idx, actor_policy(sac[(SAC_PLAIN, seed)]["actor"], env_none))
        traces[HYBRID] = rollout(env_full, eval_idx, actor_policy(sac[(HYBRID, seed)]["actor"], env_full))
        rows += [dict(metrics(name, tr), Tohum=seed) for name, tr in traces.items()]
    long = pd.DataFrame(rows)
    params = dict(E=E, Pb=Pb, P_lim=P_lim, eval_hours=len(eval_idx), train_hours=len(train_idx), seeds=seeds)
    return aggregate(long), long, params


def contributions(long):
    """Katman katkısı: aynı tohumdaki iki kontrolcünün farkı (eşleştirilmiş), tohumlar üzerinden ort ± std."""
    pairs = [("K1 — tahmin katkısı: (CNN-BiLSTM + SAC) − SAC", HYBRID, SAC_PLAIN),
             ("K2 — karar katkısı: SAC − kural tabanlı", SAC_PLAIN, RULE)]
    rows = []
    for metric in ["Yakalanan potansiyel (%)", "İhlal düşüşü (%)", "Tasarruf (%)"]:
        piv = long.pivot(index="Tohum", columns="Kontrolcü", values=metric)
        for label, a, b in pairs:
            d = piv[a] - piv[b]
            rows.append(dict(Katkı=label, Metrik=metric, Ort=d.mean(), Std=d.std(ddof=0)))
    return pd.DataFrame(rows)


# 8) Rapor ve ana akış
def markdown_report(results):
    out = ["# CNN-BiLSTM + SAC — karşılaştırma tabloları", ""]
    for r in results:
        fm = pd.DataFrame(r["fold_metrics"][FULL])[["RMSE", "MAE", "R2", "MAPE_filtreli"]]
        out += [f"## {r['name']}", "",
                f"Kaynak: {r['source']}{' · SENTETİK' if r['synthetic'] else ''} · {r['n_hours']:,} saat · "
                f"test {r['params']['eval_hours']} saat · tohumlar {r['params']['seeds']}", "",
                "**Katman 1 (CNN-BiLSTM), 5 kat ortalaması ± standart sapma**", "",
                "```", pd.DataFrame({"Ort": fm.mean(), "Std": fm.std(ddof=0)}).round(4).to_string(), "```", "",
                "**Kontrolcüler (5. kat test dönemi)**", "",
                "```", r["table"].round(4).to_string(index=False), "```", "",
                "**Katman katkıları (eşleştirilmiş fark)**", "",
                "```", r["contrib"].round(3).to_string(index=False), "```", ""]
    return "\n".join(out)


def main():
    os.makedirs(CFG["out_dir"], exist_ok=True)
    os.makedirs(CFG["data_dir"], exist_ok=True)
    set_seed(CFG["seed"])
    rng = np.random.default_rng(CFG["seed"])
    gpu = torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "yok"
    print(f"PyTorch {torch.__version__} · cihaz={DEVICE} ({gpu}) · QUICK={QUICK} · tohum={CFG['n_seeds']} · "
          f"SAC süreçleri={CFG['sac_workers']}")

    results = []
    for spec in DATASETS:
        t0 = time.time()
        print(f"\n=== {spec['name']} ===")
        df, source, synthetic, spec = prepare_dataset(spec, rng)
        print(f"    kaynak: {source} · {len(df):,} saat · {df.index[0]} → {df.index[-1]}")

        print("  [A] Katman 1: CNN-BiLSTM, 5 katlı zaman serisi doğrulaması")
        fold_metrics, histories, oof, folds = run_forecasting_isolated(df)

        print("  [B] Kontrolcüler: rastgele, kural tabanlı, SAC, CNN-BiLSTM + SAC")
        table, long, params = run_operation(df, oof, folds, spec)
        contrib = contributions(long)
        print(table[["Kontrolcü", "Ort. saatlik maliyet", "Tasarruf (%)", "Yakalanan potansiyel (%)", "Toplam ihlal",
                     "İhlal düşüşü (%)"]].round(2).to_string(index=False))
        long.to_csv(os.path.join(CFG["out_dir"], f"controllers_{spec['key']}.csv"), index=False)
        contrib.to_csv(os.path.join(CFG["out_dir"], f"contributions_{spec['key']}.csv"), index=False)
        results.append(dict(name=spec["name"], source=source, synthetic=synthetic, n_hours=len(df),
                            fold_metrics=fold_metrics, table=table, contrib=contrib, params=params))
        print(f"    süre: {time.time() - t0:.0f}s")

    report = markdown_report(results)
    with open(os.path.join(CFG["out_dir"], "benchmark_v4_tables.md"), "w", encoding="utf-8") as f:
        f.write(report)
    with open(os.path.join(CFG["out_dir"], "benchmark_v4_raw.json"), "w", encoding="utf-8") as f:
        json.dump([dict(name=r["name"], source=r["source"], synthetic=r["synthetic"], params=r["params"],
                        fold_metrics=r["fold_metrics"], controllers=r["table"].to_dict("records"),
                        contributions=r["contrib"].to_dict("records")) for r in results],
                  f, ensure_ascii=False, indent=2)
    print("\n" + report, flush=True)


if __name__ == "__main__":
    main()
