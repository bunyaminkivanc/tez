# Enerji Sistemleri İçin Yapay Zeka Tabanlı Optimizasyon ve Tahmine Dayalı Modelleme

Şırnak Üniversitesi Lisansüstü Eğitim Enstitüsü yüksek lisans tezinin deney kodları ve sonuç dosyaları.

Tezde önerilen çerçeve, Tahmin İsabetine Dayalı Kontrolcü Seçimi (FACS; Forecast-Accuracy-Based Controller Selection), iki katmanlıdır. Birinci katmanda Random Forest gelecek 24 saatin yükünü tahmin eder. İkinci katman, bu tahminin önceki katlardaki isabetine (R²) bakarak kararı ya Soft Actor-Critic (SAC) ajanına ya da her saat 24 saatlik doğrusal programı yeniden çözen model öngörülü kontrole (MPC) verir: R² ortalaması 0,5'in altındaysa SAC, değilse MPC. Ortam, şebekeye tek bir bağlantı noktasından bağlı toplu bir yük ve bataryadan oluşur. Bağlantı noktasındaki güç dengesi ve kapasite sınırı modellenir.

## Dosyalar

| Dosya | İşlev |
|---|---|
| `thesis_benchmark_stats_v6.py` | Tezin ana sonuçlarını üreten betik: kural tabanlı yönetim, MPC, teorik optimum, on tohumlu SAC, birleşik amaç, Holm düzeltmesi; dördüncü ve beşinci test katı |
| `thesis_benchmark_experiment_v4.py` | Veri hazırlama, tahmin katmanı, bağlantı noktası ve batarya ortamı ile SAC uygulaması; v6 betikleri bu modülü kullanır |
| `thesis_benchmark_stats_v5.py` | Ağaç tabanlı tahminler ve istatistik testleri; v6 bu betiğin işlevlerini kullanır |
| `thesis_benchmark_lambda_v6.py` | Duyarlılık deneyleri (beşinci kat): SAC'ın ceza katsayısı (`VIOL_PENALTY`), fiyat yapısı (`FIYAT`) ve batarya boyutu (`BATARYA_KAT`) değiştirilerek yeniden eğitilmesi |
| `thesis_benchmark_experiment_v2.py` | Beş derin öğrenme mimarisinin karşılaştırması (Tablo 6) |
| `thesis_benchmark_experiment.py`, `thesis_benchmark_experiment_v3.py` | Ağaç tabanlı referanslarla ilk tahmin karşılaştırması ve tahminsiz SAC ile CNN-BiLSTM + SAC'ın üç tohumla karşılaştırması (ara sürümler) |
| `thesis_benchmark_v7_outputs/` | Tezdeki ana koşunun ham sonuçları (`stats_v6_raw.json`), tohum bazında ölçümler, örneklem dışı tahminler (`oof_*.npz`) ve çalışma kaydı (`v7_full.log`) |
| `thesis_benchmark_v7_lambda/` | Ceza katsayısı duyarlılığı (Tablo 17) ve Şekil 6'nın saatlik izi |
| `thesis_benchmark_v7_tarife/` | Fiyat yapısı duyarlılığı (Tablo 18) |
| `thesis_benchmark_v7_batarya_x0.5/`, `thesis_benchmark_v7_batarya_x2/` | Batarya boyutu duyarlılığı (Tablo 19) |
| `thesis_benchmark_v6_outputs/` | Tahmin modellerinin eğitildiği ilk koşu; sonraki koşular örneklem dışı tahminleri buradan alır. Bu koşuda SAC'ın 168 saatlik bölüm sonu gerçek bir bitiş gibi ele alınmıştı; v7'de zaman sınırı olarak ele alınır (Pardo ve diğ., 2018). Kural, MPC ve teorik optimum iki koşuda aynıdır |
| `tablolari_uret_v6.py` | Tablo 7-16 ve FACS'ın seçimi |
| `sekilleri_uret.py` | Şekil 1-5 (Şekil 3-5 ham sonuç dosyasından, Şekil 2 Tablo 6 değerlerinden) |
| `lambda_tablosu.py`, `duyarlilik_tablosu.py`, `tek_gun_sekli.py` | Tablo 17-19 ve Şekil 6 |

## Çalıştırma

Ana sonuçlar Python 3.11 ile, `requirements.txt` içindeki sürümlerle üretilmiştir.

```bash
pip install -r requirements.txt
# ana koşu: iki test katı, on tohum; tahminler ilk koşudan alınır (OOF_FROM verilmezse yeniden eğitilir)
OOF_FROM=thesis_benchmark_v6_outputs OUT_DIR=thesis_benchmark_v7_outputs python thesis_benchmark_stats_v6.py
python tablolari_uret_v6.py > tablolar_v7.md     # Tablo 7-16, seçim eşiği 0,5
python sekilleri_uret.py                         # Şekil 1-5

# duyarlılık deneyleri: beşinci kat, SAC her durumda on tohumla yeniden eğitilir,
# tahminler thesis_benchmark_v6_outputs/ içinden okunur (SRC_DIR)
OUT_DIR=thesis_benchmark_v7_lambda VIOL_PENALTY=2  python thesis_benchmark_lambda_v6.py
OUT_DIR=thesis_benchmark_v7_lambda VIOL_PENALTY=10 python thesis_benchmark_lambda_v6.py
OUT_DIR=thesis_benchmark_v7_lambda SEEDS=1 ONLY=micro python thesis_benchmark_lambda_v6.py   # Şekil 6'nın izi
OUT_DIR=thesis_benchmark_v7_tarife FIYAT=tou    ONLY=macro python thesis_benchmark_lambda_v6.py
OUT_DIR=thesis_benchmark_v7_tarife FIYAT=piyasa ONLY=micro python thesis_benchmark_lambda_v6.py
OUT_DIR=thesis_benchmark_v7_tarife FIYAT=piyasa ONLY=meso  python thesis_benchmark_lambda_v6.py
OUT_DIR=thesis_benchmark_v7_batarya_x0.5 BATARYA_KAT=0.5 python thesis_benchmark_lambda_v6.py
OUT_DIR=thesis_benchmark_v7_batarya_x2   BATARYA_KAT=2   python thesis_benchmark_lambda_v6.py
python lambda_tablosu.py                         # Tablo 17
python duyarlilik_tablosu.py                     # Tablo 18 ve 19
python tek_gun_sekli.py                          # Şekil 6
```

`FIYAT=piyasa`, şebeke ölçeğinin DE-LU gün öncesi fiyatını konut ve bina yüküne saat sırasıyla eşler; takvim günleri ve gün içi saatler birebir örtüşmez. `BATARYA_KAT`, batarya kapasitesini ve gücünü birlikte ölçekler (dört saatlik süre değişmez). `ZAMAN_SINIRI_KES=1`, SAC eğitiminde bölüm sonunu ilk koşudaki gibi gerçek bitiş olarak ele alır.

Veri kümeleri ilk çalıştırmada indirilir: UCI Individual Household Electric Power Consumption, Building Data Genome Project 2 (`Robin_office_Gayle` sayacı) ve Open Power System Data (Almanya yükü, DE-LU gün öncesi fiyatı). Betik gerçek veri yüklenemezse sentetik veriye geçmez, durur.

Rastgelelik tohumları sabittir (42-51). Derin öğrenme sonuçları kayan nokta işlemlerinin sırası nedeniyle koşular arasında küçük farklar gösterebilir.
