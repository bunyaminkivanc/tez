# v6: düzeltilmiş kural, MPC, bitiş koşulu, birleşik amaç, Holm düzeltmesi

## Mikro-Konut (UCI) · test katı 5

Kaynak: UCI Household Power (gerçek) + örnekleyici 3 zamanlı tarife · test 2010-07-27 15:00:00 – 2010-11-25 22:00:00 (2912 saat)
Hat sınırı aşımı (bataryasız): eğitim %13.4, test %10.8 · kural k = 4 saat

```
                       Kontrolcü  Birleşik amaçta yakalanan (%)  Birleşik amaçta yakalanan (%) std  Birleşik amaçta yakalanan (%) min  Fatura tasarrufu / teorik optimum (%)  İhlal düşüşü (%)  Tasarruf (%)  Bitiş SoC
           Bataryasız [referans]                           0.00                               0.00                               0.00                                   0.00              0.00          0.00       0.50
               Rastgele politika                         -27.48                               4.29                             -33.78                                 -39.16            -13.85         -2.17       0.55
      Eski kural (yüzdelik eşik)                          17.25                               0.00                              17.25                                  22.03              9.18          1.22       0.34
           Kural tabanlı yönetim                          35.55                               0.00                              35.55                                  84.86             11.53          4.69       0.10
                MPC + CNN-BiLSTM                          37.00                               0.00                              37.00                                  87.00             12.24          4.81       0.34
             MPC + Random Forest                          44.32                               0.00                              44.32                                  89.79             17.37          4.97       0.34
                   MPC + XGBoost                          44.07                               0.00                              44.07                                  88.74             17.37          4.91       0.34
               MPC + naif tahmin                          48.58                               0.00                              48.58                                  92.48             20.15          5.12       0.10
MPC + kusursuz tahmin [referans]                         100.08                               0.00                             100.08                                 100.33             58.38          5.55       0.34
                 SAC (tahminsiz)                          49.73                               3.34                              44.46                                  58.92             27.31          3.26       0.12
             Random Forest + SAC                          56.37                               2.59                              52.99                                  65.53             31.19          3.63       0.10
                   XGBoost + SAC                          55.53                               2.06                              51.11                                  60.45             31.50          3.34       0.12
     CNN-BiLSTM + SAC [önerilen]                          49.68                               5.53                              36.45                                  68.91             25.40          3.81       0.10
       Teorik optimum [referans]                         100.00                               0.00                             100.00                                 100.00             58.38          5.53       0.50
```

```
                                      Karşılaştırma                                Metrik     fark   ci_alt   ci_ust  p_wilcoxon  p_holm  pozitif
CNN-BiLSTM + SAC [önerilen] − Kural tabanlı yönetim         Birleşik amaçta yakalanan (%)  14.1297  10.1770  18.0823      0.0020  0.0527       10
     CNN-BiLSTM + SAC [önerilen] − MPC + CNN-BiLSTM         Birleşik amaçta yakalanan (%)  12.6811   8.7285  16.6337      0.0039  0.0527        9
        CNN-BiLSTM + SAC [önerilen] − MPC + XGBoost         Birleşik amaçta yakalanan (%)   5.6075   1.6549   9.5601      0.0273  0.2734        8
    CNN-BiLSTM + SAC [önerilen] − Rastgele politika         Birleşik amaçta yakalanan (%)  77.1542  72.1433  82.1652      0.0020  0.0527       10
      CNN-BiLSTM + SAC [önerilen] − SAC (tahminsiz)         Birleşik amaçta yakalanan (%)  -0.0509  -4.7778   4.6760      0.4922  1.0000        7
  CNN-BiLSTM + SAC [önerilen] − Random Forest + SAC         Birleşik amaçta yakalanan (%)  -6.6888 -10.3863  -2.9914      0.0020  0.0527        0
        CNN-BiLSTM + SAC [önerilen] − XGBoost + SAC         Birleşik amaçta yakalanan (%)  -5.8538  -9.8964  -1.8112      0.0020  0.0527        0
            SAC (tahminsiz) − Kural tabanlı yönetim         Birleşik amaçta yakalanan (%)  14.1806  11.7903  16.5708      0.0020  0.0527       10
 Kural tabanlı yönetim − Eski kural (yüzdelik eşik)         Birleşik amaçta yakalanan (%)  18.2930      NaN      NaN         NaN  1.0000       10
CNN-BiLSTM + SAC [önerilen] − Kural tabanlı yönetim Fatura tasarrufu / teorik optimum (%) -15.9555 -24.4916  -7.4194      0.0020  0.0527        0
     CNN-BiLSTM + SAC [önerilen] − MPC + CNN-BiLSTM Fatura tasarrufu / teorik optimum (%) -18.0912 -26.6273  -9.5552      0.0020  0.0527        0
        CNN-BiLSTM + SAC [önerilen] − MPC + XGBoost Fatura tasarrufu / teorik optimum (%) -19.8368 -28.3729 -11.3008      0.0020  0.0527        0
    CNN-BiLSTM + SAC [önerilen] − Rastgele politika Fatura tasarrufu / teorik optimum (%) 108.0697  99.1723 116.9672      0.0020  0.0527       10
      CNN-BiLSTM + SAC [önerilen] − SAC (tahminsiz) Fatura tasarrufu / teorik optimum (%)   9.9849   0.6091  19.3606      0.0840  0.9238        9
  CNN-BiLSTM + SAC [önerilen] − Random Forest + SAC Fatura tasarrufu / teorik optimum (%)   3.3749  -4.2220  10.9717      0.1602  1.0000        8
        CNN-BiLSTM + SAC [önerilen] − XGBoost + SAC Fatura tasarrufu / teorik optimum (%)   8.4537  -1.6500  18.5573      0.1055  1.0000        8
            SAC (tahminsiz) − Kural tabanlı yönetim Fatura tasarrufu / teorik optimum (%) -25.9404 -30.8712 -21.0096      0.0020  0.0527        0
 Kural tabanlı yönetim − Eski kural (yüzdelik eşik) Fatura tasarrufu / teorik optimum (%)  62.8370      NaN      NaN         NaN  1.0000       10
CNN-BiLSTM + SAC [önerilen] − Kural tabanlı yönetim                      İhlal düşüşü (%)  13.8749  12.2433  15.5065      0.0020  0.0527       10
     CNN-BiLSTM + SAC [önerilen] − MPC + CNN-BiLSTM                      İhlal düşüşü (%)  13.1578  11.5262  14.7894      0.0020  0.0527       10
        CNN-BiLSTM + SAC [önerilen] − MPC + XGBoost                      İhlal düşüşü (%)   8.0321   6.4005   9.6637      0.0020  0.0527       10
    CNN-BiLSTM + SAC [önerilen] − Rastgele politika                      İhlal düşüşü (%)  39.2576  36.9059  41.6093      0.0020  0.0527       10
      CNN-BiLSTM + SAC [önerilen] − SAC (tahminsiz)                      İhlal düşüşü (%)  -1.9066  -4.1450   0.3317      0.0840  1.0000        2
  CNN-BiLSTM + SAC [önerilen] − Random Forest + SAC                      İhlal düşüşü (%)  -5.7868  -7.3671  -4.2065      0.0020  0.0527        0
        CNN-BiLSTM + SAC [önerilen] − XGBoost + SAC                      İhlal düşüşü (%)  -6.0930  -7.5659  -4.6201      0.0020  0.0527        0
            SAC (tahminsiz) − Kural tabanlı yönetim                      İhlal düşüşü (%)  15.7815  14.4953  17.0677      0.0020  0.0527       10
 Kural tabanlı yönetim − Eski kural (yüzdelik eşik)                      İhlal düşüşü (%)   2.3480   2.3480   2.3480      0.0020  0.0527       10
CNN-BiLSTM + SAC [önerilen] − Kural tabanlı yönetim                          Tasarruf (%)  -0.8827  -1.3549  -0.4104      0.0020  0.0527        0
     CNN-BiLSTM + SAC [önerilen] − MPC + CNN-BiLSTM                          Tasarruf (%)  -1.0008  -1.4730  -0.5286      0.0020  0.0527        0
        CNN-BiLSTM + SAC [önerilen] − MPC + XGBoost                          Tasarruf (%)  -1.0974  -1.5696  -0.6252      0.0020  0.0527        0
    CNN-BiLSTM + SAC [önerilen] − Rastgele politika                          Tasarruf (%)   5.9784   5.4862   6.4706      0.0020  0.0527       10
      CNN-BiLSTM + SAC [önerilen] − SAC (tahminsiz)                          Tasarruf (%)   0.5524   0.0337   1.0710      0.0840  0.8398        9
  CNN-BiLSTM + SAC [önerilen] − Random Forest + SAC                          Tasarruf (%)   0.1867  -0.2336   0.6070      0.1602  1.0000        8
        CNN-BiLSTM + SAC [önerilen] − XGBoost + SAC                          Tasarruf (%)   0.4677  -0.0913   1.0266      0.1055  0.9492        8
            SAC (tahminsiz) − Kural tabanlı yönetim                          Tasarruf (%)  -1.4350  -1.7078  -1.1623      0.0020  0.0527        0
 Kural tabanlı yönetim − Eski kural (yüzdelik eşik)                          Tasarruf (%)   3.4762      NaN      NaN         NaN  1.0000       10
```

## Mikro-Konut (UCI) · test katı 4

Kaynak: UCI Household Power (gerçek) + örnekleyici 3 zamanlı tarife · test 2010-03-28 07:00:00 – 2010-07-27 14:00:00 (2912 saat)
Hat sınırı aşımı (bataryasız): eğitim %13.5, test %7.1 · kural k = 4 saat

```
                       Kontrolcü  Birleşik amaçta yakalanan (%)  Birleşik amaçta yakalanan (%) std  Birleşik amaçta yakalanan (%) min  Fatura tasarrufu / teorik optimum (%)  İhlal düşüşü (%)  Tasarruf (%)  Bitiş SoC
           Bataryasız [referans]                           0.00                               0.00                               0.00                                   0.00              0.00          0.00       0.50
               Rastgele politika                         -34.56                               4.06                             -39.98                                 -35.79            -23.73         -2.00       0.55
      Eski kural (yüzdelik eşik)                           3.09                               0.00                               3.09                                  25.77             -5.67          1.44       0.10
           Kural tabanlı yönetim                          36.83                               0.00                              36.83                                  86.90              8.45          4.85       0.73
                MPC + CNN-BiLSTM                          33.16                               0.00                              33.16                                  93.51              2.35          5.22       0.90
             MPC + Random Forest                          33.60                               0.00                              33.60                                  91.49              3.50          5.10       0.90
                   MPC + XGBoost                          36.90                               0.00                              36.90                                  93.88              6.12          5.24       0.90
               MPC + naif tahmin                          41.47                               0.00                              41.47                                  89.98             12.24          5.02       0.90
MPC + kusursuz tahmin [referans]                          99.83                               0.00                              99.83                                  99.50             69.90          5.55       0.90
                 SAC (tahminsiz)                          38.27                               3.52                              33.71                                  58.86             19.64          3.28       0.21
             Random Forest + SAC                          37.87                               5.42                              29.79                                  50.32             22.17          2.81       0.48
                   XGBoost + SAC                          36.33                               3.75                              29.89                                  48.57             21.17          2.71       0.19
     CNN-BiLSTM + SAC [önerilen]                          39.27                               3.58                              34.24                                  65.70             18.32          3.66       0.43
       Teorik optimum [referans]                         100.00                               0.00                             100.00                                 100.00             69.90          5.58       0.50
```

```
                                      Karşılaştırma                                Metrik     fark   ci_alt   ci_ust  p_wilcoxon  p_holm  pozitif
CNN-BiLSTM + SAC [önerilen] − Kural tabanlı yönetim         Birleşik amaçta yakalanan (%)   2.4417  -0.1165   5.0000      0.1055  1.0000        7
     CNN-BiLSTM + SAC [önerilen] − MPC + CNN-BiLSTM         Birleşik amaçta yakalanan (%)   6.1062   3.5479   8.6644      0.0020  0.0527       10
        CNN-BiLSTM + SAC [önerilen] − MPC + XGBoost         Birleşik amaçta yakalanan (%)   2.3719  -0.1863   4.9302      0.1055  1.0000        7
    CNN-BiLSTM + SAC [önerilen] − Rastgele politika         Birleşik amaçta yakalanan (%)  73.8238  70.1958  77.4518      0.0020  0.0527       10
      CNN-BiLSTM + SAC [önerilen] − SAC (tahminsiz)         Birleşik amaçta yakalanan (%)   0.9964  -2.5206   4.5134      0.6250  1.0000        5
  CNN-BiLSTM + SAC [önerilen] − Random Forest + SAC         Birleşik amaçta yakalanan (%)   1.3989  -2.3923   5.1902      0.4922  1.0000        7
        CNN-BiLSTM + SAC [önerilen] − XGBoost + SAC         Birleşik amaçta yakalanan (%)   2.9357  -0.7630   6.6345      0.1602  1.0000        7
            SAC (tahminsiz) − Kural tabanlı yönetim         Birleşik amaçta yakalanan (%)   1.4453  -1.0717   3.9623      0.4316  1.0000        6
 Kural tabanlı yönetim − Eski kural (yüzdelik eşik)         Birleşik amaçta yakalanan (%)  33.7376      NaN      NaN         NaN  1.0000       10
CNN-BiLSTM + SAC [önerilen] − Kural tabanlı yönetim Fatura tasarrufu / teorik optimum (%) -21.2007 -25.9775 -16.4240      0.0020  0.0527        0
     CNN-BiLSTM + SAC [önerilen] − MPC + CNN-BiLSTM Fatura tasarrufu / teorik optimum (%) -27.8096 -32.5863 -23.0328      0.0020  0.0527        0
        CNN-BiLSTM + SAC [önerilen] − MPC + XGBoost Fatura tasarrufu / teorik optimum (%) -28.1819 -32.9586 -23.4051      0.0020  0.0527        0
    CNN-BiLSTM + SAC [önerilen] − Rastgele politika Fatura tasarrufu / teorik optimum (%) 101.4855  96.9026 106.0683      0.0020  0.0527       10
      CNN-BiLSTM + SAC [önerilen] − SAC (tahminsiz) Fatura tasarrufu / teorik optimum (%)   6.8342  -0.2241  13.8926      0.0840  0.4199        8
  CNN-BiLSTM + SAC [önerilen] − Random Forest + SAC Fatura tasarrufu / teorik optimum (%)  15.3818   6.1732  24.5904      0.0098  0.0781        9
        CNN-BiLSTM + SAC [önerilen] − XGBoost + SAC Fatura tasarrufu / teorik optimum (%)  17.1238  10.5952  23.6524      0.0020  0.0527       10
            SAC (tahminsiz) − Kural tabanlı yönetim Fatura tasarrufu / teorik optimum (%) -28.0350 -33.3358 -22.7341      0.0020  0.0527        0
 Kural tabanlı yönetim − Eski kural (yüzdelik eşik) Fatura tasarrufu / teorik optimum (%)  61.1329  61.1329  61.1329      0.0020  0.0527       10
CNN-BiLSTM + SAC [önerilen] − Kural tabanlı yönetim                      İhlal düşüşü (%)   9.8688   7.6463  12.0912      0.0020  0.0527       10
     CNN-BiLSTM + SAC [önerilen] − MPC + CNN-BiLSTM                      İhlal düşüşü (%)  15.9768  13.7544  18.1993      0.0020  0.0527       10
        CNN-BiLSTM + SAC [önerilen] − MPC + XGBoost                      İhlal düşüşü (%)  12.2060   9.9835  14.4284      0.0020  0.0527       10
    CNN-BiLSTM + SAC [önerilen] − Rastgele politika                      İhlal düşüşü (%)  42.0529  38.7893  45.3165      0.0020  0.0527       10
      CNN-BiLSTM + SAC [önerilen] − SAC (tahminsiz)                      İhlal düşüşü (%)  -1.3189  -4.0370   1.3992      0.3750  1.0000        4
  CNN-BiLSTM + SAC [önerilen] − Random Forest + SAC                      İhlal düşüşü (%)  -3.8494  -6.2821  -1.4167      0.0020  0.0527        0
        CNN-BiLSTM + SAC [önerilen] − XGBoost + SAC                      İhlal düşüşü (%)  -2.8460  -5.5842  -0.1079      0.0488  0.5371        2
            SAC (tahminsiz) − Kural tabanlı yönetim                      İhlal düşüşü (%)  11.1876   9.1230  13.2523      0.0020  0.0527       10
 Kural tabanlı yönetim − Eski kural (yüzdelik eşik)                      İhlal düşüşü (%)  14.1248  14.1248  14.1248      0.0020  0.0527       10
CNN-BiLSTM + SAC [önerilen] − Kural tabanlı yönetim                          Tasarruf (%)  -1.1827  -1.4491  -0.9162      0.0020  0.0527        0
     CNN-BiLSTM + SAC [önerilen] − MPC + CNN-BiLSTM                          Tasarruf (%)  -1.5513  -1.8178  -1.2849      0.0020  0.0527        0
        CNN-BiLSTM + SAC [önerilen] − MPC + XGBoost                          Tasarruf (%)  -1.5721  -1.8386  -1.3056      0.0020  0.0527        0
    CNN-BiLSTM + SAC [önerilen] − Rastgele politika                          Tasarruf (%)   5.6613   5.4056   5.9169      0.0020  0.0527       10
      CNN-BiLSTM + SAC [önerilen] − SAC (tahminsiz)                          Tasarruf (%)   0.3812  -0.0125   0.7750      0.0840  0.4199        8
  CNN-BiLSTM + SAC [önerilen] − Random Forest + SAC                          Tasarruf (%)   0.8581   0.3444   1.3717      0.0098  0.0781        9
        CNN-BiLSTM + SAC [önerilen] − XGBoost + SAC                          Tasarruf (%)   0.9552   0.5910   1.3194      0.0020  0.0527       10
            SAC (tahminsiz) − Kural tabanlı yönetim                          Tasarruf (%)  -1.5639  -1.8596  -1.2682      0.0020  0.0527        0
 Kural tabanlı yönetim − Eski kural (yüzdelik eşik)                          Tasarruf (%)   3.4102      NaN      NaN         NaN  1.0000       10
```

## Makro-Şebeke (ENTSO-E/EPİAŞ) · test katı 5

Kaynak: OPSD / ENTSO-E DE-LU (gerçek) · test 2020-05-31 16:00:00 – 2020-09-29 23:00:00 (2912 saat)
Hat sınırı aşımı (bataryasız): eğitim %12.3, test %0.1 · kural k = 4 saat

```
                       Kontrolcü  Birleşik amaçta yakalanan (%)  Birleşik amaçta yakalanan (%) std  Birleşik amaçta yakalanan (%) min  Fatura tasarrufu / teorik optimum (%)  İhlal düşüşü (%)  Tasarruf (%)  Bitiş SoC
           Bataryasız [referans]                           0.00                               0.00                               0.00                                   0.00              0.00          0.00       0.50
               Rastgele politika                         -26.77                               2.85                             -32.60                                 -27.32             -0.32         -0.06       0.55
      Eski kural (yüzdelik eşik)                          20.91                               0.00                              20.91                                  21.34              0.00          0.05       0.10
           Kural tabanlı yönetim                          71.55                               0.00                              71.55                                  73.04              0.00          0.16       0.10
                MPC + CNN-BiLSTM                          97.25                               0.00                              97.25                                 100.94            -79.54          0.22       0.19
             MPC + Random Forest                          98.18                               0.00                              98.18                                 100.87            -30.67          0.22       0.19
                   MPC + XGBoost                          98.50                               0.00                              98.50                                 100.86            -14.36          0.22       0.19
               MPC + naif tahmin                          97.12                               0.00                              97.12                                 100.81            -79.54          0.22       0.19
MPC + kusursuz tahmin [referans]                         100.71                               0.00                             100.71                                 100.72            100.00          0.22       0.19
                 SAC (tahminsiz)                          34.95                               7.67                              24.85                                  35.32             16.91          0.08       0.24
             Random Forest + SAC                          37.43                               7.24                              24.40                                  37.46             36.04          0.08       0.26
                   XGBoost + SAC                          36.88                               4.50                              27.87                                  36.83             39.53          0.08       0.33
     CNN-BiLSTM + SAC [önerilen]                          36.20                               5.23                              28.67                                  36.88              3.52          0.08       0.25
       Teorik optimum [referans]                         100.00                               0.00                             100.00                                 100.00            100.00          0.22       0.50
```

```
                                      Karşılaştırma                                Metrik     fark   ci_alt   ci_ust  p_wilcoxon  p_holm  pozitif
CNN-BiLSTM + SAC [önerilen] − Kural tabanlı yönetim         Birleşik amaçta yakalanan (%) -35.3484 -39.0901 -31.6067      0.0020  0.0527        0
     CNN-BiLSTM + SAC [önerilen] − MPC + CNN-BiLSTM         Birleşik amaçta yakalanan (%) -61.0547 -64.7963 -57.3130      0.0020  0.0527        0
        CNN-BiLSTM + SAC [önerilen] − MPC + XGBoost         Birleşik amaçta yakalanan (%) -62.3057 -66.0474 -58.5640      0.0020  0.0527        0
    CNN-BiLSTM + SAC [önerilen] − Rastgele politika         Birleşik amaçta yakalanan (%)  62.9715  58.2770  67.6660      0.0020  0.0527       10
      CNN-BiLSTM + SAC [önerilen] − SAC (tahminsiz)         Birleşik amaçta yakalanan (%)   1.2533  -3.2267   5.7334      0.6250  1.0000        6
  CNN-BiLSTM + SAC [önerilen] − Random Forest + SAC         Birleşik amaçta yakalanan (%)  -1.2342  -6.8462   4.3779      0.5566  1.0000        3
        CNN-BiLSTM + SAC [önerilen] − XGBoost + SAC         Birleşik amaçta yakalanan (%)  -0.6857  -5.7808   4.4094      0.9219  1.0000        6
            SAC (tahminsiz) − Kural tabanlı yönetim         Birleşik amaçta yakalanan (%) -36.6018 -42.0871 -31.1165      0.0020  0.0527        0
 Kural tabanlı yönetim − Eski kural (yüzdelik eşik)         Birleşik amaçta yakalanan (%)  50.6384  50.6384  50.6384      0.0020  0.0527       10
CNN-BiLSTM + SAC [önerilen] − Kural tabanlı yönetim Fatura tasarrufu / teorik optimum (%) -36.1590 -40.2129 -32.1052      0.0020  0.0527        0
     CNN-BiLSTM + SAC [önerilen] − MPC + CNN-BiLSTM Fatura tasarrufu / teorik optimum (%) -64.0603 -68.1141 -60.0064      0.0020  0.0527        0
        CNN-BiLSTM + SAC [önerilen] − MPC + XGBoost Fatura tasarrufu / teorik optimum (%) -63.9781 -68.0320 -59.9243      0.0020  0.0527        0
    CNN-BiLSTM + SAC [önerilen] − Rastgele politika Fatura tasarrufu / teorik optimum (%)  64.2047  58.9601  69.4492      0.0020  0.0527       10
      CNN-BiLSTM + SAC [önerilen] − SAC (tahminsiz) Fatura tasarrufu / teorik optimum (%)   1.5587  -2.5343   5.6517      0.4922  1.0000        6
  CNN-BiLSTM + SAC [önerilen] − Random Forest + SAC Fatura tasarrufu / teorik optimum (%)  -0.5817  -6.1982   5.0349      0.6250  1.0000        3
        CNN-BiLSTM + SAC [önerilen] − XGBoost + SAC Fatura tasarrufu / teorik optimum (%)   0.0510  -5.0357   5.1376      0.5566  1.0000        6
            SAC (tahminsiz) − Kural tabanlı yönetim Fatura tasarrufu / teorik optimum (%) -37.7177 -43.2328 -32.2027      0.0020  0.0527        0
 Kural tabanlı yönetim − Eski kural (yüzdelik eşik) Fatura tasarrufu / teorik optimum (%)  51.6945  51.6945  51.6945      0.0020  0.0527       10
CNN-BiLSTM + SAC [önerilen] − Kural tabanlı yönetim                      İhlal düşüşü (%)   3.5194 -34.7634  41.8022      0.4199  1.0000        8
     CNN-BiLSTM + SAC [önerilen] − MPC + CNN-BiLSTM                      İhlal düşüşü (%)  83.0611  44.7783 121.3439      0.0098  0.1465        8
        CNN-BiLSTM + SAC [önerilen] − MPC + XGBoost                      İhlal düşüşü (%)  17.8825 -20.4003  56.1653      0.4199  1.0000        8
    CNN-BiLSTM + SAC [önerilen] − Rastgele politika                      İhlal düşüşü (%)   3.8434 -52.7180  60.4048      0.8457  1.0000        4
      CNN-BiLSTM + SAC [önerilen] − SAC (tahminsiz)                      İhlal düşüşü (%) -13.3883 -55.0778  28.3013      0.4961  1.0000        4
  CNN-BiLSTM + SAC [önerilen] − Random Forest + SAC                      İhlal düşüşü (%) -32.5205 -68.9665   3.9255      0.1289  1.0000        2
        CNN-BiLSTM + SAC [önerilen] − XGBoost + SAC                      İhlal düşüşü (%) -36.0084 -59.3577 -12.6592      0.0137  0.1777        2
            SAC (tahminsiz) − Kural tabanlı yönetim                      İhlal düşüşü (%)  16.9077  -7.6137  41.4291      0.1895  1.0000        7
 Kural tabanlı yönetim − Eski kural (yüzdelik eşik)                      İhlal düşüşü (%)   0.0000      NaN      NaN         NaN  1.0000        0
CNN-BiLSTM + SAC [önerilen] − Kural tabanlı yönetim                          Tasarruf (%)  -0.0799  -0.0889  -0.0709      0.0020  0.0527        0
     CNN-BiLSTM + SAC [önerilen] − MPC + CNN-BiLSTM                          Tasarruf (%)  -0.1416  -0.1505  -0.1326      0.0020  0.0527        0
        CNN-BiLSTM + SAC [önerilen] − MPC + XGBoost                          Tasarruf (%)  -0.1414  -0.1503  -0.1324      0.0020  0.0527        0
    CNN-BiLSTM + SAC [önerilen] − Rastgele politika                          Tasarruf (%)   0.1419   0.1303   0.1535      0.0020  0.0527       10
      CNN-BiLSTM + SAC [önerilen] − SAC (tahminsiz)                          Tasarruf (%)   0.0034  -0.0056   0.0125      0.4922  1.0000        6
  CNN-BiLSTM + SAC [önerilen] − Random Forest + SAC                          Tasarruf (%)  -0.0013  -0.0137   0.0111      0.6250  1.0000        3
        CNN-BiLSTM + SAC [önerilen] − XGBoost + SAC                          Tasarruf (%)   0.0001  -0.0111   0.0114      0.5566  1.0000        6
            SAC (tahminsiz) − Kural tabanlı yönetim                          Tasarruf (%)  -0.0833  -0.0955  -0.0712      0.0020  0.0527        0
 Kural tabanlı yönetim − Eski kural (yüzdelik eşik)                          Tasarruf (%)   0.1142   0.1142   0.1142      0.0020  0.0527       10
```

## Makro-Şebeke (ENTSO-E/EPİAŞ) · test katı 4

Kaynak: OPSD / ENTSO-E DE-LU (gerçek) · test 2020-01-31 08:00:00 – 2020-05-31 15:00:00 (2912 saat)
Hat sınırı aşımı (bataryasız): eğitim %11.8, test %9.4 · kural k = 4 saat

```
                       Kontrolcü  Birleşik amaçta yakalanan (%)  Birleşik amaçta yakalanan (%) std  Birleşik amaçta yakalanan (%) min  Fatura tasarrufu / teorik optimum (%)  İhlal düşüşü (%)  Tasarruf (%)  Bitiş SoC
           Bataryasız [referans]                           0.00                               0.00                               0.00                                   0.00              0.00          0.00       0.50
               Rastgele politika                         -19.76                               4.36                             -25.71                                 -23.58             -3.27         -0.08       0.55
      Eski kural (yüzdelik eşik)                          15.63                               0.00                              15.63                                  12.20              3.17          0.04       0.90
           Kural tabanlı yönetim                          52.54                               0.00                              52.54                                  71.50              7.89          0.23       0.90
                MPC + CNN-BiLSTM                          77.42                               0.00                              77.42                                 100.89             12.04          0.33       0.90
             MPC + Random Forest                          84.19                               0.00                              84.19                                 101.13             13.86          0.33       0.90
                   MPC + XGBoost                          84.97                               0.00                              84.97                                 100.19             14.16          0.33       0.90
               MPC + naif tahmin                          75.03                               0.00                              75.03                                  99.59             11.50          0.32       0.90
MPC + kusursuz tahmin [referans]                         100.08                               0.00                             100.08                                 100.26             18.29          0.33       0.90
                 SAC (tahminsiz)                          57.88                               3.87                              48.30                                  40.07             12.19          0.13       0.79
             Random Forest + SAC                          56.24                               4.86                              47.00                                  41.60             11.61          0.14       0.85
                   XGBoost + SAC                          58.89                               2.70                              52.97                                  44.01             12.11          0.14       0.85
     CNN-BiLSTM + SAC [önerilen]                          56.56                               4.46                              51.26                                  43.46             11.52          0.14       0.81
       Teorik optimum [referans]                         100.00                               0.00                             100.00                                 100.00             18.29          0.33       0.90
```

```
                                      Karşılaştırma                                Metrik     fark   ci_alt   ci_ust  p_wilcoxon  p_holm  pozitif
CNN-BiLSTM + SAC [önerilen] − Kural tabanlı yönetim         Birleşik amaçta yakalanan (%)   4.0157   0.8254   7.2059      0.0488  0.7324        7
     CNN-BiLSTM + SAC [önerilen] − MPC + CNN-BiLSTM         Birleşik amaçta yakalanan (%) -20.8673 -24.0575 -17.6770      0.0020  0.0527        0
        CNN-BiLSTM + SAC [önerilen] − MPC + XGBoost         Birleşik amaçta yakalanan (%) -28.4164 -31.6066 -25.2261      0.0020  0.0527        0
    CNN-BiLSTM + SAC [önerilen] − Rastgele politika         Birleşik amaçta yakalanan (%)  76.3193  72.0500  80.5886      0.0020  0.0527       10
      CNN-BiLSTM + SAC [önerilen] − SAC (tahminsiz)         Birleşik amaçta yakalanan (%)  -1.3209  -4.7350   2.0932      0.6250  1.0000        5
  CNN-BiLSTM + SAC [önerilen] − Random Forest + SAC         Birleşik amaçta yakalanan (%)   0.3142  -1.9850   2.6134      0.9219  1.0000        5
        CNN-BiLSTM + SAC [önerilen] − XGBoost + SAC         Birleşik amaçta yakalanan (%)  -2.3322  -5.2917   0.6274      0.2754  1.0000        4
            SAC (tahminsiz) − Kural tabanlı yönetim         Birleşik amaçta yakalanan (%)   5.3366   2.5716   8.1016      0.0098  0.1562        9
 Kural tabanlı yönetim − Eski kural (yüzdelik eşik)         Birleşik amaçta yakalanan (%)  36.9127  36.9127  36.9127      0.0020  0.0527       10
CNN-BiLSTM + SAC [önerilen] − Kural tabanlı yönetim Fatura tasarrufu / teorik optimum (%) -28.0473 -34.0162 -22.0783      0.0020  0.0527        0
     CNN-BiLSTM + SAC [önerilen] − MPC + CNN-BiLSTM Fatura tasarrufu / teorik optimum (%) -57.4295 -63.3984 -51.4606      0.0020  0.0527        0
        CNN-BiLSTM + SAC [önerilen] − MPC + XGBoost Fatura tasarrufu / teorik optimum (%) -56.7374 -62.7064 -50.7685      0.0020  0.0527        0
    CNN-BiLSTM + SAC [önerilen] − Rastgele politika Fatura tasarrufu / teorik optimum (%)  67.0413  60.3290  73.7536      0.0020  0.0527       10
      CNN-BiLSTM + SAC [önerilen] − SAC (tahminsiz) Fatura tasarrufu / teorik optimum (%)   3.3909  -2.1126   8.8945      0.1602  0.6406        7
  CNN-BiLSTM + SAC [önerilen] − Random Forest + SAC Fatura tasarrufu / teorik optimum (%)   1.8604  -1.9947   5.7154      0.3223  0.9668        7
        CNN-BiLSTM + SAC [önerilen] − XGBoost + SAC Fatura tasarrufu / teorik optimum (%)  -0.5564  -7.1122   5.9994      0.9219  1.0000        6
            SAC (tahminsiz) − Kural tabanlı yönetim Fatura tasarrufu / teorik optimum (%) -31.4382 -35.3360 -27.5404      0.0020  0.0527        0
 Kural tabanlı yönetim − Eski kural (yüzdelik eşik) Fatura tasarrufu / teorik optimum (%)  59.3054      NaN      NaN         NaN  1.0000       10
CNN-BiLSTM + SAC [önerilen] − Kural tabanlı yönetim                      İhlal düşüşü (%)   3.6299   2.6283   4.6315      0.0020  0.0527       10
     CNN-BiLSTM + SAC [önerilen] − MPC + CNN-BiLSTM                      İhlal düşüşü (%)  -0.5138  -1.5154   0.4878      0.3223  1.0000        4
        CNN-BiLSTM + SAC [önerilen] − MPC + XGBoost                      İhlal düşüşü (%)  -2.6384  -3.6400  -1.6368      0.0020  0.0527        0
    CNN-BiLSTM + SAC [önerilen] − Rastgele politika                      İhlal düşüşü (%)  14.7933  13.5747  16.0120      0.0020  0.0527       10
      CNN-BiLSTM + SAC [önerilen] − SAC (tahminsiz)                      İhlal düşüşü (%)  -0.6671  -1.5261   0.1920      0.1602  1.0000        4
  CNN-BiLSTM + SAC [önerilen] − Random Forest + SAC                      İhlal düşüşü (%)  -0.0822  -0.7233   0.5589      0.6953  1.0000        4
        CNN-BiLSTM + SAC [önerilen] − XGBoost + SAC                      İhlal düşüşü (%)  -0.5868  -1.2137   0.0400      0.0645  0.6445        3
            SAC (tahminsiz) − Kural tabanlı yönetim                      İhlal düşüşü (%)   4.2970   3.5253   5.0686      0.0020  0.0527       10
 Kural tabanlı yönetim − Eski kural (yüzdelik eşik)                      İhlal düşüşü (%)   4.7274      NaN      NaN         NaN  1.0000       10
CNN-BiLSTM + SAC [önerilen] − Kural tabanlı yönetim                          Tasarruf (%)  -0.0912  -0.1106  -0.0718      0.0020  0.0527        0
     CNN-BiLSTM + SAC [önerilen] − MPC + CNN-BiLSTM                          Tasarruf (%)  -0.1867  -0.2061  -0.1673      0.0020  0.0527        0
        CNN-BiLSTM + SAC [önerilen] − MPC + XGBoost                          Tasarruf (%)  -0.1845  -0.2039  -0.1651      0.0020  0.0527        0
    CNN-BiLSTM + SAC [önerilen] − Rastgele politika                          Tasarruf (%)   0.2180   0.1961   0.2398      0.0020  0.0527       10
      CNN-BiLSTM + SAC [önerilen] − SAC (tahminsiz)                          Tasarruf (%)   0.0110  -0.0069   0.0289      0.1602  0.6406        7
  CNN-BiLSTM + SAC [önerilen] − Random Forest + SAC                          Tasarruf (%)   0.0060  -0.0065   0.0186      0.3223  0.9668        7
        CNN-BiLSTM + SAC [önerilen] − XGBoost + SAC                          Tasarruf (%)  -0.0018  -0.0231   0.0195      0.9219  1.0000        6
            SAC (tahminsiz) − Kural tabanlı yönetim                          Tasarruf (%)  -0.1022  -0.1149  -0.0895      0.0020  0.0527        0
 Kural tabanlı yönetim − Eski kural (yüzdelik eşik)                          Tasarruf (%)   0.1928   0.1928   0.1928      0.0020  0.0527       10
```

## Mezo-Bina (ASHRAE/BDG2) · test katı 5

Kaynak: Yerel CSV (meso.csv, ör. ASHRAE) + örnekleyici 3 zamanlı tarife · test 2017-08-31 17:00:00 – 2017-12-31 00:00:00 (2912 saat)
Hat sınırı aşımı (bataryasız): eğitim %16.2, test %15.0 · kural k = 4 saat

```
                       Kontrolcü  Birleşik amaçta yakalanan (%)  Birleşik amaçta yakalanan (%) std  Birleşik amaçta yakalanan (%) min  Fatura tasarrufu / teorik optimum (%)  İhlal düşüşü (%)  Tasarruf (%)  Bitiş SoC
           Bataryasız [referans]                           0.00                               0.00                               0.00                                   0.00              0.00          0.00       0.50
               Rastgele politika                         -41.88                               5.86                             -50.37                                 -31.50            -27.80         -1.37       0.55
      Eski kural (yüzdelik eşik)                           9.15                               0.00                               9.15                                  41.40             -0.93          1.80       0.81
           Kural tabanlı yönetim                          53.05                               0.00                              53.05                                 122.52             18.44          5.32       0.34
                MPC + CNN-BiLSTM                          88.62                               0.00                              88.62                                  99.54             52.14          4.32       0.34
             MPC + Random Forest                          91.69                               0.00                              91.69                                  99.77             54.61          4.33       0.34
                   MPC + XGBoost                          91.60                               0.00                              91.60                                  99.71             54.55          4.33       0.34
               MPC + naif tahmin                          75.76                               0.00                              75.76                                  99.88             41.58          4.34       0.34
MPC + kusursuz tahmin [referans]                         100.02                               0.00                             100.02                                 100.07             61.34          4.35       0.43
                 SAC (tahminsiz)                          82.16                               2.69                              77.04                                  52.43             56.43          2.28       0.22
             Random Forest + SAC                          80.67                               4.28                              72.98                                  54.04             54.89          2.35       0.22
                   XGBoost + SAC                          79.81                               4.27                              72.41                                  53.42             54.31          2.32       0.12
     CNN-BiLSTM + SAC [önerilen]                          80.14                               2.88                              75.93                                  53.25             54.62          2.31       0.20
       Teorik optimum [referans]                         100.00                               0.00                             100.00                                 100.00             61.34          4.34       0.50
```

```
                                      Karşılaştırma                                Metrik     fark   ci_alt   ci_ust  p_wilcoxon  p_holm  pozitif
CNN-BiLSTM + SAC [önerilen] − Kural tabanlı yönetim         Birleşik amaçta yakalanan (%)  27.0961  25.0338  29.1583      0.0020  0.0527       10
     CNN-BiLSTM + SAC [önerilen] − MPC + CNN-BiLSTM         Birleşik amaçta yakalanan (%)  -8.4749 -10.5371  -6.4126      0.0020  0.0527        0
        CNN-BiLSTM + SAC [önerilen] − MPC + XGBoost         Birleşik amaçta yakalanan (%) -11.4581 -13.5204  -9.3959      0.0020  0.0527        0
    CNN-BiLSTM + SAC [önerilen] − Rastgele politika         Birleşik amaçta yakalanan (%) 122.0256 117.6832 126.3680      0.0020  0.0527       10
      CNN-BiLSTM + SAC [önerilen] − SAC (tahminsiz)         Birleşik amaçta yakalanan (%)  -2.0169  -3.9402  -0.0935      0.0488  0.4395        3
  CNN-BiLSTM + SAC [önerilen] − Random Forest + SAC         Birleşik amaçta yakalanan (%)  -0.5240  -3.0477   1.9996      0.3750  1.0000        3
        CNN-BiLSTM + SAC [önerilen] − XGBoost + SAC         Birleşik amaçta yakalanan (%)   0.3353  -2.2929   2.9634      0.9219  1.0000        6
            SAC (tahminsiz) − Kural tabanlı yönetim         Birleşik amaçta yakalanan (%)  29.1130  27.1857  31.0402      0.0020  0.0527       10
 Kural tabanlı yönetim − Eski kural (yüzdelik eşik)         Birleşik amaçta yakalanan (%)  43.8924      NaN      NaN         NaN  1.0000       10
CNN-BiLSTM + SAC [önerilen] − Kural tabanlı yönetim Fatura tasarrufu / teorik optimum (%) -69.2638 -73.9855 -64.5420      0.0020  0.0527        0
     CNN-BiLSTM + SAC [önerilen] − MPC + CNN-BiLSTM Fatura tasarrufu / teorik optimum (%) -46.2906 -51.0123 -41.5689      0.0020  0.0527        0
        CNN-BiLSTM + SAC [önerilen] − MPC + XGBoost Fatura tasarrufu / teorik optimum (%) -46.4575 -51.1793 -41.7358      0.0020  0.0527        0
    CNN-BiLSTM + SAC [önerilen] − Rastgele politika Fatura tasarrufu / teorik optimum (%)  84.7520  78.7941  90.7099      0.0020  0.0527       10
      CNN-BiLSTM + SAC [önerilen] − SAC (tahminsiz) Fatura tasarrufu / teorik optimum (%)   0.8272  -4.3047   5.9590      0.6953  1.0000        6
  CNN-BiLSTM + SAC [önerilen] − Random Forest + SAC Fatura tasarrufu / teorik optimum (%)  -0.7908  -7.3257   5.7442      0.8457  1.0000        6
        CNN-BiLSTM + SAC [önerilen] − XGBoost + SAC Fatura tasarrufu / teorik optimum (%)  -0.1660  -6.5268   6.1948      0.7695  1.0000        5
            SAC (tahminsiz) − Kural tabanlı yönetim Fatura tasarrufu / teorik optimum (%) -70.0909 -75.1169 -65.0650      0.0020  0.0527        0
 Kural tabanlı yönetim − Eski kural (yüzdelik eşik) Fatura tasarrufu / teorik optimum (%)  81.1142      NaN      NaN         NaN  1.0000       10
CNN-BiLSTM + SAC [önerilen] − Kural tabanlı yönetim                      İhlal düşüşü (%)  36.1775  35.1290  37.2260      0.0020  0.0527       10
     CNN-BiLSTM + SAC [önerilen] − MPC + CNN-BiLSTM                      İhlal düşüşü (%)   2.4757   1.4272   3.5242      0.0039  0.0625        9
        CNN-BiLSTM + SAC [önerilen] − MPC + XGBoost                      İhlal düşüşü (%)   0.0741  -0.9744   1.1226      0.6953  1.0000        5
    CNN-BiLSTM + SAC [önerilen] − Rastgele politika                      İhlal düşüşü (%)  82.4196  79.0810  85.7582      0.0020  0.0527       10
      CNN-BiLSTM + SAC [önerilen] − SAC (tahminsiz)                      İhlal düşüşü (%)  -1.8144  -2.8472  -0.7817      0.0098  0.1465        2
  CNN-BiLSTM + SAC [önerilen] − Random Forest + SAC                      İhlal düşüşü (%)  -0.2673  -1.7966   1.2619      0.4922  1.0000        4
        CNN-BiLSTM + SAC [önerilen] − XGBoost + SAC                      İhlal düşüşü (%)   0.3074  -1.4545   2.0693      0.8457  1.0000        6
            SAC (tahminsiz) − Kural tabanlı yönetim                      İhlal düşüşü (%)  37.9919  37.0411  38.9427      0.0020  0.0527       10
 Kural tabanlı yönetim − Eski kural (yüzdelik eşik)                      İhlal düşüşü (%)  19.3713      NaN      NaN         NaN  1.0000       10
CNN-BiLSTM + SAC [önerilen] − Kural tabanlı yönetim                          Tasarruf (%)  -3.0077  -3.2127  -2.8026      0.0020  0.0527        0
     CNN-BiLSTM + SAC [önerilen] − MPC + CNN-BiLSTM                          Tasarruf (%)  -2.0101  -2.2151  -1.8051      0.0020  0.0527        0
        CNN-BiLSTM + SAC [önerilen] − MPC + XGBoost                          Tasarruf (%)  -2.0173  -2.2224  -1.8123      0.0020  0.0527        0
    CNN-BiLSTM + SAC [önerilen] − Rastgele politika                          Tasarruf (%)   3.6802   3.4215   3.9389      0.0020  0.0527       10
      CNN-BiLSTM + SAC [önerilen] − SAC (tahminsiz)                          Tasarruf (%)   0.0359  -0.1869   0.2588      0.6953  1.0000        6
  CNN-BiLSTM + SAC [önerilen] − Random Forest + SAC                          Tasarruf (%)  -0.0343  -0.3181   0.2494      0.8457  1.0000        6
        CNN-BiLSTM + SAC [önerilen] − XGBoost + SAC                          Tasarruf (%)  -0.0072  -0.2834   0.2690      0.7695  1.0000        5
            SAC (tahminsiz) − Kural tabanlı yönetim                          Tasarruf (%)  -3.0436  -3.2618  -2.8253      0.0020  0.0527        0
 Kural tabanlı yönetim − Eski kural (yüzdelik eşik)                          Tasarruf (%)   3.5222   3.5222   3.5222      0.0020  0.0527       10
```

## Mezo-Bina (ASHRAE/BDG2) · test katı 4

Kaynak: Yerel CSV (meso.csv, ör. ASHRAE) + örnekleyici 3 zamanlı tarife · test 2017-05-02 09:00:00 – 2017-08-31 16:00:00 (2912 saat)
Hat sınırı aşımı (bataryasız): eğitim %16.6, test %18.2 · kural k = 4 saat

```
                       Kontrolcü  Birleşik amaçta yakalanan (%)  Birleşik amaçta yakalanan (%) std  Birleşik amaçta yakalanan (%) min  Fatura tasarrufu / teorik optimum (%)  İhlal düşüşü (%)  Tasarruf (%)  Bitiş SoC
           Bataryasız [referans]                           0.00                               0.00                               0.00                                   0.00              0.00          0.00       0.50
               Rastgele politika                         -51.93                               5.30                             -59.51                                 -34.95            -38.42         -1.46       0.55
      Eski kural (yüzdelik eşik)                           9.67                               0.00                               9.67                                  43.50             -0.42          1.81       0.10
           Kural tabanlı yönetim                          48.45                               0.00                              48.45                                 127.63             16.40          5.32       0.90
                MPC + CNN-BiLSTM                          74.66                               0.00                              74.66                                 104.61             44.11          4.36       0.90
             MPC + Random Forest                          89.93                               0.00                              89.93                                  98.32             58.80          4.10       0.90
                   MPC + XGBoost                          90.36                               0.00                              90.36                                  97.97             59.25          4.09       0.90
               MPC + naif tahmin                          72.45                               0.00                              72.45                                  99.33             43.25          4.14       0.90
MPC + kusursuz tahmin [referans]                          99.86                               0.00                              99.86                                  99.40             67.29          4.15       0.90
                 SAC (tahminsiz)                          78.38                               3.22                              74.69                                  53.67             57.80          2.24       0.16
             Random Forest + SAC                          80.49                               6.31                              67.05                                  53.06             59.77          2.21       0.25
                   XGBoost + SAC                          80.75                               2.38                              77.25                                  50.96             60.43          2.13       0.39
     CNN-BiLSTM + SAC [önerilen]                          81.56                               1.80                              78.45                                  60.46             59.20          2.52       0.31
       Teorik optimum [referans]                         100.00                               0.00                             100.00                                 100.00             67.29          4.17       0.50
```

```
                                      Karşılaştırma                                Metrik     fark   ci_alt   ci_ust  p_wilcoxon  p_holm  pozitif
CNN-BiLSTM + SAC [önerilen] − Kural tabanlı yönetim         Birleşik amaçta yakalanan (%)  33.1148  31.8292  34.4004      0.0020  0.0527       10
     CNN-BiLSTM + SAC [önerilen] − MPC + CNN-BiLSTM         Birleşik amaçta yakalanan (%)   6.9025   5.6169   8.1881      0.0020  0.0527       10
        CNN-BiLSTM + SAC [önerilen] − MPC + XGBoost         Birleşik amaçta yakalanan (%)  -8.7941 -10.0797  -7.5085      0.0020  0.0527        0
    CNN-BiLSTM + SAC [önerilen] − Rastgele politika         Birleşik amaçta yakalanan (%) 133.4953 128.6791 138.3114      0.0020  0.0527       10
      CNN-BiLSTM + SAC [önerilen] − SAC (tahminsiz)         Birleşik amaçta yakalanan (%)   3.1835   0.3933   5.9736      0.0488  0.7324        8
  CNN-BiLSTM + SAC [önerilen] − Random Forest + SAC         Birleşik amaçta yakalanan (%)   1.0762  -3.2428   5.3953      1.0000  1.0000        4
        CNN-BiLSTM + SAC [önerilen] − XGBoost + SAC         Birleşik amaçta yakalanan (%)   0.8169  -1.5568   3.1906      0.3223  1.0000        8
            SAC (tahminsiz) − Kural tabanlı yönetim         Birleşik amaçta yakalanan (%)  29.9313  27.6270  32.2356      0.0020  0.0527       10
 Kural tabanlı yönetim − Eski kural (yüzdelik eşik)         Birleşik amaçta yakalanan (%)  38.7791      NaN      NaN         NaN  1.0000       10
CNN-BiLSTM + SAC [önerilen] − Kural tabanlı yönetim Fatura tasarrufu / teorik optimum (%) -67.1740 -69.9325 -64.4154      0.0020  0.0527        0
     CNN-BiLSTM + SAC [önerilen] − MPC + CNN-BiLSTM Fatura tasarrufu / teorik optimum (%) -44.1558 -46.9143 -41.3973      0.0020  0.0527        0
        CNN-BiLSTM + SAC [önerilen] − MPC + XGBoost Fatura tasarrufu / teorik optimum (%) -37.5130 -40.2716 -34.7545      0.0020  0.0527        0
    CNN-BiLSTM + SAC [önerilen] − Rastgele politika Fatura tasarrufu / teorik optimum (%)  95.4103  90.6893 100.1314      0.0020  0.0527       10
      CNN-BiLSTM + SAC [önerilen] − SAC (tahminsiz) Fatura tasarrufu / teorik optimum (%)   6.7843  -0.0541  13.6226      0.0488  0.3418        8
  CNN-BiLSTM + SAC [önerilen] − Random Forest + SAC Fatura tasarrufu / teorik optimum (%)   7.3999  -1.5686  16.3684      0.0488  0.3418        9
        CNN-BiLSTM + SAC [önerilen] − XGBoost + SAC Fatura tasarrufu / teorik optimum (%)   9.4945   3.7966  15.1923      0.0059  0.0527        9
            SAC (tahminsiz) − Kural tabanlı yönetim Fatura tasarrufu / teorik optimum (%) -73.9582 -80.0571 -67.8594      0.0020  0.0527        0
 Kural tabanlı yönetim − Eski kural (yüzdelik eşik) Fatura tasarrufu / teorik optimum (%)  84.1347  84.1347  84.1347      0.0020  0.0527       10
CNN-BiLSTM + SAC [önerilen] − Kural tabanlı yönetim                      İhlal düşüşü (%)  42.8053  41.8541  43.7564      0.0020  0.0527       10
     CNN-BiLSTM + SAC [önerilen] − MPC + CNN-BiLSTM                      İhlal düşüşü (%)  15.0928  14.1416  16.0439      0.0020  0.0527       10
        CNN-BiLSTM + SAC [önerilen] − MPC + XGBoost                      İhlal düşüşü (%)  -0.0410  -0.9921   0.9102      0.8457  1.0000        5
    CNN-BiLSTM + SAC [önerilen] − Rastgele politika                      İhlal düşüşü (%)  97.6238  93.8809 101.3667      0.0020  0.0527       10
      CNN-BiLSTM + SAC [önerilen] − SAC (tahminsiz)                      İhlal düşüşü (%)   1.4054  -0.3112   3.1219      0.1055  0.8438        7
  CNN-BiLSTM + SAC [önerilen] − Random Forest + SAC                      İhlal düşüşü (%)  -0.5698  -3.0343   1.8947      0.4922  1.0000        3
        CNN-BiLSTM + SAC [önerilen] − XGBoost + SAC                      İhlal düşüşü (%)  -1.2260  -2.4295  -0.0224      0.0645  0.6445        2
            SAC (tahminsiz) − Kural tabanlı yönetim                      İhlal düşüşü (%)  41.3999  40.0884  42.7114      0.0020  0.0527       10
 Kural tabanlı yönetim − Eski kural (yüzdelik eşik)                      İhlal düşüşü (%)  16.8139  16.8139  16.8139      0.0020  0.0527       10
CNN-BiLSTM + SAC [önerilen] − Kural tabanlı yönetim                          Tasarruf (%)  -2.8022  -2.9173  -2.6872      0.0020  0.0527        0
     CNN-BiLSTM + SAC [önerilen] − MPC + CNN-BiLSTM                          Tasarruf (%)  -1.8420  -1.9571  -1.7269      0.0020  0.0527        0
        CNN-BiLSTM + SAC [önerilen] − MPC + XGBoost                          Tasarruf (%)  -1.5649  -1.6800  -1.4498      0.0020  0.0527        0
    CNN-BiLSTM + SAC [önerilen] − Rastgele politika                          Tasarruf (%)   3.9802   3.7832   4.1771      0.0020  0.0527       10
      CNN-BiLSTM + SAC [önerilen] − SAC (tahminsiz)                          Tasarruf (%)   0.2830  -0.0023   0.5683      0.0488  0.3418        8
  CNN-BiLSTM + SAC [önerilen] − Random Forest + SAC                          Tasarruf (%)   0.3087  -0.0654   0.6828      0.0488  0.3418        9
        CNN-BiLSTM + SAC [önerilen] − XGBoost + SAC                          Tasarruf (%)   0.3961   0.1584   0.6338      0.0059  0.0527        9
            SAC (tahminsiz) − Kural tabanlı yönetim                          Tasarruf (%)  -3.0853  -3.3397  -2.8308      0.0020  0.0527        0
 Kural tabanlı yönetim − Eski kural (yüzdelik eşik)                          Tasarruf (%)   3.5098   3.5098   3.5098      0.0020  0.0527       10
```
