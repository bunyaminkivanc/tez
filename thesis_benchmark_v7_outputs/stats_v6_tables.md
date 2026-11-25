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
                 SAC (tahminsiz)                          54.79                               2.16                              51.36                                  68.23             29.47          3.77       0.14
             Random Forest + SAC                          59.34                               2.30                              54.42                                  72.58             32.17          4.02       0.10
                   XGBoost + SAC                          59.72                               1.14                              57.61                                  69.38             33.05          3.84       0.12
     CNN-BiLSTM + SAC [önerilen]                          56.35                               1.60                              53.90                                  81.09             28.27          4.49       0.10
       Teorik optimum [referans]                         100.00                               0.00                             100.00                                 100.00             58.38          5.53       0.50
```

```
                                      Karşılaştırma                                Metrik     fark   ci_alt   ci_ust  p_wilcoxon  p_holm  pozitif
CNN-BiLSTM + SAC [önerilen] − Kural tabanlı yönetim         Birleşik amaçta yakalanan (%)  20.8002  19.6571  21.9433      0.0020  0.0469       10
     CNN-BiLSTM + SAC [önerilen] − MPC + CNN-BiLSTM         Birleşik amaçta yakalanan (%)  19.3517  18.2086  20.4947      0.0020  0.0469       10
        CNN-BiLSTM + SAC [önerilen] − MPC + XGBoost         Birleşik amaçta yakalanan (%)  12.2780  11.1350  13.4211      0.0020  0.0469       10
    CNN-BiLSTM + SAC [önerilen] − Rastgele politika         Birleşik amaçta yakalanan (%)  83.8248  80.2705  87.3791      0.0020  0.0469       10
      CNN-BiLSTM + SAC [önerilen] − SAC (tahminsiz)         Birleşik amaçta yakalanan (%)   1.5590  -0.6716   3.7897      0.1602  0.9609        7
  CNN-BiLSTM + SAC [önerilen] − Random Forest + SAC         Birleşik amaçta yakalanan (%)  -2.9938  -4.6473  -1.3402      0.0098  0.0781        2
        CNN-BiLSTM + SAC [önerilen] − XGBoost + SAC         Birleşik amaçta yakalanan (%)  -3.3719  -4.6288  -2.1150      0.0020  0.0469        0
            SAC (tahminsiz) − Kural tabanlı yönetim         Birleşik amaçta yakalanan (%)  19.2412  17.6928  20.7895      0.0020  0.0469       10
 Kural tabanlı yönetim − Eski kural (yüzdelik eşik)         Birleşik amaçta yakalanan (%)  18.2930      NaN      NaN         NaN     NaN       10
CNN-BiLSTM + SAC [önerilen] − Kural tabanlı yönetim Fatura tasarrufu / teorik optimum (%)  -3.7708  -6.9610  -0.5807      0.0195  0.1367        1
     CNN-BiLSTM + SAC [önerilen] − MPC + CNN-BiLSTM Fatura tasarrufu / teorik optimum (%)  -5.9066  -9.0968  -2.7164      0.0039  0.0469        1
        CNN-BiLSTM + SAC [önerilen] − MPC + XGBoost Fatura tasarrufu / teorik optimum (%)  -7.6522 -10.8424  -4.4620      0.0020  0.0469        0
    CNN-BiLSTM + SAC [önerilen] − Rastgele politika Fatura tasarrufu / teorik optimum (%) 120.2544 116.7069 123.8019      0.0020  0.0469       10
      CNN-BiLSTM + SAC [önerilen] − SAC (tahminsiz) Fatura tasarrufu / teorik optimum (%)  12.8624   8.2440  17.4809      0.0020  0.0469       10
  CNN-BiLSTM + SAC [önerilen] − Random Forest + SAC Fatura tasarrufu / teorik optimum (%)   8.5133   4.4758  12.5508      0.0020  0.0469       10
        CNN-BiLSTM + SAC [önerilen] − XGBoost + SAC Fatura tasarrufu / teorik optimum (%)  11.7075   8.0295  15.3856      0.0020  0.0469       10
            SAC (tahminsiz) − Kural tabanlı yönetim Fatura tasarrufu / teorik optimum (%) -16.6333 -18.6193 -14.6473      0.0020  0.0469        0
 Kural tabanlı yönetim − Eski kural (yüzdelik eşik) Fatura tasarrufu / teorik optimum (%)  62.8370      NaN      NaN         NaN     NaN       10
CNN-BiLSTM + SAC [önerilen] − Kural tabanlı yönetim                      İhlal düşüşü (%)  16.7376  16.0302  17.4450      0.0020  0.0469       10
     CNN-BiLSTM + SAC [önerilen] − MPC + CNN-BiLSTM                      İhlal düşüşü (%)  16.0205  15.3131  16.7279      0.0020  0.0469       10
        CNN-BiLSTM + SAC [önerilen] − MPC + XGBoost                      İhlal düşüşü (%)  10.8948  10.1874  11.6022      0.0020  0.0469       10
    CNN-BiLSTM + SAC [önerilen] − Rastgele politika                      İhlal düşüşü (%)  42.1203  39.7248  44.5159      0.0020  0.0469       10
      CNN-BiLSTM + SAC [önerilen] − SAC (tahminsiz)                      İhlal düşüşü (%)  -1.2039  -2.4954   0.0876      0.0840  0.7559        3
  CNN-BiLSTM + SAC [önerilen] − Random Forest + SAC                      İhlal düşüşü (%)  -3.8997  -4.7010  -3.0984      0.0020  0.0469        0
        CNN-BiLSTM + SAC [önerilen] − XGBoost + SAC                      İhlal düşüşü (%)  -4.7886  -5.7284  -3.8488      0.0020  0.0469        0
            SAC (tahminsiz) − Kural tabanlı yönetim                      İhlal düşüşü (%)  17.9415  16.9491  18.9339      0.0020  0.0469       10
 Kural tabanlı yönetim − Eski kural (yüzdelik eşik)                      İhlal düşüşü (%)   2.3480   2.3480   2.3480      0.0020     NaN       10
CNN-BiLSTM + SAC [önerilen] − Kural tabanlı yönetim                          Tasarruf (%)  -0.2086  -0.3851  -0.0321      0.0195  0.1367        1
     CNN-BiLSTM + SAC [önerilen] − MPC + CNN-BiLSTM                          Tasarruf (%)  -0.3268  -0.5032  -0.1503      0.0039  0.0469        1
        CNN-BiLSTM + SAC [önerilen] − MPC + XGBoost                          Tasarruf (%)  -0.4233  -0.5998  -0.2468      0.0020  0.0469        0
    CNN-BiLSTM + SAC [önerilen] − Rastgele politika                          Tasarruf (%)   6.6525   6.4562   6.8487      0.0020  0.0469       10
      CNN-BiLSTM + SAC [önerilen] − SAC (tahminsiz)                          Tasarruf (%)   0.7116   0.4561   0.9670      0.0020  0.0469       10
  CNN-BiLSTM + SAC [önerilen] − Random Forest + SAC                          Tasarruf (%)   0.4710   0.2476   0.6943      0.0020  0.0469       10
        CNN-BiLSTM + SAC [önerilen] − XGBoost + SAC                          Tasarruf (%)   0.6477   0.4442   0.8511      0.0020  0.0469       10
            SAC (tahminsiz) − Kural tabanlı yönetim                          Tasarruf (%)  -0.9202  -1.0300  -0.8103      0.0020  0.0469        0
 Kural tabanlı yönetim − Eski kural (yüzdelik eşik)                          Tasarruf (%)   3.4762      NaN      NaN         NaN     NaN       10
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
                 SAC (tahminsiz)                          42.37                               4.72                              35.14                                  65.92             21.48          3.68       0.56
             Random Forest + SAC                          46.61                               4.32                              38.76                                  70.32             24.40          3.92       0.48
                   XGBoost + SAC                          43.85                               4.81                              33.50                                  64.73             23.45          3.61       0.39
     CNN-BiLSTM + SAC [önerilen]                          43.12                               4.34                              33.71                                  73.56             19.63          4.10       0.47
       Teorik optimum [referans]                         100.00                               0.00                             100.00                                 100.00             69.90          5.58       0.50
```

```
                                      Karşılaştırma                                Metrik     fark   ci_alt   ci_ust  p_wilcoxon  p_holm  pozitif
CNN-BiLSTM + SAC [önerilen] − Kural tabanlı yönetim         Birleşik amaçta yakalanan (%)   6.2959   3.1893   9.4025      0.0059  0.0703        9
     CNN-BiLSTM + SAC [önerilen] − MPC + CNN-BiLSTM         Birleşik amaçta yakalanan (%)   9.9604   6.8538  13.0670      0.0020  0.0469       10
        CNN-BiLSTM + SAC [önerilen] − MPC + XGBoost         Birleşik amaçta yakalanan (%)   6.2261   3.1195   9.3327      0.0059  0.0703        9
    CNN-BiLSTM + SAC [önerilen] − Rastgele politika         Birleşik amaçta yakalanan (%)  77.6780  74.2953  81.0608      0.0020  0.0469       10
      CNN-BiLSTM + SAC [önerilen] − SAC (tahminsiz)         Birleşik amaçta yakalanan (%)   0.7543  -3.5099   5.0185      0.6250  1.0000        5
  CNN-BiLSTM + SAC [önerilen] − Random Forest + SAC         Birleşik amaçta yakalanan (%)  -3.4928  -8.6486   1.6629      0.2324  1.0000        4
        CNN-BiLSTM + SAC [önerilen] − XGBoost + SAC         Birleşik amaçta yakalanan (%)  -0.7316  -5.5139   4.0506      0.4922  1.0000        4
            SAC (tahminsiz) − Kural tabanlı yönetim         Birleşik amaçta yakalanan (%)   5.5416   2.1638   8.9194      0.0098  0.0977        9
 Kural tabanlı yönetim − Eski kural (yüzdelik eşik)         Birleşik amaçta yakalanan (%)  33.7376      NaN      NaN         NaN     NaN       10
CNN-BiLSTM + SAC [önerilen] − Kural tabanlı yönetim Fatura tasarrufu / teorik optimum (%) -13.3384 -17.8777  -8.7991      0.0020  0.0469        0
     CNN-BiLSTM + SAC [önerilen] − MPC + CNN-BiLSTM Fatura tasarrufu / teorik optimum (%) -19.9473 -24.4866 -15.4080      0.0020  0.0469        0
        CNN-BiLSTM + SAC [önerilen] − MPC + XGBoost Fatura tasarrufu / teorik optimum (%) -20.3196 -24.8589 -15.7803      0.0020  0.0469        0
    CNN-BiLSTM + SAC [önerilen] − Rastgele politika Fatura tasarrufu / teorik optimum (%) 109.3478 105.7294 112.9661      0.0020  0.0469       10
      CNN-BiLSTM + SAC [önerilen] − SAC (tahminsiz) Fatura tasarrufu / teorik optimum (%)   7.6395  -0.0223  15.3013      0.0840  0.5156        7
  CNN-BiLSTM + SAC [önerilen] − Random Forest + SAC Fatura tasarrufu / teorik optimum (%)   3.2395  -6.6732  13.1521      0.7695  1.0000        5
        CNN-BiLSTM + SAC [önerilen] − XGBoost + SAC Fatura tasarrufu / teorik optimum (%)   8.8329   0.0944  17.5714      0.0645  0.5156        9
            SAC (tahminsiz) − Kural tabanlı yönetim Fatura tasarrufu / teorik optimum (%) -20.9780 -26.1084 -15.8475      0.0020  0.0469        0
 Kural tabanlı yönetim − Eski kural (yüzdelik eşik) Fatura tasarrufu / teorik optimum (%)  61.1329  61.1329  61.1329      0.0020     NaN       10
CNN-BiLSTM + SAC [önerilen] − Kural tabanlı yönetim                      İhlal düşüşü (%)  11.1791   9.1121  13.2461      0.0020  0.0469       10
     CNN-BiLSTM + SAC [önerilen] − MPC + CNN-BiLSTM                      İhlal düşüşü (%)  17.2872  15.2202  19.3542      0.0020  0.0469       10
        CNN-BiLSTM + SAC [önerilen] − MPC + XGBoost                      İhlal düşüşü (%)  13.5163  11.4493  15.5833      0.0020  0.0469       10
    CNN-BiLSTM + SAC [önerilen] − Rastgele politika                      İhlal düşüşü (%)  43.3632  40.5731  46.1534      0.0020  0.0469       10
      CNN-BiLSTM + SAC [önerilen] − SAC (tahminsiz)                      İhlal düşüşü (%)  -1.8497  -3.8173   0.1178      0.1055  0.8438        2
  CNN-BiLSTM + SAC [önerilen] − Random Forest + SAC                      İhlal düşüşü (%)  -4.7656  -8.1536  -1.3777      0.0195  0.2148        2
        CNN-BiLSTM + SAC [önerilen] − XGBoost + SAC                      İhlal düşüşü (%)  -3.8134  -6.9523  -0.6745      0.0488  0.4883        1
            SAC (tahminsiz) − Kural tabanlı yönetim                      İhlal düşüşü (%)  13.0288  11.0847  14.9730      0.0020  0.0469       10
 Kural tabanlı yönetim − Eski kural (yüzdelik eşik)                      İhlal düşüşü (%)  14.1248  14.1248  14.1248      0.0020     NaN       10
CNN-BiLSTM + SAC [önerilen] − Kural tabanlı yönetim                          Tasarruf (%)  -0.7441  -0.9973  -0.4908      0.0020  0.0469        0
     CNN-BiLSTM + SAC [önerilen] − MPC + CNN-BiLSTM                          Tasarruf (%)  -1.1127  -1.3660  -0.8595      0.0020  0.0469        0
        CNN-BiLSTM + SAC [önerilen] − MPC + XGBoost                          Tasarruf (%)  -1.1335  -1.3867  -0.8803      0.0020  0.0469        0
    CNN-BiLSTM + SAC [önerilen] − Rastgele politika                          Tasarruf (%)   6.0998   5.8980   6.3017      0.0020  0.0469       10
      CNN-BiLSTM + SAC [önerilen] − SAC (tahminsiz)                          Tasarruf (%)   0.4262  -0.0012   0.8536      0.0840  0.5156        7
  CNN-BiLSTM + SAC [önerilen] − Random Forest + SAC                          Tasarruf (%)   0.1807  -0.3723   0.7337      0.7695  1.0000        5
        CNN-BiLSTM + SAC [önerilen] − XGBoost + SAC                          Tasarruf (%)   0.4927   0.0053   0.9802      0.0645  0.5156        9
            SAC (tahminsiz) − Kural tabanlı yönetim                          Tasarruf (%)  -1.1702  -1.4564  -0.8840      0.0020  0.0469        0
 Kural tabanlı yönetim − Eski kural (yüzdelik eşik)                          Tasarruf (%)   3.4102      NaN      NaN         NaN     NaN       10
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
                 SAC (tahminsiz)                          40.33                               6.67                              31.04                                  40.99              8.33          0.09       0.31
             Random Forest + SAC                          44.27                               8.44                              34.53                                  44.35             40.46          0.10       0.34
                   XGBoost + SAC                          43.31                               6.30                              34.74                                  43.11             53.31          0.10       0.37
     CNN-BiLSTM + SAC [önerilen]                          40.88                               8.08                              30.56                                  41.12             29.35          0.09       0.33
       Teorik optimum [referans]                         100.00                               0.00                             100.00                                 100.00            100.00          0.22       0.50
```

```
                                      Karşılaştırma                                Metrik     fark   ci_alt   ci_ust  p_wilcoxon  p_holm  pozitif
CNN-BiLSTM + SAC [önerilen] − Kural tabanlı yönetim         Birleşik amaçta yakalanan (%) -30.6642 -36.4441 -24.8842      0.0020  0.0469        0
     CNN-BiLSTM + SAC [önerilen] − MPC + CNN-BiLSTM         Birleşik amaçta yakalanan (%) -56.3704 -62.1504 -50.5905      0.0020  0.0469        0
        CNN-BiLSTM + SAC [önerilen] − MPC + XGBoost         Birleşik amaçta yakalanan (%) -57.6215 -63.4015 -51.8415      0.0020  0.0469        0
    CNN-BiLSTM + SAC [önerilen] − Rastgele politika         Birleşik amaçta yakalanan (%)  67.6558  61.3535  73.9580      0.0020  0.0469       10
      CNN-BiLSTM + SAC [önerilen] − SAC (tahminsiz)         Birleşik amaçta yakalanan (%)   0.5561  -6.0653   7.1775      0.8457  1.0000        5
  CNN-BiLSTM + SAC [önerilen] − Random Forest + SAC         Birleşik amaçta yakalanan (%)  -3.3884 -10.2004   3.4237      0.3223  1.0000        3
        CNN-BiLSTM + SAC [önerilen] − XGBoost + SAC         Birleşik amaçta yakalanan (%)  -2.4311  -6.9250   2.0627      0.2324  1.0000        4
            SAC (tahminsiz) − Kural tabanlı yönetim         Birleşik amaçta yakalanan (%) -31.2203 -35.9926 -26.4479      0.0020  0.0469        0
 Kural tabanlı yönetim − Eski kural (yüzdelik eşik)         Birleşik amaçta yakalanan (%)  50.6384  50.6384  50.6384      0.0020     NaN       10
CNN-BiLSTM + SAC [önerilen] − Kural tabanlı yönetim Fatura tasarrufu / teorik optimum (%) -31.9158 -37.7214 -26.1102      0.0020  0.0469        0
     CNN-BiLSTM + SAC [önerilen] − MPC + CNN-BiLSTM Fatura tasarrufu / teorik optimum (%) -59.8171 -65.6227 -54.0115      0.0020  0.0469        0
        CNN-BiLSTM + SAC [önerilen] − MPC + XGBoost Fatura tasarrufu / teorik optimum (%) -59.7349 -65.5405 -53.9293      0.0020  0.0469        0
    CNN-BiLSTM + SAC [önerilen] − Rastgele politika Fatura tasarrufu / teorik optimum (%)  68.4479  62.1110  74.7848      0.0020  0.0469       10
      CNN-BiLSTM + SAC [önerilen] − SAC (tahminsiz) Fatura tasarrufu / teorik optimum (%)   0.1292  -7.1241   7.3825      1.0000  1.0000        5
  CNN-BiLSTM + SAC [önerilen] − Random Forest + SAC Fatura tasarrufu / teorik optimum (%)  -3.2274  -9.8795   3.4248      0.4316  1.0000        4
        CNN-BiLSTM + SAC [önerilen] − XGBoost + SAC Fatura tasarrufu / teorik optimum (%)  -1.9820  -6.1805   2.2164      0.4316  1.0000        4
            SAC (tahminsiz) − Kural tabanlı yönetim Fatura tasarrufu / teorik optimum (%) -32.0450 -36.7768 -27.3133      0.0020  0.0469        0
 Kural tabanlı yönetim − Eski kural (yüzdelik eşik) Fatura tasarrufu / teorik optimum (%)  51.6945  51.6945  51.6945      0.0020     NaN       10
CNN-BiLSTM + SAC [önerilen] − Kural tabanlı yönetim                      İhlal düşüşü (%)  29.3505  -6.6802  65.3811      0.0645  0.6445        8
     CNN-BiLSTM + SAC [önerilen] − MPC + CNN-BiLSTM                      İhlal düşüşü (%) 108.8922  72.8615 144.9228      0.0020  0.0469       10
        CNN-BiLSTM + SAC [önerilen] − MPC + XGBoost                      İhlal düşüşü (%)  43.7136   7.6829  79.7442      0.0273  0.3008        9
    CNN-BiLSTM + SAC [önerilen] − Rastgele politika                      İhlal düşüşü (%)  29.6745  -7.1878  66.5367      0.1055  0.8438        7
      CNN-BiLSTM + SAC [önerilen] − SAC (tahminsiz)                      İhlal düşüşü (%)  21.0248 -47.2705  89.3201      0.9219  1.0000        5
  CNN-BiLSTM + SAC [önerilen] − Random Forest + SAC                      İhlal düşüşü (%) -11.1083 -46.0776  23.8610      0.6250  1.0000        4
        CNN-BiLSTM + SAC [önerilen] − XGBoost + SAC                      İhlal düşüşü (%) -23.9639 -58.6923  10.7644      0.1602  1.0000        3
            SAC (tahminsiz) − Kural tabanlı yönetim                      İhlal düşüşü (%)   8.3257 -53.1166  69.7680      0.5703  1.0000        7
 Kural tabanlı yönetim − Eski kural (yüzdelik eşik)                      İhlal düşüşü (%)   0.0000      NaN      NaN         NaN     NaN        0
CNN-BiLSTM + SAC [önerilen] − Kural tabanlı yönetim                          Tasarruf (%)  -0.0705  -0.0834  -0.0577      0.0020  0.0469        0
     CNN-BiLSTM + SAC [önerilen] − MPC + CNN-BiLSTM                          Tasarruf (%)  -0.1322  -0.1450  -0.1194      0.0020  0.0469        0
        CNN-BiLSTM + SAC [önerilen] − MPC + XGBoost                          Tasarruf (%)  -0.1320  -0.1448  -0.1192      0.0020  0.0469        0
    CNN-BiLSTM + SAC [önerilen] − Rastgele politika                          Tasarruf (%)   0.1513   0.1372   0.1653      0.0020  0.0469       10
      CNN-BiLSTM + SAC [önerilen] − SAC (tahminsiz)                          Tasarruf (%)   0.0003  -0.0157   0.0163      1.0000  1.0000        5
  CNN-BiLSTM + SAC [önerilen] − Random Forest + SAC                          Tasarruf (%)  -0.0071  -0.0218   0.0076      0.4316  1.0000        4
        CNN-BiLSTM + SAC [önerilen] − XGBoost + SAC                          Tasarruf (%)  -0.0044  -0.0137   0.0049      0.4316  1.0000        4
            SAC (tahminsiz) − Kural tabanlı yönetim                          Tasarruf (%)  -0.0708  -0.0813  -0.0604      0.0020  0.0469        0
 Kural tabanlı yönetim − Eski kural (yüzdelik eşik)                          Tasarruf (%)   0.1142   0.1142   0.1142      0.0020     NaN       10
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
                 SAC (tahminsiz)                          62.64                               2.70                              57.77                                  47.20             12.85          0.15       0.86
             Random Forest + SAC                          61.65                               4.05                              55.11                                  44.48             12.82          0.14       0.84
                   XGBoost + SAC                          62.54                               3.57                              56.88                                  47.54             12.79          0.15       0.84
     CNN-BiLSTM + SAC [önerilen]                          61.49                               2.67                              56.78                                  47.19             12.54          0.15       0.82
       Teorik optimum [referans]                         100.00                               0.00                             100.00                                 100.00             18.29          0.33       0.90
```

```
                                      Karşılaştırma                                Metrik     fark   ci_alt   ci_ust  p_wilcoxon  p_holm  pozitif
CNN-BiLSTM + SAC [önerilen] − Kural tabanlı yönetim         Birleşik amaçta yakalanan (%)   8.9525   7.0443  10.8607      0.0020  0.0469       10
     CNN-BiLSTM + SAC [önerilen] − MPC + CNN-BiLSTM         Birleşik amaçta yakalanan (%) -15.9304 -17.8387 -14.0222      0.0020  0.0469        0
        CNN-BiLSTM + SAC [önerilen] − MPC + XGBoost         Birleşik amaçta yakalanan (%) -23.4795 -25.3878 -21.5713      0.0020  0.0469        0
    CNN-BiLSTM + SAC [önerilen] − Rastgele politika         Birleşik amaçta yakalanan (%)  81.2561  77.5210  84.9913      0.0020  0.0469       10
      CNN-BiLSTM + SAC [önerilen] − SAC (tahminsiz)         Birleşik amaçta yakalanan (%)  -1.1429  -3.9128   1.6270      0.3750  1.0000        3
  CNN-BiLSTM + SAC [önerilen] − Random Forest + SAC         Birleşik amaçta yakalanan (%)  -0.1560  -2.9062   2.5943      0.8457  1.0000        4
        CNN-BiLSTM + SAC [önerilen] − XGBoost + SAC         Birleşik amaçta yakalanan (%)  -1.0459  -3.4980   1.4063      0.3750  1.0000        4
            SAC (tahminsiz) − Kural tabanlı yönetim         Birleşik amaçta yakalanan (%)  10.0954   8.1673  12.0236      0.0020  0.0469       10
 Kural tabanlı yönetim − Eski kural (yüzdelik eşik)         Birleşik amaçta yakalanan (%)  36.9127  36.9127  36.9127      0.0020     NaN       10
CNN-BiLSTM + SAC [önerilen] − Kural tabanlı yönetim Fatura tasarrufu / teorik optimum (%) -24.3134 -30.1506 -18.4762      0.0020  0.0469        0
     CNN-BiLSTM + SAC [önerilen] − MPC + CNN-BiLSTM Fatura tasarrufu / teorik optimum (%) -53.6956 -59.5329 -47.8584      0.0020  0.0469        0
        CNN-BiLSTM + SAC [önerilen] − MPC + XGBoost Fatura tasarrufu / teorik optimum (%) -53.0036 -58.8408 -47.1663      0.0020  0.0469        0
    CNN-BiLSTM + SAC [önerilen] − Rastgele politika Fatura tasarrufu / teorik optimum (%)  70.7751  64.2894  77.2609      0.0020  0.0469       10
      CNN-BiLSTM + SAC [önerilen] − SAC (tahminsiz) Fatura tasarrufu / teorik optimum (%)  -0.0072  -7.0913   7.0768      0.9219  1.0000        6
  CNN-BiLSTM + SAC [önerilen] − Random Forest + SAC Fatura tasarrufu / teorik optimum (%)   2.7067  -5.8864  11.2997      0.6250  1.0000        6
        CNN-BiLSTM + SAC [önerilen] − XGBoost + SAC Fatura tasarrufu / teorik optimum (%)  -0.3479  -8.2063   7.5105      1.0000  1.0000        6
            SAC (tahminsiz) − Kural tabanlı yönetim Fatura tasarrufu / teorik optimum (%) -24.3062 -30.4750 -18.1373      0.0020  0.0469        0
 Kural tabanlı yönetim − Eski kural (yüzdelik eşik) Fatura tasarrufu / teorik optimum (%)  59.3054      NaN      NaN         NaN     NaN       10
CNN-BiLSTM + SAC [önerilen] − Kural tabanlı yönetim                      İhlal düşüşü (%)   4.6413   3.8428   5.4398      0.0020  0.0469       10
     CNN-BiLSTM + SAC [önerilen] − MPC + CNN-BiLSTM                      İhlal düşüşü (%)   0.4976  -0.3009   1.2961      0.2754  1.0000        7
        CNN-BiLSTM + SAC [önerilen] − MPC + XGBoost                      İhlal düşüşü (%)  -1.6271  -2.4256  -0.8286      0.0020  0.0469        0
    CNN-BiLSTM + SAC [önerilen] − Rastgele politika                      İhlal düşüşü (%)  15.8047  14.6115  16.9979      0.0020  0.0469       10
      CNN-BiLSTM + SAC [önerilen] − SAC (tahminsiz)                      İhlal düşüşü (%)  -0.3116  -1.3073   0.6842      0.4922  1.0000        5
  CNN-BiLSTM + SAC [önerilen] − Random Forest + SAC                      İhlal düşüşü (%)  -0.2870  -1.0336   0.4596      1.0000  1.0000        7
        CNN-BiLSTM + SAC [önerilen] − XGBoost + SAC                      İhlal düşüşü (%)  -0.2543  -1.1549   0.6463      0.4316  1.0000        4
            SAC (tahminsiz) − Kural tabanlı yönetim                      İhlal düşüşü (%)   4.9528   4.3538   5.5518      0.0020  0.0469       10
 Kural tabanlı yönetim − Eski kural (yüzdelik eşik)                      İhlal düşüşü (%)   4.7274      NaN      NaN         NaN     NaN       10
CNN-BiLSTM + SAC [önerilen] − Kural tabanlı yönetim                          Tasarruf (%)  -0.0790  -0.0980  -0.0601      0.0020  0.0469        0
     CNN-BiLSTM + SAC [önerilen] − MPC + CNN-BiLSTM                          Tasarruf (%)  -0.1746  -0.1936  -0.1556      0.0020  0.0469        0
        CNN-BiLSTM + SAC [önerilen] − MPC + XGBoost                          Tasarruf (%)  -0.1723  -0.1913  -0.1534      0.0020  0.0469        0
    CNN-BiLSTM + SAC [önerilen] − Rastgele politika                          Tasarruf (%)   0.2301   0.2090   0.2512      0.0020  0.0469       10
      CNN-BiLSTM + SAC [önerilen] − SAC (tahminsiz)                          Tasarruf (%)  -0.0000  -0.0231   0.0230      0.9219  1.0000        6
  CNN-BiLSTM + SAC [önerilen] − Random Forest + SAC                          Tasarruf (%)   0.0088  -0.0191   0.0367      0.6250  1.0000        6
        CNN-BiLSTM + SAC [önerilen] − XGBoost + SAC                          Tasarruf (%)  -0.0011  -0.0267   0.0244      1.0000  1.0000        6
            SAC (tahminsiz) − Kural tabanlı yönetim                          Tasarruf (%)  -0.0790  -0.0991  -0.0590      0.0020  0.0469        0
 Kural tabanlı yönetim − Eski kural (yüzdelik eşik)                          Tasarruf (%)   0.1928   0.1928   0.1928      0.0020     NaN       10
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
                 SAC (tahminsiz)                          86.98                               2.53                              83.43                                  68.09             57.19          2.96       0.41
             Random Forest + SAC                          86.31                               2.09                              83.26                                  68.69             56.53          2.98       0.29
                   XGBoost + SAC                          87.90                               1.25                              86.06                                  72.30             57.09          3.14       0.15
     CNN-BiLSTM + SAC [önerilen]                          85.85                               2.23                              81.58                                  68.52             56.18          2.98       0.31
       Teorik optimum [referans]                         100.00                               0.00                             100.00                                 100.00             61.34          4.34       0.50
```

```
                                      Karşılaştırma                                Metrik     fark   ci_alt   ci_ust  p_wilcoxon  p_holm  pozitif
CNN-BiLSTM + SAC [önerilen] − Kural tabanlı yönetim         Birleşik amaçta yakalanan (%)  32.8005  31.2062  34.3947      0.0020  0.0469       10
     CNN-BiLSTM + SAC [önerilen] − MPC + CNN-BiLSTM         Birleşik amaçta yakalanan (%)  -2.7705  -4.3648  -1.1763      0.0020  0.0469        0
        CNN-BiLSTM + SAC [önerilen] − MPC + XGBoost         Birleşik amaçta yakalanan (%)  -5.7537  -7.3480  -4.1595      0.0020  0.0469        0
    CNN-BiLSTM + SAC [önerilen] − Rastgele politika         Birleşik amaçta yakalanan (%) 127.7300 123.2190 132.2409      0.0020  0.0469       10
      CNN-BiLSTM + SAC [önerilen] − SAC (tahminsiz)         Birleşik amaçta yakalanan (%)  -1.1293  -3.0161   0.7575      0.2324  1.0000        3
  CNN-BiLSTM + SAC [önerilen] − Random Forest + SAC         Birleşik amaçta yakalanan (%)  -0.4676  -2.1279   1.1928      0.4922  1.0000        3
        CNN-BiLSTM + SAC [önerilen] − XGBoost + SAC         Birleşik amaçta yakalanan (%)  -2.0544  -3.7270  -0.3818      0.0195  0.1367        2
            SAC (tahminsiz) − Kural tabanlı yönetim         Birleşik amaçta yakalanan (%)  33.9297  32.1198  35.7397      0.0020  0.0469       10
 Kural tabanlı yönetim − Eski kural (yüzdelik eşik)         Birleşik amaçta yakalanan (%)  43.8924      NaN      NaN         NaN     NaN       10
CNN-BiLSTM + SAC [önerilen] − Kural tabanlı yönetim Fatura tasarrufu / teorik optimum (%) -53.9971 -58.5178 -49.4764      0.0020  0.0469        0
     CNN-BiLSTM + SAC [önerilen] − MPC + CNN-BiLSTM Fatura tasarrufu / teorik optimum (%) -31.0239 -35.5446 -26.5032      0.0020  0.0469        0
        CNN-BiLSTM + SAC [önerilen] − MPC + XGBoost Fatura tasarrufu / teorik optimum (%) -31.1908 -35.7115 -26.6701      0.0020  0.0469        0
    CNN-BiLSTM + SAC [önerilen] − Rastgele politika Fatura tasarrufu / teorik optimum (%) 100.0187  94.1149 105.9226      0.0020  0.0469       10
      CNN-BiLSTM + SAC [önerilen] − SAC (tahminsiz) Fatura tasarrufu / teorik optimum (%)   0.4339  -4.1591   5.0269      1.0000  1.0000        5
  CNN-BiLSTM + SAC [önerilen] − Random Forest + SAC Fatura tasarrufu / teorik optimum (%)  -0.1665  -3.7037   3.3707      1.0000  1.0000        6
        CNN-BiLSTM + SAC [önerilen] − XGBoost + SAC Fatura tasarrufu / teorik optimum (%)  -3.7817  -9.0535   1.4901      0.1934  1.0000        4
            SAC (tahminsiz) − Kural tabanlı yönetim Fatura tasarrufu / teorik optimum (%) -54.4310 -59.8929 -48.9690      0.0020  0.0469        0
 Kural tabanlı yönetim − Eski kural (yüzdelik eşik) Fatura tasarrufu / teorik optimum (%)  81.1142      NaN      NaN         NaN     NaN       10
CNN-BiLSTM + SAC [önerilen] − Kural tabanlı yönetim                      İhlal düşüşü (%)  37.7362  37.0466  38.4257      0.0020  0.0469       10
     CNN-BiLSTM + SAC [önerilen] − MPC + CNN-BiLSTM                      İhlal düşüşü (%)   4.0343   3.3448   4.7239      0.0020  0.0469       10
        CNN-BiLSTM + SAC [önerilen] − MPC + XGBoost                      İhlal düşüşü (%)   1.6328   0.9432   2.3223      0.0020  0.0469       10
    CNN-BiLSTM + SAC [önerilen] − Rastgele politika                      İhlal düşüşü (%)  83.9783  80.2916  87.6650      0.0020  0.0469       10
      CNN-BiLSTM + SAC [önerilen] − SAC (tahminsiz)                      İhlal düşüşü (%)  -1.0100  -2.0359   0.0160      0.1934  1.0000        4
  CNN-BiLSTM + SAC [önerilen] − Random Forest + SAC                      İhlal düşüşü (%)  -0.3479  -1.6834   0.9875      0.4922  1.0000        4
        CNN-BiLSTM + SAC [önerilen] − XGBoost + SAC                      İhlal düşüşü (%)  -0.9097  -2.1038   0.2844      0.1934  1.0000        4
            SAC (tahminsiz) − Kural tabanlı yönetim                      İhlal düşüşü (%)  38.7461  37.9575  39.5348      0.0020  0.0469       10
 Kural tabanlı yönetim − Eski kural (yüzdelik eşik)                      İhlal düşüşü (%)  19.3713      NaN      NaN         NaN     NaN       10
CNN-BiLSTM + SAC [önerilen] − Kural tabanlı yönetim                          Tasarruf (%)  -2.3447  -2.5410  -2.1484      0.0020  0.0469        0
     CNN-BiLSTM + SAC [önerilen] − MPC + CNN-BiLSTM                          Tasarruf (%)  -1.3472  -1.5435  -1.1509      0.0020  0.0469        0
        CNN-BiLSTM + SAC [önerilen] − MPC + XGBoost                          Tasarruf (%)  -1.3544  -1.5507  -1.1581      0.0020  0.0469        0
    CNN-BiLSTM + SAC [önerilen] − Rastgele politika                          Tasarruf (%)   4.3431   4.0868   4.5995      0.0020  0.0469       10
      CNN-BiLSTM + SAC [önerilen] − SAC (tahminsiz)                          Tasarruf (%)   0.0188  -0.1806   0.2183      1.0000  1.0000        5
  CNN-BiLSTM + SAC [önerilen] − Random Forest + SAC                          Tasarruf (%)  -0.0072  -0.1608   0.1464      1.0000  1.0000        6
        CNN-BiLSTM + SAC [önerilen] − XGBoost + SAC                          Tasarruf (%)  -0.1642  -0.3931   0.0647      0.1934  1.0000        4
            SAC (tahminsiz) − Kural tabanlı yönetim                          Tasarruf (%)  -2.3636  -2.6007  -2.1264      0.0020  0.0469        0
 Kural tabanlı yönetim − Eski kural (yüzdelik eşik)                          Tasarruf (%)   3.5222   3.5222   3.5222      0.0020     NaN       10
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
                 SAC (tahminsiz)                          82.14                               3.20                              77.30                                  64.73             58.84          2.70       0.19
             Random Forest + SAC                          85.61                               2.34                              80.36                                  68.85             61.04          2.87       0.24
                   XGBoost + SAC                          84.03                               2.43                              80.04                                  63.59             60.73          2.65       0.25
     CNN-BiLSTM + SAC [önerilen]                          85.52                               1.78                              82.54                                  73.09             60.09          3.05       0.34
       Teorik optimum [referans]                         100.00                               0.00                             100.00                                 100.00             67.29          4.17       0.50
```

```
                                      Karşılaştırma                                Metrik     fark   ci_alt   ci_ust  p_wilcoxon  p_holm  pozitif
CNN-BiLSTM + SAC [önerilen] − Kural tabanlı yönetim         Birleşik amaçta yakalanan (%)  37.0690  35.7950  38.3429      0.0020  0.0469       10
     CNN-BiLSTM + SAC [önerilen] − MPC + CNN-BiLSTM         Birleşik amaçta yakalanan (%)  10.8567   9.5828  12.1307      0.0020  0.0469       10
        CNN-BiLSTM + SAC [önerilen] − MPC + XGBoost         Birleşik amaçta yakalanan (%)  -4.8399  -6.1139  -3.5660      0.0020  0.0469        0
    CNN-BiLSTM + SAC [önerilen] − Rastgele politika         Birleşik amaçta yakalanan (%) 137.4495 133.4691 141.4299      0.0020  0.0469       10
      CNN-BiLSTM + SAC [önerilen] − SAC (tahminsiz)         Birleşik amaçta yakalanan (%)   3.3752   0.3988   6.3517      0.0371  0.3340        8
  CNN-BiLSTM + SAC [önerilen] − Random Forest + SAC         Birleşik amaçta yakalanan (%)  -0.0965  -2.5340   2.3411      0.8457  1.0000        5
        CNN-BiLSTM + SAC [önerilen] − XGBoost + SAC         Birleşik amaçta yakalanan (%)   1.4841  -0.5568   3.5250      0.1309  1.0000        7
            SAC (tahminsiz) − Kural tabanlı yönetim         Birleşik amaçta yakalanan (%)  33.6938  31.4075  35.9800      0.0020  0.0469       10
 Kural tabanlı yönetim − Eski kural (yüzdelik eşik)         Birleşik amaçta yakalanan (%)  38.7791      NaN      NaN         NaN     NaN       10
CNN-BiLSTM + SAC [önerilen] − Kural tabanlı yönetim Fatura tasarrufu / teorik optimum (%) -54.5454 -59.4838 -49.6071      0.0020  0.0469        0
     CNN-BiLSTM + SAC [önerilen] − MPC + CNN-BiLSTM Fatura tasarrufu / teorik optimum (%) -31.5273 -36.4656 -26.5889      0.0020  0.0469        0
        CNN-BiLSTM + SAC [önerilen] − MPC + XGBoost Fatura tasarrufu / teorik optimum (%) -24.8845 -29.8228 -19.9461      0.0020  0.0469        0
    CNN-BiLSTM + SAC [önerilen] − Rastgele politika Fatura tasarrufu / teorik optimum (%) 108.0389 101.9957 114.0821      0.0020  0.0469       10
      CNN-BiLSTM + SAC [önerilen] − SAC (tahminsiz) Fatura tasarrufu / teorik optimum (%)   8.3537   0.2166  16.4907      0.0645  0.5156        8
  CNN-BiLSTM + SAC [önerilen] − Random Forest + SAC Fatura tasarrufu / teorik optimum (%)   4.2377  -1.9563  10.4317      0.1602  0.8008        7
        CNN-BiLSTM + SAC [önerilen] − XGBoost + SAC Fatura tasarrufu / teorik optimum (%)   9.4988   3.4978  15.4998      0.0039  0.0469        9
            SAC (tahminsiz) − Kural tabanlı yönetim Fatura tasarrufu / teorik optimum (%) -62.8991 -68.2249 -57.5733      0.0020  0.0469        0
 Kural tabanlı yönetim − Eski kural (yüzdelik eşik) Fatura tasarrufu / teorik optimum (%)  84.1347  84.1347  84.1347      0.0020     NaN       10
CNN-BiLSTM + SAC [önerilen] − Kural tabanlı yönetim                      İhlal düşüşü (%)  43.6911  42.8429  44.5392      0.0020  0.0469       10
     CNN-BiLSTM + SAC [önerilen] − MPC + CNN-BiLSTM                      İhlal düşüşü (%)  15.9786  15.1304  16.8268      0.0020  0.0469       10
        CNN-BiLSTM + SAC [önerilen] − MPC + XGBoost                      İhlal düşüşü (%)   0.8448  -0.0033   1.6930      0.0840  0.7559        6
    CNN-BiLSTM + SAC [önerilen] − Rastgele politika                      İhlal düşüşü (%)  98.5096  95.4253 101.5939      0.0020  0.0469       10
      CNN-BiLSTM + SAC [önerilen] − SAC (tahminsiz)                      İhlal düşüşü (%)   1.2525  -0.4381   2.9430      0.2324  1.0000        7
  CNN-BiLSTM + SAC [önerilen] − Random Forest + SAC                      İhlal düşüşü (%)  -0.9518  -2.1493   0.2457      0.1602  1.0000        3
        CNN-BiLSTM + SAC [önerilen] − XGBoost + SAC                      İhlal düşüşü (%)  -0.6414  -2.1576   0.8749      0.1602  1.0000        3
            SAC (tahminsiz) − Kural tabanlı yönetim                      İhlal düşüşü (%)  42.4386  41.1092  43.7680      0.0020  0.0469       10
 Kural tabanlı yönetim − Eski kural (yüzdelik eşik)                      İhlal düşüşü (%)  16.8139  16.8139  16.8139      0.0020     NaN       10
CNN-BiLSTM + SAC [önerilen] − Kural tabanlı yönetim                          Tasarruf (%)  -2.2754  -2.4814  -2.0694      0.0020  0.0469        0
     CNN-BiLSTM + SAC [önerilen] − MPC + CNN-BiLSTM                          Tasarruf (%)  -1.3152  -1.5212  -1.1092      0.0020  0.0469        0
        CNN-BiLSTM + SAC [önerilen] − MPC + XGBoost                          Tasarruf (%)  -1.0381  -1.2441  -0.8321      0.0020  0.0469        0
    CNN-BiLSTM + SAC [önerilen] − Rastgele politika                          Tasarruf (%)   4.5070   4.2549   4.7591      0.0020  0.0469       10
      CNN-BiLSTM + SAC [önerilen] − SAC (tahminsiz)                          Tasarruf (%)   0.3485   0.0090   0.6879      0.0645  0.5156        8
  CNN-BiLSTM + SAC [önerilen] − Random Forest + SAC                          Tasarruf (%)   0.1768  -0.0816   0.4352      0.1602  0.8008        7
        CNN-BiLSTM + SAC [önerilen] − XGBoost + SAC                          Tasarruf (%)   0.3963   0.1459   0.6466      0.0039  0.0469        9
            SAC (tahminsiz) − Kural tabanlı yönetim                          Tasarruf (%)  -2.6239  -2.8461  -2.4017      0.0020  0.0469        0
 Kural tabanlı yönetim − Eski kural (yüzdelik eşik)                          Tasarruf (%)   3.5098   3.5098   3.5098      0.0020     NaN       10
```
