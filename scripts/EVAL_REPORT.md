# NIDS — Model Evaluation Report

Held-out test set: **47,912 flows** | classes: **11** | features: **78** (1D CNN)

## Headline metrics

| Metric | Value |
|---|---|
| Accuracy | **99.48%** |
| Macro F1 | **97.67%** |
| Weighted F1 | **99.47%** |
| Macro Precision | 98.29% |
| Macro Recall | 97.20% |

## Per-class metrics

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| BENIGN | 99.02% | 98.97% | 98.99% | 10,000 |
| Botnet | 94.70% | 77.75% | 85.39% | 391 |
| DDoS | 99.97% | 99.99% | 99.98% | 10,000 |
| DoS GoldenEye | 99.47% | 99.90% | 99.69% | 2,059 |
| DoS Hulk | 99.62% | 99.93% | 99.78% | 10,000 |
| DoS Slowhttptest | 98.56% | 99.64% | 99.10% | 1,100 |
| DoS slowloris | 99.74% | 98.62% | 99.18% | 1,159 |
| FTP-Patator | 100.00% | 100.00% | 100.00% | 1,587 |
| PortScan | 99.96% | 99.96% | 99.96% | 10,000 |
| SSH-Patator | 97.92% | 99.75% | 98.82% | 1,180 |
| Web Attacks | 92.19% | 94.72% | 93.44% | 436 |

## sklearn classification_report

```
                  precision    recall  f1-score   support

          BENIGN     0.9902    0.9897    0.9899     10000
          Botnet     0.9470    0.7775    0.8539       391
            DDoS     0.9997    0.9999    0.9998     10000
   DoS GoldenEye     0.9947    0.9990    0.9969      2059
        DoS Hulk     0.9962    0.9993    0.9978     10000
DoS Slowhttptest     0.9856    0.9964    0.9910      1100
   DoS slowloris     0.9974    0.9862    0.9918      1159
     FTP-Patator     1.0000    1.0000    1.0000      1587
        PortScan     0.9996    0.9996    0.9996     10000
     SSH-Patator     0.9792    0.9975    0.9882      1180
     Web Attacks     0.9219    0.9472    0.9344       436

        accuracy                         0.9948     47912
       macro avg     0.9829    0.9720    0.9767     47912
    weighted avg     0.9947    0.9948    0.9947     47912

```

## Confusion matrix (rows = true, cols = pred)

| true\pred | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 BENIGN | 9,897 | 17 | 3 | 3 | 36 | 3 | 0 | 0 | 4 | 3 | 34 |
| 1 Botnet | 87 | 304 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2 DDoS | 1 | 0 | 9,999 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 3 DoS GoldenEye | 2 | 0 | 0 | 2,057 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 4 DoS Hulk | 0 | 0 | 0 | 7 | 9,993 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5 DoS Slowhttptest | 1 | 0 | 0 | 0 | 0 | 1,096 | 3 | 0 | 0 | 0 | 0 |
| 6 DoS slowloris | 2 | 0 | 0 | 0 | 0 | 13 | 1,143 | 0 | 0 | 1 | 0 |
| 7 FTP-Patator | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1,587 | 0 | 0 | 0 |
| 8 PortScan | 2 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 9,996 | 0 | 1 |
| 9 SSH-Patator | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1,177 | 0 |
| 10 Web Attacks | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 21 | 413 |

Class index → name: 0=BENIGN, 1=Botnet, 2=DDoS, 3=DoS GoldenEye, 4=DoS Hulk, 5=DoS Slowhttptest, 6=DoS slowloris, 7=FTP-Patator, 8=PortScan, 9=SSH-Patator, 10=Web Attacks
