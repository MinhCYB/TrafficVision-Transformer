## Kết quả so sánh trên UIT-ADrone Dataset

| Model | AUC-ROC | Training Time | #Params |
|---|---|---|---|
| Baseline (CA) | ? | ? | ? |
| Proposed (STE-TTE) | ? | ? | ? |
| Paper reported | ~0.xx | — | — |

> Điền vào sau khi chạy xong inference + `compute_auc.py`

---

## Cách tính kết quả

```bash
# STE-TTE
python compute_auc.py \
    --scores-dir experiments_andt_ADrone_STE_TTE/ \
    --labels-dir ../../UIT-ADrone/test/test_frame_mask/ \
    --plot --save-plot results/roc_ste_tte.png

# Baseline CA
python compute_auc.py \
    --scores-dir experiments_andt_ADrone_baseline_CA/ \
    --labels-dir ../../UIT-ADrone/test/test_frame_mask/ \
    --plot --save-plot results/roc_baseline.png

# Vẽ comparison plot
python results/visualize.py
```

---

## Notes

- AUC-ROC cao hơn = model phát hiện anomaly tốt hơn
- Kết quả bị ảnh hưởng bởi: số epoch, learning rate, image size
- Training chỉ dùng **normal frames** (unsupervised)
