# 🔍 Model Diagnostics & Verification Tools

Direktori ini berisi skrip-skrip pengujian cepat, validasi sanitasi (*sanity checks*), verifikasi bobot model, dan diagnostik hipotesis metrik.

---

## 🛠️ Daftar Alat Diagnostik

| Skrip | Deskripsi & Fungsi |
| :--- | :--- |
| `check_anchor.py` | Memeriksa struktur angka nol dan distribusi nilai submission Anchor (0.47303). |
| `check_weights.py` | Memverifikasi kelengkapan, dimensi array, dan batas ukuran file bobot (`<= 200 MB`). |
| `check_clean_train_scales.py` | Mengaudit skala pembagi $s_p$ pada data latih 183 film bersih. |
| `check_small_scale.py` | Menganalisis sebaran film/bioskop mikro ($s_p \le 10$) pada data uji. |
| `check_small_scale_train.py` | Menganalisis sebaran film/bioskop mikro pada data latih. |
| `diagnose_scores.py` | Mendiagnosis error MASE per horizon dan per segmen bioskop dari file OOF. |
| `test_calibrations.py` | Menguji variasi fungsi peredaman kontinu (Continuous Soft Bayesian Transition). |
| `test_flop_signal.py` | Menguji kekuatan prediktif sinyal occupancy rendah terhadap probabilitas cut off. |
| `test_nelder_mead.py` | Menguji konvergensi algoritma optimasi Nelder-Mead terhadap metrik MASE. |
| `test_pure_grand_sota.py` | Menguji karakteristik prediksi model unanchored pure SOTA. |
| `test_volume_effect.py` | Menguji sensitivitas metrik MASE terhadap fluktuasi total volume tiket nasional. |

---

## 🚀 Cara Menjalankan

Seluruh skrip di folder ini dapat dijalankan langsung dari root proyek:
```bash
python diagnostics/check_anchor.py
python diagnostics/check_weights.py
```
atau dari dalam folder `diagnostics/`:
```bash
cd diagnostics
python check_anchor.py
```
Setiap skrip telah dilengkapi *root path bootstrapper* otomatis.
