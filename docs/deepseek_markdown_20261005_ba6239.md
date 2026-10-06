# 🚀 SPRINT BLUEPRINT v3: TWO-STAGE HURDLE MASE PIPELINE — UPGRADED WITH LITERATURE & COMPETITION EVIDENCE

*Kompetisi: Data Science Track JOINTS X INSPIRE UGM 2026 | Tim: JARVIS | Hardware: RTX 3050 6 GB*

---

## 📌 0. Apa yang Berubah dari v2 → v3

| # | Area di v2 | Perbaikan v3 | Sumber |
|---|---|---|---|
| 1 | Validasi GroupKFold saja | Tambah **gap-cross-validation** dengan gap 10 hari untuk semua evaluasi temporal; dokumentasi protokol gap yang eksplisit | GapKFold |
| 2 | Hurdle vs ZI tidak dibedakan | Tambah **diagnostik asumsi zero**: tentukan apakah nol struktural (tidak ada pair) atau sampling zero (pair ada tapi penjualan 0) | Hurdle vs ZI |
| 3 | Stage 2 hurdle diasumsikan selalu lebih baik | Tambah **uji ablation wajib**: bandingkan hurdle vs **single-stage SHOS-style feature augmentation** sebelum mengunci arsitektur | Feature engineering > architectural complexity |
| 4 | Tidak ada referensi ke arsitektur hurdle modern | Tambah opsi **shared hurdle head** (satu encoder, dua output head) sebagai alternatif two-model | Switch-Hurdle |
| 5 | Prior shift hanya probabilitas | Tambah **distribution-aware validation** untuk mendeteksi shift sebelum training | OAN-DA / SDFDA |
| 6 | Blend hanya grid `a` sederhana | Tambah **stacked generalization** dengan meta-learner yang menerima OOF prediksi + fitur konteks | Kaggle winning patterns + zero-inflated ensemble |
| 7 | Tidak ada baseline `DummyRegressor` | Tambah **trivial baselines** (predict s_p, predict 0, predict median) sebagai sanity floor | Praktik standar Kaggle |
| 8 | θ search step 0.01 | Perbaiki: **θ search dengan kurva plateau detection**, step 0.005 di sekitar plateau | Robustness best practice |
| 9 | Feature pruning ad-hoc | Tambah **permutation importance + SHAP** untuk pruning terdokumentasi | Interpretability best practice |
| 10 | Tidak ada referensi kompetisi serupa | Tambah **section benchmarking** vs kompetisi zero-inflated (Prozorro, Dhaka) dan paper box-office | Kaggle + arXiv |

**Target keberhasilan (tetap):** MASE terbobot tier lokal vs LB ≤ 0.03; LB < 0.46432; artefak lolos uji.

---

## 📊 1. Fakta Data & Definisi (dikunci + tambahan)

| Berkas | Isi |
|---|---|
| `train.csv` | 138.959 baris, 01 Apr–30 Sep 2025, **hanya baris aktif** (`total_ticket ≥ 1`) |
| `test_history.csv` | 32.323 baris, D1–D3 untuk 160 film test (basis `s_p`) |
| `test.csv` | 72.611 baris, grid (film × bioskop × D4–D10), 04 Okt 2025–27 Mar 2026, ±40% nol |
| Overlap | 10 film (94% cold start), 117 bioskop, 67 kota |

**Definisi:**
