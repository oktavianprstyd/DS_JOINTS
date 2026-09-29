# 🎬 DS_JOINTS - JOINTS X INSPIRE UGM 2026 Data Science Competition

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/oktavianprstyd/DS_JOINTS/blob/main/solution.ipynb)

Repositori kolaborasi tim untuk kompetisi **JOINTS X INSPIRE UGM 2026** (Data Science Track).

## 📌 Ringkasan Masalah (Problem Statement)
- **Tujuan**: Memprediksi jumlah penjualan tiket bioskop (`total_ticket`) untuk Hari 4–10 penayangan di bioskop Indonesia berdasarkan data historis 3 hari pertama (Opening Weekend: Hari 1–3).
- **Metrik Evaluasi**: **Mean Absolute Scaled Error (MASE)** dengan skala per pasangan film-bioskop ($s_p$) dihitung dari rata-rata penjualan tiket 3 hari pertama.
- **Strategi Optimasi Loss**: Model dilatih mengoptimalkan rasio terhadap skala ($z_i = y_i / s_p$) menggunakan objektif **L1 / MAE**, yang secara langsung meminimalkan metrik MASE kompetisi.

---

## 📁 Struktur Direktori
```text
.
├── data/                       # Dataset kompetisi (train.csv, test.csv, dsb.)
├── submissions/                # Berkas submission CSV hasil inferensi
├── build_solution_notebook.py  # Generator notebook otomatis
├── download_data.py            # Skrip pengunduh data via kagglehub
├── feature_engineering.py      # Modul feature engineering & alignment data
├── PROGRESS_SUMMARY.md         # Dokumentasi lengkap eksperimen & temuan
├── solution.ipynb              # Notebook alur lengkap EDA -> Training -> Submission
├── train_champion_model.py     # Pipeline training model juara (CatBoost + LightGBM + XGBoost)
├── train_ensemble.py           # Pipeline blending ensemble
├── train_lgbm.py               # Model baseline LightGBM
├── train_super_ensemble.py     # Pipeline GPU-accelerated super ensemble
├── tm_slides.pdf               # Panduan teknis & materi TM resmi
└── tm_slides_text.txt          # Transkrip teks materi TM
```

---

## 🚀 Panduan Memulai (Quick Start)

### 1. Instalasi Dependensi
Pastikan menggunakan Python 3.10+ lalu instal dependensi yang dibutuhkan:
```bash
pip install numpy pandas scikit-learn lightgbm catboost xgboost scipy kagglehub jupyter
```

### 2. Download Data (Jika Belum Ada)
```bash
python download_data.py
```

### 3. Training Model Champion
Jalankan skrip training champion 5-Fold GroupKFold:
```bash
python train_champion_model.py
```
Hasil prediksi submission akan otomatis diekspor ke folder `submissions/`.

### 4. Ekplorasi Notebook
Buka dan jalankan [solution.ipynb](file:///solution.ipynb) untuk melihat analisis visual, EDA, validasi silang, dan inferensi.

---

## 🌐 Menjalankan di Google Colab

Proyek ini sepenuhnya kompatibel dan sangat disarankan dijalankan di **Google Colab** (dengan GPU T4 gratis):

1. **Cara Cepat (1 Klik)**:
   - Klik tombol **Open in Colab** di atas atau buka link:  
     [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/oktavianprstyd/DS_JOINTS/blob/main/solution.ipynb)
   - Aktifkan GPU gratis melalui menu: **Runtime** -> **Change runtime type** -> **T4 GPU**.
   - Jalankan sel secara berurutan (*Runtime -> Run all*). Sel pertama sudah otomatis mendeteksi lingkungan Colab dan mengunduh dataset secara instan.

2. **Menjalankan Skrip Python di Colab Baru**:
   Buka notebook kosong di Colab, lalu jalankan:
   ```python
   !git clone https://github.com/oktavianprstyd/DS_JOINTS.git
   %cd DS_JOINTS
   !python train_champion_model.py
   ```

---

## 📖 Dokumentasi Lengkap
Lihat file [PROGRESS_SUMMARY.md](file:///PROGRESS_SUMMARY.md) untuk rincian formula metrik, penanganan sneak preview, arsitektur ensemble, dan rekapitulasi skor validasi.
