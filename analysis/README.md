# 📊 Exploratory Data Analysis & Statistical Reports

Folder ini berisi skrip analisis statistik, investigasi korelasi fitur, multikolinearitas, perbandingan antarsubmisi, dan berkas data laporan pendukung.

---

## 📂 Berkas & Skrip Analisis

### 1. Skrip Analisis (`.py`)
| Skrip | Deskripsi |
| :--- | :--- |
| `analysis_task1.py` | Komparasi statistik menyeluruh dari seluruh file submisi CSV di `submissions/`. |
| `analysis_task2_3.py` | Analisis komparatif sebaran data latih (`train.csv`) vs data uji (`test.csv`). |
| `analysis_task4.py` | Analisis mendalam *winning submission* vs *anchor baseline* per bioskop dan horizon. |
| `analyze_feature_relationships.py` | Menghitung korelasi Pearson fitur terhadap `is_active` dan intensitas `target_z`, serta mendeteksi multikolinearitas ($|r| \ge 0.85$). |
| `analyze_managerial_features.py` | Analisis fitur manajerial bioskop (pemotongan layar, penurunan okupansi, dan sinyal flop). |
| `analyze_z_ratios.py` | Analisis distribusi rasio target normalisasi $z = y / s_p$. |
| `generate_correlation_visualizations.py` | Membangun visualisasi matriks korelasi dan grafik pendorong survival ke `eda_plots/`. |
| `generate_final_report_data.py` | Mengekstrak data agregat untuk laporan statistik final. |
| `download_data.py` | Utilitas pengunduhan dataset resmi dari Kaggle Hub. |

### 2. Berkas Data & Laporan Statistik
| Berkas | Deskripsi |
| :--- | :--- |
| `eda_thematic_correlation_matrix.csv` | Matriks korelasi antar-kelompok tema domain (skala, okupansi, WOM, kapasitas, dsb.). |
| `eda_target_correlations.csv` | Korelasi tiap fitur terhadap status penayangan (`is_active`) dan intensitas tiket (`target_z`). |
| `eda_multicollinearity_pairs.csv` | Daftar pasangan fitur dengan tingkat korelasi tinggi ($|r| \ge 0.85$). |
| `submissions_comparison_stats.csv` | Tabel metrik ringkasan (volume, % nol, mean, max) untuk setiap submission. |
| `submissions_pearson_corr.csv` | Matriks korelasi Pearson antarsubmisi. |
| `submissions_spearman_corr.csv` | Matriks korelasi Spearman antarsubmisi. |
| `winner_vs_anchor_comparison.parquet` | Dataset gabungan baris-per-baris perbandingan prediksi model juara vs anchor. |
