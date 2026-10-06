# 📋 PROJECT OVERVIEW & CONTEXT BRIEF — JOINTS X INSPIRE UGM 2026
**Instruksi Konteks Lengkap untuk Asisten AI (Claude / ChatGPT / LLM Researcher)**  
*Dokumen ini dirancang agar AI penerima langsung memahami 100% domain, struktur data, kendala teknis, dan tujuan optimasi kompetisi tanpa ambigu.*

---

> [!IMPORTANT]
> ### 🚨 CATATAN KHUSUS UNTUK AI PENERIMA (READ THIS FIRST):
> Jika Anda menerima prompt kuesioner standar yang menanyakan tentang *"citra / resolusi / computer vision"*, **abaikan asumsi citra tersebut!**  
> Proyek ini **BUKAN proyek Computer Vision (0 citra, resolusi N/A)**. Ini adalah kompetisi **Data Science Tabular: Hierarchical Time-Series & Box Office Ticket Forecasting**. Semua data berada dalam bentuk **6 tabel relasional (CSV)**.

---

## 🎯 1. Identitas & Ringkasan Kompetisi

| Komponen | Rincian Resmi |
| :--- | :--- |
| **Kompetisi** | Data Science Track — **JOINTS X INSPIRE UGM 2026** |
| **Penyelenggara** | Universitas Gadjah Mada (UGM) |
| **Tim** | **JARVIS** |
| **Tugas Pemodelan** | *Cold-Start Sequential Forecasting*: Memprediksi jumlah tiket bioskop terjual (`total_ticket`) per film di setiap bioskop untuk **Hari ke-4 s.d. Hari ke-10 (D4–D10)** penayangan, dengan hanya berbekal histori penjualan tiket **3 hari pertama (D1–D3)**. |
| **Metrik Evaluasi** | **MASE (Mean Absolute Scaled Error)** dengan basis skala rata-rata 3 hari pertama. |
| **Batas Akhir (Deadline)**| **13 Oktober 2026, pukul 23:59 WIB** |
| **Regulasi & Format Pengumpulan** | File archive `jarvis.zip` berisi: <br>1. `jarvis.ipynb` (Notebook audit juri, harus dapat dieksekusi end-to-end)<br>2. `jarvis.pkl` (File bobot model terlatih, **wajib $\le 200$ MB**).<br>⚠️ **Dilarang keras**: Menggunakan pustaka AutoML (AutoGluon, FLAML, TPOT), LLM inference di pipeline, atau data eksternal tidak resmi. |

---

## 📊 2. Karakteristik & Volume Data (Jawaban Spesifikasi Dataset)

Data disediakan dalam folder `data/` yang terdiri dari 6 berkas CSV relasional:

```
data/
├── train.csv              (138.959 baris | 11.86 MB) -> Log penayangan aktif historis
├── test.csv               ( 72.611 baris |  5.63 MB) -> Target prediksi D4-D10 (Grid Kartesian)
├── test_history.csv       ( 32.323 baris |  2.67 MB) -> Histori D1-D3 film di test set
├── movies.csv             (    397 baris | 93.20 KB) -> Metadata detail film
├── holidays.csv           (    366 baris | 10.70 KB) -> Kalender libur nasional & akhir pekan
├── ticket_prices.csv      (    207 baris |  4.88 KB) -> Plafon & tarif harga tiket per kota
└── sample_submission.csv  ( 72.611 baris | 932.8 KB) -> Format submit (id, total_ticket)
```

### 2.1 Rincian Skema & Fitur Tiap Berkas:
1. **`train.csv` (138.959 baris)**:
   - Kolom: `date_show`, `cinema_ids`, `city_name`, `movie_title`, `total_ticket` *(target)*, `occupation_rate`, `total_show`.
   - Periode: 01 April 2025 s.d. 30 September 2025 (237 film, 117 bioskop, 67 kota).
   - Catatan: Hanya mencatat penayangan aktif (`total_ticket` $\ge 1$).

2. **`test.csv` (72.611 baris)**:
   - Kolom: `id`, `movie_title`, `cinema_ids`, `city_name`, `date_show`.
   - Periode: 04 Oktober 2025 s.d. 27 Maret 2026 (160 film, 121 bioskop, 69 kota).
   - Karakteristik: Berbentuk **Grid Kartesian Lengkap** (film $\times$ bioskop $\times$ hari penayangan D4–D10).

3. **`test_history.csv` (32.323 baris)**:
   - Kolom: `date_show`, `cinema_ids`, `city_name`, `movie_title`, `total_ticket`, `occupation_rate`, `total_show`.
   - Data riil 3 hari pertama rilis (D1, D2, D3) untuk seluruh 160 film di `test.csv`.

4. **`movies.csv` (397 film)**:
   - Kolom: `original_title`, `age_rating`, `genre`, `producer`, `director`, `writer`, `casts`.

5. **`holidays.csv` (366 hari)**:
   - Kolom: `date`, `day_tipe` (Weekday/Weekend), `holiday_tipe` (Libur Nasional/Cuti Bersama), `holiday_name`.

6. **`ticket_prices.csv` (207 baris)**:
   - Kolom: `city_name`, `ceil` (plafon harga), `price_day` (Weekday/Weekend).

---

## 🏷️ 3. Kondisi Label, Target & Tantangan Zero-Inflation

* **Tipe Target**: Regresi Kontinu / Count (`total_ticket` $\ge 0$).
* **Statistik Target (Train)**:
  - Min: 1 | Max: 32.845 | Mean: 354,12 | Median: 161,00 tiket.
* **Fenomena Hurdle / Zero-Inflation Ekstrem pada Test Set**:
  - Pada `test.csv` (72.611 baris), terdapat **~40,41% s.d. 41,86% baris bernilai 0 tiket** (~29.341 s.d. 30.044 baris).
  - **Penyebab Bisnis**: Bioskop melakukan *screening cut* (memangkas layar penayangan hingga 0 jadwal) untuk film-film yang sepi/flop setelah melewati akhir pekan pertama.
  - **Solusi Model**: Pendekatan **Two-Stage Hurdle**:
    * *Stage 1 (Classifier)*: Memprediksi probabilitas film tetap tayang ($P(\text{active}) \in [0, 1]$).
    * *Stage 2 (Regressor)*: Memprediksi intensitas penjualan tiket jika tayang, dinormalisasi terhadap skala $s_p$:
      $$z = \frac{\text{total\_ticket}}{s_p}$$
* **Covariate Shift Skala Bioskop ($s_p$)**:
  - Di Test Set, **40,63%** baris adalah bioskop mikro ($s_p \le 50$), sementara di Train hanya 27,08%.
  - Karena metrik MASE membagi error dengan $s_p$, kesalahan pada bioskop kecil memberikan penalti yang jauh lebih fatal.

---

## 🌍 4. Domain Masalah & Dinamika Industri Bioskop

Domain: **Entertainment Business & Cinema Analytics (Pasar Bioskop Indonesia: Cinema XXI, CGV, Cinepolis)**.

Faktor-faktor fisis yang harus diperhitungkan dalam feature engineering:
1. **Jedidi-Krider Box Office Decay Curve**: Tren alami penurunan eksponensial penonton sejak rilis perdana.
2. **Thursday Competitor Release Hazard**: Film-film baru di Indonesia selalu rilis serentak setiap hari **Kamis**, menyebabkan film lama mengalami pemotongan layar drastis jika occupancy rate D1-D3 rendah.
3. **Weekend Rebound Effect**: Penjualan tiket pada hari Sabtu & Minggu melonjak 2x–4x lipat dibanding hari kerja (*weekday*).
4. **Cultural & Holiday Affinity**: Film bergenre horor dan religi lokal memiliki lonjakan khusus pada malam Jumat atau masa libur keagamaan (Lebaran/Natal).
5. **Star Power & Track Record**: Daya tarik box office dari sutradara dan aktor utama (misal: Joko Anwar, Reza Rahadian, dsb.).

---

## 🎯 5. Metrik Evaluasi Resmi: MASE (Mean Absolute Scaled Error)

### 5.1 Formulasi Matematis
$$\text{MASE} = \frac{1}{N} \sum_{i=1}^{N} \frac{|y_i - \hat{y}_i|}{s_p}$$
di mana:
* $y_i$: Jumlah tiket aktual di baris $i$.
* $\hat{y}_i$: Prediksi jumlah tiket di baris $i$.
* $s_p$: Skala rata-rata penjualan tiket 3 hari pertama film tersebut pada bioskop bersangkutan:
  $$s_p = \max\left(1, \frac{y_{\text{D1}} + y_{\text{D2}} + y_{\text{D3}}}{3}\right)$$

### 5.2 Implikasi Pemodelan
* Metrik ini linier terhadap error (L1/MAE) tetapi diskalakan dengan bobot $w_i = \frac{1}{s_p}$.
* Penggunaan model regresi berbasis L2/MSE akan menghasilkan prediksi berlebih (*overprediction*) pada film besar dan berujung pada skor MASE yang buruk.
* Optimasi kuantil (*Quantile Regression* dengan $\tau \in [0.42, 0.45]$) terbukti menghasilkan perbaikan MASE yang konsisten.

---

## 💻 6. Kapasitas Komputasi & Spesifikasi Hardware

* **Workstation Lokal**:
  * **GPU**: NVIDIA GeForce RTX 3050 Laptop GPU (**6 GB VRAM**, CUDA 13.0, ~5.4 GB VRAM idle).
  * **CPU**: Multi-core x86_64, Windows 11.
* **Akselerator Cloud (Tersedia)**:
  * Kaggle Notebook: 1x NVIDIA Tesla T4 (16 GB VRAM) / CPU 4-core.
* **Model Stack yang Digunakan**:
  * **CatBoost GPU** (`task_type='GPU'`, loss MAE).
  * **CUDA XGBoost** (`tree_method='hist'`, `device='cuda'`, reg:absoluteerror).
  * **LightGBM CPU** (`objective='quantile'`, $\alpha=0.45$, linear_tree).
  * **SE-ResNet-1D / PyTorch CUDA** (Representasi temporal sekuensial).
* **Ukuran Bobot Model**:
  * Berkas `jarvis.pkl` saat ini berukuran **13.86 MB** (sangat aman di bawah kuota 200 MB).

---

## 🏆 7. Status Performa Terkini & Analisis Leaderboard

### 7.1 Papan Klasemen Kaggle Terkini (04–05 Oktober 2026)
* **Peringkat #1 (SAGARAS)**: **`0.34456`** 🏆
* **Peringkat #2 (PathFinder)**: **`0.34972`**
* **Peringkat #3 s.d. #20**: Rentang **`0.37105 s.d. 0.39799`**
* **Personal Best (PB) Tim JARVIS**: **`0.46432`** (`submission_star_power_champion_master_70_30.csv`)

### 7.2 Diagnosa Kunci Kenapa Skor Tertahan di 0.464 vs Peluang Tembus 0.34xxx
1. **The Anchor Drag**:
   - Seluruh submisi tim JARVIS sebelumnya yang mencetak skor 0.464xx selalu mem-blend **70% bobot anchor baseline lama** (`submission_hurdle_top.csv` yang memiliki skor 0.47303 dan volume 12.10M tiket).
   - Anchor 0.473 inilah yang menahan skor publik tidak bisa turun ke bawah 0.46.
2. **Kebenaran Validasi Lokal (OOF) Model Pure**:
   - Skor validasi lokal (5-Fold GroupKFold bebas leakage) untuk model **murni tanpa anchor** adalah:
     * **`0.34002`** (Managerial 5-Pillar SOTA)
     * **`0.34397`** (Championship Level-3 Simplex Pipeline)
   - Angka OOF `0.340` s.d. `0.343` ini **sangat identik dengan skor tim #1 SAGARAS (0.34456)**!
   - Ini membuktikan bahwa tim top leaderboard mengirimkan **Pure Unanchored SOTA Models** dengan volume tiket nasional ~10.1M - 10.2M dan tingkat nol ~41.4%.

---

## 🚀 8. File Submisi Final yang Telah Dihasilkan

1. **`Final_A_Strict_SOTA.csv`** *(Pure SOTA Challenger — Target: 0.34xxx)*:
   - Volume: 10.111.844 tiket | Zero Rate: 41,38% (30.044 baris nol).
   - Dibangun dari Level-3 Simplex Blend: 56.3% CatBoost GPU + 21.9% CUDA XGBoost + 21.8% 7-Horizon Specialists.
   - **Tujuan**: Langsung menembus tier 0.34xxx untuk merebut peringkat 1 Leaderboard.

2. **`Final_B_Golden_Vertex.csv`** *(Conservative Champion — Target: 0.463xx)*:
   - Volume: 11.142.743 tiket (Terkunci di puncak kurva parabola MASE) | Zero Rate: 40,41% (29.341 baris nol).
   - Blend 85% PB 0.46432 + 15% Level-3 SOTA.

---

## ❓ PANDUAN TUGAS UNTUK AI PENERIMA:
Berdasarkan konteks teknis di atas, Anda dapat meminta AI penerima untuk:
1. Mengevaluasi feature engineering tambahan berbasis interaksi bioskop-film atau time-decay lanjutan.
2. Memberikan saran kalibrasi probabilitas penayangan (*survival thresholding*) per skala bioskop mikro vs makro.
3. Mengusulkan strategi post-processing atau kuantil tuning untuk menekan metrik MASE lebih dalam dari 0.340.
4. Membantu mereview arsitektur notebook juri `jarvis.ipynb` agar memenuhi kriteria penilaian metodologis juri UGM.
