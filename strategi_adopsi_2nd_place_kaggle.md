# 🏆 STRATEGI ADOPSI 2ND PLACE KAGGLE SOLUTION
## Aplikasi Filosofi, Workflow, dan Teknik Juara pada Kompetisi DS JOINTS X INSPIRE 2026

Dokumen ini merangkum analisis mendalam terhadap writeup **Kaggle 2nd Place Solution ("Predicting Smartphone Addiction" - Playground Series)** serta pemetaan teknis (*translation & execution mapping*) untuk mendongkrak performa sistem prediksi box office bioskop **DS_JOINTS**.

---

## 1. Intisari Filosofi 2nd Place Solution: Pelajaran Terpenting

Writeup 2nd Place memberikan 5 pelajaran fundamental yang sangat relevan dengan situasi proyek kita saat ini:

### 💡 Insight 1: "I thought I needed a better blender. In fact, I needed a better model."
> *Penulis menghabiskan berminggu-minggu mencoba 40 model neural net, meta-regressor, logistic regression blender, stacking bertingkat, dan hampir tidak ada yang membantu. Kesalahan terbesarnya adalah berhenti meningkatkan model terkuat terlalu dini.*
- **Refleksi DS_JOINTS**: Jangan hanya mengandalkan utak-atik bobot blending (60/40, 50/50, dsb). Kekuatan sesungguhnya lahir dari **model tunggal (*single model*) yang fundamentally lebih tajam** dalam memahami pola data!

### 💡 Insight 2: Fitur Aritmatika Rekonstruksi Domain (`fake_daily = social + work + game`)
> *Penulis mencetak terobosan skor pertamanya bukan dari algoritma canggih, melainkan dari fitur aritmatika domain sederhana: merekonstruksi total harian dari komponen aktivitasnya. Ketika `daily` hilang/noisy, `fake_daily` memulihkan sinyal tersebut.*
- **Refleksi DS_JOINTS**: Hubungan fundamental bioskop:
  $$\text{total\_ticket} \approx \text{est\_capacity} \times \text{total\_show} \times (\text{occupation\_rate} / 100)$$
  Kita dapat merekonstruksi kapasitas implisit, sisa kursi kosong (*slack capacity*), dominasi bioskop di kotanya, dan rasio pemanfaatan layar pembukaan!

### 💡 Insight 3: Transformasi Fitur Sistematis (RAW + Transform)
> *Penulis menerapkan permutasi sistematis:*
> - $\text{RAW} + \text{Ratio}$
> - $\text{RAW} + \text{Bins}$
> - $\text{RAW} + \text{Target Encoding}$
> - $\text{RAW} + \text{Round / Digits}$

### 💡 Insight 4: "Never Tune for Public LB. Trust OOF."
> *Banyak peserta naik-turun 100 peringkat karena blind blending pada Public LB. Penulis hanya mempercayai OOF lokal yang ketat, dan Public LB hanya dijadikan cross-check kewajaran.*
- **Refleksi DS_JOINTS**: Kita telah membuktikan matematis bahwa gap OOF 0.35 vs Kaggle 0.46 terjadi karena penyebut skala test 35% lebih kecil. Kita tidak boleh overfit ke Public LB, melainkan menjaga OOF dan metrik *Low-Scale MASE*.

### 💡 Insight 5: Keep Experiments Simple
> *Satu eksperimen = Satu tujuan jelas (Satu script fitur, satu script tuning parameter, satu script validasi).*

---

## 2. Pemetaan Fitur Domain: Mengadopsi Filosofi `fake_daily` ke Bioskop

Dalam domain box office bioskop Indonesia, berikut fitur-fitur aritmatika rekonstruksi dan transformasi sistematis yang ekuivalen:

| Konsep 2nd Place | Padanan di Domain Bioskop (DS_JOINTS) | Formulasi Matematis / Kode |
| :--- | :--- | :--- |
| **`fake_daily` (Rekonstruksi Total)** | **`implied_total_capacity`** | $\text{est\_capacity} \times \text{show\_mean}$ (Kapasitas total kursi yang tersedia per hari) |
| **`fake_work / fake_game` (Komponen Slack)** | **`slack_seats_d1..d3`** | $\text{implied\_total\_capacity} - \text{ticket\_d1..d3}$ (Jumlah kursi kosong yang terbuang saat pembukaan) |
| **Pilar Dekomposisi Harian** | **`opening_ticket_share_d1..d3`** | $\frac{\text{ticket\_d1}}{\sum \text{ticket}_{D1..D3}}$ (Porsi penonton di hari Jumat vs Sabtu vs Minggu) |
| **Dominasi Lokal vs Kota** | **`city_ticket_slack`** | $\text{city\_prior\_tickets} - \text{cinema\_prior\_tickets}$ (Kekuatan bioskop kompetitor di kota yang sama) |
| **Dominasi Lokal vs Kota** | **`city_dominance_ratio`** | $\frac{\text{scale}}{\text{city\_prior\_tickets} + 1.0}$ (Berapa persen pasar kota yang dikuasai bioskop ini) |
| **Momentum Akselerasi Lanjutan** | **`ticket_accel_normalized`** | $\frac{(D3 - D2) - (D2 - D1)}{\text{scale} + 1.0}$ (Kelengkungan laju penonton dinormalisasi skala) |
| **RAW + Bins** | **`scale_quantile_bin`** | Binning persentil skala ($s_p$ deciles: 1-10) untuk membedakan bioskop mikro, menengah, mega |
| **RAW + Log Transform** | **`log_scale`, `log_nat_scale`, `log_tickets`** | Mengurangi skewness ekstrim pada distribusi film raksasa vs film mikro |
| **Interaksi Bin x Kalender** | **`scale_bin_x_dow`** | $\text{scale\_bin} \times 7 + \text{day\_of\_week}$ (Pola bioskop kecil di hari kerja vs bioskop besar di weekend) |

---

## 3. Rencana Eksekusi Nyata: Memperkuat Model Tunggal (Strong Single Model)

Sesuai saran penulis writeup (*"I stopped improving strong models too early... I needed a better model"*), kita akan mengeksekusi:

1. **Modul Rekayasa Fitur Baru**: Tambahkan 15 fitur aritmatika rekonstruksi dan transformasi log/bin ke [`feature_engineering.py`](file:///C:/Users/oktav/BOT_clean/JOINTS/jarvis/feature_engineering.py).
2. **Penguatan Model Utama (Deep Hyperparameter Tuning di GPU)**:
   - Bangun model **XGBoost CUDA & CatBoost GPU** yang lebih dalam (`max_depth=7`, `learning_rate=0.02`, `n_estimators=750`).
   - Latih dengan **Sample-Weighting** terkalibrasi untuk memproteksi bioskop kecil ($s_p \le 15$).
   - Aktifkan **Two-Tier Scale Hurdle + L1 Capping**.
3. **Uji Validasi OOF Bebas Bocor**:
   - Ukur penurunan OOF MASE dan Low-Scale MASE.
4. **Ensembling Terukur (Bukan Blind Blend)**:
   - Padukan model tunggal super kuat ini dengan Anchor terverifikasi (mengunci 29.341 baris nol) untuk menghasilkan kandidat submisi pemecah rekor baru!

---
*Dibuat untuk panduan eksekusi teknis tim Jarvis DS_JOINTS 2026.*
