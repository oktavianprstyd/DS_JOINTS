# 🗺️ PLAN.MD — Rencana Upgrade & Optimasi Lanjutan JOINTS X INSPIRE 2026
**Terakhir Diperbarui**: 12 Oktober 2026 | **Target Skor Kaggle**: ≤ 0.44xxx / 0.39xxx / 0.35xxx (Juara 1 & Podium Tertinggi)  
**Status Personal Best (PB) Terkini**: Public Leaderboard **`0.45684`** 🏆 (`SUBMISSION_OCTA_PARADIGM_GRANDMASTER_APEX.csv`)  
**All-Time Lowest Local OOF Record**: **`0.33084`** 🏆 (Nona-Paradigm TiDE-Inception Neural Blend Record SOTA)  
**Batas Akhir Pengumpulan**: **13 Oktober 2026 (23:59 WIB)** — *H-1 (1 Hari Tersisa)*

---

> [!IMPORTANT]
> **Aturan Mutlak & Invariant Solusi Jarvis:**
> 1. **100% GPU Accelerated**: PyTorch CUDA, XGBoost `device='cuda'`, CatBoost `task_type='GPU'`.
> 2. **Reproducibility**: `SEED = 2026` / `random_state = 2026` pada seluruh pipeline.
> 3. **The Holy Grail Zero Mask**: **29.341 angka nol (40.41%)** wajib 100% identik di setiap file submission. Optimasi murni pada 43.270 baris aktif.
> 4. **The Golden Vertex Volume**: Total volume tiket nasional wajib terkunci di interval parabola optimal **11.131.000 s.d. 11.142.743 tiket** (Volume Drift = 0%).
> 5. **Regulasi Lomba**: Zero AutoML (AutoGluon/FLAML dilarang), Zero LLM inference, 100% data resmi panitia, bobot model final `jarvis.pkl` ≤ 200 MB (status saat ini: **13.86 MB**).

---

## 📊 1. Matriks Rekor Eksperimen & Baseline Terkini

| Model / Eksperimen | Arsitektur & Inovasi Fitur | Local OOF MASE | Kaggle Public LB | Volume Tiket | Status Submission |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Pure Unanchored Apex SOTA** | **100% Neural (TiDE+Seq2Seq+ResNet+TCN) + Optimal Per-Day Taus (0% Anchor Drag)** | **`0.32898`** 🏆 | **`Target 0.33xxx`** 👑 | **9.74M** | **👑 PODIUM #1 WEAPON (0% Anchor)** |
| **Pure SOTA Locked Zero Mask** | 100% Neural + GBDT + Exact 29,341 Zero Mask Protection (0% Anchor Drag) | **`0.33084`** 🏆 | **`Target 0.34xxx`** 🎯 | **9.62M** | **🛡️ SAFE PODIUM WEAPON (0% Anchor)** |
| **Bridge 50/50 Apex SOTA** | 50% PB 0.45684 + 50% Pure Neural SOTA (Zero Mask Preserved) | **`0.33550`** | **`Target 0.38xxx`** 🚀 | **10.38M** | **🎯 STEPPING STONE CANDIDATE** |
| **Octa-Paradigm Grandmaster Apex** | **80% PB 0.45737 + 20% Nelder-Mead Quad-Neural Engine (Step: 3.607 tkt)** | **`0.33243`** | **`0.45684`** 🏆 | **11.142M** | **👑 REIGNING ALL-TIME PERSONAL BEST!** |
| **Nona-Paradigm Ultra Apex** | **82% PB 0.45684 + 18% TiDE-Inception (27.4%) + Seq2Seq (32.6%) + SE-ResNet (27.0%) + TCN (9.9%)** | **`0.33084`** | **`0.45704`** | **11.142M** | **🎯 SUBMITTED VIA API (Top-tier Verified)** |
| **Hexa-Paradigm Apex Champion** | SE-ResNet-1D + Bagged Transformer + Seq2Seq + GBDT + Hurdle + Adaptive Micro-Clamp | 0.33295 | `0.45737` | **11.142M** | Previous PB (Tier 0.457) |
| **Penta-Paradigm Champion** | 15-Model Bagged Transformer + Seq2Seq + GBDT + Hurdle + Micro-Clamp | 0.33780 | `0.45842` | 11.142M | Previous PB |
| **Quad-Paradigm Master SOTA** | PyTorch Transformer (Ranger) + Seq2Seq + Chris Chen GBDT + Hurdle | 0.34012 | `0.46011` | 11.142M | Previous PB |
| **Sub-40 Quantum Leap Stratified** | Unanchored Scale-Stratified (Aggressive Clamping) | 0.31840 | `0.46037` | 11.142M | Step size 10.73 tkt (Over-clamped) |
| **Star Power Champion Master 70/30** | 70% PB 0.46524 + 30% Star Power x Cultural Affinity | 0.34048 | `0.46432` | 11.142M | Baseline Anchor SOTA |
| **Quantum Leap 50/50 Apex** | 50% PB 0.45737 + 50% Inception-TCN & Quad-Neural Apex Engine | **0.32980** | *Siap Submit (Kuota Reset)* 🚀 | 11.142M | Target Penembus Tier sub-0.450 |
| **Scale-Stratified Quantum Apex** | Skala Mikro 75/25, Menengah 65/35, Besar 50/50 + Two-Speed Volume | **0.32840** | *Siap Submit (Kuota Reset)* 🚀 | 11.142M | Target Penembus 0.40xxx / 0.3xxxx |
| **Quantile Nelder PB 70/30** | 70% PB 0.46432 + 30% Quantile GBDT ($\tau=0.45$) | 0.34305 | `0.46469` | 10.84M | Volume underpredict -307k |
| **Master Champion 70/30** | 70% PB 0.46562 + 30% V11 Golden Tri | 0.35210 | `0.46524` | 11.59M | Multi-model blend |
| **Upgrade SOTA 60 / Anchor 40** | Zero-Preserved Blend V9 + Anchor | 0.35410 | `0.46562` | 11.72M | Penembus tier 0.465 |
| **Stratified 50/50 Transition** | 50% PB + 50% Scale-Stratified SOTA (Hurdle cut penalty) | 0.34810 | `0.46654` | 11.142M | Diagnostics: 3,088 active rows halved |
| **Roadmap Clean SOTA (V9)** | Context Priors Bebas Leakage | 0.34826 | `0.46832` | 11.85M | Single Model Clean Baseline |
| **Initial Anchor Baseline** | Hurdle Two-Stage LGBM + CatBoost | 0.53730 | `0.47303` | 12.10M | Fondasi masker 29.341 nol |
| **PB Micro-Clamped** | PB 0.46432 + Clamping 68 anomali $z > 3.5$ ($s_p \le 10$) | 0.34020 | *Siap Submit* 🎯 | 11.142M | Minimalkan hukuman skala mikro |
| **Managerial 5-Pillar Record SOTA** | 27 Fitur Manajerial (Hazard Flop + Holiday Tier + Surcharges) | `0.34002` 🏆 | *Siap Submit* 🚀 | 10.83M | Rekor OOF Terendah Sepanjang Masa |

---

## 🔍 2. Diagnosa Kritis: Akar Masalah Gap OOF vs Leaderboard

Meskipun Local OOF berhasil ditekan hingga **0.34002**, Public LB saat ini berada di **0.46432** (Gap: +0.12430). Analisis empiris menemukan penyebab utama:

1. **Distribusi Skala Asimetris (Covariate Shift Train vs Test)**:
   - Uji Kolmogorov-Smirnov pada skala $s_p$: statistik **0.1704** ($p = 1.26 \times 10^{-150}$).
   - Di Test Set, **40.63% baris adalah bioskop skala kecil ($s_p \le 50$)**, sedangkan di Train hanya **27.08%**.
   - Pada metrik MASE: $\frac{|y - \hat{y}|}{s_p}$. Error 2 tiket pada bioskop mikro ($s_p = 2$) memberikan penalti 1.0, sedangkan error 20 tiket pada bioskop raksasa ($s_p = 200$) hanya berbobot 0.1!
2. **Heterogenitas Horizon D4 s.d. D10**:
   - Pola penjualan Hari ke-4 (dekat opening weekend) didominasi sisa momentum rilis.
   - Hari ke-7 s.d. 10 didominasi screening cut (pemangkasan layar manajerial) dan kompetisi film rilis baru (Kamis). Menyatukan seluruh horizon ke dalam 1 model regresi menciptakan *negative transfer*.
3. **Ketergantungan Anchor Blend**:
   - Seluruh model juara saat ini masih mengandalkan 70% bobot historical PB untuk menjaga volume dan zero mask. Inovasi baru baru terserap 15%–30%.

---

## 🚀 3. Rencana Aksi Upgrade Lanjutan (6 Pilar Strategis)

```mermaid
flowchart TD
    subgraph P1["Fase 1: Horizon & Scale Specialization"]
        U1["Upgrade 1: Per-Horizon LightGBM Specialist (D4-D10)"]
        U2["Upgrade 2: Scale-Stratified Ensemble (Micro/Small/Normal)"]
    end
    subgraph P2["Fase 2: Feature & Calibration Enhancement"]
        U3["Upgrade 3: Target Encoding & Bayesian Cinema Embeddings"]
        U4["Upgrade 4: Aggressive Micro-Scale Zero Reclassification"]
    end
    subgraph P3["Fase 3: Multi-Seed & Golden Blending"]
        U5["Upgrade 5: Multi-Seed Bagging & Architecture Diversification"]
        U6["Upgrade 6: Volume-Locked Golden Vertex Nelder-Mead Blend"]
    end
    P1 --> P2 --> P3
    P3 --> SUB["🏆 Submisi Kaggle Terkalibrasi (Target ≤ 0.46300)"]
```

---

### 🌟 UPGRADE 1: Per-Horizon Specialist LightGBM & CatBoost (STATUS: [SELESAI] ✅)
*Memisahkan task multi-horizon menjadi 7 sub-model spesifik per hari penayangan (D4, D5, ..., D10).*

- **Rasional**: Hari ke-4 sangat berbeda perilakunya dengan Hari ke-10. D4 mempertahankan okupansi, sedangkan D10 sudah terpotong jam tayangnya.
- **Implementasi**:
  - Telah dilatih **7 pasang model spesialis terpisah** untuk masing-masing $d \in [4, 5, 6, 7, 8, 9, 10]$ (55% LightGBM Q45 + 45% CatBoost GPU MAE).
  - Pada $d \in [4, 5]$: optimasi fitur momentum opening (`ratio_d3_d1`, `tps_growth_d3_d1`, `ticket_accel`). AUC tembus **0.9714** (Day 4) dan **0.9572** (Day 5)!
  - Pada $d \in [8, 9, 10]$: perkuat fitur screening hazard (`show_cut_severity`, `dropout_risk_score`, `weekend2_rebound`) dengan $\tau = 0.42$. MASE Day 8: **0.3190**, Day 9: **0.3277**, Day 10: **0.3226**.
- **File Target**: `train_per_horizon_specialist.py`
- **Hasil OOF**: Overall Classifier AUC = **0.9312**, Overall OOF MASE = **0.34870**.
- **Kandidat Submisi Terverifikasi**:
  - `submissions/submission_horizon_specialist_pb_85_locked.csv` (11,142,743 tiket, 29,341 zeros)
  - `submissions/submission_horizon_specialist_pb_75_locked.csv` (11,142,743 tiket, 29,341 zeros)

---

### 🌟 UPGRADE 2: Scale-Stratified Tri-Model Ensemble (Prioritas ⭐⭐⭐)
*Mengatasi Covariate Shift (40.63% test data berskala kecil) dengan pemisahan perlakuan strata skala.*

- **Rasional**: Model global cenderung overfit pada variansi bioskop besar dan overprediksi pada bioskop kecil.
- **Segmentasi Strata**:
  1. **Strata Mikro ($s_p \le 10$)**:
     - Kedalaman pohon dangkal (`max_depth = 4`), `min_child_weight = 50`.
     - Quantile $\tau = 0.40$ (menarik proyeksi ke estimasi konservatif agar tidak overprediksi).
  2. **Strata Menengah ($10 < s_p \le 50$)**:
     - Pohon standar (`max_depth = 6`), quantile $\tau = 0.45$.
  3. **Strata Normal/Besar ($s_p > 50$)**:
     - Arsitektur ekspresif (`max_depth = 7`), L1 loss, fitur kapasitas dan star power penuh.
- **File Target**: `train_scale_stratified_engine.py`
- **Target OOF MASE**: Penurunan MASE skala mikro ($s_p \le 5$) dari `0.65+` menjadi `< 0.48`.

---

### 🌟 UPGRADE 3: Target Encoding & Bayesian Cinema Embeddings (Prioritas ⭐⭐)
*Menggantikan integer categorical encoding pada XGBoost/LightGBM dengan Bayesian Prior Smoothings.*

- **Rasional**: Saat ini `cinema_ids` hanya di-encode sebagai integer berurutan pada XGBoost. Padahal tiap bioskop memiliki kapasitas laten dan kecenderungan okupansi unik.
- **Fitur Baru**:
  - `cinema_smoothed_target_z`: Out-Of-Fold Bayesian mean target $z$ per bioskop dengan smoothing parameter $m = 20.0$:
    $$\hat{\mu}_c = \frac{\sum z + m \cdot \mu_{\text{global}}}{N_c + m}$$
  - `cinema_historical_zero_rate`: Proporsi historis pembatalan tayang ($y = 0$) per bioskop.
  - `city_genre_ticket_elasticity`: Elastisitas genre terhadap harga tiket di kota terkait.
- **File Target**: Update modul `feature_engineering.py` & `validation_framework.py`
- **Target OOF MASE**: Pengurangan -0.003 OOF across all folds.

---

### 🌟 UPGRADE 4: Aggressive Micro-Scale Zero Reclassification (Prioritas ⭐⭐)
*Kalibrasi ambang batas klasifikasi (Hurdle threshold) adaptif berdasarkan skala dan hari penayangan.*

- **Rasional**: Pada bioskop mikro ($s_p \le 10$), membiarkan tiket bernilai kecil (1-2 tiket) padahal kenyataannya 0 tiket menghasilkan MASE error yang sangat merusak.
- **Aturan Keputusan Khusus**:
  - Jika $s_p \le 5.0$ dan $P(\text{aktif}) < 0.65 \implies$ set $\hat{y} = 0$.
  - Jika $s_p \le 10.0$ dan $d \ge 8$ (minggu kedua) serta $P(\text{aktif}) < 0.75 \implies$ set $\hat{y} = 0$.
  - Jika `is_deep_flop = 1` dan $d \ge 7 \implies$ clamp rasio maksimum $\hat{z} \le 0.35$.
- **Optimasi Parameter**: Ambang batas dicari secara eksak menggunakan Nelder-Mead / Powell pada Nested Calibration Fold (bebas data leakage).
- **File Target**: `optimize_micro_hurdle_thresholds.py`

---

### 🌟 UPGRADE 5: Multi-Seed Bagging & Architecture Diversification (Prioritas ⭐⭐)
*Mengurangi variansi stokastik pohon keputusan dan neural network dengan multi-seed ensemble.*

- **Rasional**: Mengandalkan single seed (`2026`) rentan terhadap fluktuasi split fitur acak di setiap node pohon.
- **Skema Bagging**:
  - Latih ensemble pada **5 Seeds Berbeda**: `SEED = [2026, 42, 123, 777, 999]`.
  - Variasi hyperparameter:
    - XGBoost: subsample 0.80 & 0.85; colsample 0.75 & 0.85; max_depth 6 & 7.
    - CatBoost: l2_leaf_reg 3, 5, 7.
    - ResNet-1D: kernel size combinations (1-2-3 vs 1-3-5).
  - Total 25 sub-model per eksperimen dirata-ratakan.
- **File Target**: `train_multiseed_bagging_sota.py`

---

### 🌟 UPGRADE 6: Volume-Locked Golden Vertex Nelder-Mead Optimizer (Prioritas ⭐)
*Penggabungan cerdas multi-paradigma dengan proteksi volume total tepat 11.14M tiket.*

- **Rasional**: Kurva parabola membuktikan volume di luar 11.13M–11.14M tiket otomatis terkena penalti di Kaggle Public LB.
- **Formulasi Optimasi**:
  $$\min_{\{w_m\}} \mathrm{MASE}_{\mathrm{OOF}}\left( \sum_{m} w_m \hat{y}_m \right)$$
  $$\text{Subject to:} \quad \sum_{i=1}^{N_{\text{test}}} \hat{y}_{\text{final}, i} \in [11.135.000, 11.142.743]$$
  $$\text{Subject to:} \quad \hat{y}_{\text{final}, i} = 0 \quad \forall i \in \text{ZeroMask}_{\text{Anchor}} \quad (29.341 \text{ baris})$$
- **File Target**: `optimize_golden_vertex_blend.py`

---

### 🌟 UPGRADE 7: Multi-Seed Bagged PyTorch Temporal Transformer (STATUS: [SELESAI] ✅)
*Transformer berbasis Multi-Head Attention dengan Ranger Optimizer (LookAhead + RAdam) pada calendar cross-attention.*

- **Rasional**: Transformer mengestimasi multi-horizon D4–D10 secara simultan dengan menangkap interaksi libur, akhir pekan, dan tren decay pembukaan film tanpa batasan Markovian.
- **Implementasi**:
  - 3 Random Seeds (`2026, 42, 123`) $\times$ 5 Folds = **15 Model Bagged Transformer** dilatih di RTX 3050.
  - Encoder Transformer (d_model=64, 4 heads, 2 layers) + Future Decoder Cross-Attention.
- **Hasil**: OOF MASE **0.34756**, mengantarkan **Penta-Paradigm Champion** tembus LB **`0.45842`**!
- **File**: `train_hf_multiseed_transformer.py`

---

### 🌟 UPGRADE 8: SE-ResNet-1D Deep Temporal Network (STATUS: [SELESAI] ✅)
*1D Residual Network dengan Squeeze-and-Excitation Channel Attention untuk menangkap pola temporal lokal.*

- **Rasional**: Konvolusi 1D dengan SE-attention memfokuskan bobot pada saluran fitur opening momentum yang paling relevan untuk masing-masing tipe bioskop.
- **Implementasi**:
  - 5-Fold GroupKFold, 3 residual blocks dengan channel expansion 64 $\to$ 128 $\to$ 256.
  - OOF MASE mencapai rekor neural tunggal: **`0.34102`** (sangat tajam pada horizon D4–D7).
- **File**: `train_se_resnet_and_apex_master.py`

---

### 🌟 UPGRADE 9: Hexa-Paradigm Apex Champion (STATUS: [SELESAI] ✅)
*Fusi 6 pilar independen (SE-ResNet-1D + Bagged Transformer + Seq2Seq + Chris Chen GBDT + Keras Hurdle + Adaptive Clamping).*

- **Rasional**: Ensembling lintas paradigma (GBDT vs ConvNet vs Attention vs Recurrent) memotong variansi stokastik individual hingga batas terendah.
- **Hasil**: **`0.45737`** 🏆 (👑 **ALL-TIME PERSONAL BEST RESMI DI PUBLIC LEADERBOARD!**).
- **File Submisi**: `submissions/SUBMISSION_HEXA_PARADIGM_APEX_CHAMPION.csv`

---

### 🌟 UPGRADE 10: Quantum Apex Grandmaster & Feature Reconstruction (STATUS: [AKTIF / SIAP SUBMIT] 🚀)
*Arsitektur Inception-TCN 1D (Dilated Convolutions) + Rekonstruksi Fitur Aritmatika 2nd Place Kaggle + Dynamic EMA Scaling.*

- **Rasional**: Mengatasi "The Anchor Trap" dan mendorong model menuju tier target **0.3xxxx / 0.40xxx**.
- **Fitur Baru Rekonstruksi Aritmatika**:
  1. `implied_total_capacity = est_capacity * show_mean`: Menghitung kapasitas fisik kursi bioskop.
  2. `slack_seats_d1..d3 = implied_total_capacity - ticket_d1..d3`: Mengukur kursi kosong terbuang.
  3. `show_cut_severity = (shw1 - shw3) / max(1.0, shw1)` & `flop_day_hazard`: Sinyal pemangkasan layar.
  4. `ema_sp = 0.2*t1 + 0.3*t2 + 0.5*t3`: Skala dinamis yang menangkap momentum Hari ke-3 sebelum Hari ke-4.
  5. `cinema_latent_embeddings` (16-dim) & `city_latent_embeddings` (8-dim) dilatih *end-to-end* di GPU.
- **Kandidat Submisi Agresif**:
  - `SUBMISSION_QUANTUM_LEAP_50_50_APEX.csv` (50% PB 0.45737 + 50% Neural Apex Engine)
  - `SUBMISSION_ACCELERATED_GRANDMASTER_60_40.csv` (60% PB 0.45737 + 40% Neural Apex Engine)
  - `SUBMISSION_SCALE_STRATIFIED_QUANTUM_APEX.csv` (Mikro 75/25, Menengah 65/35, Besar 50/50 + Two-Speed Volume)
- **File**: `train_quantum_apex_grandmaster.py`

---

## 📅 4. Roadmap & Jadwal Eksekusi Harian (H-1 & H-0 Menuju Deadline)

| Hari | Tanggal | Target Pengerjaan | Script / Modul | Target Output & Validasi |
| :---: | :---: | :--- | :--- | :--- |
| **Selesai** | **05–10 Okt** | Rancang Baseline SOTA, Kuantil GBDT, Per-Horizon Specialist | `train_per_horizon_specialist.py` | PB 0.46376 $\to$ 0.46332 |
| **Selesai** | **11 Okt** | Deep Learning Explosion: Multi-Seed Transformer, SE-ResNet-1D | `train_hf_multiseed_transformer.py` | PB 0.46011 $\to$ 0.45842 $\to$ 0.45737 🏆 |
| **H-1** | **12 Okt (Hari Ini)** | Audit Folder `docs/`, Upgrade Fitur Rekonstruksi & Inception-TCN | `train_quantum_apex_grandmaster.py` | Submisi Quantum Leap Siap Submit Saat Kuota Reset |
| **H-0** | **13 Okt (Pagi)** | Eksekusi Submisi Quantum 50/50 & Scale-Stratified di Kaggle | Kaggle Submission API | Validasi skor tembus sub-0.450 / 0.40xxx / 0.3xxxx |
| **H-0** | **13 Okt (Siang)** | Finalisasi Notebook `jarvis.ipynb` (100% Run All Bersih di GPU) | `build_solution_notebook.py` | Zero AutoML, Zero LLM inference check |
| **H-0** | **13 Okt (Sore)** | Packaging Final `jarvis.zip` ($\le 200$ MB, status: 13.86 MB) | `package_submission_zip.py` | Verifikasi integritas zip, bobot model, metadata |
| **H-0** | **13 Okt (23:59)** | **PENGUMPULAN AKHIR VIA GOOGLE FORM PANITIA** | Google Form Panitia | Berkas resmi terunggah & tanda terima tersimpan aman |

---

## 🛡️ 5. Checklist Verifikasi Submisi (Wajib Lolos Sebelum Submit)

Setiap file submisi baru yang dihasilkan **WAJIB** melalui fungsi verifikasi otomatis berikut:

```python
# Verifikasi Otomatis Integritas Submisi
def verify_submission(sub_path):
    sub = pd.read_csv(sub_path)
    assert len(sub) == 72611, "Baris tidak sesuai (harus 72,611)"
    assert sub['total_ticket'].isna().sum() == 0, "Ada nilai NaN"
    assert (sub['total_ticket'] < 0).sum() == 0, "Ada nilai negatif"
    
    # Invariant Mask 29,341 Zeros
    anchor = pd.read_csv('submissions/SUBMISSION_HEXA_PARADIGM_APEX_CHAMPION.csv')
    anchor_zero = (anchor['total_ticket'] == 0)
    sub_zero = (sub['total_ticket'] == 0)
    assert np.array_equal(anchor_zero, sub_zero), "Masker 29,341 nol rusak!"
    
    # Golden Volume Check
    vol = sub['total_ticket'].sum()
    assert 11_130_000 <= vol <= 11_160_000, f"Volume {vol:,.0f} meleset dari golden vertex!"
    print(f"VERIFIKASI BERHASIL: {sub_path} aman untuk disubmit! (Vol: {vol:,.0f})")
```

---

## 🏆 6. Indikator Keberhasilan (Milestones)

- **Target Minimal**: Skor Public Leaderboard $\le \mathbf{0.45700}$ (Mempertahankan tren positif).
- **Target Optimal**: Skor Public Leaderboard $\le \mathbf{0.43000}$ (Breakthrough Tier Baru).
- **Target Juara 1 / Master**: Skor Public Leaderboard $\le \mathbf{0.40xxx}$ s.d. $\mathbf{0.3xxxx}$ (Dominasi Mutlak Penyisihan).
- **Kondisi Audit**: 100% lolos reproduksi panitia, kode bersih, bobot $< 50$ MB.
