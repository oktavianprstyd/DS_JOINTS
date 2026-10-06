# 🧪 Experiments & Research Archive

Folder ini berisi seluruh skrip riset, pengujian hipotesis arsitektur, pelatihan model historis per iterasi, dan generator submisi kandidat selama kompetisi **JOINTS X INSPIRE UGM 2026**.

---

## 📂 Struktur Isi `experiments/`

### 1. Advanced Architecture & Training Engines (Iterasi SOTA)
- `train_star_power_affinity_sota.py`: Engine pelatihan Star Power $\times$ WOM + Kalender (pencetak Kaggle PB **0.46432**).
- `train_managerial_holiday_sota.py`: Engine 5-Pilar Manajerial & Hierarki Libur (rekor OOF terendah **0.34002**).
- `train_per_horizon_specialist.py`: Sub-model terpisah per hari penayangan D4 s.d. D10 (LGBM Q45 + CatBoost GPU).
- `train_quantile_nelder_combo.py`: GBDT Pinball Loss ($\tau=0.45$) + optimasi metrik langsung Nelder-Mead.
- `train_swa_resnet_clustering_engine.py`: PyTorch 1D ConvNet dengan Stochastic Weight Averaging (SWA) & entity clustering.
- `train_upgrade1_cultural_affinity_engine.py`: Multiplier cultural affinity genre $\times$ kota/bioskop (OOF 0.34120).
- `tune_cnn_xgboost_master.py`: Hybrid PyTorch SE-ResNet-1D + CUDA XGBoost dengan 2D grid search.
- `train_scale_aware_sota.py`: Pemodelan Two-Tier Scale-Aware Hurdle untuk bioskop mikro vs makro.
- `train_deep_sota_model.py`: Pelatihan Deep GBDT (Depth 7 XGBoost CUDA + CatBoost GPU).
- `train_cnn_xgboost_hybrid.py`: Arsitektur hibrida awal CNN feature extraction + XGBoost head.
- `train_grandmaster_internal_sota.py`: Pipeline internal Grandmaster dengan fitur interaksi non-linear.
- `train_grandmaster_sota_30s.py`: Iterasi Grandmaster 30-fitur utama.
- `train_podium_90f_gpu.py`: Model 90-fitur GPU (Skor Leaderboard 0.46890).
- `train_plan_b_clean_consecutive.py`: Implementasi fondasi 183 film beruntun bebas jeda (*clean consecutive*).
- `run_roadmap_experiments.py`: Script runner otomatis untuk serangkaian eksperimen roadmap.

### 2. Candidate Builders & Calibration Generators
- `create_star_power_champion_submission.py`: Generator submisi blend Star Power Champion (PB 0.46432).
- `create_upgrade1_champion_submission.py`: Generator submisi Cultural Affinity Upgrade #1.
- `create_managerial_record_submissions.py`: Generator submisi berbasis model manajerial OOF 0.34002.
- `create_master_candidate.py`: Generator kandidat Master Champion 70/30 (PB 0.46524).
- `create_next_gen_candidates.py`: Generator variasi bobot kandidat generasi berikutnya.
- `create_breakthrough_candidates.py`: Eksperimen variasi penembus batas akurasi.
- `generate_breakthrough_master.py`: Generator master submisi breakthrough.
- `execute_step1_soft_calibration.py`: Eksekusi peredaman kontinu Soft Bayesian Transition (Step 1).
- `execute_step2_jedidi_decay.py`: Eksekusi kurva peluruhan eksponensial Jedidi-Krider (Step 2).
- `execute_step3_exhibitor_hazard.py`: Eksekusi hazard pemotongan layar bioskop Kamis (Step 3).
- `build_grand_champion.py`: Script blending submisi Grand Champion awal.
- `build_gacor_submission.py`: Generator submisi eksperimen gacor.

### 3. Historical Training Suites & Benchmarks
- `benchmark_trio_ensemble_bayes.py`: Pipeline pengujian Trio Ensemble (LGBM, CatBoost, XGBoost) + Bayesian shrinkage.
- `benchmark_improvements_on_unbiased.py`: Eksperimen penambahan 90+ fitur pada dataset unbiased.
- `benchmark_sota_night_research.py`: Eksperimen optimasi hyperparameter semalam.
- `train_clean_hard_hurdle.py`: Training suite Clean Consecutive + Hard Hurdle (threshold diskret).
- `train_deep_hurdle_gpu.py`: Pelatihan PyTorch Deep ResHurdleNet di CUDA.
- `train_hurdle_ensemble.py`: Pipeline ensemble Two-Stage Hurdle awal (skor Kaggle: 0.47303).
- `train_plan_b_7horizon_bayes.py`: Implementasi 14-Segmen Bayesian Shrinkage.
- `train_grandmaster_system.py`, `train_gpu_master.py`, `train_champion_model.py`, `train_super_ensemble.py`, `train_triple_hurdle_master.py`, `train_lgbm.py`, `train_deep_model.py`, `train_ensemble.py`, `train_grandmaster_unbiased.py`: Arsip pipeline pelatihan iterasi sebelumnya.

---

## 🚀 Cara Menjalankan

Seluruh eksperimen di folder ini telah dilengkapi *root path bootstrapper*. Anda dapat menjalankannya dari root direktori proyek:
```bash
python experiments/train_managerial_holiday_sota.py
python experiments/compare_subs.py
```
atau langsung dari dalam folder `experiments/`:
```bash
cd experiments
python compare_subs.py
```
