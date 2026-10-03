# 🧪 Experiments & Research Archive

Folder ini berisi seluruh script riset, benchmark, pengujian hipotesis, dan skrip pelatihan historis selama pengembangan model kompetisi **JOINTS X INSPIRE UGM 2026**.

---

## 📂 Struktur Isi `experiments/`

### 1. Benchmark & Research
- `benchmark_trio_ensemble_bayes.py`: Pipeline pengujian Trio Ensemble (LGBM, CatBoost, XGBoost) + Bayesian shrinkage.
- `benchmark_improvements_on_unbiased.py`: Eksperimen penambahan 90+ fitur pada dataset unbiased.
- `benchmark_sota_night_research.py`: Eksperimen optimasi hyperparameter semalam.
- `reproduce_and_improve_top.py`: Uji reproduksi skor benchmark awal.
- `experiment_hurdle_survival.py`: Eksperimen survival analysis untuk probabilitas screening dropout.

### 2. Diagnostics, Analysis & Stress Testing
- `analyze_clean_strategies.py`: Analisis perbandingan strategi pembersihan data rilis sneak preview.
- `analyze_discrepancy.py`: Investigasi selisih skala pembagi $s_p$ dan prediksi tiket.
- `compare_all_remedies.py`: Komparasi komprehensif berbagai strategi mitigasi defisit volume tiket.
- `compare_subs.py`: Utility perbandingan statistik antar file submisi CSV.
- `diagnose_bad_days.py`: Analisis error per hari penayangan (D4 s.d. D10).
- `evaluate_day_by_day.py`: Evaluasi metrik MASE breakdown per hari penayangan.
- `evaluate_quad_ensemble.py`: Evaluasi ensemble 4 model.
- `evaluate_quick_fixes.py`: Evaluasi perbaikan cepat pasca-proses.
- `internal_stress_test.py`: Uji ketahanan model terhadap anomali input.
- `test_volume.py`: Pengujian dan validasi total volume tiket nasional pada data uji.
- `scratch_test_zero_mase.py`: Uji matematis pengaruh prediksi nol terhadap nilai MASE.

### 3. Hypothesis & Architecture Testing
- `test_bayes_shrinkage.py`: Pengujian peredaman daya Bayes kontinu (Continuous Bayesian Power Shrinkage).
- `test_deep_residual.py`: Pengujian arsitektur PyTorch Deep Residual Hurdle.
- `test_deep_on_unbiased.py`: Uji deep learning pada data unbiased.
- `test_empirical_and_bayes.py`: Komparasi transisi empiris vs Bayesian shrinkage.
- `test_full_features_hurdle.py`: Pengujian Two-Stage Hurdle dengan fitur lengkap.
- `test_knn_archetype_matching.py`: Eksperimen kemiripan profil film dengan KNN archetype.
- `test_alternative_tweedie_quantile.py`: Eksperimen loss function alternatif (Tweedie vs Quantile).

### 4. Historical Training Suites
- `train_clean_hard_hurdle.py`: Training suite Clean Consecutive + Hard Hurdle (threshold diskret).
- `train_deep_hurdle_gpu.py`: Pelatihan PyTorch Deep ResHurdleNet di CUDA.
- `train_hurdle_ensemble.py`: Pipeline ensemble Two-Stage Hurdle awal (skor Kaggle: 0.47303).
- `train_plan_b_7horizon_bayes.py`: Implementasi 14-Segmen Bayesian Shrinkage.
- `train_grandmaster_system.py`: Pelatihan sistem Grandmaster multi-model.
- `train_gpu_master.py`: Pipeline master GPU.
- `train_champion_model.py`: Pelatihan model Champion terdahulu.
- `train_super_ensemble.py`, `train_triple_hurdle_master.py`, `train_lgbm.py`, `train_deep_model.py`, `train_ensemble.py`, `train_grandmaster_unbiased.py`: Pipeline pelatihan iterasi sebelumnya.

### 5. Builders & Visualizations
- `build_grand_champion.py`: Script blending submisi Grand Champion terdahulu.
- `build_gacor_submission.py`: Script generator submisi eksperimen.
- `generate_eda_visualizations.py`: Generator grafik dan visualisasi di folder `eda_plots/`.
