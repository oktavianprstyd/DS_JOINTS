# DS JOINTS X INSPIRE UGM 2026
## ROADMAP PERBAIKAN & LANJUTAN MENUJU PERFORMA LEADERBOARD MAKSIMAL

*Berdasarkan audit repo terbaru `DS_JOINTS-main`*

---

### STATUS TERKINI
Public leaderboard best resmi saat ini: **`0.46432`** (🏆 **ALL-TIME PERSONAL BEST!**).
Dihasilkan oleh: `submissions/submission_star_power_champion_master_70_30.csv` (Perpaduan 70% PB 0.46524 + 30% Star Power x Cultural Affinity SOTA dengan Invariant Zero-Preservation).
Local OOF all-time record terendah yang dicatat: **`0.34002`** (Managerial 5-Pillar SOTA: CatBoost GPU MASE 0.34482, AUC 0.9295).

**Riwayat Kemajuan Submisi Resmi Terkini:**
1. `submission_star_power_champion_master_70_30.csv` : **`0.46432`** (All-Time PB Resmi) 🏆
2. `submission_quantile_nelder_pb_70_30.csv`         : **`0.46469`** (Nelder-Mead + Quantile GBDT tau=0.45)
3. `submission_master_champion_70_30.csv`            : **`0.46524`** (Previous PB)
4. `submission_upgrade_sota60_anchor40.csv`          : **`0.46562`**
5. `submission_deep_sota_v11_golden_tri.csv`         : **`0.46605`** (V11 Deep SOTA + Capacity Reconstruction)
6. `submission_breakthrough_precision_dual.csv`      : **`0.46615`** (Failed zero-mask alteration -> Established 29,341 zero invariant)
7. `submission_upgrade_v10_golden_tri_blend.csv`     : **`0.46665`** (V10 Scale-Aware Tri-Blend)
8. `submission_clean_sota_roadmap.csv`               : **`0.46832`** (Roadmap Clean SOTA Pure V9) 🚀
9. `submission_podium_roadmap_blend.csv`             : **`0.46888`**
10. `submission_podium_blend_anchor_80_90f_20_zp.csv`: **`0.46890`**
11. `submission_hurdle_top.csv`                      : **`0.47303`** (Initial Anchor)
12. `submission_clean_consecutive_master.csv`        : **`0.48908`** (Over-Shrunk Baseline)

Dokumen ini adalah checklist eksekusi teknis. Tujuannya bukan menjanjikan peringkat tertentu, melainkan memaksimalkan kualitas prediksi dengan eksperimen yang terukur, validasi yang ketat, dan submission yang benar-benar reproducible.

> **Legenda Prioritas:**
> - $P0 =$ Wajib dibereskan
> - $P1 =$ Prioritas tinggi
> - $P2 =$ Optimasi
> - $P3 =$ Eksplorasi

---

## 1. Executive Summary

| Area | Kondisi Sekarang | Keputusan |
| :--- | :--- | :--- |
| **Public LB** | **`0.46432`** | 🏆 **ALL-TIME PERSONAL BEST!** Pertahankan `submission_star_power_champion_master_70_30.csv` sebagai patokan utama. |
| **Local OOF** | **`0.34002`** | Rekor terendah dicapai oleh Managerial 5-Pillar SOTA (27 fitur manajerial). |

| **Local OOF** | $0.34826$ | Berguna untuk diagnosis, tetapi jangan dianggap sebagai estimasi hidden-test tanpa *nested calibration / temporal checks*. |
| **Leak-free priors** | Sudah diterapkan di `run_roadmap_experiments.py` | Pertahankan. |
| **Safe categorical encoding** | Sudah diterapkan untuk XGBoost | Pertahankan. |
| **Per-horizon hurdle** | Sudah diterapkan | Pertahankan dan uji hanya dengan validation yang lebih realistis. |
| **Final notebook** | Telah disinkronkan dengan pipeline 98-fitur dan PB terbaru | **P0** - Pastikan sinkron 100% dengan pipeline terbaik. |
| **Final artifact** | Telah memuat model ensemble dan fungsi inference | **P0** - Satukan model + weights + thresholds + blend + feature pipeline. |
| **Calibration & Validation** | Modul 3-lapis validasi (Cold-Start, Temporal, Cold-Start+Temporal) & Nested Calibration telah diimplementasikan di `validation_framework.py`. | **P1 (SELESAI)** - Siap digunakan untuk benchmark arsitektur berikutnya. |

### KESIMPULAN UTAMA
Jangan masuk ke eksperimen model baru sebelum *source of truth* final sudah tunggal. Saat ini risiko terbesar adalah mismatch antara pipeline terbaik, notebook yang dikumpulkan, artifact inference, dan metode validasi.

---

## 2. Apa yang Sudah Bagus dan Jangan Dirusak

* Target ratio $z = \text{total\_ticket} / \text{scale}$ sudah selaras secara matematis dengan MASE ketika objective menggunakan $\text{MAE} / \text{L1}$.
* *Clean consecutive alignment* sudah digunakan untuk membentuk konteks $D1-D3$ dan target $D4-D10$ pada movie yang memenuhi jendela berurutan.
* *Context priors* pada roadmap sudah dihitung per training fold melalui `fit_context_priors()`.
* XGBoost categorical mapping sudah memiliki *vocab* dari training fold dan *unknown bucket* $0$.
* *Shared hurdle + per-horizon hurdle* sudah tersedia dan *GPU-enabled*.
* *Validation framework* sudah melaporkan overall MASE, horizon MASE, zero-rate error, volume ratio, low-scale MASE, dan cold-start MASE.
* Semua randomness utama diarahkan ke $\text{SEED} = 2026$.

> ⚠️ **JANGAN DILAKUKAN SEKARANG:** Jangan mengganti seluruh arsitektur hanya karena ingin model yang lebih kompleks. Pertahankan baseline DS JOINTS 2026 yang menghasilkan $0.46832$ sebagai *anchor* eksperimen.

---

## 3. P0 - Perbaikan Wajib Sebelum Eksperimen Baru

### 3.1 Jadikan `run_roadmap_experiments.py` sebagai Source of Truth
File yang menjadi pusat pipeline terbaru adalah `run_roadmap_experiments.py`. `solution.ipynb` masih memuat score/pipeline lama. Jangan mengklaim notebook sebagai final sebelum notebook menjalankan pipeline yang sama dengan submission PB terbaru.

| File | Masalah | Tindakan Konkret | Acceptance Check |
| :--- | :--- | :--- | :--- |
| `solution.ipynb` | Masih menyebut $0.46890$ dan memakai pipeline lama di beberapa sel. | Regenerate notebook dari pipeline final; update tabel skor, model, artifact path, dan inference. | *Run All* selesai tanpa manual patch; output akhir identik dengan script final. |
| `run_roadmap_experiments.py` | Menghasilkan V9 dan submission terbaru. | Pertahankan sebagai training reference; refactor bagian finalization agar artifact menyimpan seluruh state V9. | Satu command menghasilkan `metrics.json`, model package, dan submission. |
| `feature_engineering.py` | Sudah kaya fitur dan fold-safe input via priors. | Jaga API `build_features`; jangan hard-code target. | Unit check tidak menggunakan target future saat transform validation/test. |
| `validation_framework.py` | Sudah punya metric suite. | Tambahkan *calibration-safe split* dan *temporal windows*. | Tidak ada parameter threshold/weight dipilih dari evaluation fold. |
| `.gitignore` / `weights` | Weights dikecualikan dari repo, tetapi paket final harus memuat artifact. | Pisahkan repo development vs submission package; pastikan artifact final $< 200$ MB. | Unzip submission dan jalankan inference pada *clean environment*. |

### 3.2 Rebuild Artifact agar Benar-benar Mereplikasi V9
Saat ini artifact *shared ensemble* disimpan lengkap, tetapi fungsi `predict_from_artifact()` hanya memakai shared hurdle dengan threshold $0.50$. Itu belum identik dengan V9 yang memakai shared + per-horizon + optimal weights + optimal thresholds.

Artifact final harus menyimpan minimal:
1. *Shared fold models*: XGBoost classifier/regressor + CatBoost classifier/regressor + `cat_maps` per fold.
2. *Per-horizon models* $D4-D10$ dan `cat_maps` yang sesuai.
3. *Feature list* dan feature version/hash.
4. *Blend weights* $w_{\text{shared}}$ dan $w_{\text{horizon}}$.
5. *Threshold* $D4-D10$.
6. SEED, model hyperparameters, dan metadata dataset/version.
7. Preprocessing state yang dibutuhkan untuk membentuk fitur test.

```python
artifact = {
    'version': 'roadmap_v2',
    'features': features,
    'shared_fold_models': shared_fold_models,
    'per_horizon_models': per_horizon_models,
    'blend_weights': {'shared': w_shared, 'horizon': w_horizon},
    'thresholds': opt_thresholds,
    'seed': SEED,
}
```

---

## 4. P1 - Validasi Harus Lebih Dekat ke Kondisi Hidden Test

### 4.1 Masalah Saat Ini
GroupKFold by movie sudah bagus untuk menguji cold-start movie. Namun V9 memilih blend weights dan threshold dari OOF yang sama lalu melaporkan OOF yang sudah dikalibrasi pada data tersebut. Ini dapat membuat angka lokal lebih optimistis.

### 4.2 Validation Baru: 3 Lapis

| Validation | Tujuan | Desain |
| :--- | :--- | :--- |
| **A - Movie Cold Start** | Menguji film yang tidak pernah terlihat saat training. | `GroupKFold` by `movie_title`; priors fit hanya dari fold train. |
| **B - Temporal** | Menguji drift antar periode. | Rolling windows: train periode awal $\rightarrow$ 3-day history $\rightarrow$ 7-day forecast pada periode sesudahnya. |
| **C - Cold Start + Temporal** | Meniru kombinasi kondisi test. | Pilih film yang baru muncul pada periode validasi, berikan hanya history $D1-D3$, prediksi $D4-D10$ pada periode sesudahnya. |

> **TARGET VALIDATION:** Model baru hanya dianggap "lebih baik" bila peningkatan tidak hanya muncul pada satu validation. Catat Overall MASE, $D4-D10$ MASE, zero-rate error, volume ratio, low-scale MASE, dan cold-start MASE.

---

## 5. P1 - Nested / Calibration-Safe Threshold & Blend

Ganti proses optimasi weight/threshold yang mengambil keputusan langsung dari seluruh OOF menjadi prosedur yang memisahkan data untuk fit model dan calibration.

| Langkah | Data | Yang Boleh Dilakukan |
| :--- | :--- | :--- |
| **Inner train** | Fold A | Fit model dan feature priors. |
| **Calibration set** | Fold B | Cari blend weight dan threshold. |
| **Outer validation** | Fold C | Ukur MASE final; tidak boleh dipakai memilih parameter. |
| **Repeat** | Berputar pada beberapa split | Rata-rata/median performa outer validation. |

```python
# Konsep sederhana
for outer_train, outer_val in temporal_or_group_splits:
    train_part, calib_part = split_calibration(outer_train)
    models = fit_models(train_part)
    calib_pred = predict(models, calib_part)
    params = optimize_blend_and_threshold(calib_pred, calib_part.y, calib_part.scale)
    val_pred = apply_params(predict(models, outer_val), params)
    score = compute_mase(outer_val.y, val_pred, outer_val.scale)
```

---

## 6. P2 - Eksperimen yang Dilanjutkan Setelah P0/P1

| Urutan | Eksperimen | Kenapa | Kriteria Lanjut |
| :---: | :--- | :--- | :--- |
| **1 & 2** | Normalized target: raw-ticket vs $z = y/\text{scale}$<br>Per-horizon direct hurdle $D4-D10$ | Langsung selaras dengan MASE; sudah cocok dengan desain sekarang.<br>Horizon punya pola berbeda; sudah terbukti berguna secara lokal. | Pilih hanya jika menang konsisten pada validation $A/B/C$.<br>Pertahankan jika stabil pada temporal validation. |
| **3** | Context priors multi-level | Film, cinema, city, pair-context memberi sinyal saat cold-start. | Tidak boleh memakai future target pada validation/test. |
| **4** | Weekend/holiday interaction | Target mencakup hari yang berbeda dan holiday effects. | Uji per-horizon, jangan diasumsikan universal. |
| **5** | KNN archetype transfer | Memindahkan kurva retensi dari film historis yang mirip. | Harus dihitung dari data yang available sebelum forecast; validasi terhadap leakage. |
| **6** | LightGBM sebagai engine tambahan | Sudah ada ide di plan.md, tetapi bukan prioritas sebelum validation rapi. | Masuk ensemble hanya jika OOF/temporal benar-benar meningkat. |
| **7** | Meta-stacking | Potensial mengoptimalkan blend per kondisi, tetapi risiko overfit tinggi. | Hanya dengan OOF bertingkat / cross-fit meta learner. |
| **8** | Weekend spike calibration | Menarik, tetapi mudah overfit jika hard-coded. | Wajib OOF/temporal evidence sebelum digunakan. |

---

## 7. P2 - Feature Engineering Prioritas Tinggi

| Kelompok | Fitur Inti | Catatan |
| :--- | :--- | :--- |
| **Opening dynamics** | `ticket_d1/d2/d3`, `ratio_d2_d1`, `ratio_d3_d2`, `ratio_d3_d1`, `ticket_accel` | Sudah ada; lakukan ablation agar tidak menambah noise tanpa bukti. |
| **Cinema context** | `cinema_prior_tickets`, `cinema_prior_occ`, `cinema_prior_shows`, `local_vs_nat_occ`, `cinema_to_city_share` | Wajib fold-safe pada training/validation. |
| **Movie context** | `nat_scale`, `nat_cinemas`, `nat_cities`, `nat_avg_occ`, `nat_avg_shows`, `trend` | Penting untuk cold-start film. |
| **Calendar** | `day_of_week`, weekend flags, holiday adjacency, bridge day, `effective_weekend` | Uji per horizon. |
| **Movie metadata** | genre, age_rating, director/producer experience, major studio, star director | Berguna saat movie unseen, tetapi hindari target-derived metadata. |
| **Format** | `is_3d`, `is_imax`, `is_uncut`, `ceil` | Pertahankan format sebagai signal terpisah. |
| **Archetype** | fingerprint + nearest-neighbor retention | Tambahkan setelah validation framework siap. |

---

## 8. P2 - Error Analysis: Cari Lokasi Error, Bukan Sekadar Skor

Setelah setiap run, simpan prediksi OOF dan buat tabel error per kondisi. Fokuskan investigasi pada kondisi yang paling merusak MASE.

| Slice | Metric | Pertanyaan |
| :--- | :--- | :--- |
| **$D4-D10$** | MASE per horizon | Hari mana paling sulit? |
| **Low scale $\le 5$** | MASE | Apakah model mengorbankan pasangan kecil? |
| **Active vs zero** | MASE / zero-rate error | Apakah classifier terlalu agresif atau terlalu shrink? |
| **Cold start** | MASE | Apakah performa turun drastis pada movie unseen? |
| **Weekend vs weekday** | MASE | Apakah spike/rebound under-predict? |
| **Sparse history** | MASE | Apa yang terjadi saat history test hanya 1-2 baris? |
| **Cinema/city buckets** | MASE | Apakah model terlalu bergantung pada prior tertentu? |

> **ATURAN EKSPERIMEN:** Satu eksperimen = satu hipotesis. Simpan feature version, model version, validation A/B/C, OOF score, submission diagnostics, dan keputusan keep/revert.

---

## 9. PO - Final Submission & Reproducibility Checklist

* [ ] **Notebook standalone:** Bisa dijalankan dari awal di clean environment tanpa file lokal tersembunyi.
* [ ] **Data source:** Hanya data yang diizinkan panitia; tidak ada external data yang melanggar cutoff.
* [ ] **Randomness:** `SEED = 2026` / `random_state` konsisten pada semua komponen stochastic.
* [ ] **Artifact:** Model package $\le 200$ MB dan berisi seluruh komponen final inference.
* [ ] **Inference parity:** Prediksi dari artifact sama dengan submission final pada byte/angka yang dapat diverifikasi.
* [ ] **Submission schema:** Kolom tepat: `id`, `total_ticket`; row count $72,611$; id unik dan urut.
* [ ] **Non-negative:** Semua ticket $\ge 0$ dan tidak ada NaN/inf.
* [ ] **Version lock:** CatBoost / XGBoost / Python environment tercatat.
* [ ] **Experiment log:** Simpan commit hash / version name yang menghasilkan submission.
* [ ] **Final backup:** Simpan anchor $0.46832$ candidate dan semua predecessor penting sebelum submission baru.

---

## 10. File-by-File Action List

| File | Prioritas | Perubahan |
| :--- | :---: | :--- |
| `run_roadmap_experiments.py` | **P0** | Refactor artifact final; simpan shared + horizon + blend weights + thresholds + feature metadata; buat single final inference path. |
| `validation_framework.py` | **P1** | Tambahkan temporal split, calibration split, nested evaluation, dan helper untuk sparse-history cases. |
| `feature_engineering.py` | **P1** | Pertahankan fold-safe priors. Tambahkan versioning dan unit tests untuk $D1-D3$ mapping, missing history, serta no-future-target guarantee. |
| `solution.ipynb` | **P0** | Regenerate dari pipeline final; hapus narasi/score $0.46890$ sebagai final; tampilkan score dan artifact path terbaru. |
| `build_solution_notebook.py` | **P0** | Jadikan generator notebook sebagai satu-satunya sumber build notebook final. |
| `plan.md` | **P2** | Ubah dari daftar ide menjadi experiment backlog dengan status KEEP/REJECT dan evidence. |
| `PROGRESS_SUMMARY.md` | **P2** | Sinkronkan angka, nama submission, dan pipeline final agar tidak ada score lama yang dianggap terbaru. |
| `README.md` | **P2** | Update quick start agar benar-benar cocok dengan file yang ada; hapus referensi script yang tidak ada. |
| `package_submission_zip.py` | **P0** | Paketkan notebook + artifact final dan jalankan smoke test unpack/re-run. |
| `.gitignore` | **P2** | Tidak harus commit weights ke repo, tetapi pastikan packaging script menangkap artifact final. |

---

## 11. Urutan Eksekusi yang Disarankan

1. **Freeze anchor:** simpan metadata submission $0.46832$ dan jangan menimpa file tersebut.
2. Perbaiki artifact final agar benar-benar sama dengan V9 yang menghasilkan submission terbaru.
3. Regenerate `solution.ipynb` dari pipeline final dan tes reproduksi pada sample test.
4. Bangun validation A/B/C: movie cold-start, temporal, dan cold-start + temporal.
5. Pindahkan threshold/weight calibration ke calibration split atau nested procedure.
6. Jalankan baseline lama pada validation baru. Ini menjadi angka pembanding yang jujur.
7. Jalankan normalized-target comparison, per-horizon hurdle, dan context priors sebagai eksperimen utama.
8. Baru jalankan KNN archetype, LightGBM, stacking, dan weekend calibration secara selektif.
9. Bandingkan bukan hanya Overall MASE, tetapi horizon, low-scale, cold-start, zero-rate error, volume ratio, dan stability antar split.
10. Pilih kandidat submission berdasarkan bukti validation + diagnostics, lalu uji leaderboard secara bertahap sambil menjaga anchor.

---

## 12. Definition of Done

**KANDIDAT FINAL SIAP SUBMIT HANYA JIKA SEMUA TERPENUHI:**
1. Notebook dan artifact identik pipeline-nya.
2. Prediction artifact dapat mereplikasi submission final.
3. Validation $A/B/C$ tersedia.
4. Calibration tidak memakai outer validation secara langsung.
5. Submission lolos schema checks.
6. Model package $\le 200$ MB.
7. Semua randomness terkunci.
8. Eksperimen terbaru memiliki evidence lebih baik daripada anchor secara lokal sebelum dipertimbangkan untuk leaderboard.

> 💡 **PRINSIP KERJA:**  
> $\text{VALIDATE} \rightarrow \text{DIAGNOSE} \rightarrow \text{CHANGE ONE THING} \rightarrow \text{REVALIDATE} \rightarrow \text{SUBMIT} \rightarrow \text{LOG}$