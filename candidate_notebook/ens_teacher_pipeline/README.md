# Ensemble-teacher pipeline (PREPARED, NOT RUN)

| Step | Folder | Kaggle kernel id | What it does |
|---|---|---|---|
| 1a | raptorteacher_m224/ | hosen42/rsna-knee-m224-teacher | M224 (+ mirror) predictions on the 4,407 train studies → m224_train_raw.csv (~1–1.5 h GPU) |
| 1b | r7teacher_0/, r7teacher_1/ | hosen42/rsna-knee-r7-teacher-0 / -1 | R7 predictions on train, 2 shards → r7_train_shard{0,1}.csv (~3–4.5 h each) |
| 2 | gate_ens_teacher.py | local CPU | checks the teacher dumps, writes ens_train_raw.csv (T2 = rank blend 0.325/0.175/0.5 on Mirror's marginal) |
| 3 | — | dataset hosen42/rsna-knee-ens-teacher-train | upload T2 (not created yet) |
| 4 | gpjwstudent4_hosen42/ | hosen42/rsna-knee-gpjw-student4 | trains students E75t / E75s on 0.25·v0 + 0.75·T2 (~8 h); needs hosen42's private cache kernels rsna-knee-gpjw-cache-0..4 |

**Caveat:** the T2 teacher copies the old 0.956 blend (Mirror + M224 + R7). The current 0.958 parent uses R6, so a re-run should rebuild the teacher around R6. Each kernel folder is pushable with `kaggle kernels push -p <folder>`. The id inside kernel-metadata.json is hosen42's; change it for another account.
