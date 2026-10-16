# Report 17 ও বর্তমান H960 draft: validation

তারিখ: ১৭ অক্টোবর ২০২৬, Asia/Dhaka। Scope: report/code-এর static validation। Notebook cells, builder, archived scripts, inference, training, blending evaluation বা Kaggle submission চালানো হয়নি। Report 17 এবং existing implementation পরিবর্তন করা হয়নি।

**Verdict: মূল E1 hypothesis যুক্তিসংগত; report-এর কয়েকটি দাবি সংশোধন প্রয়োজন। H960/H958M syntax ও baseline preservation checks pass; runtime correctness, predictive improvement এবং submission readiness প্রতিষ্ঠিত নয়।**

## আগে সমাধানযোগ্য findings

### 1. [P1] Reader setup failure হলে H960-এর promised baseline fallback পাওয়া যায় না

Location: `ours_leg/build_h960.py:94`; `ours_leg/student_leg.py:59`; generated H960 **c13** এবং **c14**। Report 17-এর fallback দাবি: line172।

Builder reader stage-কে baseline-এর শেষ blend cell-এর **আগে** ঢোকায়। H960-তে original c8 এখন c14; optional reader setup চলে c13-তে। `_find("cnxt_v0_fold0.pt")`, `_find("test_series.csv")` বা `_ckpts(...)` ব্যর্থ হলে exception ওঠে; ওই setup পুরোটা try/fallback দ্বারা guarded নয়। ফলে final baseline `submission.csv` তৈরি হওয়ার আগেই notebook থামতে পারে। Inference subprocess timeout handle করা আছে; setup exception-এর একই fallback নেই।

এটি missing dependency-এর concrete failure path; actual hidden run failure ঘটেছে—এমন দাবি নয়। **Proposed correction:** baseline c8 আগে সফলভাবে শেষ করে immutable baseline CSV snapshot রাখা; optional reader-এর পুরো setup/run guarded করা; failure হলে experiment invalid receipt দিয়ে baseline restore। কোনো change apply হয়নি।

### 2. [P1] `H957_FINAL blended` একটি সম্পূর্ণ পাঁচ-student experiment-এর নিশ্চয়তা দেয় না

Location: H960 **c16**, source lines16–30 ও41–42; patch উৎস `ours_leg/student_leg.py:118`। Report 17-এর go condition line251।

Final cell target ID-তে reindex করার পরে cell-level not-null fraction <0.97 হলে reject করে, কিন্তু small incomplete subset থাকতে দেয়। R5 missing row-এর student rank parent rank থেকে linear fit করে পূরণ করে, তারপরও status=`blended` লেখে। ফলে receipt-এ `reader_failed_rows>0` থাকলেও শুধু `blended` দেখে go দেওয়া যায়। Constant-column check-এ `.any()` আছে: একটি target varying হলেই check পাস, সব ১২টি varying হওয়ার নিশ্চয়তা নেই।

এটি reported complete-row formula থেকে আলাদা fallback behavior। **Proposed correction:** initial controlled E1-এর জন্য exact UID set/unique IDs, ১২ target-এর valid finite probabilities, `reader_failed_rows=0`, সব target nonconstant এবং active checkpoint identities check করতে হবে। Partial/imputed output হলে complete E1 বলে গ্রহণ নয়। কোনো predictions evaluate করা হয়নি।

### 3. [P2] Same-score team-best switch-কে improvement/rejection rule করা সমর্থিত নয়

Location: `audit/17_research_0958_to_0959.md:206`, `:256`; historical inference `:100`–`:103`।

একই displayed 0.958-এ ID switch হলে unrounded improvement এবং switch না হলে reject—এই দুই সিদ্ধান্তের জন্য precise scores ও verified best-selection/tie policy দরকার। বর্তমান evidence package-এ সে evidence নেই। নতুন snapshot-এ rank/cutoff claims-ও পুনরায় প্রমাণ করা যায়নি।

**Correction:** same displayed score → **Inconclusive**। Displayed improvement → **Public LB Observed**। Private generalization বা Validation Verified label নয়। Public best-selection signal থাকলে secondary evidence হিসেবে তার সীমা লিখতে হবে।

### 4. [P2] Baseline-এর learned postprocessing-কে absent বলা বিভ্রান্তিকর

Location: report17 line72 ও88; evidence `audit/18_evidence/r6_manifest_download.json`, `r6_fluid_download.json`, `r6_state_download.json`।

Final c8-এ আলাদা calibrator নেই। তবে active R6 manifest-এ **Effusion fluid stacker** এবং **Baker’s state stacker** আছে, যা final rank-এর আগেই model output বদলায়। “কোনো calibration নেই” pipeline-wide অর্থে ঠিক নয়।

**Correction:** “Final stage: rank blend, no additional calibrator; R6 internally uses learned Effusion/Baker’s stackers.” Strictly monotone output remap বনাম rank-changing feature stacker আলাদা রাখতে হবে।

### 5. [P2] Resolution table actual input contract দেখায় না

Location: report17 line86; baseline **c2** source lines119/232/279।

Checkpoint নাম `r384`, কিন্তু active `INPUT_RES=320`; loader সেটাই ফেরায় এবং TTA input `(94,3,320,320)` assert করে। R6 manifest `size=384`, `crop=352`।

**Correction:** “R384-named checkpoint: 384 canvas →320 crop; R224: 0.8 crop →224; R6:384 render →352 crop.” এটি observation; training mismatch বা bug প্রমাণ নয়।

### 6. [P2] Runtime ও statistical uncertainty সংখ্যার পর্যাপ্ত ভিত্তি নেই

Location: report17 line168,179,198,203।

20–30 GPU-minute, +0.5h এবং submission SD≈0.003-এর পক্ষে current baseline-এর measured hidden runtime, public sample size/prevalence বা paired uncertainty artifacts পাওয়া যায়নি। Hypothesis label থাকলেও নির্দিষ্ট সংখ্যা decision-কে অতিরিক্ত নিশ্চিত করে।

**Correction:** exact runtime/uncertainty **Unknown**; measured stage receipt পাওয়ার পরে estimate। 224 mirror অতিরিক্ত branch forward যোগ করে—এটুকু source থেকে বলা যায়, total wall time নয়।

আরও একটি code সীমা: `student_leg.py:74`-এ `max(60, remaining_budget)` budget শেষ হলেও অন্তত60s reader চালাবে; asset lookup/install timeout-এর বাইরে। তাই “reader কখনো9h অতিক্রম করাবে না” guarantee নয়।

### 7. [P2] Checkpoint availability এবং validation readiness এক নয়

Location: report17 line165 ও210; limitation line199–202।

পাঁচ student file local আছে, কিন্তু same-sample full baseline/student held-out predictions নেই। Old reader OOF বর্তমান full-data C-family checkpoints-এর OOF নয়। একই report-এ missing OOF স্বীকার করে “সব asset আছে” বলা execution এবং validation-এর অবস্থা গুলিয়ে ফেলে।

**Correction:** checkpoint-backed candidate পরিকল্পনা সম্ভব; **Requires new inference / Cannot currently validate**। Smoke/commit logs operational correctness সমর্থন করতে পারে; AUC improvement প্রমাণ নয়।

## ছোট factual corrections

| Report17 location | বর্তমান দাবি | সঠিক সীমিত বিবরণ |
|---|---|---|
| line75 | চার dataset public; missing dependency নেই | Notebook metadata-তে **তিন dataset + এক competition source**। Full dependency closure/runtime availability independently verified নয় |
| line87 | পাঁচ finding-এ per-finding weights | পাঁচ dictionary entries; Medial OA=.5 default-এর সমান; **চার non-default overrides** |
| line138 | GP26 reader range .008–.035 | Actual map **.003–.035**; Synovitis=.003, Effusion=.004; GP26 c21 |
| line138 | পাঁচ student ×.2 | পাঁচ student-এর **probability mean**, তারপর ওই aggregate-এর rank-কে .2 weight; প্রতি student .2 নয় |
| line153 | R7 proven redundant | Author-এর three-decimal no-difference observation; precise effect/error complementarity Unknown; lower priority বলা যায় |
| line163 | সম্পূর্ণ আলাদা model family | Different reader representation/target mix; R6-এর সঙ্গে ConvNeXt family, Raptor-এর সঙ্গে teacher ancestry shared; independence প্রমাণিত নয় |
| line194/240 | নয় cell byte-identical | Generated H960-তে **সাতটি original code cell exact**; title markdown c0 বদলেছে |
| line223 | 12-hex SHA prefix freeze | Current file full SHA সত্যিই মেলে; reproducible freeze assertion-এ **পুরো64-hex digest equality** ব্যবহার ভালো |
| line279 | current rank317, silver cutoff305 | এখন পরিদর্শিত evidence থেকে **Unverified**; historical snapshot থাকলে timestampসহ আলাদা cite প্রয়োজন |

## যেগুলো এই turn-এ pass হয়েছে

- তিন ZIP-এর **২৭৩/২৭৩ original file** এখনো disk copy-এর সঙ্গে byte-identical। Original notebook source বদলায়নি।
- Baseline full SHA-256: `00ca614beec23736346df921e3305742bf8533b7cfd69a208880022f899fa1d7`।
- H960: **১৭ cell/১৩ code cell**। Baseline c2,c3,c4,c5,c6,c7 হুবহু same indices-এ, c8 হুবহু H960 c14-এ।
- H958M: **৯ cell/৭ code cell**। Original code c2,c3,c6,c7,c8 unchanged; c4 mirror এবং c5 timeout পরিবর্তিত।
- দুই candidate-এর মোট **২০টি code cell** syntax-only AST parse পাস। Magics-এর marker line বাদ দিয়ে Python parse করা হয়েছে; কোনো cell চালানো হয়নি।
- `build_h960.py`, `student_leg.py`, modified `build_h958.py` AST parse পাস; import/execute করা হয়নি।
- H960 metadata-তে baseline-এর তিন dataset + reader assets + private student dataset আছে; internet off, pinned Docker আছে। Version-specific mounting/hash completeness runtime check বাকি।
- H960 এবং H958M notebook-এ recorded execution output নেই। Files থাকা runtime success বা scored result-এর প্রমাণ নয়।

## সিদ্ধান্ত

**Report validation: সংশোধন প্রয়োজন। Static syntax/baseline preservation: pass। Runtime ও model-performance validation: চালানো হয়নি। Submission readiness: অনুমোদনযোগ্য বলে প্রতিষ্ঠিত নয়।**

E1 এখনও সবচেয়ে যুক্তিসংগত first research candidate। আগে fallback/data-integrity acceptance criteria ঠিক করা এবং report-এর unsupported claims সীমিত করা দরকার। বর্তমান source বা notebook পরিবর্তন করা হয়নি; নতুন inference/experimental validation-এর আগের permission সীমা বহাল আছে।

সম্পূর্ণ reconstruction ও corrected research context: [Report18](18_verified_0958_to_0959_research_bn.md)। এই review report18-এর decision ব্যাখ্যা করে; report18-এর পরে workspace-এ দেখা নতুন drafts-ও সীমিত static inspection-এর অন্তর্ভুক্ত।
