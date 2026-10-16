# RSNA Knee: 0.958 baseline যাচাই ও 0.959+ গবেষণা পরিকল্পনা

তারিখ: **১৭ অক্টোবর ২০২৬, Asia/Dhaka**। কাজের ধরন: **read-only research, static code audit এবং experiment planning**।

**সিদ্ধান্ত:** প্রথম predictive experiment হিসেবে frozen 0.958 ensemble-এর শেষে আমাদের পাঁচটি existing Mirror-student-এর probability mean-কে rank করে **0.8 baseline / 0.2 students** যোগ করা সবচেয়ে যুক্তিসংগত। এটি **বর্তমান baseline-এর জন্য নতুন combination**, নতুন architecture আবিষ্কার নয়। আগের R7/S6 parent-এ student ব্যবহারের historical সমর্থন আছে; বর্তমান R6 parent-এর সঙ্গে complementary errors-এর প্রমাণ নেই। **এটি Requires new inference এবং Cannot currently validate; existing artifacts দিয়ে এখনই offline improvement মাপা সম্ভব নয়।** 0.959+, rank বা medal-এর নিশ্চয়তা নেই।

এই turn-এ কোনো training, inference, notebook cell, archived script, blending search, AUC evaluation, bootstrap বা Kaggle submission চালানো হয়নি। কোনো original notebook বা আগের report পরিবর্তন করা হয়নি। নতুন ফাইলগুলো শুধু এই report, static inventory ও evidence snapshots। Small JSON এবং source text পড়া হয়েছে; checkpoint download করা হয়নি। Token-এর contents report বা logs-এ রাখা হয়নি।

## প্রমাণ পড়ার নিয়ম

**Code Verified** = actual source পড়া/তুলনা; **Metadata Verified** = archive index, manifest বা API metadata; **Public LB Observed** = submission record-এ displayed score; **Validation Verified** = reproducible held-out evaluation। **Hypothesis** = পরীক্ষা বাকি; **Unknown** = প্রয়োজনীয় প্রমাণ নেই। এই গবেষণায় নতুন কোনো Validation Verified improvement তৈরি হয়নি। Manifest/notebook-এর পুরোনো AUC সংখ্যাগুলো recorded results, independent validation নয়।

সেলের নাম **c0 থেকে শুরু**; যেমন c8 হলো notebook-এর নবম cell। Baseline = `our_base_notebook/0.958_kishormorol__rsna-knee-helios-blend-fork.ipynb`; H958B = `our_submission_notebook/submission_h958b/rsna-knee-h958b.ipynb`; GP26 = `public_notebooks/0.958_217v_goodpjw2008__rsna-knee-coatnet-r6-2-5d-mil-lb-0-958.ipynb`। Embedded source-এর উল্লেখ যেমন `baseline c6 / SOURCES['compact_infer.py']`।

আগের `17_research_0958_to_0959.md` সিদ্ধান্তের সূত্র হিসেবে পড়েছি, মূল evidence হিসেবে গ্রহণ করিনি। নিচের নতুন checks সেই report-এর কয়েকটি অতিরিক্ত নিশ্চিত দাবি সংশোধন করে।

## Report A — সম্পূর্ণ archive inventory

| Archive | মূল ফাইল | ধরন | Uncompressed bytes | macOS ফাইল | ডিস্কে byte-identical |
|---|---:|---|---:|---:|---:|
| `our_base_notebook.zip` | 2 | 2 notebooks | 1,260,214 | 3 | 2/2 |
| `our_submission_notebook.zip` | 121 | 33 notebooks, 58 JSON, 25 Python, 3 Markdown, 1 CSV, 1 log | 12,028,121 | 182 | 121/121 |
| `public_notebooks.zip` | 150 | 114 notebooks, 34 JSON, 2 CSV | 41,788,015 | 152 | 150/150 |

**Metadata Verified:** ২৭৩টি মূল ফাইল hash করা হয়েছে; ১৪৯টি notebook JSON parse করে মোট ৩,৪৩৪টি cell-এর source hash ও recorded-output count সংগ্রহ করা হয়েছে। JSON parse failure নেই। Directory entries এবং ৩৩৭টি macOS metadata ফাইল আলাদা। File-by-file archive/path/type/size/hash/readability/author/slug/version/score/submission/dependency evidence আছে [archive_inventory.csv](18_evidence/archive_inventory.csv)-এ। Unknown field অনুমান করে পূরণ করা হয়নি। Checkpoint reference column-এ inactive helper-এর নামও থাকতে পারে; সেটি final ensemble membership-এর প্রমাণ নয়।

**পরিদর্শনের গভীরতা:** সব মূল ফাইলের byte/hash, সব notebook-এর cell structure/source এবং সব JSON-এর parseability পরীক্ষা হয়েছে। Baseline-এর সব code cell, embedded compact inference/trainer/pixel-cache source, আটটি pinned mainline module, GP26-এর final dataflow, H957/H958 family এবং নির্বাচিত public alternatives-এর relevant execution path গভীরভাবে পড়া হয়েছে। অন্য প্রতিটি public notebook-এর পূর্ণ methodology audit হয়েছে—এমন দাবি করা হচ্ছে না।

Archive SHA-256:

```text
our_base_notebook.zip       99394246314f0c0786964b7cdb8047d6f04987d1e799846b116a92d7f138a43e
our_submission_notebook.zip c46411bf915d5d9584c5eb92f9c94c4185d8ffeb47c54ecb421f9d6b3ff72536
public_notebooks.zip        71e6d66a61977cba2a97262efb71d1afc607096b7ba72c69bf9733456e076bed
```

**Code Verified: প্রধান duplicates**

| একই bytes-এর ফাইল | SHA-256 | তাৎপর্য |
|---|---|---|
| kishormorol 0.958 baseline ↔ heliosli `0.958_157v...blend-mine...` | `00ca614beec23736346df921e3305742bf8533b7cfd69a208880022f899fa1d7` | একই implementation; দুই independent strategy নয় |
| base archive-এর foysal H958B ↔ submission archive-এর H958B | `3759d660c7c800de653c03f0b1ba59f86bd6c974bd7e4d9a69e84e0a8c084dd2` | scored archive-কে newer builder output দিয়ে প্রতিস্থাপন করা যাবে না |
| submission_anchor944 ↔ public goodpjw 0.944 | `582f4a1c768cb4b347483797cfff4dc2a3e7648f0bb603d8e878eeb9ccc2695f` | folder ownership independent model authorship নয় |
| submission_blendgold ↔ heliosli public 0.951 | `2a4549eb2e50b31827ba6ac7e6dc88f2d5f61f88bfd70f85308c5532da8777ad` | local copy স্বাধীন implementation নয় |
| submission_h954 ↔ heliosli public 0.954 | `eb8f8db0aa9e85132d7046c14fac787a2ad7f1bda01688a45485eaf9b4ad3348` | baseline-copy সম্পর্ক |

সব file type মিলে ১১টি full-byte duplicate group এবং ৬টি all-code-cell duplicate group পাওয়া গেছে; [duplicate_groups.json](18_evidence/duplicate_groups.json)-এ তালিকা। Output/formatting ভিন্ন হলেও code একই হতে পারে; দুই hash আলাদা রাখা হয়েছে।

**Version এবং missing evidence:**

- `_index.csv`-এ H957/H958 পরিবারের 0.957 records আছে, **0.958 record নেই**। Archive README ব্যাখ্যা করে public verbatim forks অন্য folder-এ থাকে; তাই index-এ না থাকা missing submission-এর প্রমাণ নয়।
- GP26-এর indexed scored version **26**, indexed/latest API version **27**। Archive-এর **সব code cell** recorded v26 pull-এর সঙ্গে হুবহু মেলে; recorded/fresh v27 code ভিন্ন। Direct version-specific REST query এই turn-এ HTTP 400 দিয়েছে; সেই ব্যর্থতা লুকানো হয়নি। v26 binding-এর evidence হলো archived index + আগে সংরক্ষিত v26 metadata/source equality।
- v27-এ optional R7 setup/run ও compact-slot mixing বাদ গেছে; R6-only compact slot হয়েছে। Figure-ও বদলেছে। [Version source hashes](18_evidence/goodpjw_version_source_comparison.json), [readable diff](18_evidence/goodpjw_v26_v27.diff.txt)। **v27-কে v26-এর 0.958 result দেওয়া যাবে না।**
- Heliosli old `0.9573` index-এ scored v1/latest v2; kiyoshiohno template scored v15/latest v18; leoprovorov scored v2/latest v12। Filename-এর score দিয়ে archived/latest source-এর scored identity প্রতিষ্ঠিত হয় না।
- `laimerbotrolenk`-এর 0.957-named file-এর নিজের markdown বলছে এটি unmeasured V33 enhancement। `prvsiyan`-এর 0.957-named file-এ unvalidated 55/45 contrast আছে। এই দুই ক্ষেত্রে score/source attribution **Unknown**।

## Report B — exact 0.958 reconstruction

### পরিচয় এবং scored provenance

| বিষয় | পাওয়া প্রমাণ | শ্রেণি |
|---|---|---|
| Team submission | **57252502**, `kishormorol` | Public LB Observed |
| UTC timestamp | **2026-10-16 02:57:40.447 UTC**; Dhaka **08:57:40.447** | Metadata Verified |
| Scored kernel link | `/code/kishormorol/rsna-knee-helios-blend-fork?scriptVersionId=358866294` | Metadata Verified |
| Notebook ordinal version | API `currentVersionNumber=1` | Metadata Verified |
| Status/score | `complete`, displayed **0.958**; private score unavailable | Public LB Observed |
| Archive ↔ live source | সব cell source এবং code একই | Code Verified |
| Upstream notebook | `heliosli/blend-mine-in-private-lb-luck-to-all-upvoters`, v1 | Code + Metadata Verified |
| Fork owner | kishormorol; notebook recipe/source heliosli | Metadata + Code Verified |
| Formal parent-ID graph | REST response-এ explicit parent field নেই; submission description নিজেকে unchanged fork বলে, source পুরো একই | Metadata Verified সীমার মধ্যে |

এখন **notebook/version/submission linkage যাচাইকৃত**, শুধু Provisional Baseline রাখার প্রয়োজন নেই। তবে hidden execution-এর dataset version IDs, full worker receipts, failure counts এবং final CSV hash পাওয়া যায়নি। সেই execution-level reproducibility অংশ **Unknown**। দুইটি 0.958 submission দেখাই সম্পূর্ণ pipeline সক্রিয় থাকার প্রমাণ নয়।

দ্বিতীয় historical submission **57252333**, foysalemonshanto, **2026-10-16 02:50:43.343 UTC**, version-ID **358866931**, displayed **0.958**। Record যাচাই হয়েছে; ওই private fork-এর exact source এই audit-এ recover করা হয়নি। একই score ও CSV byte size দিয়ে একই predictions প্রমাণ করা যায় না। [Filtered submission evidence](18_evidence/submission_history.json)।

Original model authorship পৃথক: Raptor recipe dreaddevelopment; ব্যবহৃত দুই retrained CoAtNet checkpoint nartaa; R6 bundle ও compact pipeline heliosli। goodpjw reader এবং আমাদের distilled students baseline-এ নেই।

### Final prediction-এ কোন model বাস্তবে পৌঁছায়

| Component | Exact identifier/checkpoint | Input ও role | Availability/provenance |
|---|---|---|---|
| Raptor “384” branch | `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`; `raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt` | 384 canvas থেকে **320×320 centre crop**, 94 three-slice windows; plain+anatomical mirror probability mean; final Raptor slot-এ যায় | c2/c3। Remote file-list-এ 292,840,787 bytes; local checkpoint নেই। Bound SHA `7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef` |
| Raptor 224 branch | CoAtNet/MIL; `raptor_ft_alldata_t16_blendjev_oai_d96_c08_r224_swa.pt` | 384 canvas-এর **0.8 crop**, bilinear resize 224; 94 windows; **plain only**; final Raptor slot-এ যায় | c4/c5। Remote file-list-এ 292,842,755 bytes; local নেই। Bound SHA `3394fdd7e668286956e85af2554bef0386df5cd79d1c1af223b095c828b815d6`। Exact stored arch field checkpoint load ছাড়া নিশ্চিত নয় |
| R6 compact | `convnext_small`; bundle **fe055406fad6a12f**, arm **E_lrprior** | native 384 rendering, **352 crop**, 3 slices/physical 3.5 mm windows, `eval_stride=1`, `views=v4`; four-fold mean logits + flip; final compact slot | c6/c7 + fetched manifest। `E_lrprior_s0_f0.pt`…`f3.pt`, প্রতিটি 200,379,391 bytes remote-listed; local নেই |
| R6 side classifier | `side_cls.pt` | trustworthy laterality না থাকলে orientation correction-এর সহায়ক; independent final vote নয় | Manifest + compact loader; remote-listed, local নেই |
| Anatomy model | code dataset-এর `anatomy.pt` | v4 joint/patella anchors; independent finding vector নয় | c6 active `views=v4` branch, remote-listed; local নেই |
| R6 learned stackers | `fluid_stacker.json` Effusion; `state_stacker.json` Baker’s | adopted finding-এর compact output বদলায়; তারপর final rank blend | Small JSON fetched; hash manifest-এর সঙ্গে মেলে |
| R7, reader/students, extra pure ViT | **Absent from final baseline path** | কোনো final contribution নেই | baseline-এর সমস্ত code এবং active manifest পরীক্ষা করে সিদ্ধান্ত; optional helper definition সক্রিয় model নয় |

**গুরুত্বপূর্ণ correction:** checkpoint-এর “384” নামকে model input resolution বলা ভুল। c2 `INPUT_RES=320`, `load_model()` ওই resolution ফেরায়, `gpu_windows()` centre crop করে, `infer_pair_probs()` shape `(94,3,320,320)` assert করে। এটিকে bug বলা হচ্ছে না: এটি observed scored code path। Training source/paired preprocessing equivalence ছাড়া “384 input ঠিক করে দেওয়া” নতুন improvement নয়।

R6 manifest SHA-256 **6508ce6921eda986c885b2226fa55dd4d2aa8e5fc570b4c007c9a60c05bfc648**। Metadata-তে এক seed, চার fold, ১৬ epoch, batch ২, ১৭৬ train windows, ৪,৩৪৪ pool study, ৫৮ gold study এবং `gold_in_training=0` লেখা আছে। এগুলো **recorded training provenance**, এই audit-এর independent training verification নয়। `models2=None`: optional second architecture loader উপস্থিত থাকলেও active নয়। `compile=True`; compile failure-এ eager fallback-এর code আছে।

### Preprocessing ও inference dataflow

**Raptor c2/c4 — Code Verified:** DICOM header-এর orientation/position থেকে slice order; modality LUT/rescale এবং MONOCHROME1 handling; per-series 2/98 percentile intensity mapping; 140 mm centre field → INTER_AREA 384 canvas। Five slots `[26,22,18,12,18]` = 96 slices; adjacent triplets-এর 94 windows। ImageNet mean/std। Active fast reader `study_lut`; base64 payload কেবল text হিসেবে decode করে পড়া হয়েছে, execute হয়নি। 384 branch mirror হয় **normalization-এর আগে**: sagittal triplet-channel reversal, coronal/axial width flip। দুই pass-এর **probabilities** average হয়। Study batch ১; 384 encoder windows chunk ১৬; GPU-গুলো পৃথক study process করে।

**Precision:** 384 branch CUDA fp16 এবং nonfinite logits assertion ব্যবহার করে; এই active function-এ fp32 retry নেই। 224 `infer_probs`-এ fp16 RuntimeError হলে fp32 retry আছে। পুরোনো comments-এর “42/24 windows” actual `K_EVAL=94`-এর evidence নয়।

**R6 c6 embedded modules — Code Verified:** canonical native geometry, DICOM rescale/photometric handling, exact-duplicate series resolver, `series_companded` intensity mapping, full-field এবং 100 mm knee-centred/PF/posterior/bilateral views। Thin stacks-এর slab averaging এবং physical-order windowing আছে। Laterality classifier conditional correction করে। Eight downloaded mainline source hashes c6 `CODE_SHA`-এর সঙ্গে **8/8 মেলে**; source text ও [hash evidence](18_evidence/mainline_source_hashes.json) সংরক্ষিত। Training cache/targets বাস্তবে unavailable হওয়ায় end-to-end train/inference equivalence **Unknown**, mismatch প্রমাণিত নয়।

**R6 flip:** `FLIP_TTA=True`; `mirror_batch_v4c` sagittal slice reversal ও coronal/axial width flip করে, metadata/medial-lateral outputs reindex করে। প্রতি fold plain/mirror **logits** average, **MCL plain only**; তারপর folds-এর mean logits। এটি Raptor probability-TTA-এর সমান নয়। Loader batch study-প্রতি; controller `--loader-workers 2`, worker AMP `auto`; hard-coded universal image batch-size ধরে runtime গণনা করা যাবে না।

**R6 stackers আগে থেকেই আছে:** Effusion-এ image logit, fluid volume ও missingness; Baker’s-এ logit + state probabilities ব্যবহার হয়। Final rank blend-এ আলাদা probability calibration নেই, কিন্তু “baseline-এ calibration/learned postprocessing সম্পূর্ণ absent” বলা ঠিক নয়। Fluid JSON-এ নিজের preregistered adopt rule fail হওয়ার পর user override recorded আছে; state JSON অন্য class-এ adopt=False, Baker’s adopt=True। এই recorded AUC-কে আমাদের validation বা নতুন gain হিসেবে ব্যবহার করা যাবে না।

### Exact final ensemble mathematics

c8-এ `pct_on` প্রতিটি finding-এর জন্য সেই model-এর **নিজের successful studies**-এ `rank(method='average', pct=True)` নেয়। Complete row-এর ক্ষেত্রে:

```text
r384 = pct_on(Raptor384 raw probability)
r224 = pct_on(Raptor224 raw probability)
rr   = 0.65*r384 + 0.35*r224
ro   = pct_on(R6 probability after its adopted stackers)
B_c  = w_c*rr + (1-w_c)*ro
```

| Finding | w_c: Raptor slot | R384 effective share | R224 effective share | R6 share |
|---|---:|---:|---:|---:|
| Synovitis | 0.80 | 0.5200 | 0.2800 | 0.20 |
| Lateral OA | 0.20 | 0.1300 | 0.0700 | 0.80 |
| PF OA | 0.60 | 0.3900 | 0.2100 | 0.40 |
| Baker’s | 0.35 | 0.2275 | 0.1225 | 0.65 |
| বাকি ৮ finding, Medial OA-সহ | 0.50 | 0.3250 | 0.1750 | 0.50 |

এখানে 0.8/0.2 **Synovitis-এর Raptor-slot/R6 rank ratio**; student weight নয়। Overall pipeline-এর জন্য একটি global 0.8/0.2 বলা ভুল। Medial OA dictionary-তে আছে কিন্তু তার 0.5 default-এর সমান; **চারটি non-default finding override**, পাঁচটি improvement নয়।

**Failure handling, c8:** Raptor raw row সব finding-এ 0.5 হলে failed বলে; R6 success worker predictions থেকে নির্ধারিত। একটি Raptor branch fail করলে অন্যটির rank, দুইটি fail করলে R6 rank; R6 fail করলে Raptor slot; সব fail হলে 0.5। পৃথক Raptor branch-এ >3% failure অথবা R6-তে >50% failure হলে stop। Raw/sub rank consistency check আছে। এগুলো safeguards; hidden run-এ এগুলো কতবার সক্রিয় হয়েছিল **Unknown**।

সব input StudyInstanceUID test.csv-এর সঙ্গে মিলিয়ে reindex; target columns sample submission থেকে; output float32, finite/range/order assertions; `/kaggle/working/submission.csv`, `blend_receipt.json`, `submission_receipt.json` লেখা হয়। Raw `raptor_raw.csv`, `raptor224_raw.csv`, `ours_sub.csv` final score-এর operative sources। Notebook-এ ওই filenames থাকাই locally recovered predictions থাকার প্রমাণ নয়।

### Technical audit table

| Component | Status | Code location / evidence | পরবর্তী অর্থ |
|---|---|---|---|
| Base Raptor/CoAtNet | Present | c2/c3, single active ARMS entry | আবার CoAtNet যোগ বলা redundant |
| Raptor 224 | Present | c4/c5, A224=0.35 c8 | multi-resolution ইতোমধ্যে আছে |
| 384 mirror | Present | c2 consumer/infer_pair_probs | পুনরায় recommend নয় |
| 224 mirror | Absent | c4 consumer plain `infer_probs` only | E2-তে specific delta |
| R6 ConvNeXt four folds | Present | c6/c7 + manifest | নতুন CNN source নয় |
| R6 flip TTA | Present | c6 switch + worker | identical flip recommendation redundant |
| R7 | Absent | active manifest + final path | যোগ করলে cost; benefit অজানা |
| Our/student reader | Absent | সব baseline cells + final inputs | E1-তে new combination |
| Multiple seeds | R6-তে absent beyond seed0 | manifest models s0_f0…f3 | fold ও seed গুলিয়ে নয় |
| Per-finding weights | Present | c8 W_RAPTOR_C | পুরোনো technique |
| Rank averaging | Present | c8 pct_on | নতুন rank blend generic suggestion নয় |
| Learned per-finding stackers | Present internally | R6 Effusion/Baker’s | already implemented differently from final calibration |
| Additional final calibrator | Absent | c8 | AUC gain-এর জন্য monotone remap যথেষ্ট নয় |
| Clean full-ensemble OOF | Unknown/unavailable locally | artifact inventory | validation gate unresolved |

## Report C — historical experiment ledger

Fresh submission records ও archived index-এর public scores মেলে। নিচের পার্থক্যগুলো actual code থেকে; demonstrated causal gain নয়। [History JSON](18_evidence/submission_history.json)।

| Experiment | Actual configuration/change | Submission ID / displayed Public | Validation evidence | সিদ্ধান্ত |
|---|---|---|---|---|
| H955C | S6+Mirror parent; single C student; parent/student 0.8/0.2 | 57082452 / 0.955 | clean paired full-stack result নেই | 0.954 parent-এর তুলনায় Public-LB-only improvement observation |
| H957 | R7+Raptor384 0.5/0.5; C,C2,C3 probability mean; nested 0.8/0.2 rank blend | 57128358 / 0.957 | নেই | 0.956 parent-এর তুলনায় Public-LB-only observation |
| H957X5 | H957-এর সঙ্গে S75,T75s1, মোট পাঁচ student | 57151640 / 0.957 | নেই | displayed score-এ no observed difference; exact metric Unknown |
| H957X5W30 | শুধু final parent/student weight 0.7/0.3 | 57168472 / 0.957 | নেই | Public-LB-only; inconclusive |
| H957X5W40 | final 0.6/0.4 | 57168474 / 0.957 | নেই | Public-LB-only; inconclusive |
| H958 | R7 parent-এ 224+mirror 35% Raptor slot; পাঁচ student 20% | 57213895 / 0.957 | নেই | displayed difference নেই; baseline family আলাদা |
| H958B | H958-এর R7 manifest-এ `flip_tta=True` | 57213403 / 0.957 | নেই | displayed difference নেই; exact gain Unknown |
| H958B-W40 | H958B final student 40% | 57213404 / 0.957 | নেই | repeat weight search-এর শক্ত সমর্থন নেই |
| বর্তমান 0.958 | R7→R6+flip, 224 mirror নেই, চার class weight override, student বাদ | 57252502 / 0.958 | নেই | নতুন best displayed score observed; কোন একটি পরিবর্তন দায়ী বলা যায় না |

**Exact code checks:** H957→H957X5-এ reader checkpoint glob বদলায়, final formula একই। X5→W30/W40-এ final cell-এর ratio বদলায়। H958→H958B-এ R7 setup cell-এর flip switch বদলায়; 384, 224, reader, final cells একই। H958B→বর্তমান 0.958-এ একাধিক পরিবর্তন, তাই 0.001-কে R6 বা weight-এর causal effect বলা যাবে না।

Student models-এর training configuration: C = ConvNeXt-Tiny, target `0.5 v0 + 0.5 Mirror`; C2 একই seed1; C3 target `0.25 v0 + 0.75 Mirror`; S75 ConvNeXt-Small ওই 75% teacher target; T75s1 Tiny seed1। `training/gpjwstudent*/gpjwstudent.py` এবং H957X5 c10/c11 evidence। Full-data student-কে পাঁচ-fold OOF ensemble বলা যাবে না। Student output **probability mean**, তারপর final blend-এ percentile rank। Teacher baseline-এর Raptor family-র হওয়ায় statistically independent source নয়।

**আগে করা হয়েছে:** additional students; 20/30/40% ratios; R7 flip; 224 mirror; S6/R7 parent replacement; teacher-target changes। **বর্তমান R6+per-finding parent-এ পাঁচ student যোগের result archive/history-এ পাওয়া যায়নি**। এই scope-এ নতুন; সমস্ত account-এর অদেখা experiment সম্পর্কে দাবি নয়।

একই displayed 0.957-কে “verified no effect”, “verified regression” বা “equal predictions” বলা যাবে না। আগের report-এর team-best-ID switch থেকে exact improvement inference এখানে প্রতিষ্ঠা করা হয়নি: tie policy, concurrent submission এবং precise score ছাড়া এটি noisy secondary signal।

## Report D — public comparison ও সত্যিকার novelty

### Our 0.958 বনাম GP26

| Component | Our 0.958 | Public GP26 | Exact difference / evidence |
|---|---|---|---|
| Raptor384 | একই bound nartaa checkpoint, mirror | একই checkpoint/mirror | মূল model নতুন নয়; ছোট wrapper/format differences আছে |
| Raptor224 | plain, 0.8 crop, 224 | plain+mirror probability average | GP26 c6-এর consumer অতিরিক্ত mirror forward করে |
| Compact | R6+flip | **0.6 rank(R6+flip)+0.4 rank(R7)** | GP26 c12/c13 optional R7; c15 R7W=0.4; runtime skip হলে R6-only |
| Resolution | 320 input +224; R6 352 crop | inherited equivalent branch contracts | title-এর 384 actual input নয় |
| Per-finding Raptor weight | চার non-default override | একই weights | GP26 c15; already implemented identically |
| Reader/student | নেই | public `cnxt_v0_fold[012].pt` তিন fold | GP26 c18–c21; আমাদের C-family checkpoint নয় |
| Final reader weights | নেই | finding-specific **0.003–0.035** | `OWN_W=0.15` কার্যকর global weight নয়; full map overrides every finding |
| Calibration | internal R6 stackers; final rank blend | inherited compact + final rank transform | নতুন calibrated source evidence নেই |
| Validation | clean full-ensemble predictions নেই | markdown historical results; clean paired held-out artifacts নেই | contribution proven নয় |
| Runtime | hidden actual Unknown | optional R7 >5h gate, reader >8.2h skip | source presence ≠ scored execution inclusion |

GP26 reader weight map: Baker’s .035, Contusion .030, Medial OA/ACL .025, Medial Meniscus/MCL .020, Lateral Meniscus/Fracture/Lateral OA/PF OA .008, Effusion .004, Synovitis .003। আগের report-এর .008–.035 range অসম্পূর্ণ ছিল। Reader mean probabilities rank হয়; parent rank-এর সঙ্গে mix-এর পর output আবার rank হয়।

GP26 markdown-এ v21/v22/v24=0.957 এবং R7 share 0.4/0.5-তে three-decimal difference না হওয়ার দাবি আছে। এটি **recorded author observation**, independently recovered paired validation নয়। R7-কে “প্রমাণিত redundant” বলা অতিরিক্ত নিশ্চিত; এখানে lower priority বলা যুক্তিসংগত। GP26 নিজেই দুর্বল reader full weight-এ 0.956→0.954 recorded করেছে; এই parent-এ আমাদের student 20% নিরাপদ—এমন নিশ্চয়তা দেয় না।

### অন্যান্য relevant solutions

| Source | Actual difference | Technique classification | Score/source সীমাবদ্ধতা |
|---|---|---|---|
| heliosli `0.9576` | same R6+flip, same 384/224 plain; class weights .5; পুরোনো mutable R6 dataset slug | mostly Already implemented; weights/dataset binding Implemented differently | c6/c8 diff verify; filename decimal precision নয় |
| heliosli `0.9573` | R6 without flip | Previously tested/inferior novelty | scored/latest v1/v2 mismatch খেয়াল করতে হবে |
| varshithakolupuri `one-free-flip` | R7 parent, 224 anatomical mirror | Previously tested on old parent; Potentially complementary on R6 | H958 lineage; displayed 0.956 alone isolates no precise gain |
| bliverrigpdoge `r7-raptor-knee-centred-convnext` | extra knee-centred ConvNeXt-Small/EffNetV2-S leg, 0.8/0.2 | Potentially complementary; Missing dependencies | c0 নিজেই private weights ও fork fallback 0.956 বলে; 0.957 label transferable নয় |
| leoprovorov | R7+S6 all-data compact slot 0.6/0.4; both Raptor branches mirrored | Implemented differently; extra trained source | scored v2/latest v12; archived source-version binding Unknown |
| laimerbotrolenk | R7 .30 + S6 .20 + R384-TTA .325 + R224 plain/TTA ranks .0875/.0875 | Genuinely different aggregation; Hypothesis | file-এর V33 enhancement unscored বলা আছে; 0.957 filename endorsement নয় |
| kiyoshiohno template | external dataset pipeline/manifest launcher | Unknown methodology; Missing source dependencies | notebook ছোট হওয়া simple model প্রমাণ করে না; v15/v18 mismatch |
| picond fixed-readable | inherited R6+flip / 384+224; explanatory layout, same family | largely Redundant model information | model authorship বা independent strategy ধরে গণনা নয় |
| prvsiyan v47-named file | many old pipeline cells `raw`; operative c148 packed supervisor; decoded bound R6 recipe ও uniform **45% Raptor/55% R6** draft | same models; Implemented differently in weights/runtime | draft says no measured score; archived 0.957 cannot be assigned to this modified code |

prvsiyan-এর base91/LZMA package **শুধু data হিসেবে decode ও hash verify** করা হয়েছে; payload import/run হয়নি। আটটি decoded item-এর [hash manifest](18_evidence/bee_decoded_payload_hashes.json)। তাই raw inactive DINO/RadImageNet cells দেখে final model count বাড়ানো হয়নি। Packed weight contrast-এর upstream .5/.5 recipe আমাদের class-weighted .958 baseline নয়।

Public technique-এর practical সিদ্ধান্ত: CoAtNet, R6, rank blend, multi-resolution, R6 flip, 384 mirror, class weights আগে থেকেই আছে। 224 mirror এবং আমাদের reader leg **current baseline-এ অনুপস্থিত**। Private checkpoint, unknown source version, published score অথবা architecture name alone দিয়ে recommendation বানানো হয়নি। এই audit কোনো source-কে definitively Rule-incompatible ঘোষণা করেনি; eligibility Unknown হলে Unknown-ই রাখা হয়েছে।

## Existing artifacts ও reproducibility gate

[local_artifacts.json](18_evidence/local_artifacts.json)-এ ৫টি .pt, ৪৬টি .npz, ৪টি .npy, ৩টি .parquet-এর inventory। **কোনো complete current baseline raw prediction, matching R6 OOF, current পাঁচ-student prediction CSV বা hidden-test baseline CSV পাওয়া যায়নি।** Archives-এও prediction table নেই; submission archive-এর একমাত্র CSV হলো `_index.csv`।

| Artifact | বাস্তবে পাওয়া অবস্থা | কী করা যায় / যায় না |
|---|---|---|
| Our five student checkpoints | `students_upload/cnxt_pl{C,C2,C3,S75,T75s1}_full.pt`; মোট 701,864,820 bytes; পৃথক SHA verified | checkpoint recovery প্রয়োজন নেই; predictions তৈরি করতে new inference লাগবে |
| Private Kaggle student dataset | authenticated file-list-এ ঠিক পাঁচ expected filename, local sizes-এর সঙ্গে মেলে | এই account-এর listing access verified; future notebook owner-এর share/mount access separately verify করতে হবে; remote checkpoint bytes download করে hash করা হয়নি |
| Raptor checkpoints/R6 bundle | remote file-list ও R6 manifest পাওয়া; local weights নেই | remote recoverable metadata; existing local artifacts দিয়ে executable নয় |
| R6 code/anatomy | আটটি pinned module hash 8/8 verified; anatomy remote-listed | native code dataset file-list paginated; সব dependency wheel-এর completeness এখনো Unknown |
| goodpjw decoder dataset | cp311/cp312 pylibjpeg wheels এবং তিন reader ckpt remote-listed | proposed reader-এর compressed DICOM support asset আছে; install/runtime এই phase-এ পরীক্ষা হয়নি |
| Old reader OOF CSV | tiny/small fold3/fold4, 928/927 rows; ID unnamed first column; targets 12 | এগুলো cnxt_v0 fold readers; বর্তমান full-data C-family নয় |
| Old EffNet/E13 OOF NPZ | `pred`, `truth`, `ids`, `fold`; example 870×12; gold 58×12 | old-model artifacts আছে; current baseline OOF হিসেবে ব্যবহার অযোগ্য |
| NX2 CoAtNet arrays | OOF 894 IDs, gold 58 IDs | named Raptor checkpoint-এর equivalent predictions নয় |
| Labels parquet | তিনটি file physically present | installed parquet engine নেই; contents যাচাই হয়নি, dependency install করা হয়নি |
| Patient IDs | baseline native-cache code patients.csv তৈরি করতে পারে; actual matched current table পাওয়া যায়নি | grouping capability ≠ existing patient-disjoint validation |

চার reader OOF CSV-তে duplicate Study UID নেই; fold3/fold4-এর intersection **58 gold rows**। `set='val'` রাখলে 870/869 row; gold বাদ না দিয়ে concatenate করলে repeated gold হবে। Same-fold tiny/small ID sets মেলে। Target names মেলে, কিন্তু patient IDs নেই; folds/report-derived truth/current-model identities এক না হওয়ায় current ensemble validation পরিকল্পনায় এগুলো সরাসরি blend করা যাবে না। [Prediction schemas](18_evidence/prediction_schemas.json)। কোনো AUC/correlation/error-complementarity এখানে গণনা হয়নি।

Local checkpoint SHA সংরক্ষিত [student_checkpoint_hashes.json](18_evidence/student_checkpoint_hashes.json)-এ; weights deserialize/unpickle করা হয়নি। Training script configuration এবং model-file identity chain আলাদা evidence: source config জানা গেলেও full-data file-এর training log/args-এর সবকিছু independently যাচাই হয়নি।

## Metric, validation ও rule evidence

Official competition evaluation হলো **১২টি finding-এর macro ROC AUC**, `mean(AUC_c)`। Submission key `StudyInstanceUID`। [Official overview/evaluation](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/) এবং [fresh competition metadata](18_evidence/competition_detail.json)।

API-তে Public fraction **30%**, daily submissions **5**, final selected submissions **2**, CPU/GPU limit **540 minutes**, `submission.csv`, deadline **2026-10-22 23:59 UTC = ২৩ অক্টোবর ২০২৬ 05:59 Dhaka**। `scoreTruncationNumDecimals=3`; UI display থেকে rounding/truncation formula বা exact AUC interval ধরে gain হিসাব করা হয়নি। Hidden test N এবং per-finding prevalence Unknown; তাই “~390 public studies”, “SD=0.003”, exact +0.001 expected gain বা p95 runtime ব্যবহৃত হয়নি।

**পরিকল্পিত validation, এখন চালানো হয়নি:**

1. Baseline notebook full SHA, নয়টি cell hash, checkpoints/manifest/source hashes, software image ও prediction receipts freeze। UID/column identities, complete-row mask, fallback counts এবং fold provenance record।
2. Candidate এবং baseline **একই labels-সহ একই studies**-এ। PatientID পাওয়া গেলে patient cluster ধরে disjoint split; না পাওয়া গেলে patient-disjoint claim বন্ধ। DICOM duplicate/near-duplicate study এবং report duplicates আলাদাভাবে পরীক্ষা। Report grouping একা patient grouping নয়।
3. Label provenance official image truth বনাম weak report/teacher label আলাদা। Current full-data students বা teacher যে studies দিয়ে train হয়েছে সেগুলো clean OOF নয়। Existing gold-58 repeatedly selected/inspected; checkpoint/weight selection-এ unbiased holdout বলে ব্যবহার করা যাবে না।
4. প্রতিটি class-এর AUC ও macro delta; একই patient clusters-এ paired bootstrap, আগে নির্ধারিত seed/resample count; class-only draws হলে undefined AUC কতটি ও কীভাবে handle হবে preregister করতে হবে। Tiny/reused gold হলে interval explanatory, confirmatory নয়।
5. Clean holdout থাকলে weight .2 আগে fixed; tuning প্রয়োজন হলে inner grouped split-এ সীমিত candidate set, outer untouched evaluation। OOF masks/checkpoint training membership যাচাই ছাড়া analysis নয়।
6. Success-এর শক্ত মানদণ্ড: positive macro delta এবং paired 95% interval-এর lower bound >0 on a sufficiently independent held-out set; clinically relevant per-finding degradation ও fallback/runtime budget আগেই স্থির করতে হবে। Reused gold বা weak-label-only improvement একে Validation Verified করবে না।
7. Clean baseline/candidate predictions recover না হলে এই gate **blocked by artifacts**, কিন্তু research decision দেওয়া যায়। নতুন inference ও যথার্থ held-out retraining আলাদা cost/permission; শুধু train-set prediction বানিয়ে clean validation সৃষ্টি হয় না।

**Public LB evidence:** predeclared single probe অনুমোদন পেলে exploratory evidence হতে পারে। ≥0.959 displayed হলে Public LB Observed; private generalization প্রমাণ নয়। একই 0.958 হলে exact effect **Inconclusive**। Team-best switch একা tie-break proof নয়; switch না হলে reject নয়। Score কমলে current candidate park, baseline অক্ষত; ভালো না হওয়া পর্যন্ত weights খোঁজার loop নয়।

**Rules/rank:** official overview freely/publicly available external data ও pretrained models এবং internet-off requirement বলে। Full rules body এই turn-এর web reader-এ readable হয়নি; rules API attempt HTTP 404। OAI, fastMRI, derived checkpoints/pseudo-labels-এর specific eligibility **Unverified**। Archived host/participant discussion [external data](../discussions/737950_external-data-behind-an-institutional-data-use-agreement-does-the-knee.md) ও [OAI clarification](../discussions/749145_oai-and-the-nih-geo-restriction-is-oai-allowed-under-rule-2-6-b-and-wh.md) concern দেখায়; participant interpretation host approval/ban নয়। Student-এর Mirror-distillation inherited provenance দূর করে না।

Fresh metadata-তে মোট teams **6,105**। Leaderboard REST প্রথম ২০ row ও nextPageToken দিয়েছে; তাতে আমাদের team নেই। **Current team rank, silver cutoff এবং official tie-break policy এই audit-এ Unverified**। আগের report-এর rank317/cutoff305 পুনরায় verified বলা হয়নি। 0.959 থেকে silver guarantee দেওয়া যাবে না। Public/private patient distribution difference এবং future cutoff Unknown।

## Report E — সবচেয়ে promising missing improvements

যথেষ্ট সমর্থন পাওয়া গেছে **দুটি** সুযোগে; তৃতীয়টি জোর করে যোগ করছি না।

| Rank | Proposed opportunity | Baseline novelty / historical evidence | Assets, compute, difficulty | Validation risk / confidence |
|---|---|---|---|---|
| 1 | পাঁচ existing Mirror-student leg current baseline-এর শেষে .2 | reader absent; S6 parent-এ C এবং R7 parent-এ C-family historical displayed improvement | পাঁচ weights local; extra DICOM reader GPU inference; training নয়; exact GPU minutes Unknown; engineering low–medium | current R6 complementarity Unknown; same teacher family; stronger parent-এ 20% ক্ষতি করতে পারে; **moderate confidence in prioritizing, low confidence in 0.959 outcome** |
| 2 | Raptor224 anatomical mirror only | 384/R6 mirror আছে, 224 plain; H958/GP26-তে implement করা আছে | same 224 checkpoint; অতিরিক্ত 224 forward প্রায় দ্বিগুণ branch forward-count; total runtime increase Unknown; engineering medium | new information নয়, equivariance averaging; old-parent result causal নয়; **low confidence** |

অন্য model architecture, higher resolution, per-finding weight search, logit blending ও ensemble-teacher retraining এখন top-three evidence-supported improvement নয়। R6-তে ConvNeXt, four-fold mean logits এবং learned state/fluid information আগে থেকেই আছে। Scalar strictly monotone calibration final AUC বদলায় না; rank-এর আগে একই ordering রেখে calibration করলেও novelty নেই। Rank-changing stacker/extra information আলাদা বিষয়, যা বর্তমান R6-তে কিছু findings-এ ইতোমধ্যে active।

## Report F — ঠিক একটি First Recommended Experiment

**E1: Frozen-B0958 + fixed five-student overlay, W_STUDENT=0.20।**

Hypothesis: wider-field reader preprocessing, alternative pooled representation, teacher-target mix ও seed/backbone variation current R6/Raptor ensemble-এর কিছু ordering errors কমাতে পারে। এটি **Hypothesis**; shared teacher, report sources এবং ConvNeXt family substantial correlation ঘটাতে পারে।

**কেন প্রথমে:** পাঁচ trained assets পাওয়া গেছে; training দরকার নেই; scored historical integration path আছে; current baseline cells অপরিবর্তিত রেখে append করা যায়। 224 mirror-এর causal gain evidence আরও দুর্বল এবং পুরো branch forward বাড়ায়। Bare architecture diversity-এর চেয়ে historical add-on evidence এখানে বেশি প্রাসঙ্গিক, তবে R6 parent-এ transfer নিশ্চিত নয়।

Frozen baseline full SHA: `00ca614beec23736346df921e3305742bf8533b7cfd69a208880022f899fa1d7`। Baseline-এর weights, crops, TTA, stackers, failure routing ও cell order c0–c8 অপরিবর্তিত রাখার প্রস্তাব।

**Required files:** তিন original baseline datasets; five named `cnxt_pl*_full.pt`; `goodpjw2008/rsna-knee-2-5d-convnext-reader` offline decoder assets; competition DICOM/test tables; H957X5 reader source c7–c11। Local student weights পাওয়া; baseline weights ও competition DICOM local নেই। Kaggle remote file listings availability সমর্থন করে; owner permission, exact mounted versions ও dependencies runtime-এ যাচাই বাকি।

**Exact change:** baseline c8 সফল হওয়ার পরে baseline CSV snapshot; H957X5 reader implementation নতুন candidate-এ append; পাঁচ student-এর per-study per-finding **probability mean**; final `0.8*rank(B0958) + 0.2*rank(student_mean)`। এটি complete-row nested formula; baseline-এর flat Raptor/R6 weights-এর সঙ্গে সরাসরি 0.8 গুণ করে arithmetic flatten করা যাবে না, কারণ `rank(B0958)` nonlinear।

**GPU/compute:** new inference প্রয়োজন; new training নয়। Cached matched predictions থাকলে CPU-only overlay সম্ভব হতো, কিন্তু সেগুলো পাওয়া যায়নি। Total Kaggle run baseline + five-reader leg; exact runtime/VRAM Unknown। 9h-এর মধ্যে reader timeout-এর জন্য reserved write/cleanup margin পরিকল্পনা করতে হবে; guard থাকা guarantee নয়। আগের report-এর 20–30min/+0.5h estimate এই audit independently corroborate করতে পারেনি।

**Validation ও decision:** clean same-sample current-model outputs নেই, তাই **আজ E1 execute/validate/submit করার go দেওয়া হচ্ছে না**। প্রথমে অনুমোদনের পরে artifact recovery/access ও technically complete candidate run; তারপর clean validation সম্ভব কি না নির্ধারণ। সত্যিকার held-out comparison পাওয়া গেলে উপরের paired protocol। না পাওয়া গেলে user আলাদা করে exploratory public-LB-only probe অনুমোদন করলে শুধু সেই সীমার মধ্যে একটি predeclared comparison।

**Success:** frozen components/hashes অপরিবর্তিত, পাঁচ checkpoint active, fresh full-coverage prediction, no silent fallback; appropriate independent validation positive ও stable। Public score উন্নত হলেও label Public LB Observed। **Reject/park:** identity mismatch, hidden student skip, incomplete outputs, runtime budget exceed, held-out regression; displayed same score alone Inconclusive। Neither missing files nor a failed reader is a successful E1 run।

**Feasible fallback:** baseline reuse/preservation ও artifact provenance checks এখনই feasible এবং সেগুলো সম্পন্ন হয়েছে। Missing matched predictions-এর বদলে older OOF blend করে proxy improvement দাবি করব না। E1 checkpoints unavailable হলে E2-ও GPU inference-নির্ভর conditional alternative; permissions ছাড়া কোনো predictive experiment-এর safe executable shortcut নেই।

## Report G — proposed implementation, apply করা হয়নি

এই section-এর snippets **review draft**। নতুন notebook/builder বানানো, cells execute বা original source patch করা হয়নি।

**E1 affected locations:** original c0–c8 পুরো freeze; পরে baseline snapshot, H957X5 c7–c11 reader stage, শেষে overlay। Existing `build_h958.py` R7 parent ধরে hard-code করে; সেটি সরাসরি চালালে .958 baseline integration তৈরি হয় না। দরকার হলে পরে আলাদা builder-এ baseline nine-cell SHA assertions ও append-only assembly করা হবে।

Reader-এর প্রয়োজনীয় engineering: নাম ধরে exact পাঁচ checkpoint খোঁজা/hash check; R2-style remaining-clock timeout; R3-style multi-frame slice count; fresh output namespace। R1–R3 source বর্তমান builder-এ আছে কিন্তু archived H958B-তে নেই। এগুলো identical normal-path math বজায় রাখার উদ্দেশ্য; এই phase-এ test করা হয়নি। R5 missing-rank fit source reuse করলে data-dependent behavior যোগ হয়; initial controlled candidate-এ **সব expected row complete না হলে experiment invalid**, baseline fallback রেখে receipt-এ failure লিখব।

Proposed overlay core—শুধু previously generated, complete, aligned tables-এর জন্য; **এখন execute করা হয়নি**:

```python
# Draft only: B is a frozen baseline snapshot; S is the five-model probability mean.
labels = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
key = "StudyInstanceUID"
assert list(B.columns) == [key] + labels
assert B[key].is_unique and S[key].is_unique
assert set(B[key]) == set(S[key])
S = S.set_index(key).loc[B[key], labels]
assert np.isfinite(S.to_numpy()).all()
assert ((S >= 0) & (S <= 1)).all().all()
assert (S.nunique() > 1).all()
parent_rank = B[labels].rank(method="average", pct=True)
student_rank = S.reset_index(drop=True).rank(method="average", pct=True)
candidate = B.copy()
candidate[labels] = (0.8 * parent_rank + 0.2 * student_rank).astype("float32")
# Later implementation: candidate-specific path + receipt first, retain baseline;
# final submission.csv replacement only after that experiment is explicitly authorized.
```

`S` aggregation H957X5 c10-এর `torch.stack([sigmoid(model_i(...))]).mean(0)`; `mean(logits)` দিয়ে বদলানো যাবে না। Existing H957 final c12-এর broad exception fallback baseline রাখে; E1 success গণনা করতে `H957_FINAL=blended`, explicit component receipt এবং unchanged baseline-copy hash দরকার। শুধু submission.csv থাকা যথেষ্ট নয়।

**E2 affected location:** baseline c4-এর `gpu_windows(..., mirror=False)` path-এ H958B c15-এর `_slot_index/_mirror_tri` ও mirror-before-normalization support; consumer-এ দুই predictions average। c5 timeout শুধু measured headroom অনুযায়ী পরে review; c8 A224/weights/failure routing unchanged। একটি `flip(-1)` সব plane-এ করা anatomically equivalent নয়।

```python
# Draft delta inside the Raptor224 CUDA consumer after importing its known mirror helper:
x_plain = gpu_windows(vol, mask, K_EVAL, res, device)
x_mirror = gpu_windows(vol, mask, K_EVAL, res, device, mirror=True)
p224 = 0.5 * (infer_probs(model, x_plain, device)
              + infer_probs(model, x_mirror, device))
# Preserve checkpoint, crop=0.8, resize=224, window centres, final class weights.
```

Permission পেলে প্রয়োজনীয় meaningful checks: nine original cell source equality, exact file/version bindings, ID mismatch/duplicate/missing-row rejection, prediction mean/nested-rank mathematical chain, time-guard/reader failure preserving baseline, compressed DICOM decode parity, unchanged transforms। এই phase-এ synthetic chain execution-ও করা হয়নি।

## Report H — ranked controlled roadmap

সর্বোচ্চ চারটি predictive candidate নিচে; এর মধ্যে E3/E4 initial execution থেকে **Defer**, evidence/validation gate পেরোলে বিবেচনা। Artifact/provenance verification কোনো model experiment নয়।

| Priority / ID | একটিমাত্র প্রধান change | Prerequisites / availability | Compute ও validation | Go / no-go |
|---|---|---|---|---|
| **1 / E1** | B0958 শেষে fixed five-student .20 overlay | পাঁচ checkpoint local; matched predictions missing; remote baseline assets listed | Requires new inference; Cannot currently validate | Go only after authorization + exact assets/clean comparison বা explicitly permitted exploratory probe; incomplete leg হলে no-go |
| **2 / E2** | B0958 c4-এ 224 anatomical mirror | existing remote checkpoint, helper code; extra prediction missing | Requires new inference; same samples/metric, new mirror pass | Baseline-এ একা compare; E1-এর সঙ্গে একসঙ্গে যোগ নয়; runtime uncertain হলে defer |
| **3 / E3** | E1-এর এক fixed lower student share .10 বনাম .20 | E1 matched cached prediction + clean held-out labels দরকার | CPU reblend after recovery, new inference নয় যদি caches থাকে | **Defer**; strong-parent shrinkage hypothesis মাত্র; no LB weight sweep; historical .30/.40 repeat নয় |
| **4 / E4** | চার non-default w_c-কে default .5-এর দিকে fixed 50% shrink | exact baseline raw predictions + clean validation দরকার | CPU if recovered; validation বর্তমানে unavailable | **Defer**; private-overfit robustness hypothesis; .80→.65, .20→.35, .60→.55, .35→.425; blind replacement নয় |

E2 প্রথমে B0958-এর ওপর independent comparison। E1/E2 আলাদা evidence পাওয়ার পরে combined test নতুন experiment হিসেবে গণনা করতে হবে; আগেই দুই change মিশিয়ে causal conclusion নয়।

**বর্তমানে বাদ/স্থগিত:** wider .30/.40 student weight repeats; R7 second compact without new complementarity evidence; same R6 TTA আবার যোগ; public per-finding weights আবার “নতুন” বলা; private knee-centred checkpoint fork-কে executable ধরা; ensemble-teacher/student retraining। শেষটির নতুন teacher predictions/targets, full-data weights ও clean evaluation নেই; training/GPU budget Unknown এবং permission নেই। Architecture নাম দেখে benefit ranking নয়।

## Report I — চূড়ান্ত research decision

**বর্তমান .958 সম্পর্কে নিশ্চিত:** unchanged heliosli recipe, kishormorol v1, submission57252502। Raptor384-named checkpoint-এ actual 320 crop + mirror; Raptor224 plain; R6 four-fold/352 crop/flip এবং learned Effusion/Baker’s stackers; class-specific rank blend; current students absent। Code/version/submission linkage strong, hidden execution receipt incomplete।

**এখনো Unknown:** exact hidden dataset versions/receipt/runtime, final prediction bytes, current baseline clean OOF, R6-student error complementarity, proper patient-disjoint labels, second teammate fork source, precise AUC gains, rank/cutoff/tie-break এবং inherited OAI/fastMRI eligibility।

**সবচেয়ে evidence-supported পরবর্তী predictive পদক্ষেপ E1**, current frozen parent-এর ওপর existing পাঁচ student fixed .20 overlay। এটি evidence-supported research priority, demonstrated improvement নয়। Existing predictions missing হওয়ায় “এখন CPU-only blend করে .959 বের করা যাবে” বলা সম্ভব নয়। প্রথম operational কাজ হলো exact dependencies ও current-model predictions-এর recoverability নিশ্চিত করা; নতুন inference চালাতে আলাদা অনুমতি লাগবে।

**Kaggle submission যুক্তিসংগত হবে** explicit অনুমতির পরে exact-version candidate, all active components, complete aligned outputs, deadline-safe execution ও appropriate evaluation evidence থাকলে। Independent held-out evidence না পাওয়া গেলে একটি explicitly accepted public-LB-only probe-এর সীমা পরিষ্কার রাখতে হবে। Smoke success alone generalization evidence নয়।

**Private risk:** baseline class weights-এর selection provenance অস্পষ্ট; accumulated public probes overfit করতে পারে; teacher-derived students shared errors ও inherited eligibility বহন করে; 30% public result remaining 70%-এর ordering নিশ্চিত করে না। New model leg থাকাই private improvement-এর প্রমাণ নয়।

**Confidence:** baseline/code reconstruction **high**; first experiment priority **moderate**; validation-backed gain **not established**; displayed .959+ ও silver **Unknown**। তিনটি দুর্বল recommendation দিয়ে সংখ্যা পূরণ করা হয়নি।

### Evidence package

সব supporting artifacts [18_evidence/](18_evidence/)। প্রধানগুলো: archive/cell inventory, duplicate groups, filtered fresh submission history, live notebook metadata, version-source hashes/diff, R6 manifest/stacker JSON, eight verified source texts, local artifact schemas ও student hashes। [originals_unchanged.json](18_evidence/originals_unchanged.json) অনুযায়ী **২৭৩/২৭৩ original file ZIP-এর bytes-এর সঙ্গে এখনও identical**। আগের `17_research_0958_to_0959.md` untouched আছে।
