# Our leg: D4-এর পদ্ধতি, আমাদের নিজস্ব মডেল দিয়ে

অবস্থা (2026-09-27): **কোড তৈরি এবং লোকালি নকল ডেটায় পরীক্ষিত। Kaggle-এ এখনো কিছু push, train বা submit হয়নি।** কোনো স্কোরের প্রতিশ্রুতি নেই।

## ১. D4 কীভাবে 0.943 থেকে 0.946-এ গেল

| ধাপ | কী বদলেছে | প্রমাণ |
|---|---|---|
| Jiwei v11 | public pipeline | public LB 0.943 |
| D4 Lite | উপরেরটা, সাথে Fracture token বদল আর **নিজস্ব EfficientNetV2-S মডেল** (384px, 5 slot, 4 checkpoint), rank-blend 0.45 ওজনে | 0.945 |
| D4 Blend | Lite, সাথে CoAtNet আর ConvNeXt-L arm (একই 0.45 ওজন) | 0.946 |
| আমাদের E02 (26-09) | D4 Blend থেকে private মডেলটা বাদ, অর্থাৎ শুধু Jiwei v11 + Fracture token | **0.943** |

- 0.943 → 0.946-এর পুরো পার্থক্যটা আসে ওই আলাদা ভাবে train করা CNN মডেল থেকে।
- মেশানোর সূত্র: `final = rank(0.55·rank(public) + 0.45·rank(ours))`।
- ওদের weights, training code আর inference script সবই private। তাই কপি সম্ভব নয়, নিজেদেরটা বানাতে হবে।
- বিস্তারিত বিশ্লেষণ: `../audit/d4/`।

## ২. আমরা যা বানালাম

D4-এর মডেলের সেটিং (384px, ৯৬ slice, MAX_WIN 36) Raptor pipeline-এর মতো। Raptor-এর training-এর তিনটা অংশই public (CC0):
- **ডেটা:** `dreaddevelopment/knee-raptor-corpus` আর `-ext`। 4,407টা স্টাডি, প্রতিটা 44 × 336 × 336 slice stack।
- **Label:** `dreaddevelopment/rsna-knee-labels`, `labels_llm_soft.parquet`। 4,349টা স্টাডির soft label।
- **কোড:** `dreaddevelopment/knee-mri-training-the-twelve-finding-model` (v8)।

আমরা এই recipe রেখে **backbone বদলে `tf_efficientnetv2_s.in21k_ft_in1k` @384** (D4 Lite-এর architecture) বসিয়েছি।

| ফাইল | কাজ |
|---|---|
| `folds.py`, `folds_v1.json` | স্থির ৫-fold ভাগ। gold-58 বাদ, একই report একই fold-এ, প্রতিটা label-এর অনুপাত সমান। sha256 `f4292c4a…` |
| `train_ours_leg.py` | Kaggle GPU training script। কী কী বদলানো হয়েছে তা script-এর শুরুতে লেখা |
| `infer_kernel.py` | test DICOM থেকে হুবহু corpus-এর নিয়মে 44-slice stack বানায়, তারপর সব checkpoint দিয়ে predict করে `submission.csv` লেখে |
| `build_submission_notebook.py` | `../rsna-knee-d4-blend_patched.ipynb` থেকে (মূল ফাইলে হাত না দিয়ে) নতুন নোটবুক বানায় |
| `submission/rsna-knee-base-plus-ours.ipynb` | cell 0–30 হুবহু আগের মতো (0.943 বেইস), তার পরে আমাদের মডেল আর blend |
| `kaggle/*.kernel-metadata.json`, `kaggle/push.sh` | Kaggle-এ push করার জন্য। সব dataset/kernel বেইসের হুবহু version-এ আটকানো, একই Docker image, T4×2 |

**recipe থেকে বদলগুলো, কারণসহ:**
1. backbone হলো EfficientNetV2-S। CNN বলে backbone-এর LR 3e-5 থেকে বাড়িয়ে 1e-4 করা হয়েছে।
2. T4-এ bf16 নেই, তাই fp16 + GradScaler।
3. backbone-এর নিজস্ব normalization (mean/std 0.5)। এটা checkpoint-এর ভেতরে রাখা হয়।
4. সেরা epoch বাছা হয় OOF fold দিয়ে। label-এর অনিশ্চিত অংশ (0.25–0.7) metric থেকে বাদ। **gold-58 দিয়ে কিছুই বাছা হয় না**, শুধু diagnostic হিসেবে দেখা হয়।
5. 2×T4-এ DataParallel, আর ১১ ঘণ্টার সময়সীমা।
6. top-3 epoch-এর weights গড় করার পর BatchNorm-এর পরিসংখ্যান নতুন করে হিসাব করা হয়।

## ৩. লোকালি যা যাচাই হয়েছে (নকল ডেটা, CPU)

- **fold:** আলাদা hash seed দিয়েও একই checksum আসে। Kaggle-এর পথে (pandas + parquet) পড়লেও একই।
- **training (smoke):** epoch, OOF, gold, সেরা checkpoint, SWA আর BatchNorm, সব ফাইল প্রত্যাশিত layout-এ।
- **inference:** নকল DICOM থেকে stack বানায়, slot সীমানা 0/12/22/30/36/44। slice সাজানো হয় জ্যামিতিক অবস্থান দিয়ে, InstanceNumber দিয়ে নয়। ২টা worker দিয়ে চলে। schema ঠিক।
- **নোটবুক cell 30–35:** checkpoint থাকলে "public 0.70 + ours 0.30" blend হয়। না থাকলে হুবহু public ফলাফল থাকে।
- **যা যাচাই হয়নি:** আসল GPU-তে গতি ও মেমরি, আসল ডেটায় মডেলের মান, আর Kaggle-এ পুরো নোটবুকের রান।

## রানের ইতিহাস

| version | সময় | ফল |
|---|---|---|
| v1 | 2026-09-27 12:42 | **ব্যর্থ, ~৩০ মিনিটে।** প্রথম মূল্যায়নের শুরুতে host RAM শেষ হয়ে "Killed"। গতি ছিল মাত্র 2.1 স্টাডি/সেকেন্ড, কারণ `/kaggle/input`-এ memmap-এর page fault। setup ঠিক ছিল: 2×T4, fold checksum মিলেছে, train 3479 / OOF 870 / gold 58 |
| v2 | 2026-09-27 13:21 | **ব্যর্থ, ~৫৬ মিনিটে ("Killed")।** ডেটার অপেক্ষা 0%, তবু গতি 1.4 স্টাডি/সেকেন্ড। RAM প্রতি স্টাডিতে ~৫ MB করে বেড়েছে (4 → 20 GB) |
| diag v1/v2 | 2026-09-27 14:20–14:45 | **কারণ মাপা গেছে।** (১) GPU সুস্থ (fp16 matmul 24 TFLOPS, ResNet-50 292 ছবি/সেকেন্ড), কিন্তু এই মডেলে DataParallel এক GPU-র চেয়ে ৩ গুণ ধীর (1.46 বনাম 4.46 স্টাডি/সেকেন্ড; forward-এ ১০ গুণ)। (২) NFS page cache ছাড়ানো যায়, আসল সমস্যা DataLoader worker-এ glibc malloc: প্রতি স্টাডিতে ~1.5–1.75 MB বাড়ে, worker বন্ধ হলে ফেরত আসে |
| v3 | 2026-09-27 14:57 | **সফল, 4.67 ঘণ্টা।** বদল: DataParallel বাদ, প্রতিটা GPU-তে আলাদা fold (fold 0 আর fold 1 একসাথে); worker একটাই buffer বারবার ব্যবহার করে (`preadv`), সাথে `mallopt`; pin_memory বন্ধ। RAM পুরো সময় 5.5–6 GB-এ স্থির; প্রতিটা GPU-তে 3.8 স্টাডি/সেকেন্ড; এক epoch 17.3 মিনিট |

### v3-এর ফল (fold 0 + 1; অনিশ্চিত band বাদ দিয়ে soft label-এর বিপরীতে OOF)

| | fold 0 | fold 1 | দুটো fold একসাথে (n=1740) |
|---|---|---|---|
| best epoch (OOF দিয়ে বাছা) | ep13: 0.8728 | ep6: 0.8668 | 0.8684 |
| **SWA** (top-3 epoch, BN পুনর্গণনাসহ) | 0.8724 | 0.8729 | **0.8721**, তাই এটাই বাছা হলো |
| gold-58 (শুধু যাচাই, বাছাইয়ে ব্যবহৃত নয়) | 0.8858 | 0.8943 | দুটো মডেলের গড় 0.9000 |

- **finding অনুযায়ী (একসাথে OOF):** ACL 0.890, MCL 0.846, Med. Meniscus 0.918, Lat. Meniscus 0.838, Med. OA 0.885, Lat. OA 0.832, PF OA 0.845, Effusion 0.908, Synovitis 0.848, Baker's 0.926, Contusion 0.829, Fracture 0.903।
- **দুটো fold মডেলের মিল:** gold-58-এ rank correlation 0.85।
- **তুলনা, সতর্কভাবে পড়ুন:** Raptor-এর একক CoAtNet gold-58-এ 0.905 আর LB-তে 0.914। কিন্তু ওটা gold দেখে বাছা ছিল, আমাদেরটা নয়।
- আমাদের মডেল একা LB-তে কত পাবে, তা জানা নেই।
- **ফাইল:** `results/train_v3/`-এ json, OOF আর gold npz, আর Kaggle log। checkpoint আছে Kaggle-এ, `hosen42/rsna-knee-ours-leg-train-effv2s` v3-এর আউটপুটে।
- **সাবমিশন নোটবুক:** এটা ওই version 3-এ আটকানো। পরে fold 2–4 অন্য version-এ train হলে যাতে f0/f1 হারিয়ে না যায়, সেজন্য ওগুলো আলাদা kernel-এ চালাতে হবে।

## ৪. চালানোর পরিকল্পনা আর সিদ্ধান্তের নিয়ম

1. **fold 0 training:** `kaggle/push.sh train`। T4×2-এ আনুমানিক ৩–৪ ঘণ্টা; প্রথম epoch-এর সময় দেখে নিশ্চিত হওয়া যাবে। ফল হিসেবে আসবে fold-0 OOF macro AUC, প্রতিটা finding-এর AUC, আর gold-58 (diagnostic)।
2. **সিদ্ধান্ত:** OOF যুক্তিসঙ্গত হলে fold 1–4 train হবে (`OURS_FOLD` বদলে)। না হলে থামব, আর কারণ খুঁজব।
3. **সাবমিশন নোটবুক:** `kaggle/push.sh submission`। আগে commit run হবে (৩টা দৃশ্যমান test স্টাডি), log-এ `[ours] final: public 0.70 + ours 0.30` আছে কি না দেখা হবে।
4. **Blend ওজন আগে থেকে ঠিক করা:** প্রথমে 0.30, তারপর D4-এর 0.45, **সর্বোচ্চ ২টা সাবমিশন**। 0.45 রাখা হবে কেবল যদি সেটা LB আর gold-58 দুটোতেই কম না হয়।
5. **প্রতিটা সাবমিশনের আগে আপনার অনুমতি লাগবে।**

**ঝুঁকি:**
- আমাদের মডেল D4-এর মডেলের মতো শক্তিশালী না-ও হতে পারে। esfersami50-এর 0.889 মডেল 0.30 ওজনে মেশানোয় স্কোর 0.940-এ নেমেছিল।
- anchor-এর কোনো OOF নেই, তাই blend-এর ওজন OOF দিয়ে শেখা যায় না।
- ১ ঘরের LB পার্থক্য rounding আর নয়েজের মধ্যেই পড়ে।
