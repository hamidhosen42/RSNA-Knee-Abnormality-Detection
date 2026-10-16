# Audit 18 evidence

১৭ অক্টোবর ২০২৬ (client date, Asia/Dhaka)-এর read-only audit-এর evidence। মূল বাংলা রিপোর্ট: [18_verified_0958_to_0959_research_bn.md](../18_verified_0958_to_0959_research_bn.md)।

- `archive_inventory.csv`: তিন ZIP-এর ২৭৩টি মূল ফাইল; অজানা metadata-তে Unknown। Author/score-এর archive-index attribution execution proof নয়। Checkpoint references active model membership-এর প্রমাণ নয়।
- `macos_metadata_inventory.csv`: বাদ দেওয়া ৩৩৭টি macOS file entry। Directory entry এতে নেই।
- `notebook_cells.csv`: ১৪৯টি notebook-এর ৩,৪৩৪টি cell source hash; c0 থেকে numbering।
- `duplicate_groups.json`: full bytes এবং all-code-cell hash অনুযায়ী দুই ধরনের duplicate grouping।
- `baseline_live_source_verification.json`: archive baseline ও live private kishormorol v1-এর নয়টি cell-source equality। Whole-file serialization hash একই দাবি নয়।
- `submission_history.json`: fresh API থেকে task-relevant records; অন্য submissions বাদ। Displayed score exact AUC নয়।
- `kishor_kernel_metadata.json`, `goodpjw_latest_metadata.json`: live API metadata। Dataset source name version pin নয়।
- `goodpjw_v26_recorded_metadata.json`, `goodpjw_v27_recorded_metadata.json`: আগের সংরক্ষিত source pull-এর metadata; এই turn-এর fresh v26 request সফল হয়নি।
- `goodpjw_version_source_comparison.json`, `goodpjw_v26_v27.diff.txt`: archived code/recorded v26 equality এবং v27 difference। Long embedded source line diff-এ abbreviated; cell hashes সম্পূর্ণ।
- `helios_09576_to_0958.diff.txt`: archived same-family predecessor বনাম baseline-এর static diff।
- `r6_manifest_download.json`, `r6_fluid_download.json`, `r6_state_download.json`: ছোট public bundle JSON সরাসরি পড়া হয়েছে; recorded training/evaluation claims independently reproduced নয়।
- `mainline_source_hashes.json`, আটটি `*.py.txt`: public mainline source text; baseline c6-এর আটটি CODE_SHA-র সঙ্গে 8/8 মেলে। কোনো source import/execute হয়নি।
- `r6_files.json`, `student_files.json`, `reader_files.json`, `nartaa_files.json`, `code_files.json`: remote file listings। Code dataset listing **paginated**; এই snapshot পুরো listing নয়। Local/remote same filename/size একা byte equality নয়।
- `student_checkpoint_hashes.json`: পাঁচ local student file-এর SHA-256; weights load/unpickle করা হয়নি।
- `local_artifacts.json`, `prediction_schemas.json`: local array shapes, CSV schema ও split counts; AUC, blending search বা statistical validation চালানো হয়নি।
- `bee_decoded_payload_hashes.json`: public notebook-এর packed source কেবল data হিসেবে decode করে hash করা হয়েছে; decoded scripts run হয়নি।
- `competition_detail.json`: fresh read-only GetCompetition response।
- `leaderboard_first_page.json`: প্রথম ২০ team এবং nextPageToken; এটি দিয়ে আমাদের rank বা medal cutoff পাওয়া যায় না।
- `originals_unchanged.json`: তিন ZIP-এর সব মূল ফাইল ডিস্কের archived bytes-এর সঙ্গে মেলে।

## Source access

Read-only REST requests: `https://www.kaggle.com/api/v1/kernels/pull/{owner}/{slug}`, `datasets/list/{owner}/{slug}`, individual small `datasets/download/{owner}/{slug}/{filename}`, `competitions/submissions/list/rsna-knee-abnormality-detection`, `competitions/rsna-knee-abnormality-detection/leaderboard/view`; read-only internal GetCompetition। Authentication token output/snapshot-এ নেই।

সীমাবদ্ধতা: anonymous competition list HTTP401; version-specific public kernel pull query HTTP400; guessed GetLeaderboard/GetCompetitionRules service routes HTTP404। Standard REST leaderboard fallback সফল, কিন্তু প্রথম page মাত্র। Web overview/rules কিছু requests-এ empty render দিয়েছে; official search result-এ evaluation formula ও code requirements পাওয়া গেছে। Full rules, current team rank এবং tie-break policy তাই verified বলে চিহ্নিত হয়নি।

Original notebooks, scripts, indexes এবং আগের audit report অপরিবর্তিত। এই package কোনো generated experiment notebook বা submission নয়।
