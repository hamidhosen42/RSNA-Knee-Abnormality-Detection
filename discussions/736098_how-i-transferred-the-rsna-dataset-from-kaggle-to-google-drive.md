# How I Transferred the RSNA Dataset from Kaggle to Google Drive

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/736098
- **Author:** **Y.** (@xiongdaxiongmao, Contributor) · _topic author_ · 2026-08-19 14:12 UTC · ▲3
- **Topic votes:** 3 · **Comments/replies:** 0 · **Pinned:** no
- **Last comment:** 2026-08-19 14:12 UTC by Y.
- **Cloned:** 2026-10-09 09:46 UTC (topic id 736098, raw JSON `_raw/736098.json`)

---

The RSNA dataset was too large to download through Colab or my local PC due to disk space limitations.

Instead, I used a Kaggle Notebook + rclone workflow. I connected rclone to Google Drive using OAuth, stored the authentication information securely in Kaggle Secrets, and then transferred the dataset directly from Kaggle’s /kaggle/input storage to Google Drive.

The overall workflow was:

Kaggle Dataset → Kaggle Notebook → rclone → Google Drive

I first tested the transfer with a small file and a single study, then started the full dataset transfer. This approach avoids storing the entire dataset on Colab or a local machine and also allows the transfer to be resumed if the Kaggle session stops.

---

## Comments (0, threaded, oldest first)

_No comments._
