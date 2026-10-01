# [Baseline Notebook] 2.5D ResNet-34 PyTorch Pipeline (Fold 0) | Preprocessed 3D Volumes

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733934
- **Author:** **Barun Kumar** (@barun2104, Expert) · _topic author_ · 2026-08-09 06:32 UTC · LB rank 3687
- **Topic votes:** 0 · **Comments/replies:** 1 · **Pinned:** no
- **Last comment:** 2026-09-26 00:29 UTC by Chris Deotte
- **Cloned:** 2026-10-01 19:24 UTC (topic id 733934, raw JSON `_raw/733934.json`)

---

Hey Everyone! 👋
To help jumpstart model exploration in the RSNA Knee Abnormality Detection Challenge, I've published a complete, end-to-end [PyTorch baseline notebook](https://www.kaggle.com/code/barun2104/baseline-rsna-knee-mri-2-5d-resnet34) using a **2.5D ResNet-34 architecture** trained on Fold 0.

Handling 3D DICOM volumes directly during training can cause severe I/O bottlenecks. To make training fast, clean, and reproducible, this baseline runs off preprocessed, memory-efficient `(24, 160, 160)` volume arrays.

### 💡 Key Notebook Highlights
- **2.5D ResNet-34 Backbone:** Instead of computationally heavy 3D convolutions, we map the 24 depth slices across input channels, adapting ImageNet pre-trained weights to leverage transfer learning.

- **12 Multi-Label Pathology Targets:** Trains directly on all primary targets (`ACL`, `MCL`, `Meniscus tears`, `OA`, `Effusion`, etc.) using `BCEWithLogitsLoss`.

- **Fast Batch Training (AMP FP16):** Uses Automatic Mixed Precision and batch size of 32 for rapid epoch iteration times on Kaggle GPU (T4 / P100).

- **Cross-Validation Metric:** Tracks Mean Macro ROC-AUC across Fold 0.

### 🔗 Related Resources & Datasets
- **📓 Training Notebook:** [[Baseline] RSNA Knee MRI — 2.5D ResNet34](https://www.kaggle.com/code/barun2104/baseline-rsna-knee-mri-2-5d-resnet34)

- **📊 Metadata CSV Dataset:** [RSNA Knee - Stratified Folds & LLM Soft Labels](https://www.kaggle.com/datasets/barun2104/rsna-knee-stratified-folds-and-llm-soft-labels)

- **🖼️ Processed 3D Volumes Dataset:** [RSNA Knee MRI - Processed 3D Volumes](https://www.kaggle.com/datasets/barun2104/rsna-knee-mri-processed-3d-volumes)

### 🚀 What's Next on the Roadmap?
- **Semi-Supervised Pre-Training:** Leveraging the `pseudo_*` targets in the metadata file to pre-train on unannotated studies before fine-tuning on ground truth folds.

- **Backbone Benchmarking:** Testing EfficientNet and ConvNeXt 2.5D variants.

- **Data Augmentation:** Adding 3D spatial transformations (random flipping, slight rotations, intensity scaling).

Check out the notebook, feel free to fork/copy it for your own experiments, and let me know if you have any feedback or ideas to improve the CV score!

**What model architectures or slice depths are you planning to experiment with first?**

---

## Comments (1, threaded, oldest first)

### [1] Chris Deotte

**Chris Deotte** (@cdeotte, Grandmaster) · 2026-09-26 00:29 UTC · ▲-1 · LB rank 92

Thanks for sharing!
