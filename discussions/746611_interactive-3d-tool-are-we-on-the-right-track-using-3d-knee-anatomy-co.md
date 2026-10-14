# [Interactive 3D Tool] Are we on the right track? Using 3D Knee Anatomy & Continuous Depth Slicing to test our ML assumptions

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/746611
- **Author:** **Phil Gear** (@philgear, Contributor) · _topic author_ · 2026-10-07 16:03 UTC · LB rank 1996
- **Topic votes:** 0 · **Comments/replies:** 1 · **Pinned:** no
- **Last comment:** 2026-10-08 14:47 UTC by Phil Gear
- **Cloned:** 2026-10-14 18:23 UTC (topic id 746611, raw JSON `_raw/746611.json`)

---

**Live Interactive 3D Simulator**: [https://www.kaggle.com/code/philgear/3d-knee-anatomy-slicing-simulator](https://www.kaggle.com/code/philgear/3d-knee-anatomy-slicing-simulator)

(Sorry for the scroll)

---

### Hi everyone,

Like many of you, we started this competition treating knee MRIs primarily as bags of 2D slices (MIL pooling, 2.5D channel stacking, multi-view convolutions). 

However, as we pushed past the ~0.93 mark, we noticed our models were occasionally treating visually similar textures as interchangeable, ignoring the fact that **an MRI scanner samples a continuous 3D physical volume**.

To test whether our anatomical intuition aligns with clinical reality, we built a **zero-dependency interactive 3D WebGL Knee Anatomy & MRI Slicing Simulator** right inside a Kaggle notebook. 

👉 **You can orbit the 3D model and scrub slice depth controls here**: [https://www.kaggle.com/code/philgear/3d-knee-anatomy-slicing-simulator](https://www.kaggle.com/code/philgear/3d-knee-anatomy-slicing-simulator)

Before we push further into this architectural direction, **we'd love to ask the community: are we on the right track with these three core hypotheses?**

---

### ❓ Question 1: Continuous Slice Depth ($z$) vs. Standard Pooling

In the 3D simulator, scrubbing the sagittal slicing plane demonstrates how deterministic anatomical landmarks are:
- $z \in [0.10, 0.25]$: Medial Meniscus (posterior & anterior horns)
- $z \in [0.42, 0.58]$: Intercondylar Notch (ACL & PCL crossing)
- $z \in [0.75, 0.90]$: Lateral Meniscus & Popliteus

When we injected continuous slice depth $z \in [0.0, 1.0]$ into a 2.5D feature extractor via a simple sinusoidal + MLP gating module (shared in the notebook), our cross-validation jumped from **0.8208 to 0.8782 Macro AUC**.

👉 **Our Question**: Are other top teams explicitly conditioning on physical slice depth (e.g., from `ImagePositionPatient` or normalized slice index), or are your 3D/MIL pooling heads learning this implicitly? Does physical depth conditioning help prevent false-positive tears in peripheral slices on your validation sets?

---

### ❓ Question 2: Laterality Canonicalization (Left vs. Right Knee Mirroring)

A subtle spatial trap we noticed when aligning coordinate frames:
- In a **Right Knee**, $z = 0.15$ is **Medial** and $z = 0.85$ is **Lateral**.
- In a **Left Knee**, $z = 0.15$ is **Lateral** and $z = 0.85$ is **Medial**!

Without laterality canonicalization, feeding a non-mirrored Left Knee into a depth-conditioned model inverts the spatial meaning of the features for roughly 50% of patients. When we canonicalized all studies to a standard Right Knee orientation (mirroring left knees based on DICOM metadata), spatial alignment snapped into place.

👉 **Our Question**: Have others found that laterality canonicalization provides a clean boost, or does standard random horizontal flip augmentation during training already make your models invariant to this?

---

### ❓ Question 3: Kinematic Triads & Biomechanical Co-Occurrence

In the simulator's findings carousel, selecting an abnormality reveals its **Companion Triad**:
- **ACL Tear** $\rightarrow$ Co-occurs with **Bone Contusion (~88%)** (pivot-shift impaction of the lateral femoral condyle against the posterior tibial plateau) and **Medial Meniscus Tear (~64%)** (O'Donoghue's Triad).
- **Baker's Cyst** $\rightarrow$ Co-occurs with **Joint Effusion (~72%)** (unidirectional fluid valve into the popliteal fossa).

We've been experimenting with **Asymmetric Loss (ASL)** ($\gamma_- = 4.0, \gamma_+ = 1.0$) and Bayesian prior smoothing to handle sparse positive findings and sharpen borderline predictions.

👉 **Our Question**: Are teams using explicit graph neural networks, Bayesian priors, or multi-task loss weightings to capture these physical dependencies? Or does standard multi-label cross-entropy already capture enough co-occurrence signal in practice?

---

### 🎨 Features in the Simulator:

If you want to test these anatomical relationships yourself in the notebook (runs 100% locally in browser, zero GPU required):
1. **Interactive Plane Slicing**: Sagittal, Coronal, and Axial planes with real-time clipping.
2. **Procedural Tissue Layers**: Toggle Bone, Musculotendinous units, and Skin envelopes.
3. **Kaizen Visual Management (*Mieruka*)**: 
   - 🔴 **Andon Red**: Acute mechanical tears (ACL, bucket-handle meniscal tears, fractures)
   - 🟡 **Kaizen Amber**: Secondary reactive effusion, bone marrow edema contusion, Baker's cysts
   - 🟢 **Kaizen Green**: Intact baseline homeostatic architecture
4. **Doctor ⟷ Patient Mode Switch**: Toggle between high-density radiological heuristics (*Blumensaat Line*, *Bow-Tie 2-Slice Rule*) and patient-facing 3-Act recovery narratives.

We'd love to hear your thoughts, feedback, and whether this matches what you're seeing in your own validation splits! 🚀

---

## Comments (1, threaded, oldest first)

### [1] Phil Gear

**Phil Gear** (@philgear, Contributor) · _topic author_ · 2026-10-08 14:47 UTC · LB rank 1996

I added a game too to help teach patients, one for the waiting room.
