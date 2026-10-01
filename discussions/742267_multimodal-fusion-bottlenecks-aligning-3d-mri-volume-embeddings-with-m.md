# Multimodal Fusion Bottlenecks: Aligning 3D MRI Volume Embeddings with Multilingual Report Transformers

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/742267
- **Author:** **Lavin Wins** (@lavinwins, Contributor) · _topic author_ · 2026-09-21 03:44 UTC · ▲-10 · LB rank 279
- **Topic votes:** -10 · **Comments/replies:** 8 · **Pinned:** no
- **Last comment:** 2026-09-24 13:28 UTC by Lavin Wins
- **Cloned:** 2026-10-01 19:24 UTC (topic id 742267, raw JSON `_raw/742267.json`)

---

## Introduction:

Hi all,
As the leaderboard accelerates past 0.95, I’ve been experimenting with the multimodal aspect of the RSNA Knee Abnormality Detection dataset. While visual-only backbones give a solid baseline, properly fusing the 3D MRIs with the multi-language text reports presents a unique architecture bottleneck.

I wanted to share a specific roadblock I hit regarding cross-attention alignment, along with a lightweight fusion snippet, to see how others are tackling this.


## The Roadblock / Insight

When feeding the 3D MRI embeddings (extracted via a 3D-EfficientNet) and text embeddings (via a multilingual BERT variant) into a late-fusion Cross-Attention layer, I noticed a severe gradient imbalance. The model aggressively overfits to the text reports within the first 3 epochs, completely ignoring subtle visual cues for rarer abnormalities like MCL tears.
To mitigate this, I implemented a custom Layer-wise Learning Rate Decay (LLRD) specifically for the text encoder to prevent it from dominating early training.

## Quick preview of the Multimodal Cross-Attention Block I am testing

`

import torch
import torch.nn as nn

class RSNAKneeFusion(nn.Module):
    def __init__(self, visual_dim=1024, text_dim=768, embed_dim=512):
        super().__init__()
        self.vis_proj = nn.Linear(visual_dim, embed_dim)
        self.txt_proj = nn.Linear(text_dim, embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        
    def forward(self, image_feats, text_feats):
        # image_feats shape: [B, S, 1024] (3D context tokens)
        # text_feats shape:  [B, T, 768]  (Radiology report tokens)
        v = self.vis_proj(image_feats)
        t = self.txt_proj(text_feats)
        
        # Query: Image, Key/Value: Text
        attn_out, _ = self.cross_attn(v, t, t)
        return attn_out.mean(dim=1) # Global fused representation
`

###The Peer-Level Questions :

Q1 ) Text Bias: Is anyone else finding that multimodal models overfit drastically faster than vision-only models on this dataset? Are you utilizing a warm-up strategy for the image backbone before introducing text?

Q2) Tokenization: With the reports being in a dozen different languages, did you find standard translation APIs helpful, or are native multilingual models (like XLM-RoBERTa) retaining the clinical nuances better?

Q3) CV-to-LB Correlation: My local 5-fold CV tracks perfectly with the text-dominant model, but I suspect a private LB shakeup if the hidden test set contains missing or brief text reports. How are you handling potential missing text modalities at inference?

---

## Comments (8, threaded, oldest first)

### [1] Wayne_127

**Wayne_127** (@wayne127, Expert) · 2026-09-21 04:29 UTC · LB rank 786

During inference on the test set, diagnostic reports are not provided.🤣

#### ↳ [1.1] Lavin Wins

**Lavin Wins** (@lavinwins, Contributor) · _topic author_ · 2026-09-21 08:15 UTC · ▲-1 · LB rank 279

Lol forgot to add

### [2] (unknown)

**(unknown)** · 2026-09-21 06:35 UTC

_(no content — deleted or empty)_

#### ↳ [2.1] Lavin Wins

**Lavin Wins** (@lavinwins, Contributor) · _topic author_ · 2026-09-21 08:16 UTC · LB rank 279

Hmm... Exactly

### [3] Enai

**Enai** (@enaimaan, Contributor) · 2026-09-23 11:08 UTC · LB rank 3042

have you tried to completely freeze the text encoder and train just the image encoder? This way, you don't need reports in testing. you are just using them to get a study-wise embedding that you can later train with a small set of high-quality labels. This is an idea I was contemplating.

#### ↳ [3.1] Lavin Wins

**Lavin Wins** (@lavinwins, Contributor) · _topic author_ · 2026-09-23 12:08 UTC · ▲-1 · LB rank 279

I tried this too , but I did not find this something special cause many discussion forum have already discussed this lol

##### ↳ ↳ [3.1.1] Enai

**Enai** (@enaimaan, Contributor) · 2026-09-23 13:19 UTC · LB rank 3042

Oh, thanks. I didn't notice that. I guess I should read the discussions more :)

##### ↳ ↳ [3.1.2] Lavin Wins

**Lavin Wins** (@lavinwins, Contributor) · _topic author_ · 2026-09-24 13:28 UTC · LB rank 279

no problem
