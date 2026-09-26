# Working with our own GPU

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/739000
- **Author:** **fran garcia** (@frangarcia02, Contributor) · _topic author_ · 2026-09-02 12:33 UTC · ▲1 · LB rank 3021
- **Topic votes:** 1 · **Comments/replies:** 11 · **Pinned:** no
- **Last comment:** 2026-09-08 04:52 UTC by Moonwon12
- **Cloned:** 2026-09-26 16:24 UTC (topic id 739000, raw JSON `_raw/739000.json`)

---

Hi! I'm new to Kaggle, this is my first time participating in a competition, and I have a couple of questions I'd like to ask. Can I use my own GPU to train the model, or does it have to run exclusively on a notebook? Thanks for your help!

---

## Comments (11, threaded, oldest first)

### [1] PC Jimmmy

**PC Jimmmy** (@pcjimmmy, Master) · 2026-09-02 12:56 UTC · ▲1 · LB rank 641

You can use your GPU's, if your a relative of Elon Musk maybe he would let you use the Nashville facility.

Almost 100% of kaggle competitions let you train local.  You will need to attached your model files in a data set, but lots of examples of how to do that and most LLM's know enough about kaggle to help you create a zip file local to use in the data set.

Only the final inference and creation of the submission.csv needs to be on a kaggle notebook.

have fun

### [2] Seung Sup Lee

**Seung Sup Lee** (@seungsuplee, Contributor) · 2026-09-02 14:48 UTC · ▲1 · LB rank 2633

Yes, you can train on your own GPU.

You can set up Ubuntu locally with Python, PyTorch, CUDA, and the libraries you need, then do most of your training there.

If you are starting now, I recommend targeting Kaggle’s **Tesla T4 x2** environment rather than the P100, since the P100 is being retired.

One important thing: each T4 has **16 GB of VRAM**. T4 x2 means two separate 16 GB GPUs, not one 32 GB GPU.

For a beginner, I strongly recommend starting with **one GPU first**. Make sure the whole training or inference pipeline finishes correctly within 16 GB VRAM. Once that works reliably, then modify the code to use both T4 GPUs in parallel.

Multi-GPU code adds extra complexity, so debugging is much easier if the single-GPU version already works.

Also, if your local GPU has much more VRAM, your code may work locally but fail on Kaggle. You may need to reduce the batch size, input size, model size, or memory usage before uploading it.

So the safest approach is:

**Local Ubuntu → one GPU → fit within 16 GB VRAM → complete the full pipeline → then optimize for T4 x2.**

#### ↳ [2.1] fran garcia

**fran garcia** (@frangarcia02, Contributor) · _topic author_ · 2026-09-02 23:13 UTC · LB rank 3021

Thanks! I have another question. If I train locally, do I have to download the Kaggle dataset?

##### ↳ ↳ [2.1.1] PC Jimmmy

**PC Jimmmy** (@pcjimmmy, Master) · 2026-09-03 02:14 UTC · ▲1 · LB rank 641

Yes - you need to download the kaggle data.

##### ↳ ↳ [2.1.2] Komil Parmar

**Komil Parmar** (@komilparmar, Expert) · 2026-09-03 04:59 UTC · ▲1 · LB rank 59

I don't understand this question. If you want to train locally, why or how wouldn't you need to have the data locally as well? 
Are you planning to make some data stream? Or train a model without using the competition data?

##### ↳ ↳ [2.1.3] Seung Sup Lee

**Seung Sup Lee** (@seungsuplee, Contributor) · 2026-09-04 11:08 UTC · ▲1 · LB rank 2633

Yes. If you want to work on it locally, you need to download the competition data to your PC. In my case, the compressed ZIP file is about 247 GB, and downloading such a large file from overseas can be quite slow and inconvenient.

Kaggle also appears to be using Google-hosted infrastructure, and from what I have observed, the server location may differ depending on the runtime. For example, CPU-only sessions and Tesla T4 ×2 GPU sessions seem to be assigned to different server regions within the United States. This may also affect download and transfer speeds for overseas participants.

### [3] Dread Development

**Dread Development** (@dreaddevelopment, Expert) · 2026-09-02 15:42 UTC · ▲1 · LB rank 145

Yes you absolutely can! If you’re looking for cheap options outside of your own local GPU’s vast.ai is what I generally use

#### ↳ [3.1] Phuc

**Phuc** (@phucbb, Contributor) · 2026-09-03 03:02 UTC · ▲1 · LB rank 2153

Hi, How can you transfer data between and within new instances in Vastai? I wonder if anyone can use spot instances effectively.

##### ↳ ↳ [3.1.1] Dread Development

**Dread Development** (@dreaddevelopment, Expert) · 2026-09-03 03:15 UTC · ▲1 · LB rank 145

Hey! So I treat vast as disposable compute, not storage so small files & code go directly over SSH with scp on Linux
rsync -avP is better because interrupted transfers resume.
For large datasets, upload them once to a persistent storage like a vast volume, S3 etc then let every new instance pull from there instead of uploading repeatedly.
I do use the spot instances with little to no issue ever, you just have to make sure you're monitoring & saving checkpoints.
I have also found that chunking & sending parallel transfers is much more effective for large files.
I have used the spot instances for a number of comps as well as my own distillation, pre training & fine tuning with less issues than I have finger so, hope this helped!
If you have any more questions feel free!

### [4] Linda PerezP

**Linda PerezP** (@lindaperezp, Contributor) · 2026-09-02 16:10 UTC · ▲1 · LB rank 3076

Hi Fran, 

You can use Kaggle, AWS, GCP and even your local computer without GPUs. Take into account, the time, complexity and cost associated to the project. Ask your LLM, what are the Data Engineering and Deployment aspects that you should cover for a multimodal project like this provide the Competition link. 


Good Luck!

### [5] Moonwon12

**Moonwon12** (@moonwon12, Contributor) · 2026-09-08 04:52 UTC · ▲-1 · LB rank 1221

Hi! I'm also new to kaggle and this is my first competition.
I think you can use your own GPU but it is not recomended. I don't know your GPU availablity but for me 30h capcity that kaggle support was enough.
