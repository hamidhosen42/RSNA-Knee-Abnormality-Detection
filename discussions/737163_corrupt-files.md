# corrupt files

- **URL:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/737163
- **Author:** **Yee Ng** (@yeeseng, Expert) · _topic author_ · 2026-08-23 22:22 UTC · ▲10 · LB rank 3026
- **Topic votes:** 10 · **Comments/replies:** 4 · **Pinned:** no
- **Last comment:** 2026-08-27 07:09 UTC by Yiming Ma
- **Cloned:** 2026-10-09 09:46 UTC (topic id 737163, raw JSON `_raw/737163.json`)

---

There are two files in the training set that crashed my runs:
                                                                                                                 
  - StudyInstanceUID: 1.2.826.0.1.3680043.8.498.37833587429731221455928642963031995680                                                                                                                                     
  - SeriesInstanceUID: 1.2.826.0.1.3680043.8.498.31160785238781353848727311763596115703                                                                                                                                    
  - SOPInstanceUID: 1.2.826.0.1.3680043.8.498.66460523372602171398246641420761543626.dcm                                                                                                                        
                                                                                                                                                                                                                                                                                                 
  - StudyInstanceUID: 1.2.826.0.1.3680043.8.498.34685905030370793639196564723935583035                                                                                                                                     
  - SeriesInstanceUID: 1.2.826.0.1.3680043.8.498.39396636671446532796538574315802928348                                                                                                                                    
  - SOPInstanceUID: 1.2.826.0.1.3680043.8.498.41947538458330624797291612781685746320.dcm                                                                                                                        

Did anyone else experience the same issue, or is it just me? I'm trying to decide whether I should redownload the dataset.

---

## Comments (4, threaded, oldest first)

### [1] takeharu-yoshikawa

**takeharu-yoshikawa** (@takeharuyoshikawa, Contributor) · 2026-08-25 10:41 UTC · LB rank 52

Yes, I experienced exactly the same issue with exactly those two DICOM files. Both files contain Pixel Data that is exactly half the size expected from their DICOM metadata, so this appears to be an issue in the distributed dataset rather than a failed download. Redownloading is therefore unlikely to fix these two files.

#### ↳ [1.1] Yee Ng

**Yee Ng** (@yeeseng, Expert) · _topic author_ · 2026-08-25 15:58 UTC · LB rank 3026

Thank you! I’m going to blacklist both during training. Cheers.

### [2] Jason Sho

**Jason Sho** (@shosys, Contributor) · _competition host_ · 2026-08-26 15:47 UTC · ▲2

Hello,

Thank you kindly for surfacing this issue--we will review these files and provide an update shortly.

### [3] Yiming Ma

**Yiming Ma** (@yimingma1024, Contributor) · 2026-08-27 07:09 UTC

Same issue here.
