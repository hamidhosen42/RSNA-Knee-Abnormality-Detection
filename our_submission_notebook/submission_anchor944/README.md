# Anchor 0.944 (verbatim fork)

`rsna-knee-anchor944.ipynb` is byte-identical to goodpjw2008/rsna-knee-stack-2-5d-convnext-mil-lb-0-944 **v2**
(the scored version, public LB 0.944; medvax reproduced it at 0.944 too). It is our E02 anchor code
(D4 public part = Jiwei v11 + Fracture flip; all 23 base cells hash-match) plus the public Apache-2.0
2.5D ConvNeXt-T reader (dataset goodpjw2008/rsna-knee-2-5d-convnext-reader v1), global rank blend
0.70 stack / 0.30 reader, weight fixed by the author before submitting.

Data sources use the same version pins as our E02/E11 submission metadata, plus the reader dataset v1.
Push: `kaggle kernels push -p submission_anchor944`. Analysis: audit/12_new_0944_notebooks.md.
