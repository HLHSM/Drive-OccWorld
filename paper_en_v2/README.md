# AgriOcc English manuscript v2

English translation of `../paper_cn_v2/main.tex`. This directory contains the manuscript, nine figure assets, bibliography, CAS template, and an independent build script. The older `../paper_en/` manuscript is separate.

Build:

```bash
TEXLIVE_BIN=/data/HL/texlive/2026/bin/x86_64-linux bash paper_en_v2/build.sh
```

The output is `paper_en_v2/main.pdf`; compilation diagnostics are retained in `compile.log` and `main.log`.

`translation_review.md` records the independent bilingual review. `translation_checks.json` records structural comparisons (labels, references, citations, figures, equations, and numeric table rows). `source_snapshot.sha256` identifies the Chinese source used for the latest synchronization.

## Synchronization on 2026-11-01

The English version includes the author's Chinese-source corrections: a 15 m forward extent and 75 × 75 × 25 output grid; Base crop IoU of 30.03%; Base + TSA mIoU/crop IoU of 62.76%/35.76%; the present-class averaging protocol for the ORAD-3D farmland subset; and five, rather than six, best class IoUs for refinement-head adaptation.

Two associated inconsistencies still present in the Chinese source were corrected in English using the stated spatial extent and table values:

- The task-definition ROI uses `[0,15] × [-7.5,7.5] × [-2,3] m`, consistent with the 15 × 15 × 5 m region, experimental protocol, and 75 × 75 × 25 grid. The Chinese equation still has `[-10,10]` laterally.
- Agri-AMoE gains over Base + TSA are 1.44 mIoU percentage points (64.20 − 62.76) and 3.17 crop-IoU percentage points (38.93 − 35.76). The Chinese prose still gives the older gains of 0.44 and 1.17.

The Chinese manuscript was not edited by the translation process. The translation preserves the author's reported results; this task does not independently validate experimental provenance.
