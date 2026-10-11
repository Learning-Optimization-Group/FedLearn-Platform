# ResNet server step and NIH threshold diagnosis

The fixed offline ResNet comparison suggests testing a smaller FedOpt server
step next. The six-round comparison retains the same initialization, seed,
64-image client packages,320-image disjoint validation package, Python CPU
runtime and client Adam rate0.001. Reducing server Adam's rate from0.01 to0.001
changes final validation from33/320 to166/320 correct. Training reaches128/128;
validation still misclassifies almost half its examples, and class recalls remain
uneven. Complete trajectories and exact checkpoint reload are in the
[control receipt](../../results/on-device/resnet-server-step-controls-2026-10-10.json).

This is an offline two-client Python comparison. It has not qualified that server
rate on the phone, changed platform defaults, or used the official test set to
choose settings. The next useful check is a fresh mixed-device run at the candidate
rate, followed by independent evaluation after settings are fixed.

The NIH diagnosis keeps the existing ten-epoch last-block checkpoint unchanged.
Its checkpoint hash, four input packages, default confusion counts and AUROC
reproduce the previous evaluation. The threshold is selected only from balanced
validation to retain at least80% of its positive examples. The proportional
reference is evaluated after the threshold is fixed; it is not an independent
cohort from balanced validation, which shares169 held-out patients with it.
Neither held-out set overlaps training patients or prepared image content.

At threshold0.322976, reference counts are5TP,1FN,142TN and364FP. The previous
0.5 threshold had the same5TP and1FN but222TN and284FP. Lowering the threshold
therefore adds80 false positives without recovering a positive. Reference AUROC
remains0.554018. The512-image reference contains only6 positives, so its measured
sensitivity is especially uncertain. The
[threshold receipt](../../results/on-device/nih-threshold-diagnosis-2026-10-10.json)
supports neither useful NIH classification nor deployment of this threshold.
This is operating-threshold diagnosis, not probability calibration or improved
model learning.

Further NIH work should improve ranking on a larger patient-disjoint validation
cohort and compare against a fixed centralized baseline. Moving the threshold
alone does not repair the current weak model.
