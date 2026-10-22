# Medical feature candidate: development-only real-image input check

22 October 2026. This extends the [synthetic candidate check](medical-backbone-candidate-2026-10-21.md)
to eight real NIH development images. It verifies that source-bound X-rays reach the
pinned CheXpert DenseNet-121 under an explicit preprocessing contract and yield frozen
pooled features. It selects no condition, fits no classifier, reads no test manifest
or label column and reports no accuracy/AUROC. It is not a better baseline result.

## What the helper enforces

`repro/probe_xray_development.py` refuses before touching images unless:

- the cache `report.json` matches its pin and the train/validation manifests match the
  report's payload hashes; `test.csv` is never opened (a regression test makes it
  unreadable and selection still succeeds);
- every manifest row has a canonical `images_NNN/images/<index>` path whose patient
  prefix equals `patient_id`, and the two splits share no patient or image across the
  whole sets, not only the sample;
- the original image-audit report and `images.csv` match their pins, and each selected
  row's split, patient and path agree with its audit row.

Per image, raw bytes are read inside the dataset root, hashed and compared with the
audit `file_sha256` and byte count **before** decoding. The decoded RGB identity must
then reproduce the audit `rgb_sha256`, and an independent imageio decode must equal
the Pillow pixels exactly.

**Sample rule.** Per split, patients are ordered by
SHA256(`capstone-xray-development-probe-v1` NUL split NUL patient_id); the first four
are taken, using each patient's lowest image index. This rule doesn't use labels,
findings, image content or model output. A refused image fails the probe and is recorded;
it is never swapped for another sample.

**Preprocessing contract.** Accept 8-bit `L`, or RGB/RGBA whose three colour channels
are identical and whose alpha is fully opaque, then keep channel 0. This matches the
pinned XRV `normalize(reshape=True)` channel rule while refusing genuine colour or
16-bit input instead of converting it. Next: `xrv.utils.normalize(maxval=255)` to
[-1024, 1024], XRV center crop, then skimage resize to 1 × 224 × 224. Features
come from `features2()`, not from the diagnostic outputs.

## Observed result

| Control | Result |
| --- | --- |
| Whole-set separation | 22,399 train / 5,600 validation patients, no overlap |
| Sample | 4 train + 4 validation images, 8 distinct patients |
| Raw-byte and decoded-pixel bindings | 8/8 match the original audit |
| Decoder agreement (Pillow vs imageio) | 8/8 exactly equal |
| Observed PNG mode | All 8 are 8-bit `L`; the RGB/RGBA branch was exercised only by synthetic tests |
| Pooled features | 8 × 1,024, finite, all rows distinct |
| Repeat / manual ReLU-pool / single-vs-batch | Exactly equal / exactly equal / max abs difference 0.0 |
| Parameters and buffers | Unchanged; state hash equals across runs |
| Fresh-process replay | Receipt byte-identical (SHA256 `b5cb4429…a02b8`) |
| XRV module sources | Same hashes as the synthetic receipt |

The helper has eight standalone tests: six written first and watched failing
against the earlier stub, plus two added afterwards for the unreadable test
manifest and the decode contract. The full reproduction suite passes 17/17.

## Limits

Eight images on one Mac CPU, so these are execution and provenance checks, not
throughput, device-generality or representation-quality evidence. It says nothing about whether these features separate any condition. The private
receipt holds image indices and stays outside the repository. The 512-wide historical
cache, its report and the historical study are unchanged. A new 1,024-wide cache still needs its own
version, report and dimension-aware qualification before use.
