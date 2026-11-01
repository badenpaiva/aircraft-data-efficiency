# Detection study results

**Provisional dataset and HQGAN reconstruction: not a certified paper reproduction.**

Plan: `030ea934951e35d8809e0c9e2be1c0aed99e9d2e7fc0d4dc2280c68f4a693222`
Code: `0d49d04e0bf40032d0094d7f1a05123009f98e889b3bb2d05344c708103e461d`
Planned detector cells: 60. Status: {'complete': 60}.
Completed validation metrics: 60; test metrics: 0.

## Measured results

| Split | Model | Arm | Fraction | Seeds | mAP50 mean | mAP50-95 mean |
|---|---|---|---:|---:|---:|---:|
| val | yolo11 | hqgan_0p5 | 0.05 | 1 | 0.0766 | 0.0298 |
| val | yolo11 | hqgan_0p5 | 0.1 | 1 | 0.1214 | 0.0400 |
| val | yolo11 | hqgan_0p5 | 0.25 | 1 | 0.2361 | 0.0781 |
| val | yolo11 | hqgan_0p5 | 0.5 | 1 | 0.3861 | 0.1375 |
| val | yolo11 | hqgan_0p5 | 1 | 1 | 0.4739 | 0.1744 |
| val | yolo11 | hqgan_1p0 | 0.05 | 1 | 0.0862 | 0.0283 |
| val | yolo11 | hqgan_1p0 | 0.1 | 1 | 0.1211 | 0.0443 |
| val | yolo11 | hqgan_1p0 | 0.25 | 1 | 0.2566 | 0.0899 |
| val | yolo11 | hqgan_1p0 | 0.5 | 1 | 0.3934 | 0.1424 |
| val | yolo11 | hqgan_1p0 | 1 | 1 | 0.4680 | 0.1812 |
| val | yolo11 | hqgan_2p0 | 0.05 | 1 | 0.0908 | 0.0279 |
| val | yolo11 | hqgan_2p0 | 0.1 | 1 | 0.1415 | 0.0514 |
| val | yolo11 | hqgan_2p0 | 0.25 | 1 | 0.2103 | 0.0749 |
| val | yolo11 | hqgan_2p0 | 0.5 | 1 | 0.3492 | 0.1262 |
| val | yolo11 | hqgan_2p0 | 1 | 1 | 0.4543 | 0.1654 |
| val | yolo26 | hqgan_0p5 | 0.05 | 1 | 0.0690 | 0.0242 |
| val | yolo26 | hqgan_0p5 | 0.1 | 1 | 0.1152 | 0.0385 |
| val | yolo26 | hqgan_0p5 | 0.25 | 1 | 0.2452 | 0.0869 |
| val | yolo26 | hqgan_0p5 | 0.5 | 1 | 0.3851 | 0.1441 |
| val | yolo26 | hqgan_0p5 | 1 | 1 | 0.4775 | 0.1802 |
| val | yolo26 | hqgan_1p0 | 0.05 | 1 | 0.0977 | 0.0317 |
| val | yolo26 | hqgan_1p0 | 0.1 | 1 | 0.1162 | 0.0412 |
| val | yolo26 | hqgan_1p0 | 0.25 | 1 | 0.2325 | 0.0795 |
| val | yolo26 | hqgan_1p0 | 0.5 | 1 | 0.3799 | 0.1326 |
| val | yolo26 | hqgan_1p0 | 1 | 1 | 0.4732 | 0.1826 |
| val | yolo26 | hqgan_2p0 | 0.05 | 1 | 0.0961 | 0.0294 |
| val | yolo26 | hqgan_2p0 | 0.1 | 1 | 0.1163 | 0.0423 |
| val | yolo26 | hqgan_2p0 | 0.25 | 1 | 0.2470 | 0.0783 |
| val | yolo26 | hqgan_2p0 | 0.5 | 1 | 0.3764 | 0.1383 |
| val | yolo26 | hqgan_2p0 | 1 | 1 | 0.4944 | 0.1878 |
| val | yolov8 | hqgan_0p5 | 0.05 | 1 | 0.0728 | 0.0266 |
| val | yolov8 | hqgan_0p5 | 0.1 | 1 | 0.1150 | 0.0400 |
| val | yolov8 | hqgan_0p5 | 0.25 | 1 | 0.2376 | 0.0860 |
| val | yolov8 | hqgan_0p5 | 0.5 | 1 | 0.4124 | 0.1506 |
| val | yolov8 | hqgan_0p5 | 1 | 1 | 0.4852 | 0.1940 |
| val | yolov8 | hqgan_1p0 | 0.05 | 1 | 0.0820 | 0.0262 |
| val | yolov8 | hqgan_1p0 | 0.1 | 1 | 0.1250 | 0.0404 |
| val | yolov8 | hqgan_1p0 | 0.25 | 1 | 0.2552 | 0.0854 |
| val | yolov8 | hqgan_1p0 | 0.5 | 1 | 0.3844 | 0.1515 |
| val | yolov8 | hqgan_1p0 | 1 | 1 | 0.4786 | 0.1915 |
| val | yolov8 | hqgan_2p0 | 0.05 | 1 | 0.0935 | 0.0307 |
| val | yolov8 | hqgan_2p0 | 0.1 | 1 | 0.1451 | 0.0470 |
| val | yolov8 | hqgan_2p0 | 0.25 | 1 | 0.2430 | 0.0817 |
| val | yolov8 | hqgan_2p0 | 0.5 | 1 | 0.4086 | 0.1500 |
| val | yolov8 | hqgan_2p0 | 1 | 1 | 0.4304 | 0.1717 |
| val | yolox | hqgan_0p5 | 0.05 | 1 | 0.0292 | 0.0091 |
| val | yolox | hqgan_0p5 | 0.1 | 1 | 0.0810 | 0.0265 |
| val | yolox | hqgan_0p5 | 0.25 | 1 | 0.1287 | 0.0411 |
| val | yolox | hqgan_0p5 | 0.5 | 1 | 0.2561 | 0.0851 |
| val | yolox | hqgan_0p5 | 1 | 1 | 0.2570 | 0.0840 |
| val | yolox | hqgan_1p0 | 0.05 | 1 | 0.0487 | 0.0186 |
| val | yolox | hqgan_1p0 | 0.1 | 1 | 0.0627 | 0.0204 |
| val | yolox | hqgan_1p0 | 0.25 | 1 | 0.1521 | 0.0493 |
| val | yolox | hqgan_1p0 | 0.5 | 1 | 0.2198 | 0.0755 |
| val | yolox | hqgan_1p0 | 1 | 1 | 0.3141 | 0.1069 |
| val | yolox | hqgan_2p0 | 0.05 | 1 | 0.0451 | 0.0129 |
| val | yolox | hqgan_2p0 | 0.1 | 1 | 0.0468 | 0.0168 |
| val | yolox | hqgan_2p0 | 0.25 | 1 | 0.1029 | 0.0335 |
| val | yolox | hqgan_2p0 | 0.5 | 1 | 0.1785 | 0.0600 |
| val | yolox | hqgan_2p0 | 1 | 1 | 0.1085 | 0.0325 |

## Reporting notes

- AP uses the same pycocotools evaluator and original image area bins for all models. Scores are fractions, not percentages.
- summary.csv uses Student-t intervals over seeds. No interval is claimed from a single seed. Intervals are not clipped to [0,1].
- Per-run fixed-point precision/recall/F1 intervals resample related-image groups. These are not AP confidence intervals.
- Paired tests match model, fraction, split and seed; Holm correction spans all available contrasts per test type. Missing pairs are reported, not imputed.
- Fewer than all planned seeds/cells means an incomplete comparison. Three seeds have weak inferential power.
- Classical augmentation is offline with transformed boxes; its image count and optimizer exposure differ from baseline. Compare cost as well as accuracy.
- YOLOX-S and the Ultralytics nano models are different parameter sizes. This is a family comparison, not a size-matched causal comparison.
- YOLOX uses the official architecture/loss with a local AdamW trainer; the paper used SGD and 300 detector epochs. This grid is not an exact reproduction.
- Latency uses batch one wall time including preprocessing after warmup; compare only on the same device/environment.
- Generated samples must pass semantic, placement and final composite review before HQGAN detector training.
- No CHPDNet, segmentation scores, FID or IS are inferred from detection metrics.

Artifacts: runs.csv, per_class.csv, per_image_counts.csv, summary.csv, paired_comparisons.csv, coverage.csv, dataset_inventory.csv, subsets.csv, all_metrics.json and figures/.
