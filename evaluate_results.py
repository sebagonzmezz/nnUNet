import argparse
from pathlib import Path
from itertools import chain
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import SimpleITK as sitk
from skimage.morphology import skeletonize

def compute_iou(gt, pred):
    intersection = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()
    return intersection / union if union != 0 else 0.0


def compute_dice(gt, pred):
    intersection = np.logical_and(gt, pred).sum()
    s = gt.sum() + pred.sum()
    return (2.0 * intersection) / s if s != 0 else 0.0


def cl_score(v, s):
    return (v & s).sum() / v.sum() if v.sum() != 0 else 0.0


def compute_cldice(gt, pred):
    gt = gt.astype(bool)
    pred = pred.astype(bool)

    skel_gt = skeletonize(gt)
    skel_pred = skeletonize(pred)

    tprec = cl_score(skel_pred, gt)
    tsens = cl_score(skel_gt, pred)

    if tprec + tsens == 0:
        return 0.0

    return 2 * tprec * tsens / (tprec + tsens)

def process_case(pred_path, gt_dir, labels, cldice_flag):
    gt_path = Path(gt_dir) / pred_path.name

    gt = sitk.GetArrayFromImage(sitk.ReadImage(str(gt_path)))
    pred = sitk.GetArrayFromImage(sitk.ReadImage(str(pred_path)))

    case_id = pred_path.stem.replace(".nii", "")
    case_result = {}
    neg_case_result = {}

    for label in labels:
        gt_bin = (gt == label)
        pred_bin = (pred == label)

        if gt_bin.sum() > 0:
            case_result[f"label_{label}_iou"] = compute_iou(gt_bin, pred_bin)
            case_result[f"label_{label}_dice"] = compute_dice(gt_bin, pred_bin)
            if cldice_flag:
                case_result[f"label_{label}_cldice"] = compute_cldice(gt_bin, pred_bin)
        else:
            neg_case_result[f"label_{label}_size"] = pred_bin.sum()

    print(f"{pred_path} processed")
    fold = str(pred_path).split("fold")[-1][1]
    return case_id, fold, case_result, neg_case_result

def compute_metrics(gt_dir, pred_dir, num_folds, labels, cldice_flag, workers):
    if num_folds is not None:
        pred_paths = list(chain.from_iterable(
            (Path(pred_dir) / f"fold_{i}").rglob("*.nii.gz")
            for i in range(num_folds)
        ))
    else:
        pred_paths = list(Path(pred_dir).rglob("*.nii.gz"))

    results = {}
    neg_results = {}

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                process_case,
                pred_path,
                gt_dir,
                labels,
                cldice_flag
            )
            for pred_path in pred_paths
        ]

        for future in futures:
            case_id, fold, case_result, neg_case_result = future.result()
            results[f"{fold}_{case_id}"] = case_result
            neg_results[f"{fold}_{case_id}"] = neg_case_result

    neg_df = pd.DataFrame(neg_results).T
    neg_df = neg_df[neg_df.notna().all(axis=1)]

    return pd.DataFrame(results).T, neg_df

def main():
    parser = argparse.ArgumentParser(
        description="Compute segmentation metrics per label for nnUNet output."
    )

    parser.add_argument("--gt_dir", type=str, required=True)
    parser.add_argument("--pred_dir", type=str, required=True)
    parser.add_argument("--num_folds", type=int, default=None)

    parser.add_argument(
        "--labels",
        type=int,
        nargs="+",
        required=True,
        help="List of labels to compute metrics for. Example: --labels 1 2 3"
    )

    parser.add_argument(
        "--cldice_flag",
        action="store_true",
        help="Compute clDice metric."
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel worker processes"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path"
    )

    args = parser.parse_args()

    if args.output is None:
        args.output = Path(args.pred_dir) / "metrics.csv"
        neg_output = Path(args.pred_dir) / "neg_metrics.csv"

    df, neg_df = compute_metrics(
        args.gt_dir,
        args.pred_dir,
        args.num_folds,
        args.labels,
        args.cldice_flag,
        args.workers
    )

    df.to_csv(args.output, index=True)
    print(df.describe())
    if not neg_df.empty:
        neg_df.to_csv(neg_output, index=True)
        print(neg_df)


if __name__ == "__main__":
    main()
