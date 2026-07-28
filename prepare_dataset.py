"""
Prepare a paired volume/segmentation dataset for nnU-Net v2.

Takes two flat folders (scans and labels with matching filenames) and produces
a valid nnU-Net raw dataset: split into train/holdout, renamed to nnU-Net's
required convention, compressed to .nii.gz, with a dataset.json.

Usage:
    python prepare_dataset.py \
        --volumes  /path/to/volumes \
        --labels   /path/to/segmentations \
        --out      /path/to/nnUNet_raw \
        --dataset-id 100 \
        --dataset-name SPINE
"""

import argparse
import json
import random
import shutil
import sys
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import SimpleITK as sitk

# Vertebra label map, verified against the data by checking that label index
# increases monotonically from superior (head) to inferior (sacrum).
VERTEBRA_LABELS = (
    [f"C{i}" for i in range(1, 8)]      # 1-7
    + [f"T{i}" for i in range(1, 13)]   # 8-19
    + [f"L{i}" for i in range(1, 6)]    # 20-24
    + ["S1"]                            # 25
)


def find_pairs(volumes_dir: Path, labels_dir: Path):
    """Match scans to labels by filename. Refuses to proceed on any orphan."""
    volumes = {p.name: p for p in sorted(volumes_dir.glob("*.nii*"))}
    labels = {p.name: p for p in sorted(labels_dir.glob("*.nii*"))}

    only_volumes = sorted(set(volumes) - set(labels))
    only_labels = sorted(set(labels) - set(volumes))
    if only_volumes or only_labels:
        print("ERROR: unpaired files found.", file=sys.stderr)
        for n in only_volumes[:10]:
            print(f"  scan with no label:  {n}", file=sys.stderr)
        for n in only_labels[:10]:
            print(f"  label with no scan:  {n}", file=sys.stderr)
        sys.exit(1)

    return [(volumes[n], labels[n]) for n in sorted(volumes)]


def geometry_matches(scan: sitk.Image, label: sitk.Image) -> bool:
    """nnU-Net rejects the whole dataset if a scan and its label disagree
    on size, spacing, origin or orientation. Check before we waste GPU hours."""
    return (
        scan.GetSize() == label.GetSize()
        and np.allclose(scan.GetSpacing(), label.GetSpacing(), atol=1e-4)
        and np.allclose(scan.GetOrigin(), label.GetOrigin(), atol=1e-4)
        and np.allclose(scan.GetDirection(), label.GetDirection(), atol=1e-4)
    )


def convert_case(task, prefix: str):
    """Write one case out under nnU-Net's naming convention.

    The trailing _0000 is the modality channel index. CT is single-channel so
    it is always _0000, but nnU-Net still requires it to be there.

    Runs in a worker process; returns a plain tuple so it pickles cleanly.
    """
    split_name, scan_path, label_path, images_dir, labels_dir, case_id = task

    scan = sitk.ReadImage(str(scan_path))
    label = sitk.ReadImage(str(label_path))

    if not geometry_matches(scan, label):
        return split_name, scan_path.name, False, []

    # Labels must be integers; uint8 is enough for 25 classes and keeps them small.
    label = sitk.Cast(label, sitk.sitkUInt8)

    sitk.WriteImage(scan, str(images_dir / f"{prefix}_{case_id}_0000.nii.gz"), True)
    sitk.WriteImage(label, str(labels_dir / f"{prefix}_{case_id}.nii.gz"), True)

    present = [int(v) for v in np.unique(sitk.GetArrayFromImage(label))]
    return split_name, scan_path.name, True, present


def write_dataset_json(out_dir: Path, num_training: int):
    labels = {"background": 0}
    for index, name in enumerate(VERTEBRA_LABELS, start=1):
        labels[name] = index

    spec = {
        "channel_names": {"0": "CT"},
        "labels": labels,
        "numTraining": num_training,
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "SimpleITKIO",
    }
    (out_dir / "dataset.json").write_text(json.dumps(spec, indent=4))
    return spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volumes", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path,
                        help="nnUNet_raw directory")
    parser.add_argument("--dataset-id", type=int, default=100)
    parser.add_argument("--dataset-name", default="SPINE")
    parser.add_argument("--holdout", type=float, default=0.2,
                        help="fraction reserved for final testing (default 0.2). "
                             "nnU-Net cross-validates the remainder itself.")
    parser.add_argument("--seed", type=int, default=42,
                        help="fixed so the split is reproducible")
    parser.add_argument("--workers", type=int, default=max(1, cpu_count() - 2),
                        help="parallel conversion processes")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dataset_dir = args.out / f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    if dataset_dir.exists():
        if not args.overwrite:
            print(f"ERROR: {dataset_dir} already exists. Pass --overwrite to replace.",
                  file=sys.stderr)
            sys.exit(1)
        shutil.rmtree(dataset_dir)

    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    images_ts = dataset_dir / "imagesTs"
    labels_ts = dataset_dir / "labelsTs"
    for d in (images_tr, labels_tr, images_ts, labels_ts):
        d.mkdir(parents=True)

    pairs = find_pairs(args.volumes, args.labels)
    print(f"found {len(pairs)} matched scan/label pairs")

    # Shuffle and assign case ids BEFORE parallelising, so the output is
    # identical regardless of how many workers run or what order they finish in.
    random.Random(args.seed).shuffle(pairs)
    n_holdout = int(round(len(pairs) * args.holdout))
    holdout, train = pairs[:n_holdout], pairs[n_holdout:]
    print(f"  train (nnU-Net will 5-fold this): {len(train)}")
    print(f"  holdout (never seen until eval):  {len(holdout)}")

    tasks = []
    for split_name, subset, img_dir, lbl_dir in (
        ("train", train, images_tr, labels_tr),
        ("holdout", holdout, images_ts, labels_ts),
    ):
        for i, (scan_path, label_path) in enumerate(subset, start=1):
            tasks.append((split_name, scan_path, label_path, img_dir, lbl_dir,
                          f"{i:03d}"))

    print(f"converting with {args.workers} workers...")
    skipped = []
    labels_seen = set()
    worker = partial(convert_case, prefix=args.dataset_name)

    with Pool(args.workers) as pool:
        for n, (split_name, name, ok, present) in enumerate(
                pool.imap_unordered(worker, tasks, chunksize=4), start=1):
            if ok:
                labels_seen.update(present)
            else:
                skipped.append((split_name, name))
            if n % 100 == 0 or n == len(tasks):
                print(f"  {n}/{len(tasks)}", flush=True)

    n_training = len(list(images_tr.glob("*.nii.gz")))
    spec = write_dataset_json(dataset_dir, n_training)

    print(f"\nwrote {dataset_dir}")
    print(f"  imagesTr {n_training}   labelsTr {len(list(labels_tr.glob('*.nii.gz')))}")
    print(f"  imagesTs {len(list(images_ts.glob('*.nii.gz')))}   "
          f"labelsTs {len(list(labels_ts.glob('*.nii.gz')))}")

    declared = set(spec["labels"].values())
    missing = sorted(declared - labels_seen - {0})
    if missing:
        names = [n for n, v in spec["labels"].items() if v in missing]
        print(f"\nWARNING: declared labels never present in any case: {names}")
    if skipped:
        print(f"\nWARNING: skipped {len(skipped)} cases for geometry mismatch:")
        for split_name, name in skipped[:10]:
            print(f"  [{split_name}] {name}")


if __name__ == "__main__":
    main()
