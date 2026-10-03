#!/usr/bin/env python3
"""YOLO dataset audit — read-only. Outputs to audit_out/. Re-run: python tools/audit_datasets.py"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import imagehash
import numpy as np
import yaml
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

SEED = 0
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "audit_out"
SPLITS = ("train", "valid", "test")
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tif", ".tiff"}
RF_BASE = re.compile(r"^(.+)\.rf\.[a-f0-9]{8,}\.(jpg|jpeg|png|bmp|webp)$", re.I)
PHASH_THRESHOLDS = (0, 3, 5, 8)
CROSS_THRESHOLDS = (0, 5, 10)
FRACTIONS = (1.0, 0.5, 0.25, 0.10, 0.05)
MIN_FAIL, MIN_WARN = 5, 20

# Expected Roboflow exports (folder resolved at runtime)
DATASET_SPECS: dict[str, dict[str, Any]] = {
    "D1": {
        "label": "Innovation Hangar v2 Roboflow v2 stock (no aug)",
        "expect_images": 4288,
        "folder_hint": "Innovation Hangar",
    },
    "D2": {
        "label": "DDIISc aircraft_skin_defects v1",
        "expect_images": 372,
        "folder_hint": "aircraft_skin_defects.v1-roboflow",
    },
    "D3": {
        "label": "SUTD aircraft-ai-dataset v4",
        "expect_images": 983,
        "folder_hint": "Aircraft AI Dataset.v4",
    },
    "D4": {
        "label": "airscraft-skin-crack-dent",
        "expect_images": None,
        "folder_hint": "airscraft-skin-crack-dent",
    },

}

CROSS_IDS = ("D1", "D2", "D3", "D4")

MAP_POOL = {
    "crack": "crack",
    "dent": "dent",
    "paint-off": "paint_peeling",
    "paint-peel-off": "paint_peeling",
    "rust": "corrosion",
}


def resolve_folder(hint: str) -> Path | None:
    if not RAW.is_dir():
        return None
    for p in sorted(RAW.iterdir()):
        if p.is_dir() and hint.lower() in p.name.lower():
            return p
    return None


def resolve_datasets() -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for did, spec in DATASET_SPECS.items():
        p = resolve_folder(spec["folder_hint"])
        if p is not None:
            paths[did] = p
    return paths


def read_data_yaml(ds_path: Path) -> dict[str, Any]:
    for name in ("data.yaml", "data.yml", "dataset.yaml"):
        yp = ds_path / name
        if yp.is_file():
            with open(yp, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    return {}


def parse_readme(ds_path: Path) -> dict[str, str]:
    info: dict[str, str] = {}
    for name in ("README.roboflow.txt", "README.dataset.txt"):
        rp = ds_path / name
        if not rp.is_file():
            continue
        text = rp.read_text(encoding="utf-8", errors="replace")
        info["file"] = name
        m = re.search(r"- v(\d+)", text, re.I)
        if m:
            info["version_line"] = m.group(0)
        if re.search(r"No image augmentation", text, re.I):
            info["augmentation"] = "none stated"
        aug = re.search(r"augmentation was applied[^\n]*", text, re.I)
        if aug:
            info["augmentation"] = aug.group(0).strip()
        pre = re.search(r"pre-processing was applied[^\n]*", text, re.I)
        if pre:
            info["preprocessing"] = pre.group(0).strip()
        imgs = re.search(r"includes (\d+) images", text, re.I)
        if imgs:
            info["image_count_readme"] = imgs.group(1)
        break
    return info


def stem_for_label(image_path: Path) -> str:
    return image_path.stem


def label_path_for_image(img: Path) -> Path:
    # .../split/images/foo.jpg -> .../split/labels/foo.txt
    parts = list(img.parts)
    if "images" in parts:
        idx = parts.index("images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")
    return img.with_suffix(".txt")


def roboflow_base(name: str) -> str | None:
    m = RF_BASE.match(name)
    return m.group(1) if m else None


def popcount_matrix(xor: np.ndarray) -> np.ndarray:
    x = xor.astype(np.uint64, copy=False)
    c = np.zeros(x.shape, dtype=np.uint16)
    for _ in range(64):
        c += (x & 1).astype(np.uint16)
        x >>= 1
    return c


def phash_int(h: imagehash.ImageHash) -> int:
    return int(str(h), 16)


@dataclass
class ImageRec:
    dataset_id: str
    split: str
    path: Path
    rel: str
    phash: int | None = None
    md5: str | None = None
    width: int = 0
    height: int = 0
    corrupt: bool = False
    grayscale: bool = False
    ext: str = ""


def parse_yolo_label(
    label_file: Path, nc: int
) -> tuple[list[tuple[int, float, float, float, float]], dict[str, int]]:
    stats = Counter()
    boxes: list[tuple[int, float, float, float, float]] = []
    if not label_file.is_file():
        return boxes, stats
    raw = label_file.read_text(encoding="utf-8", errors="replace").strip()
    if not raw:
        stats["empty_file"] += 1
        return boxes, stats
    for li, line in enumerate(raw.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        # Mixed YOLO rows are normalized for inventory, never auto-approved.
        from review_d1 import parse_line
        try:
            cid, kind, coordinates, bounds = parse_line(line, nc)
        except ValueError:
            stats["malformed_line"] += 1
            continue
        x1, y1, x2, y2 = bounds
        if kind == "polygon":
            stats["polygon_record"] += 1
        boxes.append((cid, (x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1))
    # duplicate boxes
    seen = Counter(boxes)
    for b, cnt in seen.items():
        if cnt > 1:
            stats["duplicate_box"] += cnt - 1
    return boxes, stats


def union_find_clusters(hashes: np.ndarray, threshold: int) -> list[list[int]]:
    n = len(hashes)
    if n == 0:
        return []
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    block = 512
    arr = hashes.astype(np.uint64, copy=False)
    for i0 in range(0, n, block):
        i1 = min(n, i0 + block)
        hi = arr[i0:i1]
        for j0 in range(i0, n, block):
            j1 = min(n, j0 + block)
            hj = arr[j0:j1]
            xor = hi.reshape(-1, 1) ^ hj.reshape(1, -1)
            dist = popcount_matrix(xor)
            ys, xs = np.where(dist <= threshold)
            for a, b in zip(ys.tolist(), xs.tolist(), strict=False):
                gi, gj = i0 + a, j0 + b
                if gi < gj:
                    union(gi, gj)
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return list(groups.values())


def cluster_stats(
    clusters: list[list[int]], splits: list[str]
) -> dict[str, Any]:
    multi = [c for c in clusters if len(c) >= 2]
    in_multi = sum(len(c) for c in multi)
    sizes = sorted((len(c) for c in multi), reverse=True)
    leak = 0
    for c in multi:
        ss = {splits[i] for i in c}
        if len(ss) > 1:
            leak += 1
    return {
        "images_in_clusters_ge2": in_multi,
        "num_clusters_ge2": len(multi),
        "top5_cluster_sizes": sizes[:5],
        "clusters_cross_split": leak,
    }


def count_cross_matches(
    ha: np.ndarray, hb: np.ndarray, threshold: int, max_examples: int = 10
) -> tuple[int, list[tuple[int, int]]]:
    """Count images in A (each index) with >=1 match in B."""
    if len(ha) == 0 or len(hb) == 0:
        return 0, []
    matched_a = np.zeros(len(ha), dtype=bool)
    pairs: list[tuple[int, int]] = []
    block = 512
    hb = hb.astype(np.uint64)
    for i0 in range(0, len(ha), block):
        i1 = min(len(ha), i0 + block)
        hi = ha[i0:i1].astype(np.uint64)
        for j0 in range(0, len(hb), block):
            j1 = min(len(hb), j0 + block)
            hj = hb[j0:j1]
            xor = hi.reshape(-1, 1) ^ hj.reshape(1, -1)
            dist = popcount_matrix(xor)
            for ai, bi in zip(*np.where(dist <= threshold), strict=False):
                gi = i0 + int(ai)
                if not matched_a[gi]:
                    matched_a[gi] = True
                if len(pairs) < max_examples:
                    pairs.append((gi, j0 + int(bi)))
    return int(matched_a.sum()), pairs


def laplacian_variance(gray: np.ndarray) -> float:
    g = gray.astype(np.float64)
    lap = (
        -4 * g
        + np.roll(g, 1, 0)
        + np.roll(g, -1, 0)
        + np.roll(g, 1, 1)
        + np.roll(g, -1, 1)
    )
    return float(lap.var())


def noise_estimate(gray: np.ndarray) -> float:
    g = gray.astype(np.float64)
    blur = (
        g
        + np.roll(g, 1, 0)
        + np.roll(g, -1, 0)
        + np.roll(g, 1, 1)
        + np.roll(g, -1, 1)
    ) / 5.0
    return float(np.median(np.abs(g - blur)))


def d3_group(name: str) -> str:
    low = name.lower()
    if low.startswith("sceneadd"):
        return "sceneadd"
    if low.startswith("scene"):
        return "scene"
    if low.startswith("image-"):
        return "image-"
    return "other"


def make_contact_sheet(paths: list[Path], out_path: Path, rng: np.random.Generator) -> None:
    if not paths:
        return
    pick = paths if len(paths) <= 25 else list(rng.choice(paths, size=25, replace=False))
    cell = 256
    sheet = Image.new("RGB", (cell * 5, cell * 5), (32, 32, 32))
    for i, p in enumerate(pick[:25]):
        try:
            im = Image.open(p).convert("RGB")
            im.thumbnail((cell, cell), Image.Resampling.LANCZOS)
            ox = (i % 5) * cell + (cell - im.width) // 2
            oy = (i // 5) * cell + (cell - im.height) // 2
            sheet.paste(im, (ox, oy))
        except OSError:
            continue
    sheet.save(out_path, quality=85)


def infer_d4_classes(ds_path: Path) -> tuple[int, list[str]]:
    max_id = -1
    for lf in ds_path.rglob("labels/*.txt"):
        if not lf.is_file():
            continue
        for line in lf.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split()
            if len(parts) >= 1:
                try:
                    max_id = max(max_id, int(float(parts[0])))
                except ValueError:
                    pass
    nc = max_id + 1 if max_id >= 0 else 0
    config = yaml.safe_load((ROOT / "config/experiment.yaml").read_text())
    names = config["datasets"]["sources"]["xiong_zenodo_crack_dent"].get("source_label_names")
    if not names or len(names) != nc:
        raise ValueError("D4 class mapping missing or inconsistent with observed IDs")
    return nc, names


def audit_one_dataset(did: str, ds_path: Path) -> dict[str, Any]:
    yaml_data = read_data_yaml(ds_path)
    readme = parse_readme(ds_path)
    names: list[str] = list(yaml_data.get("names") or [])
    nc = int(yaml_data.get("nc") or len(names) or 0)
    if did == "D4" and nc == 0:
        nc, names = infer_d4_classes(ds_path)

    inv: dict[str, Any] = {}
    class_boxes: dict[str, Counter] = {s: Counter() for s in SPLITS}
    class_images: dict[str, Counter] = {s: Counter() for s in SPLITS}
    label_stats_total: Counter = Counter()
    sizes: Counter = Counter()
    aspects: list[float] = []
    exts: Counter = Counter()
    area_by_class: dict[int, list[float]] = defaultdict(list)
    boxes_per_image: dict[str, list[int]] = {s: [] for s in SPLITS}
    cooccur: dict[str, Counter] = {s: Counter() for s in SPLITS}
    distinct_class_counts: dict[str, Counter] = {s: Counter() for s in SPLITS}

    records: list[ImageRec] = []
    md5_groups: dict[str, list[str]] = defaultdict(list)
    rf_bases: Counter = Counter()

    for split in SPLITS:
        img_dir = ds_path / split / "images"
        if not img_dir.is_dir():
            inv[split] = {"missing_images_dir": True}
            continue
        images = sorted(img_dir.rglob("*"))
        images = [p for p in images if p.suffix.lower() in IMG_EXT]
        label_files = sorted((ds_path / split / "labels").rglob("*.txt")) if (ds_path / split / "labels").is_dir() else []
        label_stems = {p.stem for p in label_files}
        img_stems = {stem_for_label(p) for p in images}

        empty_labels = orphan_labels = corrupt = 0
        orphan_images = 0

        for lf in label_files:
            raw = lf.read_text(encoding="utf-8", errors="replace").strip()
            if not raw:
                empty_labels += 1
            if lf.stem not in img_stems:
                orphan_labels += 1

        for img in images:
            st = stem_for_label(img)
            lp = label_path_for_image(img)
            if st not in label_stems:
                orphan_images += 1
            rec = ImageRec(did, split, img, str(img.relative_to(ds_path)), ext=img.suffix.lower())
            try:
                with Image.open(img) as im:
                    im.load()
                    rec.width, rec.height = im.size
                    rgb = im.convert("RGB")
                    arr = np.asarray(rgb)
                    rec.grayscale = bool(
                        np.all(arr[:, :, 0] == arr[:, :, 1]) and np.all(arr[:, :, 1] == arr[:, :, 2])
                    )
                    rec.phash = phash_int(imagehash.phash(rgb))
                    raw = img.read_bytes()
                    rec.md5 = hashlib.md5(raw).hexdigest()
            except OSError:
                rec.corrupt = True
                corrupt += 1
                records.append(rec)
                continue

            sizes[(rec.width, rec.height)] += 1
            if rec.height:
                aspects.append(rec.width / rec.height)
            exts[rec.ext] += 1
            md5_groups[rec.md5].append(rec.rel)
            rb = roboflow_base(img.name)
            if rb:
                rf_bases[rb] += 1

            boxes, lstats = parse_yolo_label(lp, nc if nc else 1000)
            label_stats_total.update(lstats)
            boxes_per_image[split].append(len(boxes))
            classes_in_img = set()
            for cid, cx, cy, w, h in boxes:
                if nc and cid >= nc:
                    continue
                cname = names[cid] if cid < len(names) else str(cid)
                class_boxes[split][cname] += 1
                classes_in_img.add(cname)
                area_by_class[cid].append(w * h)
            for cname in classes_in_img:
                class_images[split][cname] += 1
            k = len(classes_in_img)
            bucket = "3+" if k >= 3 else str(k)
            distinct_class_counts[split][bucket] += 1
            if classes_in_img:
                cooccur[split]["+".join(sorted(classes_in_img))] += 1
            records.append(rec)

        inv[split] = {
            "images": len(images),
            "labels": len(label_files),
            "empty_labels": empty_labels,
            "orphan_images": orphan_images,
            "orphan_labels": orphan_labels,
            "corrupt_images": corrupt,
        }

    rf_multi = sum(1 for c in rf_bases.values() if c > 1)
    rf_total = sum(rf_bases.values())
    gray_frac = (
        sum(1 for r in records if not r.corrupt and r.grayscale) / max(1, sum(1 for r in records if not r.corrupt))
    )
    md5_dup_files = sum(len(v) - 1 for v in md5_groups.values() if len(v) > 1)
    md5_dup_groups = sum(1 for v in md5_groups.values() if len(v) > 1)

    readme_text = (
        (ds_path / readme["file"]).read_text(encoding="utf-8", errors="replace") if readme.get("file") else ""
    )
    readme_no_aug = bool(re.search(r"No image augmentation", readme_text, re.I))
    readme_yes_aug = bool(re.search(r"augmentation was applied", readme_text, re.I))
    looks_aug = (
        readme_yes_aug
        or (rf_multi > 0 and not readme_no_aug)
        or (".aug." in ds_path.name.lower() or ds_path.name.lower().endswith("-aug.yolo26"))
    )
    if readme_no_aug and not readme_yes_aug and rf_multi == 0:
        looks_aug = False

    aug_verdict = "looks augmented" if looks_aug else "looks un-augmented"
    aug_evidence = []
    if rf_multi:
        aug_evidence.append(f"rf_base_groups_gt1={rf_multi}")
    if gray_frac > 0.05:
        aug_evidence.append(f"grayscale_frac={gray_frac:.3f}")
    if md5_dup_groups:
        aug_evidence.append(f"md5_dup_groups={md5_dup_groups}")
    if "augmentation was applied" in readme_text.lower():
        aug_evidence.append("readme_aug=yes")
    if re.search(r"No image augmentation", readme_text, re.I):
        aug_evidence.append("readme_aug=none")

    # phash clusters per threshold
    valid_recs = [r for r in records if r.phash is not None]
    hashes = np.array([r.phash for r in valid_recs], dtype=np.uint64)
    split_list = [r.split for r in valid_recs]
    phash_report = {}
    for th in PHASH_THRESHOLDS:
        clusters = union_find_clusters(hashes, th)
        phash_report[str(th)] = cluster_stats(clusters, split_list)

    # aspect histogram summary
    asp = np.array(aspects) if aspects else np.array([1.0])
    aspect_summary = {
        "min": float(asp.min()),
        "p25": float(np.percentile(asp, 25)),
        "median": float(np.median(asp)),
        "p75": float(np.percentile(asp, 75)),
        "max": float(asp.max()),
    }

    area_pct: dict[str, dict[str, float]] = {}
    for cid, areas in area_by_class.items():
        cname = names[cid] if cid < len(names) else str(cid)
        a = np.array(areas)
        area_pct[cname] = {
            "p5": float(np.percentile(a, 5)),
            "p25": float(np.percentile(a, 25)),
            "p50": float(np.percentile(a, 50)),
            "p75": float(np.percentile(a, 75)),
            "p95": float(np.percentile(a, 95)),
        }

    bpi = {
        s: {"mean": float(np.mean(v)) if v else 0.0, "max": int(max(v)) if v else 0}
        for s, v in boxes_per_image.items()
    }

    top_sizes = sizes.most_common(5)
    co_top = {s: cooccur[s].most_common(5) for s in SPLITS}

    return {
        "id": did,
        "path": str(ds_path),
        "yaml": {"nc": nc, "names": names, "roboflow": yaml_data.get("roboflow")},
        "readme": readme,
        "inventory": inv,
        "class_boxes": {s: dict(class_boxes[s]) for s in SPLITS},
        "class_images": {s: dict(class_images[s]) for s in SPLITS},
        "top_sizes": [(list(k), v) for k, v in top_sizes],
        "aspect_summary": aspect_summary,
        "extensions": dict(exts),
        "label_issues": dict(label_stats_total),
        "box_area_percentiles_by_class": area_pct,
        "boxes_per_image": bpi,
        "augmentation_fingerprint": {
            "verdict": aug_verdict,
            "evidence": aug_evidence,
            "rf_base_groups_gt1": rf_multi,
            "rf_named_images": rf_total,
            "grayscale_fraction": gray_frac,
            "md5_duplicate_files": md5_dup_files,
            "md5_duplicate_groups": md5_dup_groups,
        },
        "phash_within": phash_report,
        "distinct_class_buckets": {s: dict(distinct_class_counts[s]) for s in SPLITS},
        "top_cooccurrence": co_top,
        "records": records,
        "total_images": len(records),
    }


def mapped_boxes_for_image(rec: ImageRec, ds_result: dict[str, Any]) -> Counter:
    names: list[str] = ds_result["yaml"]["names"]
    lp = label_path_for_image(rec.path)
    nc = ds_result["yaml"]["nc"]
    boxes, _ = parse_yolo_label(lp, nc if nc else 1000)
    ctr: Counter = Counter()
    for cid, *_ in boxes:
        if cid >= len(names):
            continue
        mapped = MAP_POOL.get(names[cid])
        if mapped:
            ctr[mapped] += 1
    return ctr


def dedup_phash5(records: list[ImageRec]) -> list[ImageRec]:
    valid = [r for r in records if r.phash is not None]
    if not valid:
        return []
    hashes = np.array([r.phash for r in valid], dtype=np.uint64)
    clusters = union_find_clusters(hashes, 5)
    keep_idx: set[int] = set()
    for cl in clusters:
        members = sorted(cl, key=lambda i: valid[i].rel)
        keep_idx.add(members[0])
    return [valid[i] for i in sorted(keep_idx)]


def planning_table(records: list[ImageRec], ds_results: dict[str, dict[str, Any]], tag: str) -> dict[str, Any]:
    deduped = dedup_phash5(records)
    class_totals: Counter = Counter()
    for r in deduped:
        did = r.dataset_id
        class_totals.update(mapped_boxes_for_image(r, ds_results[did]))
    rows: dict[str, dict[str, Any]] = {}
    for cls, total in sorted(class_totals.items()):
        rows[cls] = {}
        for frac in FRACTIONS:
            est = total * 0.70 * frac
            status = "OK"
            if est < MIN_FAIL:
                status = "FAIL"
            elif est < MIN_WARN:
                status = "WARN"
            rows[cls][str(frac)] = {"est_train_boxes": round(est, 2), "status": status}
    return {
        "tag": tag,
        "images_after_phash5_dedup": len(deduped),
        "class_box_totals_after_dedup": dict(class_totals),
        "plan": rows,
    }


def color_features(path: Path) -> np.ndarray | None:
    try:
        with Image.open(path) as im:
            im = im.convert("RGB").resize((64, 64), Image.Resampling.BILINEAR)
            arr = np.asarray(im, dtype=np.float32) / 255.0
    except OSError:
        return None
    feats: list[float] = []
    for ch in range(3):
        c = arr[:, :, ch].ravel()
        hist, _ = np.histogram(c, bins=16, range=(0.0, 1.0))
        feats.extend((hist / hist.sum()).tolist())
        feats.append(float(c.mean()))
        feats.append(float(c.std()))
    return np.array(feats, dtype=np.float64)


def cv_predict_dataset(records: list[ImageRec], labels: list[str], mask: np.ndarray | None) -> dict[str, Any]:
    idxs = [i for i, r in enumerate(records) if (mask is None or mask[i])]
    X_list, y_list = [], []
    for i in idxs:
        feat = color_features(records[i].path)
        if feat is None:
            continue
        X_list.append(feat)
        y_list.append(labels[i])
    if len(X_list) < 20 or len(set(y_list)) < 2:
        return {"error": "insufficient samples", "n": len(X_list)}
    X = np.vstack(X_list)
    y = np.array(y_list)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=SEED, multi_class="multinomial")),
        ]
    )
    pred = cross_val_predict(pipe, X, y, cv=cv)
    acc = float(accuracy_score(y, pred))
    labs = sorted(set(y_list))
    cm = confusion_matrix(y, pred, labels=labs).tolist()
    return {"accuracy": acc, "confusion_matrix": cm, "labels_order": labs, "n": len(y_list)}


def build_report_md(data: dict[str, Any], runtime_s: float, versions: dict[str, str]) -> str:
    lines: list[str] = []
    lines.append("# Dataset audit report")
    lines.append("> Geometry inventory only, not admission certification. D1 policies and visual findings: docs/dataset_audit.md. Planning estimates precede admission exclusions.")
    lines.append("")
    lines.append("## TOP FINDINGS")
    for b in data["top_findings"][:10]:
        lines.append(f"- {b}")
    lines.append("")

    for sec in data["sections_md"]:
        lines.extend(sec)
        lines.append("")

    lines.append(f"## Runtime")
    lines.append(f"- elapsed_s: {runtime_s:.1f}")
    for k, v in versions.items():
        lines.append(f"- {k}: {v}")
    text = "\n".join(lines)
    if len(text.splitlines()) > 200:
        text = "\n".join(text.splitlines()[:199] + ["- (truncated to 200 lines)"])
    return text


def main() -> int:
    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    paths = resolve_datasets()
    missing = [d for d in DATASET_SPECS if d not in paths]
    results: dict[str, Any] = {}
    ds_results: dict[str, dict[str, Any]] = {}

    for did in ("D1", "D2", "D3", "D4"):
        if did not in paths:
            continue
        print(f"Auditing {did} ...", flush=True)
        ds_results[did] = audit_one_dataset(did, paths[did])

    # D3 extras
    d3_extra: dict[str, Any] = {}
    if "D3" in ds_results:
        rng = np.random.default_rng(SEED)
        groups: dict[str, list[ImageRec]] = defaultdict(list)
        for r in ds_results["D3"]["records"]:
            groups[d3_group(r.path.name)].append(r)
        gsum: dict[str, Any] = {}
        for gname, recs in groups.items():
            sharp, noise = [], []
            box_ctr: Counter = Counter()
            img_paths = [r.path for r in recs if not r.corrupt]
            names = ds_results["D3"]["yaml"]["names"]
            nc = ds_results["D3"]["yaml"]["nc"]
            for r in recs:
                if r.corrupt:
                    continue
                boxes, _ = parse_yolo_label(label_path_for_image(r.path), nc)
                for cid, *_ in boxes:
                    cname = names[cid] if cid < len(names) else str(cid)
                    box_ctr[cname] += 1
                try:
                    with Image.open(r.path) as im:
                        gray = np.asarray(im.convert("L"), dtype=np.float64)
                    sharp.append(laplacian_variance(gray))
                    noise.append(noise_estimate(gray))
                except OSError:
                    pass
            out_img = OUT / f"d3_contact_{gname.replace('-', '_')}.jpg"
            make_contact_sheet(img_paths, out_img, rng)
            gsum[gname] = {
                "images": len(recs),
                "box_counts": dict(box_ctr),
                "mean_sharpness": float(np.mean(sharp)) if sharp else None,
                "mean_noise": float(np.mean(noise)) if noise else None,
                "contact_sheet": str(out_img.relative_to(ROOT)),
            }
        rust_boxes = sum(ds_results["D3"]["class_boxes"][s].get("rust", 0) for s in SPLITS)
        rust_images = sum(ds_results["D3"]["class_images"][s].get("rust", 0) for s in SPLITS)
        rust_areas: list[float] = []
        names = ds_results["D3"]["yaml"]["names"]
        rust_id = names.index("rust") if "rust" in names else -1
        if rust_id >= 0:
            for r in ds_results["D3"]["records"]:
                boxes, _ = parse_yolo_label(label_path_for_image(r.path), ds_results["D3"]["yaml"]["nc"])
                for cid, cx, cy, w, h in boxes:
                    if cid == rust_id:
                        rust_areas.append(w * h)
        rust_co = Counter()
        for s in SPLITS:
            for combo, cnt in ds_results["D3"]["top_cooccurrence"].get(s, []):
                if "rust" in combo.split("+"):
                    rust_co[combo] += cnt
        d3_extra = {
            "filename_groups": gsum,
            "rust": {
                "boxes": rust_boxes,
                "images": rust_images,
                "cooccurrence_top": rust_co.most_common(10),
                "box_area_pct": {
                    k: float(v)
                    for k, v in zip(
                        ("p5", "p25", "p50", "p75", "p95"),
                        np.percentile(rust_areas, [5, 25, 50, 75, 95]) if rust_areas else [0] * 5,
                        strict=False,
                    )
                },
            },
        }

    # Cross-dataset
    cross: dict[str, Any] = {"pairs": {}, "matrix_t5": {}, "subset_notes": []}
    cross_recs: dict[str, list[ImageRec]] = {
        did: [r for r in ds_results[did]["records"] if r.phash is not None] for did in CROSS_IDS if did in ds_results
    }
    hashes_cross = {did: np.array([r.phash for r in cross_recs[did]], dtype=np.uint64) for did in cross_recs}

    for a in CROSS_IDS:
        if a not in cross_recs:
            continue
        cross["matrix_t5"][a] = {}
        for b in CROSS_IDS:
            if b not in cross_recs:
                continue
            cross["matrix_t5"][a][b] = 0
    for i, a in enumerate(CROSS_IDS):
        if a not in cross_recs:
            continue
        for b in CROSS_IDS[i + 1 :]:
            if b not in cross_recs:
                continue
            pair_key = f"{a}-{b}"
            pair_data: dict[str, Any] = {}
            for th in CROSS_THRESHOLDS:
                c_ab, ex_ab = count_cross_matches(hashes_cross[a], hashes_cross[b], th)
                c_ba, ex_ba = count_cross_matches(hashes_cross[b], hashes_cross[a], th)
                ex = []
                for ia, ib in ex_ab[:10]:
                    ex.append((cross_recs[a][ia].rel, cross_recs[b][ib].rel))
                pair_data[str(th)] = {
                    f"{a}_in_{b}": c_ab,
                    f"{b}_in_{a}": c_ba,
                    "examples": ex,
                }
                if th == 5:
                    cross["matrix_t5"][a][b] = c_ab
                    cross["matrix_t5"][b][a] = c_ba
            cross["pairs"][pair_key] = pair_data
            na, nb = len(cross_recs[a]), len(cross_recs[b])
            c_ba_t5 = pair_data["5"][f"{b}_in_{a}"]
            frac_a = c_ab / na if na else 0
            frac_b = c_ba_t5 / nb if nb else 0
            if frac_a > 0.9 and na < nb:
                cross["subset_notes"].append(f"{a} is ~subset of {b} (t5 match {frac_a:.2%} of {a})")
            elif frac_b > 0.9 and nb < na:
                cross["subset_notes"].append(f"{b} is ~subset of {a} (t5 match {frac_b:.2%} of {b})")

    # Planning
    plan_d1 = planning_table(cross_recs.get("D1", []), ds_results, "D1_only") if "D1" in ds_results else {}
    pooled_recs = []
    for did in ("D1", "D2", "D3", "D4"):
        if did in cross_recs:
            for r in cross_recs[did]:
                pooled_recs.append(r)
    plan_pool = planning_table(pooled_recs, ds_results, "D1+D2+D3+D4")

    # Source predictability
    all_pool = []
    ds_labels = []
    defect_mask = []
    for did in ("D1", "D2", "D3", "D4"):
        if did not in ds_results:
            continue
        for r in ds_results[did]["records"]:
            if r.corrupt:
                continue
            all_pool.append(r)
            ds_labels.append(did)
            ctr = mapped_boxes_for_image(r, ds_results[did])
            defect_mask.append(sum(ctr.values()) > 0)
    defect_mask_arr = np.array(defect_mask, dtype=bool)
    src_all = cv_predict_dataset(all_pool, ds_labels, None)
    src_def = cv_predict_dataset(all_pool, ds_labels, defect_mask_arr)

    # Top findings
    findings: list[str] = []
    if "D1" in ds_results:
        n5 = ds_results["D1"]["total_images"]
        if n5 != DATASET_SPECS["D1"]["expect_images"]:
            findings.append(
                f"D1 mismatch: expected v2 stock ~{DATASET_SPECS['D1']['expect_images']} imgs, found {n5} at `{paths['D1'].name}` (README aug export)."
            )
        ph5 = ds_results["D1"]["phash_within"]["5"]
        if ph5["clusters_cross_split"] > 0:
            findings.append(
                f"D1 split leakage: {ph5['clusters_cross_split']} phash-5 clusters span train/valid/test."
            )
    if missing:
        findings.append(f"Missing dataset folders: {', '.join(missing)}")
    if cross.get("subset_notes"):
        findings.extend(cross["subset_notes"][:3])
    for did, dr in ds_results.items():
        af = dr["augmentation_fingerprint"]
        if did in CROSS_IDS and af["verdict"] == "looks augmented":
            findings.append(f"{did} {af['verdict']}: {', '.join(af['evidence'][:3])}")
    for did, dr in ds_results.items():
        for sp, iv in dr["inventory"].items():
            if isinstance(iv, dict) and iv.get("orphan_images", 0) + iv.get("orphan_labels", 0) > 0:
                findings.append(f"{did} {sp}: orphan images={iv.get('orphan_images')} labels={iv.get('orphan_labels')}")
    if src_all.get("accuracy", 0) > 0.5:
        findings.append(f"Pooled source predictability accuracy={src_all.get('accuracy', 0):.3f} (domain shift risk).")
    for tag, plan in [("D1", plan_d1), ("pool", plan_pool)]:
        if not plan:
            continue
        fails = [
            f"{cls}@{fr}"
            for cls, frs in plan.get("plan", {}).items()
            for fr, cell in frs.items()
            if cell["status"] == "FAIL"
        ]
        if fails:
            findings.append(f"Planning FAIL ({tag}): {', '.join(fails[:6])}")

    # Serialize (strip records from json — too large; keep summaries)
    json_out: dict[str, Any] = {
        "datasets": {},
        "cross_dataset": cross,
        "d3_extra": d3_extra,
        "planning": {"D1": plan_d1, "pooled": plan_pool},
        "source_predictability": {"all_images": src_all, "defect_images_only": src_def},
        "missing_ids": missing,
        "paths": {k: str(v) for k, v in paths.items()},
    }
    for did, dr in ds_results.items():
        copy = {k: v for k, v in dr.items() if k != "records"}
        json_out["datasets"][did] = copy

    # Markdown sections (compact)
    sections: list[list[str]] = []

    for did, dr in ds_results.items():
        sec = [f"## {did} — items 1–7"]
        spec = DATASET_SPECS[did]
        sec.append(f"Path: `{Path(dr['path']).name}` | expect imgs: {spec.get('expect_images')} | actual: {dr['total_images']}")
        y = dr["yaml"]
        sec.append(f"Classes (nc={y['nc']}): {', '.join(y['names'])}")
        rd = dr["readme"]
        if rd:
            sec.append(f"README: {rd.get('file','')} | {rd.get('version_line','')} | aug: {rd.get('augmentation','n/a')}")
        sec.append("| split | imgs | labels | empty_lbl | orphan_img | orphan_lbl | corrupt |")
        sec.append("|---|---:|---:|---:|---:|---:|---:|")
        for sp in SPLITS:
            iv = dr["inventory"].get(sp, {})
            if iv.get("missing_images_dir"):
                sec.append(f"| {sp} | - | - | - | - | - | - |")
            else:
                sec.append(
                    f"| {sp} | {iv.get('images',0)} | {iv.get('labels',0)} | {iv.get('empty_labels',0)} | "
                    f"{iv.get('orphan_images',0)} | {iv.get('orphan_labels',0)} | {iv.get('corrupt_images',0)} |"
                )
        sec.append(f"Label issues (total): {dr['label_issues']}")
        af = dr["augmentation_fingerprint"]
        sec.append(f"**Aug verdict:** {af['verdict']} ({'; '.join(af['evidence'])})")
        for th in PHASH_THRESHOLDS:
            p = dr["phash_within"][str(th)]
            sec.append(
                f"phash<={th}: in_clusters={p['images_in_clusters_ge2']} cross_split_clusters={p['clusters_cross_split']} top5={p['top5_cluster_sizes']}"
            )
        sections.append(sec)

    sections.append(["## 8 — Cross-dataset (D1,D2,D3,D4)"])
    sections[-1].append("phash t=5 match counts (row has match in col):")
    hdr = "| | " + " | ".join(CROSS_IDS) + " |"
    sections[-1].append(hdr)
    sections[-1].append("|" + "---|" * (len(CROSS_IDS) + 1))
    for a in CROSS_IDS:
        row = [a]
        for b in CROSS_IDS:
            row.append(str(cross["matrix_t5"].get(a, {}).get(b, "-")))
        sections[-1].append("| " + " | ".join(row) + " |")
    for pk, pd in list(cross["pairs"].items())[:6]:
        t5 = pd.get("5", {})
        if t5.get("examples"):
            sections[-1].append(f"Examples {pk} (t=5): " + "; ".join(f"{a}<->{b}" for a, b in t5["examples"][:3]))
    for note in cross.get("subset_notes", []):
        sections[-1].append(f"- {note}")

    if d3_extra:
        sec9 = ["## 9–10 — D3 groups & rust"]
        for g, info in d3_extra.get("filename_groups", {}).items():
            sec9.append(
                f"- {g}: n={info['images']} boxes={info['box_counts']} sharp={info['mean_sharpness']:.1f} noise={info['mean_noise']:.3f} sheet={info['contact_sheet']}"
                if info["mean_sharpness"] is not None
                else f"- {g}: n={info['images']}"
            )
        r = d3_extra["rust"]
        sec9.append(f"Rust: boxes={r['boxes']} images={r['images']} area_pct={r['box_area_pct']}")
        sections.append(sec9)

    sec11 = ["## 11 — Planning (post phash5 dedup, train=0.70×fraction)"]
    for label, plan in [("D1", plan_d1), ("Pooled", plan_pool)]:
        if not plan:
            continue
        sec11.append(f"**{label}** dedup imgs={plan['images_after_phash5_dedup']} totals={plan['class_box_totals_after_dedup']}")
        sec11.append("| class | 1.0 | 0.5 | 0.25 | 0.1 | 0.05 |")
        sec11.append("|---|---|---|---|---|---|")
        for cls, frs in plan.get("plan", {}).items():
            cells = [cls]
            for fr in FRACTIONS:
                c = frs[str(fr)]
                cells.append(f"{c['est_train_boxes']:.0f}{c['status'][:1]}")
            sec11.append("| " + " | ".join(cells) + " |")
    sections.append(sec11)

    sec12 = ["## 12 — Source predictability (5-fold LR, 64² hist+moments)"]
    for name, block in [("all", src_all), ("defect_only", src_def)]:
        if block.get("error"):
            sec12.append(f"{name}: {block['error']} n={block.get('n')}")
        else:
            sec12.append(f"{name}: acc={block['accuracy']:.4f} n={block['n']} labels={block['labels_order']}")
            sec12.append(f"CM: {block['confusion_matrix']}")
    sections.append(sec12)

    versions = {
        "python": sys.version.split()[0],
        "Pillow": Image.__version__,
        "numpy": np.__version__,
        "imagehash": getattr(imagehash, "__version__", "unknown"),
        "sklearn": __import__("sklearn").__version__,
    }
    runtime = time.perf_counter() - t0
    md = build_report_md({"top_findings": findings, "sections_md": sections}, runtime, versions)
    (OUT / "REPORT.md").write_text(md, encoding="utf-8")
    json_out["runtime_s"] = runtime
    json_out["versions"] = versions
    json_out["generated_utc"] = datetime.now(timezone.utc).isoformat()
    (OUT / "report.json").write_text(json.dumps(json_out, indent=2), encoding="utf-8")
    print(f"Done in {runtime:.1f}s -> {OUT / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
