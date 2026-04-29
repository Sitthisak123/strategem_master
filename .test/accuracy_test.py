import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.strategem_detection import (
    detect_icon_boxes_new,
    detect_icon_boxes_old,
    detect_strategems_new,
    detect_strategems_old,
    get_hud_region,
    load_detection_assets,
)


DEFAULT_MANIFEST = PROJECT_ROOT / ".test" / "sample" / "samples.json"


def normalize_codes(codes):
    return [str(code).strip() for code in codes if str(code).strip()]


def compare_codes(expected, actual):
    expected_counter = Counter(expected)
    actual_counter = Counter(actual)
    correct = sum(min(expected_counter[code], actual_counter[code]) for code in expected_counter)
    positional = sum(
        1 for index, code in enumerate(expected)
        if index < len(actual) and actual[index] == code
    )

    precision = correct / len(actual) if actual else (1.0 if not expected else 0.0)
    recall = correct / len(expected) if expected else (1.0 if not actual else 0.0)
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0

    return {
        "exact": expected == actual,
        "correct": correct,
        "positional": positional,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "missing": list((expected_counter - actual_counter).elements()),
        "unexpected": list((actual_counter - expected_counter).elements()),
    }


def load_manifest(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("samples", [])


def format_percent(value):
    return f"{value * 100:.1f}%"


def run_detector(name, frame, assets, args, max_results):
    if name == "old":
        return detect_strategems_old(
            frame,
            assets,
            match_threshold=args.threshold,
            include_defaults=args.include_defaults,
            max_results=max_results,
            skip_first=args.skip_first,
            verbose=args.verbose,
        )

    return detect_strategems_new(
        frame,
        assets,
        match_threshold=args.threshold,
        include_defaults=args.include_defaults,
        max_results=max_results,
        skip_first=args.skip_first,
        phash_candidates=args.phash_candidates,
        verbose=args.verbose,
    )


def get_detector_boxes(name, frame):
    if name == "old":
        hud = frame[30:800, 30:600]
        region = {"left": 30, "top": 30}
        return detect_icon_boxes_old(hud), region

    hud, hud_region = get_hud_region(frame)
    region = {"left": hud_region.left, "top": hud_region.top}
    return detect_icon_boxes_new(hud, hud_region.scale), region


def save_debug_overlay(path, frame, boxes, region):
    output = frame.copy()
    left = region["left"]
    top = region["top"]

    for index, (y, x, w, h, _icon_region) in enumerate(boxes, start=1):
        x1 = left + x
        y1 = top + y
        x2 = x1 + w
        y2 = y1 + h
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            output,
            str(index),
            (x1 + 4, y1 + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), output)


def warn_missing_assets(expected, assets):
    missing_metadata = [code for code in expected if code not in assets.strategems_by_code]
    missing_templates = [code for code in expected if code not in assets.templates]

    if missing_metadata:
        print(f"[WARN] Expected codes missing from CSV metadata: {missing_metadata}")
    if missing_templates:
        print(f"[WARN] Expected codes missing template images in img/: {missing_templates}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare old and new strategem detection accuracy against sample images."
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--skip-first", type=int, default=2)
    parser.add_argument("--phash-candidates", type=int, default=0)
    parser.add_argument("--max-results", type=int, default=None)
    parser.add_argument("--include-defaults", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--report", default=None)
    parser.add_argument("--debug-dir", default=None)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    sample_dir = manifest_path.parent
    samples = [sample for sample in load_manifest(manifest_path) if sample.get("enabled", True)]

    if not samples:
        print(f"No enabled samples found in {manifest_path}")
        print("Add images to .test/sample and add entries to samples.json.")
        return 0

    assets = load_detection_assets()
    report = {
        "manifest": str(manifest_path),
        "threshold": args.threshold,
        "skip_first": args.skip_first,
        "include_defaults": args.include_defaults,
        "samples": [],
        "summary": {},
    }
    totals = {
        "old": {"exact": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0},
        "new": {"exact": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0},
    }
    processed = 0

    for sample in samples:
        image_path = sample_dir / sample["image"]
        expected = normalize_codes(sample.get("expected_codes", []))
        frame = cv2.imread(str(image_path))

        if frame is None:
            print(f"[SKIP] Could not read image: {image_path}")
            continue

        processed += 1
        print("")
        print(f"Sample: {sample.get('name', sample['image'])}")
        print(f"Image: {image_path}")
        print(f"Expected: {expected}")
        warn_missing_assets(expected, assets)

        max_results = args.max_results
        if max_results is None and expected:
            max_results = len(expected)

        sample_report = {
            "name": sample.get("name", sample["image"]),
            "image": str(image_path),
            "expected_codes": expected,
            "detectors": {},
        }

        for detector_name in ("old", "new"):
            boxes, region = get_detector_boxes(detector_name, frame)
            if args.verbose:
                print(f"[{detector_name}] icon boxes: {len(boxes)}")

            if args.debug_dir:
                debug_path = (
                    Path(args.debug_dir)
                    / f"{Path(sample['image']).stem}_{detector_name}_boxes.png"
                )
                save_debug_overlay(debug_path, frame, boxes, region)

            detected_entries = run_detector(detector_name, frame, assets, args, max_results)
            actual = [entry["code"] for entry in detected_entries]
            metrics = compare_codes(expected, actual)

            totals[detector_name]["exact"] += int(metrics["exact"])
            totals[detector_name]["precision"] += metrics["precision"]
            totals[detector_name]["recall"] += metrics["recall"]
            totals[detector_name]["f1"] += metrics["f1"]

            sample_report["detectors"][detector_name] = {
                "actual_codes": actual,
                "matches": detected_entries,
                "metrics": metrics,
            }

            print(
                f"{detector_name:>3}: {actual} | "
                f"exact={metrics['exact']} "
                f"precision={format_percent(metrics['precision'])} "
                f"recall={format_percent(metrics['recall'])} "
                f"f1={format_percent(metrics['f1'])}"
            )
            if metrics["missing"] or metrics["unexpected"]:
                print(
                    f"     missing={metrics['missing']} "
                    f"unexpected={metrics['unexpected']}"
                )

        report["samples"].append(sample_report)

    if processed == 0:
        print("No readable sample images were processed.")
        return 1

    print("")
    print("Summary")
    for detector_name in ("old", "new"):
        summary = {
            "exact": totals[detector_name]["exact"],
            "sample_count": processed,
            "exact_rate": totals[detector_name]["exact"] / processed,
            "precision": totals[detector_name]["precision"] / processed,
            "recall": totals[detector_name]["recall"] / processed,
            "f1": totals[detector_name]["f1"] / processed,
        }
        report["summary"][detector_name] = summary
        print(
            f"{detector_name:>3}: "
            f"exact={summary['exact']}/{processed} "
            f"precision={format_percent(summary['precision'])} "
            f"recall={format_percent(summary['recall'])} "
            f"f1={format_percent(summary['f1'])}"
        )

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Report saved: {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
