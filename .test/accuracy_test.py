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
    detect_strategems_hybrid,
    detect_strategems_new,
    detect_strategems_old,
    get_hud_region,
    is_default_strategem,
    load_detection_assets,
)


DEFAULT_MANIFEST = PROJECT_ROOT / ".test" / "sample" / "samples.json"
GROUP_SAMPLE_DIR = PROJECT_ROOT / "src" / "img" / "group"
GROUP_MANIFEST = GROUP_SAMPLE_DIR / "icon_detection_samples.json"
DETECTOR_NAMES = ("old", "new", "hybrid")
LEGACY_DETECTOR_NAMES = ("old", "new")


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
        "expected_count": len(expected),
        "actual_count": len(actual),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "missing": list((expected_counter - actual_counter).elements()),
        "unexpected": list((actual_counter - expected_counter).elements()),
    }


def load_manifest(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def get_sample_dir(manifest_path, manifest_data, sample_dir_arg):
    if sample_dir_arg:
        sample_dir = Path(sample_dir_arg)
    else:
        sample_dir = Path(manifest_data.get("sample_dir", manifest_path.parent))

    if sample_dir.is_absolute():
        return sample_dir
    return (manifest_path.parent / sample_dir).resolve()


def filter_default_codes(codes, assets):
    filtered = []
    for code in codes:
        strategem = assets.strategems_by_code.get(code)
        if strategem and is_default_strategem(strategem):
            continue
        filtered.append(code)
    return filtered


def expected_codes_for_sample(sample, assets, skip_first, include_defaults):
    expected = normalize_codes(sample.get("expected_codes", []))

    if sample.get("apply_skip_first_to_expected", False):
        expected = expected[skip_first:] if len(expected) > skip_first else []

    if not include_defaults:
        expected = filter_default_codes(expected, assets)

    return expected


def format_percent(value):
    return f"{value * 100:.1f}%"


def run_detector(name, frame, assets, args, max_results, skip_first):
    if name == "old":
        return detect_strategems_old(
            frame,
            assets,
            match_threshold=args.threshold,
            include_defaults=args.include_defaults,
            max_results=max_results,
            skip_first=skip_first,
            verbose=args.verbose,
        )

    if name == "hybrid":
        return detect_strategems_hybrid(
            frame,
            assets,
            match_threshold=args.threshold,
            include_defaults=args.include_defaults,
            max_results=max_results,
            skip_first=skip_first,
            phash_candidates=args.phash_candidates,
            hybrid_margin=args.hybrid_margin,
            verbose=args.verbose,
        )

    return detect_strategems_new(
        frame,
        assets,
        match_threshold=args.threshold,
        include_defaults=args.include_defaults,
        max_results=max_results,
        skip_first=skip_first,
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


def detector_names_from_arg(value):
    if value == "all":
        return DETECTOR_NAMES
    if value == "both":
        return LEGACY_DETECTOR_NAMES
    return (value,)


def new_totals():
    return {
        "samples": 0,
        "exact": 0,
        "correct": 0,
        "positional": 0,
        "expected": 0,
        "actual": 0,
        "boxes": 0,
        "used_boxes": 0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }


def micro_stats(correct, actual, expected):
    precision = correct / actual if actual else (1.0 if expected == 0 else 0.0)
    recall = correct / expected if expected else (1.0 if actual == 0 else 0.0)
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return precision, recall, f1


def main():
    parser = argparse.ArgumentParser(
        description="Compare Canny/edge strategem icon detection accuracy against sample images."
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument(
        "--group-samples",
        action="store_true",
        help="Use the Canny edge test manifest in src/img/group.",
    )
    parser.add_argument("--sample-dir", default=None)
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--skip-first", type=int, default=None)
    parser.add_argument("--phash-candidates", type=int, default=15)
    parser.add_argument("--hybrid-margin", type=float, default=0.02)
    parser.add_argument("--max-results", type=int, default=None)
    parser.add_argument(
        "--detectors",
        choices=["old", "new", "hybrid", "both", "all"],
        default="all",
    )
    parser.add_argument("--include-defaults", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--report", default=None)
    parser.add_argument("--debug-dir", default=None)
    args = parser.parse_args()

    if args.group_samples and args.manifest == str(DEFAULT_MANIFEST):
        args.manifest = str(GROUP_MANIFEST)

    manifest_path = Path(args.manifest)
    if args.skip_first is None:
        args.skip_first = 3 if manifest_path.resolve() == GROUP_MANIFEST.resolve() else 2

    manifest_data = load_manifest(manifest_path)
    sample_dir = get_sample_dir(manifest_path, manifest_data, args.sample_dir)
    samples = [sample for sample in manifest_data.get("samples", []) if sample.get("enabled", True)]
    detector_names = detector_names_from_arg(args.detectors)

    if not samples:
        print(f"No enabled samples found in {manifest_path}")
        print("Add images and expected_codes entries to the selected manifest.")
        return 0

    assets = load_detection_assets()
    report = {
        "manifest": str(manifest_path),
        "threshold": args.threshold,
        "skip_first": args.skip_first,
        "include_defaults": args.include_defaults,
        "hybrid_margin": args.hybrid_margin,
        "phash_candidates": args.phash_candidates,
        "detectors": list(detector_names),
        "samples": [],
        "summary": {},
    }
    totals = {detector_name: new_totals() for detector_name in detector_names}
    processed = 0

    for sample in samples:
        image_path = sample_dir / sample["image"]
        sample_skip_first = int(sample.get("skip_first", args.skip_first))
        expected = expected_codes_for_sample(
            sample,
            assets,
            sample_skip_first,
            args.include_defaults,
        )
        frame = cv2.imread(str(image_path))

        if frame is None:
            print(f"[SKIP] Could not read image: {image_path}")
            continue

        processed += 1
        print("")
        print(f"Sample: {sample.get('name', sample['image'])}")
        print(f"Image: {image_path}")
        print(f"Expected: {expected}")
        print(f"Skip first icon boxes: {sample_skip_first}")
        warn_missing_assets(expected, assets)

        max_results = args.max_results
        if max_results is None and expected:
            max_results = len(expected)

        sample_report = {
            "name": sample.get("name", sample["image"]),
            "image": str(image_path),
            "expected_codes": expected,
            "skip_first": sample_skip_first,
            "detectors": {},
        }

        for detector_name in detector_names:
            boxes, region = get_detector_boxes(detector_name, frame)
            used_box_count = max(0, len(boxes) - sample_skip_first)
            if args.verbose:
                print(f"[{detector_name}] icon boxes: {len(boxes)}")

            if args.debug_dir:
                debug_path = (
                    Path(args.debug_dir)
                    / f"{Path(sample['image']).stem}_{detector_name}_boxes.png"
                )
                save_debug_overlay(debug_path, frame, boxes, region)

            detected_entries = run_detector(
                detector_name,
                frame,
                assets,
                args,
                max_results,
                sample_skip_first,
            )
            actual = [entry["code"] for entry in detected_entries]
            metrics = compare_codes(expected, actual)

            totals[detector_name]["exact"] += int(metrics["exact"])
            totals[detector_name]["correct"] += metrics["correct"]
            totals[detector_name]["positional"] += metrics["positional"]
            totals[detector_name]["expected"] += metrics["expected_count"]
            totals[detector_name]["actual"] += metrics["actual_count"]
            totals[detector_name]["boxes"] += len(boxes)
            totals[detector_name]["used_boxes"] += used_box_count
            totals[detector_name]["precision"] += metrics["precision"]
            totals[detector_name]["recall"] += metrics["recall"]
            totals[detector_name]["f1"] += metrics["f1"]
            totals[detector_name]["samples"] += 1

            sample_report["detectors"][detector_name] = {
                "actual_codes": actual,
                "matches": detected_entries,
                "box_count": len(boxes),
                "used_box_count": used_box_count,
                "metrics": metrics,
            }

            print(
                f"{detector_name:>3}: boxes={len(boxes)} used={used_box_count} "
                f"detected={metrics['actual_count']}/{metrics['expected_count']} "
                f"codes={actual} | "
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
    for detector_name in detector_names:
        detector_total = totals[detector_name]
        sample_count = detector_total["samples"]
        micro_precision, micro_recall, micro_f1 = micro_stats(
            detector_total["correct"],
            detector_total["actual"],
            detector_total["expected"],
        )
        summary = {
            "exact": detector_total["exact"],
            "sample_count": sample_count,
            "expected_total": detector_total["expected"],
            "detected_total": detector_total["actual"],
            "correct_total": detector_total["correct"],
            "positional_total": detector_total["positional"],
            "box_total": detector_total["boxes"],
            "used_box_total": detector_total["used_boxes"],
            "exact_rate": detector_total["exact"] / sample_count,
            "avg_precision": detector_total["precision"] / sample_count,
            "avg_recall": detector_total["recall"] / sample_count,
            "avg_f1": detector_total["f1"] / sample_count,
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "micro_f1": micro_f1,
        }
        report["summary"][detector_name] = summary
        print(
            f"{detector_name:>3}: "
            f"exact={summary['exact']}/{sample_count} "
            f"correct={summary['correct_total']}/{summary['expected_total']} "
            f"detected={summary['detected_total']} "
            f"boxes={summary['box_total']} used={summary['used_box_total']} "
            f"avg_f1={format_percent(summary['avg_f1'])} "
            f"micro_precision={format_percent(summary['micro_precision'])} "
            f"micro_recall={format_percent(summary['micro_recall'])} "
            f"micro_f1={format_percent(summary['micro_f1'])}"
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
