import csv
import os
from collections import namedtuple

import cv2
import numpy as np

from src.utils.cannyEdgeImplement import apply_sobel_feldman
from src.utils.screen_regions import get_hud_region, scale_pixels


MATCH_THRESHOLD = 0.4
MIN_ICON_SIZE = 30
MAX_ICON_SIZE = 150
MIN_ICON_AREA = 1800
ICON_PADDING = 1
DEFAULT_STRATEGEM_NAMES = {
    "resupply",
    "reinforce",
    "sos beacon",
    "eagle rearm",
}

DetectionAssets = namedtuple(
    "DetectionAssets",
    ["strategems_by_code", "templates", "template_features"],
)


def is_default_strategem(strategem):
    return strategem.get("name", "").casefold() in DEFAULT_STRATEGEM_NAMES


def load_detection_assets(csv_file="./src/strategems.csv", img_dir="./img"):
    strategems_by_code = {}
    templates = {}

    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = row["Code"]
            strategems_by_code[code] = {
                "index": row["Index"],
                "name": row["Name"],
                "code": code,
                "key": code,
            }

            img_path = os.path.join(img_dir, f"{code}.png")
            if os.path.exists(img_path):
                template_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if template_img is not None:
                    templates[code] = template_img

    return DetectionAssets(
        strategems_by_code=strategems_by_code,
        templates=templates,
        template_features=build_template_features(templates),
    )


def ensure_gray(img):
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def normalize_lightness(img):
    if img.ndim == 2:
        lightness = img
    else:
        lightness = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 0]

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    balanced = clahe.apply(lightness)
    return cv2.bilateralFilter(balanced, d=7, sigmaColor=60, sigmaSpace=60)


def auto_canny(img, sigma=0.33):
    median = float(np.median(img))
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    if upper <= lower:
        lower, upper = 30, 90
    return cv2.Canny(img, lower, upper)


def close_edges(edges):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)


def preprocess_contours_old(img):
    gray = ensure_gray(img)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(gray, 50, 150)


def preprocess_contours_new(img):
    return close_edges(auto_canny(normalize_lightness(img)))


def old_edge_features(img):
    gray = ensure_gray(img)
    sobel = ensure_uint8(apply_sobel_feldman(gray))
    canny = cv2.Canny(gray, 50, 150)
    return cv2.bitwise_or(sobel, canny)


def new_edge_features(img):
    normalized = normalize_lightness(img)
    canny = auto_canny(normalized)
    sobel = ensure_uint8(apply_sobel_feldman(normalized))
    _, sobel_binary = cv2.threshold(
        sobel,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    return close_edges(cv2.bitwise_or(canny, sobel_binary))


def ensure_uint8(img):
    if img.dtype == np.uint8:
        return img
    return np.uint8(np.clip(img, 0, 255))


def image_phash(img, hash_size=8, highfreq_factor=4):
    gray = normalize_lightness(img)
    size = hash_size * highfreq_factor
    resized = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(resized))
    low_freq = dct[:hash_size, :hash_size]
    median = np.median(low_freq[1:, 1:])

    value = 0
    for bit in low_freq.flatten() > median:
        value = (value << 1) | int(bit)
    return value


def hamming_distance(left, right):
    return int(left ^ right).bit_count()


def build_template_features(templates):
    features = {}
    for code, template in templates.items():
        features[code] = {
            "gray": template,
            "edges": new_edge_features(template),
            "hash": image_phash(template),
        }
    return features


def detect_icon_boxes_old(hud_img):
    return _detect_icon_boxes(
        hud_img=hud_img,
        edges=preprocess_contours_old(hud_img),
        min_icon_size=MIN_ICON_SIZE,
        max_icon_size=MAX_ICON_SIZE,
        min_icon_area=MIN_ICON_AREA,
        icon_padding=ICON_PADDING,
        retrieval_mode=cv2.RETR_EXTERNAL,
        max_x_ratio=0.3,
    )


def detect_icon_boxes_new(hud_img, scale=1.0):
    return _detect_icon_boxes(
        hud_img=hud_img,
        edges=preprocess_contours_new(hud_img),
        min_icon_size=scale_pixels(MIN_ICON_SIZE, scale, minimum=12),
        max_icon_size=scale_pixels(MAX_ICON_SIZE, scale, minimum=13),
        min_icon_area=max(250, int(round(MIN_ICON_AREA * scale * scale))),
        icon_padding=scale_pixels(ICON_PADDING, scale, minimum=1),
        retrieval_mode=cv2.RETR_LIST,
        max_x_ratio=0.16,
    )


def _detect_icon_boxes(
    hud_img,
    edges,
    min_icon_size,
    max_icon_size,
    min_icon_area,
    icon_padding,
    retrieval_mode,
    max_x_ratio,
):
    cnts, _ = cv2.findContours(edges, retrieval_mode, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    max_icon_x = hud_img.shape[1] * max_x_ratio

    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        aspect = w / float(h) if h != 0 else 0
        area = w * h

        if not (
            0.7 < aspect < 1.3
            and area > min_icon_area
            and x < max_icon_x
            and min_icon_size <= w <= max_icon_size
            and min_icon_size <= h <= max_icon_size
        ):
            continue

        candidates.append((x, y, w, h, area))

    boxes = []
    for x, y, w, h, _area in remove_overlapping_boxes(candidates):
        y_start = max(0, y - icon_padding)
        x_start = max(0, x - icon_padding)
        y_end = min(hud_img.shape[0], y + h + icon_padding)
        x_end = min(hud_img.shape[1], x + w + icon_padding)
        icon_region = hud_img[y_start:y_end, x_start:x_end]

        if icon_region.size == 0:
            continue

        boxes.append((
            y_start,
            x_start,
            x_end - x_start,
            y_end - y_start,
            ensure_gray(icon_region),
        ))

    return sorted(boxes, key=lambda b: b[0])


def remove_overlapping_boxes(candidates, overlap_threshold=0.45):
    selected = []

    for candidate in sorted(candidates, key=lambda item: item[4], reverse=True):
        if any(overlap_ratio(candidate, existing) > overlap_threshold for existing in selected):
            continue
        selected.append(candidate)

    return selected


def overlap_ratio(left_box, right_box):
    lx, ly, lw, lh, _left_area = left_box
    rx, ry, rw, rh, _right_area = right_box

    inter_left = max(lx, rx)
    inter_top = max(ly, ry)
    inter_right = min(lx + lw, rx + rw)
    inter_bottom = min(ly + lh, ry + rh)

    inter_w = max(0, inter_right - inter_left)
    inter_h = max(0, inter_bottom - inter_top)
    inter_area = inter_w * inter_h
    smaller_area = min(lw * lh, rw * rh)

    if smaller_area == 0:
        return 0.0
    return inter_area / smaller_area


def match_edges(search_edges, template_edges, scales):
    best_val = -1.0
    best_loc = None
    best_scale = None
    best_size = None

    for scale in scales:
        resized = cv2.resize(
            template_edges,
            None,
            fx=float(scale),
            fy=float(scale),
            interpolation=cv2.INTER_LINEAR,
        )

        h, w = resized.shape
        if h > search_edges.shape[0] or w > search_edges.shape[1]:
            continue

        result = cv2.matchTemplate(search_edges, resized, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if not np.isfinite(max_val):
            continue

        if max_val > best_val:
            best_val = float(max_val)
            best_loc = max_loc
            best_scale = float(scale)
            best_size = (w, h)

            if best_val > 0.85:
                break

    return {
        "location": best_loc,
        "scale": best_scale,
        "score": best_val,
        "size": best_size,
    }


def old_template_match(icon_region, template):
    search_edges = old_edge_features(icon_region)
    template_edges = old_edge_features(template)
    scales = np.linspace(0.7, 1.8, 12)
    return match_edges(search_edges, template_edges, scales)


def new_template_match(icon_region, template_feature):
    search_edges = new_edge_features(icon_region)
    template_edges = template_feature["edges"]
    ih, iw = template_edges.shape
    fit_scale = min(search_edges.shape[0] / ih, search_edges.shape[1] / iw)
    min_scale = max(0.25, fit_scale * 0.65)
    max_scale = min(2.5, max(fit_scale * 1.35, min_scale + 0.05))
    scales = np.linspace(min_scale, max_scale, 14)
    return match_edges(search_edges, template_edges, scales)


def shortlist_by_hash(icon_region, template_features, available_codes, limit=15):
    if limit is None or limit <= 0 or limit >= len(available_codes):
        return list(available_codes)

    query_hash = image_phash(icon_region)
    ranked = sorted(
        available_codes,
        key=lambda code: hamming_distance(query_hash, template_features[code]["hash"]),
    )
    return ranked[:limit]


def detect_strategems_old(
    screenshot,
    assets,
    match_threshold=MATCH_THRESHOLD,
    include_defaults=False,
    max_results=4,
    skip_first=2,
    verbose=False,
):
    hud_img = screenshot[30:800, 30:600]
    icon_boxes = detect_icon_boxes_old(hud_img)
    return _match_detected_boxes(
        icon_boxes=icon_boxes[skip_first:] if len(icon_boxes) > skip_first else [],
        assets=assets,
        match_threshold=match_threshold,
        include_defaults=include_defaults,
        max_results=max_results,
        mode="old",
        verbose=verbose,
    )


def detect_strategems_new(
    screenshot,
    assets,
    match_threshold=MATCH_THRESHOLD,
    include_defaults=False,
    max_results=4,
    skip_first=2,
    phash_candidates=15,
    verbose=False,
):
    hud_img, hud_region = get_hud_region(screenshot)
    icon_boxes = detect_icon_boxes_new(hud_img, hud_region.scale)
    return _match_detected_boxes(
        icon_boxes=icon_boxes[skip_first:] if len(icon_boxes) > skip_first else [],
        assets=assets,
        match_threshold=match_threshold,
        include_defaults=include_defaults,
        max_results=max_results,
        mode="new",
        phash_candidates=phash_candidates,
        verbose=verbose,
    )


def _match_detected_boxes(
    icon_boxes,
    assets,
    match_threshold,
    include_defaults,
    max_results,
    mode,
    phash_candidates=None,
    verbose=False,
):
    detected = []
    matched_codes = set()

    for idx, (_y, _x, _w, _h, icon_region) in enumerate(icon_boxes, start=1):
        available_codes = [
            code for code in assets.templates
            if code not in matched_codes and code in assets.strategems_by_code
        ]

        if mode == "new":
            available_codes = shortlist_by_hash(
                icon_region,
                assets.template_features,
                available_codes,
                limit=phash_candidates,
            )

        best_score = match_threshold
        best_entry = None
        best_extra = {}

        for code in available_codes:
            if mode == "old":
                result = old_template_match(icon_region, assets.templates[code])
            else:
                result = new_template_match(icon_region, assets.template_features[code])

            if result["score"] > best_score:
                best_score = result["score"]
                best_entry = assets.strategems_by_code[code]
                best_extra = result

        if not best_entry:
            if verbose:
                print(f"[{mode}] icon {idx}: no match")
            continue

        matched_codes.add(best_entry["code"])
        if not include_defaults and is_default_strategem(best_entry):
            if verbose:
                print(f"[{mode}] icon {idx}: skipped default {best_entry['name']} ({best_score:.3f})")
            continue

        entry = best_entry.copy()
        entry["confidence"] = best_score
        entry["detector"] = mode
        entry["match_scale"] = best_extra.get("scale")
        detected.append(entry)

        if verbose:
            print(f"[{mode}] icon {idx}: matched {entry['name']} ({best_score:.3f})")

    if max_results is None or max_results <= 0:
        return detected
    return detected[-max_results:] if len(detected) > max_results else detected
