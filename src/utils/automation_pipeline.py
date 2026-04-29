import csv
import logging
import os
import re
import sys
import time
from difflib import SequenceMatcher, get_close_matches

import cv2
import numpy as np
import pytesseract
import requests
from bs4 import BeautifulSoup

from wiki_browser_fallback import fetch_strategems_via_browser


WIKI_URL = "https://helldivers.wiki.gg/wiki/Stratagems"
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://helldivers.wiki.gg/",
    "Cache-Control": "no-cache",
}
SRC_IMG_DIR = "./src/img/group"
OUTPUT_DIR = "./img"
CSV_PATH = "./src/strategems.csv"
LOG_DIR = "./output"
LOG_FILE = os.path.join(LOG_DIR, "pipeline.log")
TESSERACT_PATH = r"C:\My Programs\Tesseract-OCR\tesseract.exe"
MIN_ICON_SIZE = 30
MAX_ICON_SIZE = 150
ICON_PADDING = 0
TEXT_GAP_AFTER_ICON = 10
MIN_OCR_MATCH_SCORE = 0.55
OCR_CONFIGS = [
    r"--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-/ .",
    r"--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-/ .",
    r"--oem 3 --psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-/ .",
]

try:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
except FileNotFoundError:
    print(f"Tesseract not found at '{TESSERACT_PATH}'. OCR will fail.")


def normalize_key(value):
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def preprocess_ocr_variants(img):
    """Create OCR-friendly variants for uneven HUD lighting."""
    if img is None or img.size == 0:
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    scale = 3.0 if gray.shape[0] < 80 else 2.0
    resized = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    balanced = clahe.apply(resized)
    denoised = cv2.bilateralFilter(balanced, d=7, sigmaColor=60, sigmaSpace=60)

    variants = [resized, balanced, denoised]
    variants.extend(build_white_text_masks(denoised))
    _, otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, otsu_inv = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        7,
    )
    variants.extend([otsu, otsu_inv, adaptive])
    return variants


def build_white_text_masks(gray):
    """Extract bright HUD text from translucent dark backgrounds."""
    if gray is None or gray.size == 0:
        return []

    background_sigma = max(3.0, gray.shape[0] / 4.0)
    background = cv2.GaussianBlur(gray, (0, 0), background_sigma)
    local_bright = cv2.subtract(gray, background)
    local_bright = cv2.normalize(local_bright, None, 0, 255, cv2.NORM_MINMAX)

    kernel_width = max(9, (gray.shape[0] // 2) | 1)
    kernel_height = max(3, (gray.shape[0] // 10) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, kernel_height))
    top_hat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    top_hat = cv2.normalize(top_hat, None, 0, 255, cv2.NORM_MINMAX)

    enhanced = cv2.max(local_bright, top_hat)
    mean, stddev = cv2.meanStdDev(enhanced)
    percentile_threshold = float(np.percentile(enhanced, 82))
    dynamic_threshold = int(max(18, percentile_threshold, mean[0][0] + (0.35 * stddev[0][0])))

    _, percentile_mask = cv2.threshold(enhanced, dynamic_threshold, 255, cv2.THRESH_BINARY)
    _, otsu_mask = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.bitwise_or(percentile_mask, otsu_mask)

    cleanup_kernel = np.ones((2, 2), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cleanup_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cleanup_kernel)

    return [mask, cv2.bitwise_not(mask)]


def extract_text_candidates(img):
    """Extract multiple OCR text candidates from a ROI."""
    candidates = []
    seen = set()

    for variant in preprocess_ocr_variants(img):
        for config in OCR_CONFIGS:
            text = pytesseract.image_to_string(variant, config=config).strip()
            cleaned = re.sub(r"\s+", " ", text).strip()
            if not cleaned:
                continue
            key = normalize_key(cleaned)
            if key and key not in seen:
                candidates.append(cleaned)
                seen.add(key)

    return candidates


def find_strategem_by_ocr(name_roi, all_strategems):
    """Match OCR candidates to the nearest known stratagem name."""
    best_info = None
    best_text = ""
    best_score = 0.0

    for text in extract_text_candidates(name_roi):
        normalized_text = normalize_key(text)
        if not normalized_text:
            continue

        for candidate_key in get_close_matches(
            normalized_text,
            all_strategems.keys(),
            n=5,
            cutoff=0.30,
        ):
            score = SequenceMatcher(None, normalized_text, candidate_key).ratio()
            if normalized_text in candidate_key or candidate_key in normalized_text:
                score += 0.08

            if score > best_score:
                best_info = all_strategems[candidate_key]
                best_text = text
                best_score = score

    if best_info and best_score >= MIN_OCR_MATCH_SCORE:
        return best_info, best_text, best_score

    return None, best_text, best_score


def setup_logger():
    """Configure the logger for console and file output."""
    os.makedirs(LOG_DIR, exist_ok=True)

    logger = logging.getLogger("PipelineLogger")
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
    file_formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)
    return logger


def setup_environment(logger):
    """Prepare directories without deleting existing extracted icons."""
    logger.info(f"Preparing directory: {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)


def save_strategem_csv(logger, extracted_data, source_label="wiki"):
    """Persist extracted strategem data and return the lookup dict used by OCR."""
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=["Index", "Name", "Code"])
        writer.writeheader()
        writer.writerows(extracted_data)

    logger.info(f"[OK] Saved {len(extracted_data)} items to CSV via {source_label}.")
    return {normalize_key(entry["Name"]): entry for entry in extracted_data}


def fetch_strategem_data_via_browser(logger, reason):
    """Fallback to Playwright-driven browser extraction when requests is blocked."""
    logger.warning(f"[!] Falling back to Playwright browser fetch because {reason}")
    try:
        payload = fetch_strategems_via_browser()
        extracted_data = payload.get("entries") or []
        if not extracted_data:
            logger.error("[!] Browser fallback returned no strategem entries.")
            return {}

        logger.info(
            f"[OK] Browser fallback loaded {len(extracted_data)} items "
            f"using {payload.get('browser', 'a supported browser')}."
        )
        return save_strategem_csv(logger, extracted_data, "Playwright fallback")
    except Exception as error:
        logger.error(f"[!] Browser fallback failed: {error}")
        return {}


def fetch_strategem_data(logger):
    """Scrape data from the wiki with retry logic and a browser fallback."""
    logger.info(f"Fetching data from {WIKI_URL}...")
    direction_map = {"LEFT": "1", "UP": "2", "RIGHT": "3", "DOWN": "4"}

    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        try:
            response = requests.get(
                WIKI_URL,
                headers=REQUEST_HEADERS,
                timeout=20,
            )
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            tables = soup.find_all("table", class_="wikitable")

            if not tables:
                logger.error("[!] Critical Error: No tables with class 'wikitable' found on the page.")
                logger.error("    The website HTML may have changed or the response may be a challenge page.")
                return fetch_strategem_data_via_browser(
                    logger,
                    "no wiki tables were found in the requests response",
                )

            extracted_data = []
            global_index = 1
            for table in tables:
                for row in table.find_all("tr"):
                    cells = row.find_all(["td", "th"])
                    if len(cells) < 2:
                        continue

                    code_cell = None
                    name_cell = None
                    for index, cell in enumerate(cells):
                        if cell.find("span", class_="Stratagemcodeicon"):
                            code_cell = cell
                            if index > 0:
                                name_cell = cells[index - 1]
                            break

                    if not code_cell or not name_cell:
                        continue

                    name = name_cell.get_text(strip=True)
                    arrow_imgs = code_cell.find_all("img")
                    if (
                        not name
                        or not arrow_imgs
                        or len(name) > 40
                        or any(token in name for token in ["Cooldown", "Cost", "Uses"])
                    ):
                        continue

                    code = "".join(
                        [
                            direction_map[re.search(r"Arrow\s(\w+)", img.get("alt", ""), re.I).group(1).upper()]
                            for img in arrow_imgs
                            if re.search(r"Arrow\s(\w+)", img.get("alt", ""), re.I)
                        ]
                    )
                    if name and code:
                        extracted_data.append({"Index": global_index, "Name": name, "Code": code})
                        global_index += 1

            if not extracted_data:
                logger.warning("[!] Warning: Requests returned tables but no strategem entries were parsed.")
                return fetch_strategem_data_via_browser(
                    logger,
                    "requests parsing produced zero strategem entries",
                )

            return save_strategem_csv(logger, extracted_data, "requests")

        except requests.Timeout:
            retry_count += 1
            if retry_count < max_retries:
                logger.warning(f"[!] Request timeout. Retrying ({retry_count}/{max_retries})...")
                time.sleep(2)
            else:
                logger.error(f"[!] Network Timeout: Failed to fetch data after {max_retries} attempts.")
                return fetch_strategem_data_via_browser(logger, "requests timed out repeatedly")
        except requests.RequestException as error:
            if getattr(getattr(error, "response", None), "status_code", None) == 403:
                logger.error("[!] Wiki request was blocked with HTTP 403. The site is allowing a real browser but blocking direct requests.")
            logger.error(f"[!] Network Error: Failed to fetch data from {WIKI_URL}. Reason: {error}")
            return fetch_strategem_data_via_browser(logger, f"requests failed with {error}")
        except Exception as error:
            logger.error(f"[!] Unknown Error during web scraping: {error}", exc_info=True)
            return fetch_strategem_data_via_browser(logger, f"unexpected parser error: {error}")

    return {}


def extract_and_save_icons(logger, all_strategems):
    """OCR and extract icons, then log missing ones."""
    logger.info("\n[*] Starting icon extraction process...")
    if not all_strategems:
        logger.warning("[!] Cannot extract icons because strategem data is empty.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    source_image_files = sorted(
        [
            file_name
            for file_name in os.listdir(SRC_IMG_DIR)
            if file_name.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    )
    if not source_image_files:
        logger.warning(f"[!] No source images found in '{SRC_IMG_DIR}'. Cannot extract any icons.")
        return

    saved_codes = {
        os.path.splitext(file_name)[0]
        for file_name in os.listdir(OUTPUT_DIR)
        if file_name.lower().endswith(".png")
    }
    total_files = len(source_image_files)

    if saved_codes:
        logger.info(f"Found {len(saved_codes)} existing icon(s) in '{OUTPUT_DIR}'. They will be skipped.")

    for index, img_file in enumerate(source_image_files, start=1):
        logger.info(f"\n[{index}/{total_files}] Processing: {img_file}")
        frame = cv2.imread(os.path.join(SRC_IMG_DIR, img_file))
        if frame is None:
            continue

        frame_height, frame_width = frame.shape[:2]
        hud_top, hud_bottom = 30, min(frame_height, 800)
        hud_left, hud_right = 30, min(frame_width, 600)
        if hud_top >= hud_bottom or hud_left >= hud_right:
            logger.warning(f"[!] HUD crop is empty for '{img_file}'. Skipping file.")
            continue

        hud = frame[hud_top:hud_bottom, hud_left:hud_right]
        gray = cv2.cvtColor(hud, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected_in_file = []
        candidate_boxes = []
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            if (
                0.65 < (width / float(height)) < 1.35
                and width * height > 1800
                and x < hud.shape[1] * 0.3
                and MIN_ICON_SIZE <= width <= MAX_ICON_SIZE
                and MIN_ICON_SIZE <= height <= MAX_ICON_SIZE
            ):
                candidate_boxes.append((y, x, width, height))

        candidate_boxes.sort(key=lambda box: (box[0], box[1]))
        for y, x, width, height in candidate_boxes:
            y_start = max(0, y - ICON_PADDING)
            x_start = max(0, x - ICON_PADDING)
            y_end = min(hud.shape[0], y + height + ICON_PADDING)
            x_end = min(hud.shape[1], x + width + ICON_PADDING)
            icon_img = hud[y_start:y_end, x_start:x_end]

            text_start = min(hud.shape[1] - 1, x + width + TEXT_GAP_AFTER_ICON)
            name_roi = hud[y:y + height, text_start:]
            top_half = name_roi[:name_roi.shape[0] // 2, :]
            strategem_info, raw_text, score = find_strategem_by_ocr(top_half, all_strategems)

            if strategem_info:
                logger.info(
                    f"  Detected: '{raw_text}' -> "
                    f"{strategem_info['Name']} (Index: {strategem_info['Index']}, OCR score: {score:.3f})"
                )
                detected_in_file.append({"info": strategem_info, "icon": icon_img})
            else:
                logger.info(f"  Skipped OCR text: '{raw_text}' (best score: {score:.3f})")

        icons_saved_count = 0
        for item in detected_in_file:
            stg_code = item["info"]["Code"]
            if stg_code in saved_codes:
                continue

            filename = f"{stg_code}.png"
            output_path = os.path.join(OUTPUT_DIR, filename)
            icon_img = item["icon"]

            if icon_img is None or icon_img.size == 0:
                logger.warning(f"  Skipped empty icon for code: {stg_code}")
                continue

            if os.path.exists(output_path):
                logger.info(f"  Skipped existing: {filename}")
                saved_codes.add(stg_code)
                continue

            if not cv2.imwrite(output_path, icon_img):
                logger.error(f"  Failed to save: {filename}")
                continue

            logger.info(f"  Saved: {filename}")
            saved_codes.add(stg_code)
            icons_saved_count += 1

        if icons_saved_count > 0:
            logger.info(f"  [OK] Extracted {icons_saved_count} new icon(s)")

    logger.info("\n\n=== Processing Complete ===")
    all_codes = {info["Code"] for info in all_strategems.values()}
    missing_codes = all_codes - saved_codes
    logger.info(f"Icon extraction summary: Found {len(saved_codes)}/{len(all_codes)} unique icons.")

    if missing_codes:
        logger.warning(f"Found {len(missing_codes)} missing icons. Writing details to 'missing_icons.log'.")
        with open("missing_icons.log", "w", encoding="utf-8") as file_obj:
            file_obj.write("Strategems with missing icons:\n===============================\n")
            code_to_name_map = {info["Code"]: info["Name"] for info in all_strategems.values()}
            for code in sorted(missing_codes):
                name = code_to_name_map.get(code, "Unknown Name")
                file_obj.write(f"  - Name: {name}, Code: {code}\n")
    else:
        logger.info("[OK] All strategem icons were found and saved successfully.")


def main_pipeline():
    """Run the entire asset generation pipeline."""
    logger = setup_logger()
    start_time = time.time()
    logger.info("=== Helldivers 2 Asset Pipeline Starting ===")

    setup_environment(logger)
    data_dict = fetch_strategem_data(logger)
    extract_and_save_icons(logger, data_dict)

    logger.info(f"\n=== Pipeline Finished in {time.time() - start_time:.2f}s ===")
    logging.shutdown()


if __name__ == "__main__":
    main_pipeline()
