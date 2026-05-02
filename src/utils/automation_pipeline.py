import csv
import json
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

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from bs4 import BeautifulSoup
from wiki_browser_fallback import fetch_strategems_via_browser
from src.utils.screen_regions import get_hud_region
from src.utils.strategem_detection import detect_icon_boxes_new


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
TEXT_ROI_WIDTH = 420
MIN_OCR_MATCH_SCORE = 0.55
FAST_ACCEPT_SCORE = 0.70
HIGH_CONFIDENCE_OCR_SCORE = 0.95
SKIP_FIRST_ICON_BOXES = 3
ICON_SCORE_FILE = os.path.join(OUTPUT_DIR, "_scores.json")
GENERIC_SUFFIX_WORDS = {
    # "backpack",
    # "emplacement",
    # "exosuit",
    # "pack",
    # "sentry",
}
FAST_OCR_CONFIGS = [
    r"--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-/ .",
]
FALLBACK_OCR_CONFIGS = [
    r"--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-/ .",
    r"--oem 3 --psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-/ .",
] 

try:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
except FileNotFoundError:
    print(f"Tesseract not found at '{TESSERACT_PATH}'. OCR will fail.")


def normalize_key(value):
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def make_display_name(name):
    """Create the short name used by runtime UI from the original wiki name."""
    cleaned_name = " ".join(str(name).split())
    if not cleaned_name:
        return ""

    words = cleaned_name.split()
    if len(words) > 1 and re.search(r"[-/\d]", words[0]):
        tail_words = words[1:]
        if len(tail_words) > 1 and tail_words[-1].casefold() in GENERIC_SUFFIX_WORDS:
            return " ".join(tail_words[:-1])
        return " ".join(tail_words)

    return cleaned_name


def get_original_name(entry):
    return (entry.get("OriginalName") or entry.get("Name") or "").strip()


def get_display_name(entry):
    original_name = get_original_name(entry)
    return (entry.get("Name") or make_display_name(original_name) or original_name).strip()


def prepare_strategem_entry(entry):
    original_name = get_original_name(entry)
    display_name = make_display_name(original_name) or original_name
    return {
        "Index": entry.get("Index", ""),
        "OriginalName": original_name,
        "Name": display_name,
        "Code": entry.get("Code", ""),
    }


def iter_name_match_values(entry):
    """Yield OCR match names derived from CSV data."""
    original_name = get_original_name(entry)
    display_name = get_display_name(entry)

    for value in (display_name, original_name):
        if value:
            yield value


def build_strategem_lookup(entries):
    lookup = {}
    for entry in entries:
        prepared_entry = prepare_strategem_entry(entry)
        for match_value in iter_name_match_values(prepared_entry):
            key = normalize_key(match_value)
            if key and key not in lookup:
                lookup[key] = prepared_entry
    return lookup


def unique_strategems_by_code(all_strategems):
    return {info["Code"]: info for info in all_strategems.values()}


def preprocess_ocr_variants(img, fast_only=False):
    """Create OCR-friendly variants for uneven HUD lighting."""
    if img is None or img.size == 0:
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    scale = 3.0 if gray.shape[0] < 80 else 2.0
    resized = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    balanced = clahe.apply(resized)
    denoised = cv2.bilateralFilter(balanced, d=7, sigmaColor=60, sigmaSpace=60)
    masks = build_white_text_masks(denoised)

    _, otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if fast_only:
        variants = [denoised, otsu]
        if masks:
            variants.append(masks[0])
        return variants

    variants = [resized, balanced, denoised]
    variants.extend(masks)
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


def extract_text_candidates(img, fast_only=False, seen=None):
    """Extract multiple OCR text candidates from a ROI."""
    candidates = []
    seen = seen if seen is not None else set()
    configs = FAST_OCR_CONFIGS if fast_only else FALLBACK_OCR_CONFIGS

    for variant in preprocess_ocr_variants(img, fast_only=fast_only):
        for config in configs:
            text = pytesseract.image_to_string(variant, config=config).strip()
            cleaned = re.sub(r"\s+", " ", text).strip()
            if not cleaned:
                continue
            key = normalize_key(cleaned)
            if key and key not in seen:
                candidates.append(cleaned)
                seen.add(key)

    return candidates


def score_ocr_text(text, all_strategems):
    normalized_text = normalize_key(text)
    if not normalized_text:
        return None, 0.0

    if normalized_text in all_strategems:
        return all_strategems[normalized_text], 1.08

    best_info = None
    best_score = 0.0
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
            best_score = score

    return best_info, best_score


def find_strategem_by_ocr(name_roi, all_strategems):
    """Match OCR candidates to the nearest known stratagem name."""
    best_info = None
    best_text = ""
    best_score = 0.0
    seen = set()

    for text in extract_text_candidates(name_roi, fast_only=True, seen=seen):
        strategem_info, score = score_ocr_text(text, all_strategems)
        if score > best_score:
            best_info = strategem_info
            best_text = text
            best_score = score

        if best_info and best_score >= HIGH_CONFIDENCE_OCR_SCORE:
            return best_info, best_text, best_score

    if best_info and best_score >= FAST_ACCEPT_SCORE:
        return best_info, best_text, best_score

    for text in extract_text_candidates(name_roi, fast_only=False, seen=seen):
        strategem_info, score = score_ocr_text(text, all_strategems)
        if score > best_score:
            best_info = strategem_info
            best_text = text
            best_score = score

        if best_info and best_score >= HIGH_CONFIDENCE_OCR_SCORE:
            return best_info, best_text, best_score

    if best_info and best_score >= MIN_OCR_MATCH_SCORE:
        return best_info, best_text, best_score

    return None, best_text, best_score


def load_icon_scores():
    if not os.path.exists(ICON_SCORE_FILE):
        return {}

    try:
        with open(ICON_SCORE_FILE, "r", encoding="utf-8") as file_obj:
            data = json.load(file_obj)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_icon_scores(icon_scores):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(ICON_SCORE_FILE, "w", encoding="utf-8") as file_obj:
        json.dump(icon_scores, file_obj, indent=2, sort_keys=True)


def get_recorded_score(icon_scores, code):
    value = icon_scores.get(code)
    if isinstance(value, dict):
        value = value.get("score")

    try:
        return float(value)
    except (TypeError, ValueError):
        return -1.0


def remember_best_candidate(best_candidates, strategem_info, icon_img, raw_text, score, source_image, logger):
    code = strategem_info["Code"]
    previous = best_candidates.get(code)
    if previous and previous["score"] >= score:
        logger.info(
            f"  Candidate kept: {strategem_info['Name']} "
            f"existing score {previous['score']:.3f} >= {score:.3f}"
        )
        return

    best_candidates[code] = {
        "info": strategem_info,
        "icon": icon_img,
        "text": raw_text,
        "score": score,
        "source": source_image,
    }


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
    prepared_data = [prepare_strategem_entry(entry) for entry in extracted_data]
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=["Index", "OriginalName", "Name", "Code"])
        writer.writeheader()
        writer.writerows(prepared_data)

    logger.info(f"[OK] Saved {len(prepared_data)} items to CSV via {source_label}.")
    return build_strategem_lookup(prepared_data)


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
    """OCR and extract icons, then log missing ones using Advanced HUD Detection."""
    logger.info("\n[*] Starting icon extraction process with Advanced HUD Detection...")
    if not all_strategems:
        logger.warning("[!] Cannot extract icons because strategem data is empty.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    icon_scores = load_icon_scores()
    best_candidates = {}
    unique_strategems = unique_strategems_by_code(all_strategems)
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

        # ==========================================
        # 🌟 อัปเกรด: ใช้ระบบตรวจจับ HUD ขั้นสูง
        # ==========================================
        try:
            from src.utils.screen_regions import get_hud_region
            from src.utils.strategem_detection import detect_icon_boxes_new
        except ImportError:
            from screen_regions import get_hud_region
            from strategem_detection import detect_icon_boxes_new

        # 1. ครอปเฉพาะ HUD (ตัดขยะบนหน้าจอทิ้ง)
        hud, hud_region = get_hud_region(frame)
        
        # 2. ค้นหาไอคอนด้วยระบบ Grid Alignment + Smart Top-Left Normalization
        icon_boxes = detect_icon_boxes_new(hud, hud_region.scale)

        if not icon_boxes:
            logger.warning(f"  [!] No icons detected in '{img_file}'")
            continue

        # icon_boxes ส่งค่าพิกัดที่ดัดทรงแล้วกลับมา
        for box_index, (y, x, width, height, icon_gray) in enumerate(icon_boxes):
            if box_index < SKIP_FIRST_ICON_BOXES:
                logger.info(f"  Skipped first/default icon slot {box_index + 1}")
                continue

            # 3. ตัดภาพไอคอน (ความกว้าง/สูง ถูก Normalization มาแล้วอย่างสมบูรณ์แบบ)
            icon_img = hud[y:y + height, x:x + width]

            # 4. กำหนดจุดสำหรับดึง Text (OCR) 
            # อิงพิกัด x + width ซึ่งตอนนี้มั่นใจได้ว่าเป็นขอบขวาของตัวไอคอนจริงๆ 
            text_start = min(hud.shape[1] - 1, x + width + TEXT_GAP_AFTER_ICON)
            text_end = min(hud.shape[1], text_start + TEXT_ROI_WIDTH)
            
            # ป้องกัน Error กรณีกล่องอยู่ชิดขอบจอเกินไป
            if text_start >= text_end or text_start >= hud.shape[1] or text_end <= 0:
                 continue
                 
            name_roi = hud[y:y + height, text_start:text_end]
            top_half = name_roi[:name_roi.shape[0] // 2, :]
            
            # 5. วิเคราะห์ OCR เพื่อหาชื่อ Stratagem
            strategem_info, raw_text, score = find_strategem_by_ocr(top_half, all_strategems)

            if strategem_info:
                logger.info(
                    f"  Detected: '{raw_text}' -> "
                    f"{strategem_info['Name']} (Index: {strategem_info['Index']}, OCR score: {score:.3f})"
                )
                remember_best_candidate(
                    best_candidates,
                    strategem_info,
                    icon_img,
                    raw_text,
                    score,
                    img_file,
                    logger,
                )
            else:
                logger.info(f"  Skipped OCR text: '{raw_text}' (best score: {score:.3f})")

    # ==========================================
    # ส่วนบันทึกและสรุปผล (คงไว้ตามเดิม)
    # ==========================================
    icons_saved_count = 0
    icons_replaced_count = 0
    for stg_code, item in sorted(best_candidates.items()):
        filename = f"{stg_code}.png"
        output_path = os.path.join(OUTPUT_DIR, filename)
        icon_img = item["icon"]
        score = item["score"]
        previous_score = get_recorded_score(icon_scores, stg_code)
        output_exists = os.path.exists(output_path)

        if icon_img is None or icon_img.size == 0:
            logger.warning(f"  Skipped empty icon for code: {stg_code}")
            continue

        if output_exists and score <= previous_score:
            logger.info(
                f"  Skipped existing: {filename} "
                f"(candidate score {score:.3f} <= saved score {previous_score:.3f})"
            )
            saved_codes.add(stg_code)
            continue

        if not cv2.imwrite(output_path, icon_img):
            logger.error(f"  Failed to save: {filename}")
            continue

        action = "Replaced" if output_exists else "Saved"
        logger.info(
            f"  {action}: {filename} "
            f"(score: {score:.3f}, text: '{item['text']}', source: {item['source']})"
        )
        icon_scores[stg_code] = {
            "score": score,
            "text": item["text"],
            "source": item["source"],
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        saved_codes.add(stg_code)
        if output_exists:
            icons_replaced_count += 1
        else:
            icons_saved_count += 1

    save_icon_scores(icon_scores)
    logger.info(f"  [OK] Saved {icons_saved_count} new icon(s), replaced {icons_replaced_count} icon(s).")

    logger.info("\n\n=== Processing Complete ===")
    all_codes = set(unique_strategems.keys())
    missing_codes = all_codes - saved_codes
    logger.info(f"Icon extraction summary: Found {len(saved_codes)}/{len(all_codes)} unique icons.")

    if missing_codes:
        logger.warning(f"Found {len(missing_codes)} missing icons. Writing details to 'missing_icons.log'.")
        with open("missing_icons.log", "w", encoding="utf-8") as file_obj:
            file_obj.write("Strategems with missing icons:\n===============================\n")
            code_to_name_map = {code: info["Name"] for code, info in unique_strategems.items()}
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
