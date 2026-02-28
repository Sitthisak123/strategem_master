import os
import cv2
import csv
import shutil
import requests
import re
import time
import logging
from bs4 import BeautifulSoup
from pathlib import Path
import pytesseract
from difflib import get_close_matches

# --- Configuration ---
WIKI_URL = "https://helldivers.wiki.gg/wiki/Stratagems"
SRC_IMG_DIR = "./src/img/group"
OUTPUT_DIR = "./img"
CSV_PATH = "./src/strategems.csv"
LOG_DIR = "./output"
LOG_FILE = os.path.join(LOG_DIR, "pipeline.log")
TESSERACT_PATH = r"C:\My Programs\Tesseract-OCR\tesseract.exe"

# --- Tesseract Setup ---
try:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
except FileNotFoundError:
    print(f"Tesseract not found at '{TESSERACT_PATH}'. OCR will fail.")
    # No exit here, to allow other parts of the pipeline to run if needed.

# --- Logger Setup ---
def setup_logger():
    """Configures the logger for console and file output."""
    os.makedirs(LOG_DIR, exist_ok=True)
    
    logger = logging.getLogger('PipelineLogger')
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # File Handler
    file_handler = logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8')
    file_formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console Handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(message)s') # No timestamp for console
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger

def setup_environment(logger):
    """1. Remove old icons and prepare directories"""
    logger.info(f"Cleaning up directory: {OUTPUT_DIR}")
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

def fetch_strategem_data(logger):
    """2. Scrape data from Wiki - MORE ROBUST with retry logic"""
    logger.info(f"Fetching data from {WIKI_URL}...")
    direction_map = {'LEFT': '1', 'UP': '2', 'RIGHT': '3', 'DOWN': '4'}
    extracted_data = []
    
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            response = requests.get(WIKI_URL, timeout=20)  # Increased timeout from 15 to 20
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            tables = soup.find_all('table', class_='wikitable')
            
            if not tables:
                logger.error("[!] Critical Error: No tables with class 'wikitable' found on the page.")
                logger.error("    The website's HTML structure may have changed, preventing data scraping.")
                return {}

            global_index = 1
            for table in tables:
                for row in table.find_all('tr'):
                    cells = row.find_all(['td', 'th'])
                    if len(cells) < 2: continue

                    code_cell, name_cell = None, None
                    for i, cell in enumerate(cells):
                        if cell.find('span', class_='Stratagemcodeicon'):
                            code_cell = cell
                            if i > 0: name_cell = cells[i-1]
                            break
                    
                    if code_cell and name_cell:
                        name = name_cell.get_text(strip=True)
                        arrow_imgs = code_cell.find_all('img')
                        if not name or not arrow_imgs or len(name) > 40 or any(x in name for x in ['Cooldown', 'Cost', 'Uses']):
                            continue

                        code = "".join([direction_map[re.search(r'Arrow\s(\w+)', img.get('alt', ''), re.I).group(1).upper()] 
                                       for img in arrow_imgs if re.search(r'Arrow\s(\w+)', img.get('alt', ''), re.I)])
                        if name and code:
                            extracted_data.append({"Index": global_index, "Name": name, "Code": code})
                            global_index += 1
            
            if not extracted_data:
                logger.warning("[!] Warning: Found tables but could not extract any strategem data.")
                logger.warning("    The internal structure of the tables may have changed.")
                return {}

            with open(CSV_PATH, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=["Index", "Name", "Code"])
                writer.writeheader()
                writer.writerows(extracted_data)
            logger.info(f"[✓] Saved {len(extracted_data)} items to CSV.")
            return {d['Name'].lower().replace(' ', ''): d for d in extracted_data}
            
        except requests.Timeout:
            retry_count += 1
            if retry_count < max_retries:
                logger.warning(f"[!] Request timeout. Retrying ({retry_count}/{max_retries})...")
                time.sleep(2)  # Wait 2 seconds before retrying
            else:
                logger.error(f"[!] Network Timeout: Failed to fetch data after {max_retries} attempts.")
                return {}
                
        except requests.RequestException as e:
            logger.error(f"[!] Network Error: Failed to fetch data from {WIKI_URL}. Reason: {e}")
            return {}
        except Exception as e:
            logger.error(f"[!] Unknown Error during web scraping: {e}", exc_info=True)
            return {}

def extract_and_save_icons(logger, all_strategems):
    """3. OCR and Extract Icons, then log missing ones"""
    logger.info("\n[*] Starting icon extraction process...")
    if not all_strategems:
        logger.warning("[!] Cannot extract icons because strategem data is empty.")
        return

    source_image_files = sorted([f for f in os.listdir(SRC_IMG_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if not source_image_files:
        logger.warning(f"[!] No source images found in '{SRC_IMG_DIR}'. Cannot extract any icons.")
        return

    saved_codes = set()
    total_files = len(source_image_files)

    for i, img_file in enumerate(source_image_files):
        logger.info(f"\n[{i+1}/{total_files}] Processing: {img_file}")
        frame = cv2.imread(os.path.join(SRC_IMG_DIR, img_file))
        if frame is None: continue
        
        hud = frame[30:800, 30:600]
        gray = cv2.cvtColor(hud, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_in_file = []
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if 0.65 < (w/float(h)) < 1.35 and w*h > 1800 and x < hud.shape[1] * 0.3:
                name_roi = hud[y:y+h, int(hud.shape[1]*0.22):]
                top_half = name_roi[:name_roi.shape[0]//2, :]
                raw_text = pytesseract.image_to_string(top_half, config='--psm 6').strip().replace(' ', '')
                
                match = get_close_matches(raw_text.lower(), all_strategems.keys(), n=1, cutoff=0.55)
                if match:
                    stg_info = all_strategems[match[0]]
                    logger.info(f"  Detected: '{raw_text.upper()}' → {stg_info['Name']} (Index: {stg_info['Index']})")
                    icon_img = hud[y:y+h, x:x+w]
                    detected_in_file.append({'info': stg_info, 'icon': icon_img})

        icons_saved_count = 0
        for item in detected_in_file:
            stg_code = item['info']['Code']
            if stg_code not in saved_codes:
                filename = f"{stg_code}.png"
                cv2.imwrite(os.path.join(OUTPUT_DIR, filename), item['icon'])
                logger.info(f"  Saved: {filename}")
                saved_codes.add(stg_code)
                icons_saved_count += 1
        
        if icons_saved_count > 0:
            logger.info(f"  ✓ Extracted {icons_saved_count} new icon(s)")

    logger.info("\n\n=== Processing Complete ===")
    all_codes = {info['Code'] for info in all_strategems.values()}
    missing_codes = all_codes - saved_codes
    logger.info(f"Icon extraction summary: Found {len(saved_codes)}/{len(all_codes)} unique icons.")

    if missing_codes:
        logger.warning(f"Found {len(missing_codes)} missing icons. Writing details to 'missing_icons.log'.")
        with open('missing_icons.log', 'w', encoding='utf-8') as f:
            f.write("Strategems with missing icons:\n===============================\n")
            code_to_name_map = {info['Code']: info['Name'] for info in all_strategems.values()}
            for code in sorted(missing_codes):
                name = code_to_name_map.get(code, "Unknown Name")
                f.write(f"  - Name: {name}, Code: {code}\n")
    else:
        logger.info("[✓] All strategem icons were found and saved successfully!")

def main_pipeline():
    """Main function to run the entire asset generation pipeline."""
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