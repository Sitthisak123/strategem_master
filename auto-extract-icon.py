import cv2
import numpy as np
import pytesseract
from difflib import get_close_matches
import csv
import os
from pathlib import Path
import time

# Setup pytesseract (change path if needed)
pytesseract.pytesseract.tesseract_cmd = r"C:\My Programs\Tesseract-OCR\tesseract.exe"
gap = 0.22  # gap from left edge to start of text area

# Directories
SRC_IMG_DIR = "./src/img/group"
OUTPUT_DIR = "./output"
CSV_FILE = "./src/strategems.csv"
LOG_FILE = os.path.join(OUTPUT_DIR, "log.txt")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


class Logger:
    """Logger class to write logs in real-time"""
    def __init__(self, log_path):
        self.log_path = log_path
        self.file = open(log_path, 'w', encoding='utf-8')
        self.file.write(f"[LOG STARTED] {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.file.flush()
    
    def log(self, message):
        """Log message to file and print to console"""
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        self.file.write(log_msg + "\n")
        self.file.flush()
    
    def close(self):
        """Close the log file"""
        self.file.write(f"[LOG ENDED] {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.file.close()


def load_strategems_csv(csv_path):
    """Load strategems from CSV file and return as dict indexed by name"""
    strategems = {}
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['Name']:
                    name_lower = row['Name'].lower()
                    strategems[name_lower] = {
                        'index': row['Index'],
                        'name': row['Name'],
                        'code': row['Code']
                    }
        return strategems
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return {}


def preprocess_for_contours(img):
    """Preprocess image for contour detection"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    return edges


def detect_strategem_icons(hud_img):
    """Detect strategem icons from HUD image"""
    edges = preprocess_for_contours(hud_img)
    cnts, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        aspect = w / float(h)
        area = w * h
        if 0.7 < aspect < 1.3 and area > 1800 and x < hud_img.shape[1] * 0.3:
            boxes.append((y, x, w, h))
    return sorted(boxes, key=lambda b: b[0])


def extract_text(img):
    """Extract text from image using Tesseract"""
    cfg = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    return pytesseract.image_to_string(img, config=cfg).strip()


def extract_icon_region(hud_img, y, x, w, h):
    """Extract the icon region from the HUD"""
    # Return the icon (left side)
    return hud_img[y:y+h, x:x+w]


def run_ocr(img, strategems_dict, logger):
    """Run OCR on image to detect and extract strategem icons"""
    hud = img[30:800, 30:600]
    icons = detect_strategem_icons(hud)
    extracted_icons = []
    
    for (y, x, w, h) in icons:
        row = hud[y:y+h, :]
        right_th = row[:, int(hud.shape[1]*gap):]
        h_right = right_th.shape[0]
        top_roi = right_th[:h_right//2, :]
        name_text = extract_text(top_roi)
        
        match = get_close_matches(name_text.lower(),
                                  strategems_dict.keys(),
                                  n=1,
                                  cutoff=0.55)
        best_name = match[0] if match else None
        
        if best_name:
            strategem_info = strategems_dict[best_name]
            icon_img = extract_icon_region(hud, y, x, w, h)
            
            logger.log(f"  Detected: '{name_text}' → {strategem_info['name']} (Index: {strategem_info['index']})")
            
            extracted_icons.append({
                'name': best_name,
                'display_name': strategem_info['name'],
                'index': strategem_info['index'],
                'code': strategem_info['code'],
                'image': icon_img
            })
        else:
            logger.log(f"  SKIPPED: '{name_text}' - Not found in strategems.csv")
    
    return extracted_icons


def save_extracted_icons(extracted_icons, output_dir, logger):
    """Save extracted icon images to output directory"""
    for icon_data in extracted_icons:
        try:
            # Use index as filename for consistency with CSV
            filename = f"{int(icon_data['index'])}.png"
            # filename = f"{int(icon_data['index']):03d}_{icon_data['display_name']}.png"
            filepath = os.path.join(output_dir, filename)
            
            cv2.imwrite(filepath, icon_data['image'])
            logger.log(f"  Saved: {filename}")
        except Exception as e:
            logger.log(f"  ERROR saving {icon_data['display_name']}: {e}")


def get_sorted_images(img_dir):
    """Get all image files from directory sorted by name"""
    if not os.path.exists(img_dir):
        return []
    
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    images = [f for f in os.listdir(img_dir) 
              if os.path.splitext(f)[1].lower() in image_extensions]
    return sorted(images)


def main():
    # Initialize logger
    logger = Logger(LOG_FILE)
    
    logger.log("=== Strategem Icon Extraction Started ===")
    logger.log(f"Loading strategems from: {CSV_FILE}")
    
    # Load strategems from CSV
    strategems_dict = load_strategems_csv(CSV_FILE)
    if not strategems_dict:
        logger.log("ERROR: Failed to load strategems from CSV")
        logger.close()
        return
    
    logger.log(f"Loaded {len(strategems_dict)} strategems from CSV")
    
    # Get sorted image files
    image_files = get_sorted_images(SRC_IMG_DIR)
    if not image_files:
        logger.log(f"WARNING: No images found in {SRC_IMG_DIR}")
        logger.close()
        return
    
    logger.log(f"Found {len(image_files)} images to process")
    logger.log("")
    
    # Process each image
    for img_idx, img_file in enumerate(image_files, 1):
        img_path = os.path.join(SRC_IMG_DIR, img_file)
        logger.log(f"[{img_idx}/{len(image_files)}] Processing: {img_file}")
        
        try:
            # Load image
            frame = cv2.imread(img_path)
            if frame is None:
                logger.log(f"  ERROR: Could not read image {img_file}")
                continue
            
            # Extract icons
            extracted_icons = run_ocr(frame, strategems_dict, logger)
            
            # Save icons
            if extracted_icons:
                save_extracted_icons(extracted_icons, OUTPUT_DIR, logger)
                logger.log(f"  ✓ Extracted {len(extracted_icons)} icon(s)")
            else:
                logger.log(f"  ℹ No icons detected in {img_file}")
        
        except Exception as e:
            logger.log(f"  ERROR processing {img_file}: {e}")
        
        logger.log("")
    
    logger.log("=== Processing Complete ===")
    logger.close()
    print("\nLog file saved to:", LOG_FILE)


if __name__ == "__main__":
    main()
