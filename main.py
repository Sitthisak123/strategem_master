import sys
import cv2
import numpy as np
import keyboard
import pyautogui
import csv
import os
from pathlib import Path
import time
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer
from overlay_window import OverlayWindow
import functools

# Setup directories and CSV
<<<<<<< HEAD
IMG_TEMPLATE_DIR = "./img"
CSV_FILE = "./src/strategems.csv"
MATCH_THRESHOLD = 0.94
=======
IMG_DIR = "./img"
CSV_FILE = "./src/strategems.csv"
MATCH_THRESHOLD = 0.7
>>>>>>> a724b947cc0933cae862cd3d89d925d369e5c07b

# hotkeys
exit_keys = "ctrl+c"
reinforce_keys = {
    "name": "reinforce",
    "key": "g"
}
supply_keys = {
    "name": "resupply",
    "key": "v"
}
eagleRearm_keys = {
    "name": "eagle rearm",
    "key": "q"
}

top_row_keys = {
    #scanCode: top-row-num-key
    2: '1',
    3: '2',
    4: '3',
    5: '4',
    6: '5',
    7: '6',
    8: '7',
    9: '8',
    10: '9',
    11: '0',
}

allowkeys = [reinforce_keys, supply_keys, eagleRearm_keys]


def load_strategems_csv(csv_path):
<<<<<<< HEAD
    """Load strategems from CSV file"""
    strategems_by_index = {}
=======
    """Load strategems from CSV file and return as dict indexed by index and name"""
    strategems = {}
>>>>>>> a724b947cc0933cae862cd3d89d925d369e5c07b
    strategems_by_name = {}
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['Index'] and row['Name']:
<<<<<<< HEAD
                    idx = row['Index'].strip()
=======
                    idx = row['Index']
>>>>>>> a724b947cc0933cae862cd3d89d925d369e5c07b
                    name_lower = row['Name'].lower()
                    strategem_entry = {
                        'index': idx,
                        'name': row['Name'],
                        'code': row['Code'],
<<<<<<< HEAD
                        'key': row['Code']
                    }
                    strategems_by_index[idx] = strategem_entry
                    strategems_by_name[name_lower] = strategem_entry
        return strategems_by_index, strategems_by_name
=======
                        'key': row['Code']  # Code is the key sequence
                    }
                    strategems[idx] = strategem_entry
                    strategems_by_name[name_lower] = strategem_entry
        return strategems, strategems_by_name
>>>>>>> a724b947cc0933cae862cd3d89d925d369e5c07b
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return {}, {}


# Load strategems from CSV
strategems_all, strategems_by_name = load_strategems_csv(CSV_FILE)

# Default strategems that appear in default slots
strategem_default_slots = [
    "resupply",
    "reinforce",
    "sos beacon",
    "eagle rearm",
]

pyautogui.PAUSE = 0.030
<<<<<<< HEAD


def get_sorted_template_images(img_dir):
    """Get all template images from directory sorted by filename"""
    if not os.path.exists(img_dir):
        return []
    
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    images = [f for f in os.listdir(img_dir) 
              if os.path.splitext(f)[1].lower() in image_extensions]
    return sorted(images)


def load_template_library(img_dir):
    """Load and cache all template images from ./img directory"""
    templates = {}
    image_files = get_sorted_template_images(img_dir)
    
    for img_file in image_files:
        img_path = os.path.join(img_dir, img_file)
        template_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if template_img is not None:
            filename_base = os.path.splitext(img_file)[0]
            templates[filename_base] = template_img
    
    return templates


def preprocess_for_contours(img):
    """Preprocess image for contour detection (like original OCR)"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    return edges


def detect_strategem_icon_regions(hud_img):
    """Detect icon regions in HUD (like original OCR) - returns bounding boxes"""
    edges = preprocess_for_contours(hud_img)
    cnts, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        aspect = w / float(h)
        area = w * h
        # Filter for square-ish shapes (icons)
        if 0.7 < aspect < 1.3 and area > 1800 and x < hud_img.shape[1] * 0.3:
            boxes.append((y, x, w, h))
    return sorted(boxes, key=lambda b: b[0])


def extract_icon_image(hud_img, y, x, w, h):
    """Extract icon region from HUD"""
    return hud_img[y:y+h, x:x+w]


def match_icon_to_templates(icon_img_gray, templates, threshold=MATCH_THRESHOLD):
    """Match a detected icon against all template images, return best match"""
    best_match_index = None
    best_confidence = 0.0
    
    for template_index, template_img in templates.items():
        # Skip if template is larger than icon
        if template_img.shape[0] > icon_img_gray.shape[0] or \
           template_img.shape[1] > icon_img_gray.shape[1]:
            continue
        
        try:
            # Match template using normalized cross-correlation
            result = cv2.matchTemplate(icon_img_gray, template_img, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            # Track best match
            if max_val > best_confidence:
                best_confidence = max_val
                best_match_index = template_index
        except cv2.error:
            continue
    
    # Return match only if above threshold
    if best_confidence >= threshold:
        return best_match_index, best_confidence
    
    return None, 0.0


def run_template_matching(screenshot, templates):
    """
    Detect strategems using template matching on individual icons.
    1. Extract HUD from screenshot
    2. Detect icon regions
    3. For each icon, match against template library
    4. Look up in CSV by index
=======


def get_sorted_images(img_dir):
    """Get all image files from directory sorted by filename"""
    if not os.path.exists(img_dir):
        return []
    
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    images = [f for f in os.listdir(img_dir) 
              if os.path.splitext(f)[1].lower() in image_extensions]
    return sorted(images)


def match_template_image(haystack_gray, needle_gray, method=cv2.TM_CCOEFF_NORMED):
    """Match template image and return position and confidence"""
    try:
        result = cv2.matchTemplate(haystack_gray, needle_gray, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        return max_loc, max_val
    except cv2.error as e:
        return None, 0.0


def run_template_matching(screenshot):
    """
    Use template matching to detect strategems from screenshot.
    Screenshots are taken as templates, source images from ./img/* are matched.
>>>>>>> a724b947cc0933cae862cd3d89d925d369e5c07b
    """
    strategems = []
    
    # Extract HUD area from screenshot
<<<<<<< HEAD
    hud = screenshot[30:800, 30:600]
    
    # Detect icon regions (same as original OCR)
    icon_boxes = detect_strategem_icon_regions(hud)
    
    if not icon_boxes:
        print("No icon regions detected in HUD")
        return strategems
    
    # For each detected icon region
    for (y, x, w, h) in icon_boxes:
        # Extract the icon
        icon_img = extract_icon_image(hud, y, x, w, h)
        icon_gray = cv2.cvtColor(icon_img, cv2.COLOR_BGR2GRAY)
        
        # Try to match against templates
        matched_index, confidence = match_icon_to_templates(icon_gray, templates)
        
        if matched_index and matched_index in strategems_all:
            strategem_info = strategems_all[matched_index]
            print(f"Match: {strategem_info['name']} (Index: {matched_index}, Confidence: {confidence:.4f})")
            
            # Avoid duplicates
            if strategem_info['name'].lower() not in strategem_default_slots:
                strategem_copy = strategem_info.copy()
                strategem_copy['confidence'] = confidence
                strategems.append(strategem_copy)
    
    # Limit to 4 strategems
    if len(strategems) > 4:
        strategems = strategems[-4:]
=======
    screenshot_hud = screenshot[30:800, 30:600]
    screenshot_gray = cv2.cvtColor(screenshot_hud, cv2.COLOR_BGR2GRAY)
    
    # Get sorted image files from ./img
    image_files = get_sorted_images(IMG_DIR)
    
    if not image_files:
        print(f"No images found in {IMG_DIR}")
        return strategems
    
    # Track matched images to avoid duplicates
    matched_indices = set()
    
    for img_file in image_files:
        img_path = os.path.join(IMG_DIR, img_file)
        
        # Load source image
        source_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if source_img is None:
            print(f"Could not load {img_file}")
            continue
        
        # Skip if template is larger than haystack
        if source_img.shape[0] > screenshot_gray.shape[0] or \
           source_img.shape[1] > screenshot_gray.shape[1]:
            continue
        
        # Try to match this image in the screenshot
        top_left, confidence = match_template_image(screenshot_gray, source_img)
        
        if top_left and confidence >= MATCH_THRESHOLD:
            # Extract filename without extension to get index
            filename_base = os.path.splitext(img_file)[0]
            try:
                img_index = filename_base
                
                # Look up strategem by index
                if img_index in strategems_all:
                    strategem_info = strategems_all[img_index]
                    if img_index not in matched_indices:
                        matched_indices.add(img_index)
                        strategem_info_copy = strategem_info.copy()
                        strategem_info_copy['confidence'] = confidence
                        strategems.append(strategem_info_copy)
                        print(f"Match: {strategem_info['name']} (Index: {img_index}, Confidence: {confidence:.4f})")
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
    
    # Limit to 4 strategems (top 4 by confidence)
    if len(strategems) > 4:
        strategems = sorted(strategems, key=lambda x: x.get('confidence', 0), reverse=True)[:4]
>>>>>>> a724b947cc0933cae862cd3d89d925d369e5c07b
    
    return strategems


<<<<<<< HEAD
# Load template library once at startup
template_library = load_template_library(IMG_TEMPLATE_DIR)
print(f"Loaded {len(template_library)} template images")

=======
>>>>>>> a724b947cc0933cae862cd3d89d925d369e5c07b
strategems_current = []


def on_screenshot(overlay_window):
    """Take screenshot and detect strategems"""
    global strategems_current
<<<<<<< HEAD
    
=======
>>>>>>> a724b947cc0933cae862cd3d89d925d369e5c07b
    # Show loading state
    QTimer.singleShot(0, functools.partial(overlay_window.update_labels, loading=True))
    QApplication.processEvents()
    
<<<<<<< HEAD
    # Take screenshot
=======
    # Take screenshotUpdating labels
>>>>>>> a724b947cc0933cae862cd3d89d925d369e5c07b
    screenshot = pyautogui.screenshot()
    frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    
    # Run template matching detection
<<<<<<< HEAD
    stg_inSlot = run_template_matching(frame, template_library)
=======
    stg_inSlot = run_template_matching(frame)
>>>>>>> a724b947cc0933cae862cd3d89d925d369e5c07b
    strategems_current = stg_inSlot
    
    # Update overlay with detected strategems
    QTimer.singleShot(0, functools.partial(overlay_window.update_labels, stg_inSlot))
    print(f"Detected {len(strategems_current)} strategems")


def strategem_operator(key_sequence):
    """Execute key sequence for strategem input"""
<<<<<<< HEAD
    print(f"Executing key sequence: {key_sequence}")
=======
    if not key_sequence:
        print("No key sequence provided.")
        return
    print(f"\tExecuting key sequence: {key_sequence}")
>>>>>>> a724b947cc0933cae862cd3d89d925d369e5c07b
    for key in str(key_sequence):
        match key:
            case '1':
                pyautogui.keyDown("left")
                pyautogui.keyUp("left")
                pyautogui.sleep(0.1)
<<<<<<< HEAD
                print("Left")
=======
                print("←", end="")
>>>>>>> a724b947cc0933cae862cd3d89d925d369e5c07b
            case '2':
                pyautogui.keyDown("up")
                pyautogui.keyUp("up")
                pyautogui.sleep(0.1)
<<<<<<< HEAD
                print("Up")
=======
                print("↑", end="")
>>>>>>> a724b947cc0933cae862cd3d89d925d369e5c07b
            case '3':
                pyautogui.keyDown("right")
                pyautogui.keyUp("right")
                pyautogui.sleep(0.1)
<<<<<<< HEAD
                print("Right")
=======
                print("→", end="")
>>>>>>> a724b947cc0933cae862cd3d89d925d369e5c07b
            case '4':
                pyautogui.keyDown("down")
                pyautogui.keyUp("down")
                pyautogui.sleep(0.1)
<<<<<<< HEAD
                print("Down")
=======
                print("↓", end="")
>>>>>>> a724b947cc0933cae862cd3d89d925d369e5c07b
            case _:
                print(f"Unknown key: {key}")


def strategem_controller(slotnum):
    """Activate strategem in specified slot"""
    slotnum = int(slotnum)
    if len(strategems_current) < slotnum:
        print(f"Slot {slotnum}: Not enough strategems detected.")
        return
    
    strategem = strategems_current[slotnum - 1]
    print(f"Activating {strategem['name']}")
    strategem_operator(strategem['key'])


def check_hotkey(overlay_window, event):
    """Check for hotkey events"""
    if keyboard.is_pressed(exit_keys):
        print("Exiting...")
        keyboard.unhook_all()
        QApplication.exit()
        return

    if keyboard.is_pressed('ctrl+]'):
        on_screenshot(overlay_window)
        while keyboard.is_pressed('ctrl+]'):
            time.sleep(0.005)
        return

    # Check for quick-access keys (reinforce, resupply, eagle rearm)
    for ckey in allowkeys:
        if keyboard.is_pressed(f'ctrl+{ckey["key"]}'):
            strategem_name = ckey["name"].lower()
            if strategem_name in strategems_by_name:
                strategem_operator(strategems_by_name[strategem_name]['key'])
            return

    # Check for number pad (1-4) to activate strategem slots
    if event.event_type == 'down' and event.scan_code in top_row_keys:
        key = top_row_keys[event.scan_code]
        if keyboard.is_pressed(f'ctrl+{key}'):
            strategem_controller(key)
            QTimer.singleShot(0, functools.partial(overlay_window.stg_Selected, key))
            while keyboard.is_pressed(f'ctrl+{key}'):
                time.sleep(0.005)
            return


print(f"Press {exit_keys} to exit the program.")


def main():
    app = QApplication(sys.argv)
    overlay_window = OverlayWindow()
    overlay_window.show()
    
    keyboard.hook(lambda event: check_hotkey(overlay_window, event))
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
