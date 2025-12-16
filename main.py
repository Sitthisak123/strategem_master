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
from icon_regions_overlay import IconRegionsOverlay
from src.utils.cannyEdgeImplement import cannyEdgeDetection
import functools

# Setup directories and CSV
IMG_DIR = "./img"
CSV_FILE = "./src/strategems.csv"
MATCH_THRESHOLD = 0.4

# Icon size constraints (in pixels)
MIN_ICON_SIZE = 30
MAX_ICON_SIZE = 150
ICON_PADDING = 20  # Padding in pixels around detected icons

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
    """Load strategems from CSV file and return as dict indexed by index and name"""
    strategems = {}
    strategems_by_name = {}
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['Index'] and row['Name']:
                    idx = row['Index']
                    name_lower = row['Name'].lower()
                    strategem_entry = {
                        'index': idx,
                        'name': row['Name'],
                        'code': row['Code'],
                        'key': row['Code']  # Code is the key sequence
                    }
                    strategems[idx] = strategem_entry
                    strategems_by_name[name_lower] = strategem_entry
        return strategems, strategems_by_name
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


def preprocess_for_contours(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    return edges


def detect_strategem_icons(hud_img):
    """
    Detect square-like icon regions and return list of tuples:
    (y, x, w, h, icon_region_gray)
    """
    edges = preprocess_for_contours(hud_img)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        aspect = w / float(h) if h != 0 else 0
        area = w * h
        # Apply size constraints and ensure icon is on left HUD area
        if (0.7 < aspect < 1.3 and area > 1800 and x < hud_img.shape[1] * 0.3 and
                MIN_ICON_SIZE <= w <= MAX_ICON_SIZE and MIN_ICON_SIZE <= h <= MAX_ICON_SIZE):
            # Apply padding with bounds checking
            y_start = max(0, y - ICON_PADDING)
            x_start = max(0, x - ICON_PADDING)
            y_end = min(hud_img.shape[0], y + h + ICON_PADDING)
            x_end = min(hud_img.shape[1], x + w + ICON_PADDING)
            
            # crop icon region with padding and store grayscale version
            icon_region = hud_img[y_start:y_end, x_start:x_end]
            if icon_region.size == 0:
                continue
            icon_gray = cv2.cvtColor(icon_region, cv2.COLOR_BGR2GRAY)
            boxes.append((y_start, x_start, x_end - x_start, y_end - y_start, icon_gray))
    # sort by vertical position (y)
    return sorted(boxes, key=lambda b: b[0])


def get_sorted_images(img_dir):
    """Get all image files from directory sorted by filename"""
    if not os.path.exists(img_dir):
        return []
    
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    images = [f for f in os.listdir(img_dir) 
              if os.path.splitext(f)[1].lower() in image_extensions]
    return sorted(images)


def match_template_image(icon_region_gray, needle_gray, method=cv2.TM_CCOEFF_NORMED):
    """
    Match template image within a detected icon region.
    
    Args:
        icon_region_gray: grayscale image of detected icon region
        needle_gray: grayscale template image to match
        method: matching method (default: cv2.TM_CCOEFF_NORMED)
    
    Returns:
        tuple: (position, confidence) or (None, 0.0) if no match
    """
    # Check if template is larger than icon region
    if needle_gray.shape[0] > icon_region_gray.shape[0] or \
       needle_gray.shape[1] > icon_region_gray.shape[1]:
        return None, 0.0
    
    try:
        result = cv2.matchTemplate(icon_region_gray, needle_gray, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        return max_loc, max_val
    except cv2.error as e:
        return None, 0.0

# Duplicate helper definitions removed (single implementations exist above)

def run_cannyEdgeDetection(screenshot):
    """
    Use template matching to detect strategems from screenshot.
    1. Detect icon regions using contour detection
    2. Match templates within each detected icon region
    """
    strategems = []
    
    # Extract HUD area from screenshot
    screenshot_hud = screenshot[30:, 30:600]
    screenshot_gray = cv2.cvtColor(screenshot_hud, cv2.COLOR_BGR2GRAY)
    
    # Detect icon regions (coarse detection)
    icon_boxes = detect_strategem_icons(screenshot_hud)
    
    if not icon_boxes:
        print("No icon regions detected")
        return strategems
    
    # Get sorted image files from ./img
    image_files = get_sorted_images(IMG_DIR)
    
    if not image_files:
        print(f"No images found in {IMG_DIR}")
        return strategems
    
    # Track matched images to avoid duplicates
    matched_indices = set()
    
    # For each detected icon region, try to match templates
    for box_idx, box in enumerate(icon_boxes):
        # box is (y, x, w, h, icon_gray)
        try:
            y, x, w, h, icon_region = box
        except ValueError:
            # fallback for older format
            y, x, w, h = box
            icon_region = screenshot_gray[y:y+h, x:x+w]
        
        best_match = None
        best_confidence = MATCH_THRESHOLD
        best_strategem = None
        
        # Try to match each template image
        for img_file in image_files:
            img_path = os.path.join(IMG_DIR, img_file)
            
            # Load source image
            source_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if source_img is None:
                continue
            
            # Skip if template is larger than icon region
            # if source_img.shape[0] > icon_region.shape[0] or \
            #     source_img.shape[1] > icon_region.shape[1]:
            #     continue
            
            # Try to match this image in the icon region
            # top_left, confidence = match_template_image(icon_region, source_img)
            confidence = cannyEdgeDetection(icon_region, source_img)['score']
            print(f"Box {box_idx+1}, trying {img_file}: confidence={confidence:.4f}")
            if confidence > best_confidence:
                # Extract filename without extension to get index
                filename_base = os.path.splitext(img_file)[0]
                try:
                    img_index = filename_base
                    
                    # Look up strategem by index
                    if img_index in strategems_all:
                        strategem_info = strategems_all[img_index]
                        if strategem_info['name'].lower() not in strategem_default_slots:
                            best_match = img_index
                            best_confidence = confidence
                            best_strategem = strategem_info
                except Exception as e:
                    continue
        
        # Add the best match if found and not already matched
        if best_match and best_match not in matched_indices:
            matched_indices.add(best_match)
            strategem_info_copy = best_strategem.copy()
            strategem_info_copy['confidence'] = best_confidence
            strategems.append(strategem_info_copy)
            print(f"Match: {best_strategem['name']} (Index: {best_match}, Confidence: {best_confidence:.4f})")
    
    # Limit to 4 strategems (top 4 by order)
    if len(strategems) > 4:
        strategems = strategems[-4:]
    
    return strategems


strategems_current = []


def on_screenshot(overlay_window):
    global strategems_current
    # Show loading state
    QTimer.singleShot(0, functools.partial(overlay_window.update_labels, loading=True))
    QApplication.processEvents()
    
    # Take screenshotUpdating labels
    screenshot = pyautogui.screenshot()
    frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    
    # Run template matching detection
    stg_inSlot = run_cannyEdgeDetection(frame)
    strategems_current = stg_inSlot
    
    # Update overlay with detected strategems
    QTimer.singleShot(0, functools.partial(overlay_window.update_labels, stg_inSlot))
    print(f"Detected {len(strategems_current)} strategems")


def strategem_operator(key_sequence):
    """Execute key sequence for strategem input"""
    if not key_sequence:
        print("No key sequence provided.")
        return
    print(f"\tExecuting key sequence: {key_sequence}")
    for key in str(key_sequence):
        match key:
            case '1':
                pyautogui.keyDown("left")
                pyautogui.keyUp("left")
                print("←", end="")
            case '2':
                pyautogui.keyDown("up")
                pyautogui.keyUp("up")
                print("↑", end="")
            case '3':
                pyautogui.keyDown("right")
                pyautogui.keyUp("right")
                print("→", end="")
            case '4':
                pyautogui.keyDown("down")
                pyautogui.keyUp("down")
                print("↓", end="")
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


def check_hotkey(overlay_window, icon_overlay, event):
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

    # Show icon regions overlay when '[' key is pressed
    if keyboard.is_pressed('['):
        # capture and detect
        screenshot = pyautogui.screenshot()
        frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        screenshot_hud = frame[30:800, 30:600]
        icon_boxes = detect_strategem_icons(screenshot_hud)
        # convert to (y,x,w,h) for overlay display
        simple_boxes = [(y, x, w, h) for (y, x, w, h, *rest) in icon_boxes]
        # Defer to main Qt thread to avoid "Timers cannot be started from another thread"
        QTimer.singleShot(0, functools.partial(icon_overlay.display_icon_regions, frame, simple_boxes))
        while keyboard.is_pressed('['):
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
    icon_overlay = IconRegionsOverlay()
    icon_overlay.hide()
    
    keyboard.hook(lambda event: check_hotkey(overlay_window, icon_overlay, event))
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()