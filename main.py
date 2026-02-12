import sys
import cv2
import numpy as np
import keyboard
import pyautogui
import pygetwindow as gw
import csv
import os
from pathlib import Path
import time
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer
from overlay_window import OverlayWindow
from icon_regions_overlay import IconRegionsOverlay
from src.utils.cannyEdgeImplement import canny_edge_detection
import functools

# Setup directories and CSV
IMG_DIR = "./img"
CSV_FILE = "./src/strategems.csv"
MATCH_THRESHOLD = 0.4
pyautogui.PAUSE = 0.03  # Reduced delay between PyAutoGUI actions
# Icon size constraints (in pixels)
MIN_ICON_SIZE = 30
MAX_ICON_SIZE = 150
ICON_PADDING = 20  # Padding in pixels around detected icons

# hotkeys
EXIT_KEYS = "ctrl+c"
REINFORCE_KEYS = {
    "name": "reinforce",
    "key": "g"
}
SUPPLY_KEYS = {
    "name": "resupply",
    "key": "v"
}
EAGLE_REARM_KEYS = {
    "name": "eagle rearm",
    "key": "q"
}

TOP_ROW_KEYS = {
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

ALLOW_KEYS = [REINFORCE_KEYS, SUPPLY_KEYS, EAGLE_REARM_KEYS]

TEMPLATES = {} 

def load_strategems_and_templates():
    """โหลดข้อมูลจาก CSV และโหลดรูปภาพ Template เข้า Memory"""
    stg_by_code = {}
    stg_by_name = {}
    
    try:
        with open(CSV_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                code = row['Code']
                name_lower = row['Name'].lower()
                entry = {
                    'index': row['Index'],
                    'name': row['Name'],
                    'code': code,
                    'key': code 
                }
                stg_by_code[code] = entry
                stg_by_name[name_lower] = entry
                
                # โหลดภาพ Template (ชื่อไฟล์คือ Code.png)
                img_path = os.path.join(IMG_DIR, f"{code}.png")
                if os.path.exists(img_path):
                    template_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if template_img is not None:
                        TEMPLATES[code] = template_img
                        
        print(f"[✓] Loaded {len(stg_by_code)} strategems and {len(TEMPLATES)} templates.")
        return stg_by_code, stg_by_name
    except Exception as e:
        print(f"[!] Error loading data: {e}")
        return {}, {}


strategems_all, strategems_by_name = load_strategems_and_templates()


# Default strategems that appear in default slots
strategem_default_slots = [
    "resupply",
    "reinforce",
    "sos beacon",
    "eagle rearm",
]


def capture_helldivers_window():
    """
    Capture screenshot of Helldivers 2 window only.
    Returns BGR format numpy array or None if window not found.
    """
    try:
        # Try to find Helldivers 2 window by title
        helldivers_windows = gw.getWindowsWithTitle('HELLDIVERS™ 2')
        if not helldivers_windows:
            print("Warning: Helldivers 2 window not found. Trying 'helldivers'...")
            helldivers_windows = gw.getWindowsWithTitle('HELLDIVERS™ 2')
        
        if not helldivers_windows:
            print("Error: Helldivers 2 window not found. Using full screen capture.")
            return None
        
        window = helldivers_windows[0]
        
        # Activate window to ensure it's in focus
        window.activate()
        time.sleep(0.1)
        
        # Get window boundaries
        left = max(0, window.left)
        top = max(0, window.top)
        width = window.width
        height = window.height
        
        # Capture window region
        screenshot = pyautogui.screenshot(region=(left, top, width, height))
        frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        return frame
    except Exception as e:
        print(f"Error capturing Helldivers 2 window: {e}")
        return None


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

def run_canny_edge_detection(screenshot):
    """
    Detects strategems from a screenshot by matching detected icon regions
    against pre-loaded templates using Canny edge detection.
    """
    detected_stgs = []
    # Crop the screenshot to the HUD area where strategem icons appear.
    screenshot_hud = screenshot[30:800, 30:600]
    icon_boxes = detect_strategem_icons(screenshot_hud)
    
    if not icon_boxes or not TEMPLATES:
        return detected_stgs

    matched_codes = set()

    # Iterate through each detected icon box.
    for y, x, w, h, icon_region in icon_boxes:
        best_score = MATCH_THRESHOLD
        best_entry = None

        # Compare the icon region against all loaded strategem templates.
        for code, template_img in TEMPLATES.items():
            if code in matched_codes:
                continue
            
            # Use Canny edge detection to get a similarity score.
            res = canny_edge_detection(icon_region, template_img)
            
            if res['score'] > best_score:
                best_score = res['score']
                best_entry = strategems_all[code]

        # If a match is found, add it to the list of detected strategems.
        if best_entry:
            matched_codes.add(best_entry['code'])
            entry_copy = best_entry.copy()
            entry_copy['confidence'] = best_score
            detected_stgs.append(entry_copy)

    # Return the last 4 detected strategems, as that's the max in the game.
    return detected_stgs[-4:] if len(detected_stgs) > 4 else detected_stgs


strategems_current = []


def on_screenshot(overlay_window):
    global strategems_current
    # Show loading state
    QTimer.singleShot(0, functools.partial(overlay_window.update_labels, loading=True))
    QApplication.processEvents()
    
    # Capture Helldivers 2 window
    frame = capture_helldivers_window()
    if frame is None:
        print("Failed to capture window. Falling back to full screen.")
        screenshot = pyautogui.screenshot()
        frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    
    # Run template matching detection
    stg_in_slot = run_canny_edge_detection(frame)
    strategems_current = stg_in_slot
    
    # Update overlay with detected strategems
    QTimer.singleShot(0, functools.partial(overlay_window.update_labels, stg_in_slot))
    print(f"Detected {len(strategems_current)} strategems")


def strategem_operator(key_sequence):
    """Execute key sequence for strategem input"""
    if not key_sequence:
        print("No key sequence provided.")
        return
    print(f"\tExecuting key sequence: {key_sequence}")
    for key in str(key_sequence):
        # time.sleep(0.05)  # Small delay between key presses --> already used [pyautogui.PAUSE = 0.03]
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


def handle_exit_hotkey():
    """Handles the exit hotkey."""
    if keyboard.is_pressed(EXIT_KEYS):
        print("Exiting...")
        keyboard.unhook_all()
        QApplication.exit()
        return True
    return False

def handle_screenshot_hotkey(overlay_window):
    """Handles the manual screenshot hotkey."""
    if keyboard.is_pressed('ctrl+]'):
        on_screenshot(overlay_window)
        while keyboard.is_pressed('ctrl+]'):
            time.sleep(0.005)
        return True
    return False

def handle_debug_overlay_hotkey(icon_overlay):
    """Handles the debug overlay hotkey."""
    if keyboard.is_pressed('['):
        frame = capture_helldivers_window()
        if frame is None:
            print("Failed to capture window for debug overlay.")
            screenshot = pyautogui.screenshot()
            frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        
        screenshot_hud = frame[30:800, 30:600]
        icon_boxes = detect_strategem_icons(screenshot_hud)
        simple_boxes = [(y, x, w, h) for (y, x, w, h, *rest) in icon_boxes]
        
        QTimer.singleShot(0, functools.partial(icon_overlay.display_icon_regions, frame, simple_boxes))
        while keyboard.is_pressed('['):
            time.sleep(0.005)
        return True
    return False

def handle_quick_access_hotkeys():
    """Handles the quick-access strategem hotkeys."""
    for ckey in ALLOW_KEYS:
        if keyboard.is_pressed(f'ctrl+{ckey["key"]}'):
            strategem_name = ckey["name"].lower()
            if strategem_name in strategems_by_name:
                strategem_operator(strategems_by_name[strategem_name]['key'])
            return True
    return False

def handle_slot_activation_hotkeys(event, overlay_window):
    """Handles the strategem slot activation hotkeys."""
    if event.event_type == 'down' and event.scan_code in TOP_ROW_KEYS:
        key = TOP_ROW_KEYS[event.scan_code]
        if keyboard.is_pressed(f'ctrl+{key}'):
            strategem_controller(key)
            QTimer.singleShot(0, functools.partial(overlay_window.stg_Selected, key))
            while keyboard.is_pressed(f'ctrl+{key}'):
                time.sleep(0.005)
            return True
    return False

def check_hotkey(overlay_window, icon_overlay, event):
    """Check for hotkey events"""
    if handle_exit_hotkey():
        return
    if handle_screenshot_hotkey(overlay_window):
        return
    if handle_debug_overlay_hotkey(icon_overlay):
        return
    if handle_quick_access_hotkeys():
        return
    if handle_slot_activation_hotkeys(event, overlay_window):
        return


print(f"Press {EXIT_KEYS} to exit the program.")


def check_required_assets():
    """Checks if essential asset files exist before running the main application."""
    csv_exists = os.path.isfile(CSV_FILE)
    img_dir_exists = os.path.isdir(IMG_DIR)
    
    # Check if img directory has at least one png file
    img_files_exist = False
    if img_dir_exists:
        if any(f.endswith('.png') for f in os.listdir(IMG_DIR)):
            img_files_exist = True

    if not csv_exists or not img_files_exist:
        print("[!] Error: Critical assets are missing.")
        if not csv_exists:
            print(f"  - File not found: {CSV_FILE}")
        if not img_files_exist:
            print(f"  - No icon images (.png) found in: {IMG_DIR}/")
        
        print("\n    Please run the setup pipeline once to generate the required assets:")
        print("    > python src/utils/automation_pipeline.py\n")
        print("    Exiting.")
        sys.exit(1)
    
    print("[✓] All required assets found.")


def main():
    # Run asset check first
    check_required_assets()

    app = QApplication(sys.argv)
    overlay_window = OverlayWindow()
    overlay_window.show()
    icon_overlay = IconRegionsOverlay()
    icon_overlay.hide()
    
    keyboard.hook(lambda event: check_hotkey(overlay_window, icon_overlay, event))
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()