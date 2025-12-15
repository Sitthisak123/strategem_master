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
IMG_DIR = "./img"
CSV_FILE = "./src/strategems.csv"
MATCH_THRESHOLD = 0.7

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
    """
    strategems = []
    
    # Extract HUD area from screenshot
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
    stg_inSlot = run_template_matching(frame)
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
                pyautogui.sleep(0.1)
                print("←", end="")
            case '2':
                pyautogui.keyDown("up")
                pyautogui.keyUp("up")
                pyautogui.sleep(0.1)
                print("↑", end="")
            case '3':
                pyautogui.keyDown("right")
                pyautogui.keyUp("right")
                pyautogui.sleep(0.1)
                print("→", end="")
            case '4':
                pyautogui.keyDown("down")
                pyautogui.keyUp("down")
                pyautogui.sleep(0.1)
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
