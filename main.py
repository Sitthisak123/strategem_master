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
from src.utils.strategem_detection import (
    detect_strategems_hybrid,
    load_detection_assets as load_detection_assets_for_matching,
    detect_icon_boxes_new,
)
from src.utils.screen_regions import get_hud_region, scale_pixels
import functools
from threading import Thread
from queue import Queue, Empty as QueueEmpty, Full as QueueFull

# Setup directories and CSV
IMG_DIR = "./img"
CSV_FILE = "./src/strategems.csv"
MATCH_THRESHOLD = 0.5
HYBRID_MATCH_MARGIN = 0.0005
PHASH_CANDIDATES = 15

HOTKEY_DEBOUNCE_DELAY = 0.2  # 200ms debounce delay for hotkeys
last_hotkey_time = {}
pyautogui.PAUSE = 0.03  # Reduced delay between PyAutoGUI actions

# hotkeys
EXIT_KEYS = "ctrl+c"
REINFORCE_KEYS = {
    "name": "reinforce",
    "key": "g",
    "sequence": "24312"
}
SUPPLY_KEYS = {
    "name": "resupply",
    "key": "v",
    "sequence": "4423"
}
EAGLE_REARM_KEYS = {
    "name": "eagle rearm",
    "key": "q",
    "sequence": "22123"
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
SCREENSHOT_QUEUE = Queue(maxsize=2)  # Cache last 2 screenshots
LAST_SCREENSHOT_TIME = 0
SCREENSHOT_CACHE_TIMEOUT = 0.5  # Cache for 500ms 

def load_strategems_and_templates():
    """โหลดข้อมูลจาก CSV และโหลดรูปภาพ Template เข้า Memory"""
    stg_by_code = {}
    stg_by_name = {}
    
    try:
        with open(CSV_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                code = row['Code']
                display_name = (row.get('Name') or row.get('OriginalName') or '').strip()
                original_name = (row.get('OriginalName') or display_name).strip()
                name_lower = display_name.lower()
                entry = {
                    'index': row['Index'],
                    'name': display_name,
                    'original_name': original_name,
                    'code': code,
                    'key': code 
                }
                stg_by_code[code] = entry
                stg_by_name[name_lower] = entry
                if original_name:
                    stg_by_name[original_name.lower()] = entry
                
                # โหลดภาพ Template (ชื่อไฟล์คือ Code.png)
                img_path = os.path.join(IMG_DIR, f"{code}.png")
                if os.path.exists(img_path):
                    template_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if template_img is not None:
                        TEMPLATES[code] = template_img
                        
        print(f"[OK] Loaded {len(stg_by_code)} strategems and {len(TEMPLATES)} templates.")
        return stg_by_code, stg_by_name
    except Exception as e:
        print(f"[!] Error loading data: {e}")
        return {}, {}


strategems_all, strategems_by_name = load_strategems_and_templates()
DETECTION_ASSETS = load_detection_assets_for_matching(CSV_FILE, IMG_DIR)


# Default strategems that appear in default slots
strategem_default_slots = {
    "resupply",
    "reinforce",
    "sos beacon",
    "eagle rearm",
}


def is_default_strategem(strategem):
    return strategem.get("name", "").casefold() in strategem_default_slots


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

def run_enhanced_canny_template_match(screenshot):
    """
    Detect strategems with the hybrid Canny/Scharr + grayscale + pHash matcher.
    """
    if not DETECTION_ASSETS.templates:
        print(f"[DEBUG] No templates loaded. Template count: {len(DETECTION_ASSETS.templates)}")
        return []

    return detect_strategems_hybrid(
        screenshot,
        DETECTION_ASSETS,
        match_threshold=MATCH_THRESHOLD,
        include_defaults=False,
        max_results=4,
        skip_first=2,
        phash_candidates=PHASH_CANDIDATES,
        hybrid_margin=HYBRID_MATCH_MARGIN,
        verbose=True,
    )


strategems_current = []

def screenshot_worker_thread(overlay_window):
    """Background thread for screenshot capture and detection"""
    global strategems_current
    
    while True:
        try:
            frame = SCREENSHOT_QUEUE.get(timeout=0.1)
            if frame is None:
                break
                
            # Run template matching detection
            stg_in_slot = run_enhanced_canny_template_match(frame)
            
            # Update global variable (thread-safe: atomic list replacement)
            strategems_current = stg_in_slot
            
            # Update UI on main thread
            QTimer.singleShot(0, functools.partial(
                overlay_window.update_labels, stg_in_slot))
            
            print(f"\nDetected {len(stg_in_slot)} strategems:")
            for s in stg_in_slot:
                print(f"  - {s['name']}")
        except QueueEmpty:
            # This is normal - queue is empty, wait for next frame
            continue
        except Exception as e:
            print(f"[Worker Thread ERROR] {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

def on_screenshot(overlay_window):
    global strategems_current
    
    # Show loading state
    QTimer.singleShot(0, functools.partial(overlay_window.update_labels, loading=True))
    QApplication.processEvents()
    
    # Capture Helldivers 2 window asynchronously
    frame = capture_helldivers_window()
    if frame is None:
        print("Failed to capture window. Falling back to full screen.")
        screenshot = pyautogui.screenshot()
        frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    
    # Queue for background processing (Update: Keep latest frame)
    try:
        SCREENSHOT_QUEUE.put_nowait(frame)
    except QueueFull:
        try:
            # หากคิวเต็ม ให้ดึงภาพเก่าสุดทิ้งไปก่อน
            SCREENSHOT_QUEUE.get_nowait()
            # แล้วจึงยัดภาพใหม่ล่าสุดเข้าไปแทน
            SCREENSHOT_QUEUE.put_nowait(frame)
        except Exception as e:
            print(f"[Queue Error] Could not manage full queue: {e}")


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
                print("<", end="")
            case '2':
                pyautogui.keyDown("up")
                pyautogui.keyUp("up")
                print("^", end="")
            case '3':
                pyautogui.keyDown("right")
                pyautogui.keyUp("right")
                print(">", end="")
            case '4':
                pyautogui.keyDown("down")
                pyautogui.keyUp("down")
                print("v", end="")
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
        now = time.time()
        # เช็คว่าเลยระยะเวลา Debounce หรือยัง
        if now - last_hotkey_time.get('screenshot', 0) > HOTKEY_DEBOUNCE_DELAY:
            on_screenshot(overlay_window)
            last_hotkey_time['screenshot'] = now
        return True
    return False

def handle_debug_overlay_hotkey(icon_overlay):
    """Handles the debug overlay hotkey."""
    if keyboard.is_pressed('ctrl+['):
        now = time.time()
        if now - last_hotkey_time.get('debug', 0) > HOTKEY_DEBOUNCE_DELAY:
            frame = capture_helldivers_window()
            if frame is None:
                print("Failed to capture window for debug overlay.")
                screenshot = pyautogui.screenshot()
                frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            
            screenshot_hud, hud_region = get_hud_region(frame)
            icon_boxes = detect_icon_boxes_new(screenshot_hud, hud_region.scale)
            simple_boxes = [(y, x, w, h) for (y, x, w, h, *rest) in icon_boxes]
            
            QTimer.singleShot(0, functools.partial(icon_overlay.display_icon_regions, frame, simple_boxes))
            last_hotkey_time['debug'] = now
        return True
    return False

def handle_slot_activation_hotkeys(event, overlay_window):
    """Handles the strategem slot activation hotkeys."""
    if event.event_type == 'down' and event.scan_code in TOP_ROW_KEYS:
        key = TOP_ROW_KEYS[event.scan_code]
        if keyboard.is_pressed(f'ctrl+{key}'):
            now = time.time()
            action_name = f'slot_{key}'
            if now - last_hotkey_time.get(action_name, 0) > HOTKEY_DEBOUNCE_DELAY:
                strategem_controller(key)
                QTimer.singleShot(0, functools.partial(overlay_window.stg_Selected, key))
                last_hotkey_time[action_name] = now
            return True
    return False

def handle_quick_access_hotkeys():
    """Handles the quick-access strategem hotkeys using hardcoded sequences."""
    for ckey in ALLOW_KEYS:
        if keyboard.is_pressed(f'ctrl+{ckey["key"]}'):
            sequence = ckey.get("sequence")
            if sequence:
                print(f"[Hotkey] Activating {ckey['name'].title()} -> {sequence}")
                strategem_operator(sequence)
            else:
                print(f"[Hotkey] Error: No sequence defined for '{ckey['name']}'")
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
    if handle_save_hud_hotkey():
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
    
    print("[OK] All required assets found.")

def handle_save_hud_hotkey():
    """Handles the ctrl+p hotkey to save the transformed HUD image."""
    if keyboard.is_pressed('ctrl+\\'):
        now = time.time()
        if now - last_hotkey_time.get('save_hud', 0) > HOTKEY_DEBOUNCE_DELAY:
            frame = capture_helldivers_window()
            if frame is None:
                screenshot = pyautogui.screenshot()
                frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            
            from src.utils.screen_regions import get_hud_region
            from src.utils.strategem_detection import preprocess_contours_new
            
            screenshot_hud, _ = get_hud_region(frame)
            
            # แปลงภาพ HUD ด้วยวิธีเดียวกับที่ AI ใช้หา Contour เป๊ะๆ
            transformed_hud = preprocess_contours_new(screenshot_hud)
            
            os.makedirs("./output/debug", exist_ok=True)
            save_path = "./output/debug/FINAL_TRANSFORMED_HUD.png"
            cv2.imwrite(save_path, transformed_hud)
            print(f"\n[Debug] ✅ Saved final transformed HUD to: {save_path}")
            
            last_hotkey_time['save_hud'] = now
        return True
    return False

def main():
    # Run asset check first
    check_required_assets()

    app = QApplication(sys.argv)
    overlay_window = OverlayWindow()
    overlay_window.show()
    icon_overlay = IconRegionsOverlay()
    icon_overlay.hide()
    
    # Start background worker thread for detection
    worker_thread = Thread(target=screenshot_worker_thread, args=(overlay_window,), daemon=True)
    worker_thread.start()
    
    keyboard.hook(lambda event: check_hotkey(overlay_window, icon_overlay, event))
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
