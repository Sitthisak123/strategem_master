import sys
import cv2
import numpy as np
import pytesseract
import keyboard
import pyautogui
from difflib import get_close_matches
import time
# from pynput import keyboard as keyboard_pynput
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer
from overlay_window import OverlayWindow

# Setup pytesseract (change path if needed)
pytesseract.pytesseract.tesseract_cmd = r"C:\My Programs\Tesseract-OCR\tesseract.exe"

# hotkeys
reinforce_keys = {
    "name": "reinforce",
    "key": "g"
}
supply_keys = {
    "name": "resupply",
    "key": "v"
}
numpad_keys = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
allowkeys = [reinforce_keys, supply_keys]

strategems_all = {
    "mg-43 machine gun": {
        "id": 2,
        "key": 41423,
        "countdown": "480 sec"
    },
    "apw-1 anti-materiel rifle": {
        "id": 3,
        "key": 41324,
        "countdown": "480 sec"
    },
    "m-105 stalwart": {
        "id": 4,
        "key": 414221,
        "countdown": "480 sec"
    },
    "eat-17 expendable anti-tank": {
        "id": 5,
        "key": 44123,
        "countdown": "70 sec"
    },
    "gr-8 recoilless rifle": {
        "id": 6,
        "key": 41331,
        "countdown": "480 sec"
    },
    "flam-40 flamethrower": {
        "id": 7,
        "key": 41242,
        "countdown": "480 sec"
    },
    "ac-8 autocannon": {
        "id": 8,
        "key": 414223,
        "countdown": "480 sec"
    },
    "mg-206 heavy machine gun": {
        "id": 9,
        "key": 41244,
        "countdown": "480 sec"
    },
    "rl-77 airburst rocket launcher": {
        "id": 10,
        "key": 42213,
        "countdown": "480 sec"
    },
    "mls-4x commando": {
        "id": 11,
        "key": 41243,
        "countdown": "120 sec"
    },
    "rs-422 railgun": {
        "id": 12,
        "key": 434213,
        "countdown": "480 sec"
    },
    "faf-14 spear": {
        "id": 13,
        "key": 44244,
        "countdown": "480 sec"
    },
    "sta-x3 w.a.s.p. launcher": {
        "id": 14,
        "key": 44243,
        "countdown": "480 sec"
    },
    "orbital gatling barrage": {
        "id": 15,
        "key": 34122,
        "countdown": "70 sec"
    },
    "orbital airburst strike": {
        "id": 16,
        "key": 333,
        "countdown": "100 sec"
    },
    "orbital 120mm he barrage": {
        "id": 17,
        "key": 334134,
        "countdown": "180 sec"
    },
    "orbital 380mm he barrage": {
        "id": 18,
        "key": 3422144,
        "countdown": "240 sec"
    },
    "orbital walking barrage": {
        "id": 19,
        "key": 343434,
        "countdown": "240 sec"
    },
    "orbital laser": {
        "id": 20,
        "key": 34234,
        "countdown": "300 sec"
    },
    "orbital napalm barrage": {
        "id": 21,
        "key": 334132,
        "countdown": "240 sec"
    },
    "orbital railcannon strike": {
        "id": 22,
        "key": 32443,
        "countdown": "210 sec"
    },
    "eagle strafing run": {
        "id": 23,
        "key": 233,
        "countdown": "8 sec"
    },
    "eagle airstrike": {
        "id": 24,
        "key": 2343,
        "countdown": "8 sec"
    },
    "eagle cluster bomb": {
        "id": 25,
        "key": 23443,
        "countdown": "8 sec"
    },
    "eagle napalm airstrike": {
        "id": 26,
        "key": 2342,
        "countdown": "8 sec"
    },
    "lift-850 jump pack": {
        "id": 27,
        "key": 42242,
        "countdown": "480 sec"
    },
    "eagle smoke strike": {
        "id": 28,
        "key": 2324,
        "countdown": "8 sec"
    },
    "eagle 110mm rocket pods": {
        "id": 29,
        "key": 2321,
        "countdown": "8 sec"
    },
    "eagle 500kg bomb": {
        "id": 30,
        "key": 23444,
        "countdown": "8 sec"
    },
    "m-102 fast recon vehicle": {
        "id": 31,
        "key": 1434342,
        "countdown": "480 sec"
    },
    "orbital precision strike": {
        "id": 32,
        "key": 332,
        "countdown": "90 sec"
    },
    "orbital gas strike": {
        "id": 33,
        "key": 3343,
        "countdown": "75 sec"
    },
    "orbital ems strike": {
        "id": 34,
        "key": 3314,
        "countdown": "75 sec"
    },
    "orbital smoke strike": {
        "id": 35,
        "key": 3342,
        "countdown": "100 sec"
    },
    "e/mg-101 hmg emplacement": {
        "id": 36,
        "key": 421331,
        "countdown": "180 sec"
    },
    "fx-12 shield generator relay": {
        "id": 37,
        "key": 441313,
        "countdown": "90 sec"
    },
    "a/arc-3 tesla tower": {
        "id": 38,
        "key": 423213,
        "countdown": "120 sec"
    },
    "e/gl-21 grenadier battlement": {
        "id": 39,
        "key": 43413,
        "countdown": "120 sec"
    },
    "md-6 anti-personnel minefield": {
        "id": 40,
        "key": 4123,
        "countdown": "120 sec"
    },
    "b-1 supply pack": {
        "id": 41,
        "key": 414224,
        "countdown": "480 sec"
    },
    "gl-21 grenade launcher": {
        "id": 42,
        "key": 41214,
        "countdown": "480 sec"
    },
    "las-98 laser cannon": {
        "id": 43,
        "key": 41421,
        "countdown": "480 sec"
    },
    "md-i4 incendiary mines": {
        "id": 44,
        "key": 4114,
        "countdown": "120 sec"
    },
    "ax/las-5 \"guard dog\" rover": {
        "id": 45,
        "key": 421233,
        "countdown": "480 sec"
    },
    "sh-20 ballistic shield backpack": {
        "id": 46,
        "key": 414421,
        "countdown": "300 sec"
    },
    "arc-3 arc thrower": {
        "id": 47,
        "key": 434211,
        "countdown": "480 sec"
    },
    "md-17 anti-tank mines": {
        "id": 48,
        "key": 4122,
        "countdown": "120 sec"
    },
    "las-99 quasar cannon": {
        "id": 49,
        "key": 44213,
        "countdown": "480 sec"
    },
    "sh-32 shield generator pack": {
        "id": 50,
        "key": 421313,
        "countdown": "480 sec"
    },
    "md-8 gas mines": {
        "id": 51,
        "key": 4113,
        "countdown": "120 sec"
    },
    "a/mg-43 machine gun sentry": {
        "id": 52,
        "key": 42332,
        "countdown": "90 sec"
    },
    "a/g-16 gatling sentry": {
        "id": 53,
        "key": 4231,
        "countdown": "150 sec"
    },
    "a/m-12 mortar sentry": {
        "id": 54,
        "key": 42334,
        "countdown": "180 sec"
    },
    "ax/ar-23 \"guard dog\"": {
        "id": 55,
        "key": 421234,
        "countdown": "480 sec"
    },
    "a/ac-8 autocannon sentry": {
        "id": 56,
        "key": 423212,
        "countdown": "150 sec"
    },
    "a/mls-4x rocket sentry": {
        "id": 57,
        "key": 42331,
        "countdown": "150 sec"
    },
    "a/m-23 ems mortar sentry": {
        "id": 58,
        "key": 42343,
        "countdown": "180 sec"
    },
    "exo-45 patriot exosuit": {
        "id": 59,
        "key": 1432144,
        "countdown": "600 sec"
    },
    "exo-49 emancipator exosuit": {
        "id": 60,
        "key": 1432142,
        "countdown": "600 sec"
    },
    "tx-41 sterilizer": {
        "id": 61,
        "key": 41241,
        "countdown": "480 sec"
    },
    "ax/tx-13 \"guard dog\" dog breath": {
        "id": 62,
        "key": 421232,
        "countdown": "480 sec"
    },
    "sh-51 directional shield": {
        "id": 63,
        "key": 421322,
        "countdown": "300 sec"
    },
    "e/at-12 anti-tank emplacement": {
        "id": 64,
        "key": 421333,
        "countdown": "180 sec"
    },
    "a/flam-40 flame sentry": {
        "id": 65,
        "key": 423422,
        "countdown": "100 sec"
    },
    "b-100 portable hellbomb": {
        "id": 66,
        "key": 43222,
        "countdown": "300 sec"
    },
    "lift-860 hover pack": {
        "id": 67,
        "key": 422413,
        "countdown": "480 sec"
    },
    "cqc-1 one true flag": {
        "id": 68,
        "key": 41332,
        "countdown": "480 sec"
    },
    "gl-52 de-escalator": {
        "id": 69,
        "key": 43213,
        "countdown": "480 sec"
    },
    "ax/arc-3 \"guard dog\" k-9": {
        "id": 70,
        "key": 421231,
        "countdown": "480 sec"
    },
    "plas-45 epoch": {
        "id": 71,
        "key": 41213,
        "countdown": "480 sec"
    },
    "a/las-98 laser sentry": {
        "id": 72,
        "key": 423423,
        "countdown": "150 sec"
    },
    "lift-182 warp pack": {
        "id": 73,
        "key": 413413,
        "countdown": "480 sec"
    },
    "s-11 speargun": {
        "id": 74,
        "key": 434123,
        "countdown": "480 sec"
    },
    "eat-700 expendable napalm": {
        "id": 75,
        "key": 44121,
        "countdown": "140 sec"
    },
    "ms-11 solo silo": {
        "id": 76,
        "key": 42344,
        "countdown": "180 sec"
    },
    "reinforce": {
        "id": 77,
        "key": 24312,
        "countdown": 0
    },
    "sos beacon": {
        "id": 78,
        "key": 2432,
        "countdown": 0
    },
    "resupply": {
        "id": 79,
        "key": 4423,
        "countdown": 0
    },
    "eagle rearm": {
        "id": 80,
        "key": 22123,
        "countdown": 0
    },
    "sssd delivery": {
        "id": 81,
        "key": 44422,
        "countdown": 0
    },
    "prospecting drill": {
        "id": 82,
        "key": 441344,
        "countdown": 0
    },
    "super earth flag": {
        "id": 83,
        "key": 4242,
        "countdown": 0
    },
    "hellbomb": {
        "id": 84,
        "key": 42142342,
        "countdown": 0
    },
    "upload data": {
        "id": 85,
        "key": 13222,
        "countdown": 0
    },
    "seismic probe": {
        "id": 86,
        "key": 221344,
        "countdown": 0
    },
    "orbital illumination flare": {
        "id": 87,
        "key": 3311,
        "countdown": 0
    },
    "seaf artillery": {
        "id": 88,
        "key": 3224,
        "countdown": 0
    },
    "dark fluid vessel": {
        "id": 89,
        "key": 213422,
        "countdown": 0
    },
    "tectonic drill": {
        "id": 90,
        "key": 242424,
        "countdown": 0
    },
    "hive breaker drill": {
        "id": 91,
        "key": 124344,
        "countdown": 0
    }
}
gap = 0.22
pyautogui.PAUSE = 0.025


def preprocess_for_contours(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    return edges


def detect_strategem_icons(hud_img):
    edges = preprocess_for_contours(hud_img)
    cnts, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        aspect = w/float(h)
        area = w*h
        if 0.7 < aspect < 1.3 and area > 1800 and x < hud_img.shape[1]*0.3:
            boxes.append((y, x, w, h))
    return sorted(boxes, key=lambda b: b[0])


def extract_text(img):
    cfg = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    return pytesseract.image_to_string(img, config=cfg).strip()


def run_ocr(img):
    hud = img[30:800, 30:600]
    icons = detect_strategem_icons(hud)
    strategems = []
    for (y, x, w, h) in icons:
        row = hud[y:y+h, :]
        right_th = row[:, int(hud.shape[1]*gap):]
        h_right = right_th.shape[0]
        top_roi = right_th[:h_right//2, :]
        name_text = extract_text(top_roi)
        match = get_close_matches(name_text.lower(),
                                  strategems_all.keys(),
                                  n=1,
                                  cutoff=0.55)
        best_name = match[0] if match else None
        if best_name:
            print(
                f"OCR: '{name_text}' → {best_name} ({strategems_all[best_name]['key']})")
            strategemsEntry = strategems_all[best_name]
            strategemsEntry["name"] = best_name
            strategems.append(strategemsEntry)
    return strategems


strategems_current = []


def on_screenshot():
    global strategems_current
    screenshot = pyautogui.screenshot()
    frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    strategems_current = run_ocr(frame)
    print(strategems_current)
    # print("Detected:", strategems_current)


def strategem_operator(key_sequence):
    print(f"Executing key sequence: {key_sequence}")
    for key in str(key_sequence):
        match key:
            case '1':
                pyautogui.keyDown("left")
                pyautogui.keyUp("left")
                print("Left")
            case '2':
                pyautogui.keyDown("up")
                pyautogui.keyUp("up")
                print("Up")
            case '3':
                pyautogui.keyDown("right")
                pyautogui.keyUp("right")
                print("Right")
            case '4':
                pyautogui.keyDown("down")
                pyautogui.keyUp("down")
                print("Down")
            case _:  # fallback
                print(f"Unknown key: {key}")


def strategem_controller(num):
    num = int(num)
    print(f"Activating strategem slot {num}")
    if not strategems_current and num not in range(1, 5):
        print("No strategems detected or invalid slot number.")
        return
    match num:
        case 1:  # slot 1
            strategem_operator(strategems_current[-4]["key"])
            pass
        case 2:  # slot 2
            strategem_operator(strategems_current[-3]["key"])
            pass
        case 3:  # slot 3
            strategem_operator(strategems_current[-2]["key"])
            pass
        case 4:  # slot 4
            strategem_operator(strategems_current[-1]["key"])
            pass
        case 5:  # spare
            strategem_operator(strategems_current[-5]["key"])
            pass
        case 6:  # spare
            # strategem_operator(strategems_all["resupply"]["key"])
            pass
        case 7:  # spare
            # strategem_operator(strategems_current[2])
            pass
        case 8:  # spare
            pass
        case 9:  # spare
            pass
        case 0:  # spare
            pass


# def stg_supply():
#     print("Supply calling...")
#     strategem_operator(strategems_all["supply"]["key"])


# def stg_reinforce():
#     print("Reinforce calling...")
#     strategem_operator(strategems_all["reinforce"]["key"])

# keyboard.add_hotkey("ctrl+]", on_screenshot)
# keyboard.add_hotkey("/", stg_reinforce)


print("Press ESC to quit.")
# keyboard.wait("ctrl+esc")


def check_hotkey(overlay_window):
    # Check if 'Ctrl' is being held down and one of the numpad keys is pressed
    # if keyboard.is_pressed('ctrl'):
    #     overlay_window.toggle_visibility(True)
    # elif not keyboard.is_pressed('ctrl'):
    #     overlay_window.toggle_visibility(False)
    # print("Checking hotkeys...")
    if keyboard.is_pressed('ctrl+esc'):
        print("Exiting...")
        sys.exit(0)
    if keyboard.is_pressed(f'ctrl+{supply_keys["key"]}'): # resupply 
        strategem_operator(strategems_all[supply_keys["name"]]["key"])
    if keyboard.is_pressed(f'ctrl+{reinforce_keys["key"]}'): # reinforce
        strategem_operator(strategems_all[reinforce_keys["name"]]["key"])
    
    if keyboard.is_pressed('ctrl+]'):
        on_screenshot()
        overlay_window.update_labels(strategems_current)
            # print("Waiting for key release... ")
        # print("Key released.")

    for ckey in allowkeys:
        if keyboard.is_pressed(f'ctrl+{ckey["key"]}'):
            strategem_controller(strategems_all[ckey["name"]]["key"])

    for key in numpad_keys:
        if keyboard.is_pressed(f'ctrl+{key}'):  # Check for Ctrl + numpad key
            strategem_controller(key)
            overlay_window.stg_Selected(key)
            while keyboard.is_pressed(f'ctrl+{key}'):
                time.sleep(0.1)
                continue  # Wait until the key is released
                # print("Waiting for key release... ",key)
            # print("Key released.")
            break  # Once a key is pressed, break the loop to prevent multiple triggers


def main():
    app = QApplication(sys.argv)
    overlay_window = OverlayWindow()

    # Create the overlay window
    # Create a QTimer to check for hotkeys every 100ms (0.1 seconds)
    timer = QTimer()
    # Pass overlay_window to check_hotkey
    timer.timeout.connect(lambda: check_hotkey(overlay_window))
    timer.start(20)  # Check every [xx]ms

    overlay_window.show()  # Show the overlay window
    sys.exit(app.exec_())  # Start the event loop


if __name__ == '__main__':
    main()
