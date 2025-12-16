import cv2
import numpy as np
import pyautogui

# --- LOAD ICON (TEMPLATE TO FIND) ---
icon = cv2.imread('./output/192.png', cv2.IMREAD_GRAYSCALE)
assert icon is not None, "Icon not found"

icon_edges = cv2.Canny(icon, 50, 150)
ih, iw = icon_edges.shape

# --- SCREENSHOT (SEARCH IMAGE) ---
screenshot = pyautogui.screenshot()
screen_gray = cv2.imread('./output/19.png', cv2.IMREAD_GRAYSCALE)
screen_color = cv2.imread('./output/19.png', cv2.IMREAD_COLOR)
# screen_gray = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2GRAY)[30:1000, 30:550]
# screen_color = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)[30:1000, 30:550]

screen_edges = cv2.Canny(screen_gray, 50, 150)

# --- MULTI-SCALE MATCHING ---
best_val = -1
best_loc = None
best_scale = None
best_size = None

for scale in np.linspace(0.5, 2.5, 40):
    resized = cv2.resize(
        icon_edges,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_LINEAR
    )

    h, w = resized.shape

    if h > screen_edges.shape[0] or w > screen_edges.shape[1]:
        continue

    result = cv2.matchTemplate(
        screen_edges,
        resized,
        cv2.TM_CCOEFF_NORMED
    )

    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val > best_val:
        best_val = max_val
        best_loc = max_loc
        best_scale = scale
        best_size = (w, h)

# --- CHECK RESULT ---
if best_val < 0.4:
    print("❌ No match found:", best_val)
    exit()

# --- DRAW RESULT ---
top_left = best_loc
bottom_right = (
    top_left[0] + best_size[0],
    top_left[1] + best_size[1]
)

cv2.rectangle(screen_color, top_left, bottom_right, (0, 255, 0), 2)

print("✅ Match found")
print("Location:", top_left)
print("Scale:", round(best_scale, 2))
print("Score:", round(best_val, 3))

cv2.imshow("ICON MATCH", screen_color)
cv2.waitKey(0)
cv2.destroyAllWindows()
