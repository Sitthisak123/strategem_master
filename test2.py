import cv2
import numpy as np
import pyautogui

# --- Configuration ---
# 'Haystack' is the large image (Screen/Source)
# 'Needle' is the small image (Icon/Template)
silo = "./img/silo.png"
rkcn = "./img/rkcn.png"
eas = "./img/eas.png"
e110 = "./img/e110mm.png"
lc_silo = "./img/lc_silo.png"
lc_rkcn = "./img/lc_rkcn.png"

haystack_path = lc_rkcn
needle_path = rkcn
match_threshold = 0.8

# --- 1. Load Images ---
# Load Haystack in Color (for drawing the rectangle later)
haystack_color = cv2.imread(haystack_path)

# OPTIONAL: Use PyAutoGUI to capture screen instead of loading a file
# screenshot = pyautogui.screenshot()
# haystack_color = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

# Check if Haystack loaded
if haystack_color is None:
    raise Exception(f"Error: Could not load source image from {haystack_path}")

# Convert Haystack to Grayscale for matching logic
haystack_gray = cv2.cvtColor(haystack_color, cv2.COLOR_BGR2GRAY)

# Load Needle (Template) in Grayscale directly
needle_gray = cv2.imread(needle_path, cv2.IMREAD_GRAYSCALE)

# Check if Needle loaded
if needle_gray is None:
    raise Exception(f"Error: Could not load template image from {needle_path}")

# Get dimensions of the needle (Height, Width)
tH, tW = needle_gray.shape[:2]

# --- 2. Perform Matching ---
method = cv2.TM_CCOEFF_NORMED
result = cv2.matchTemplate(haystack_gray, needle_gray, method)

# Get the best match position
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# --- 3. Evaluate Result ---
if max_val >= match_threshold:
    print(f"Match Found! Confidence: {max_val:.4f}")
    
    # Calculate coordinates
    top_left = max_loc
    bottom_right = (top_left[0] + tW, top_left[1] + tH)

    # Draw Rectangle on the Color image
    # Color format is (Blue, Green, Red) -> (0, 255, 0) is Green
    cv2.rectangle(haystack_color, top_left, bottom_right, (0, 255, 0), 2)
    
    # Optional: Add text label
    cv2.putText(haystack_color, f"{max_val:.2f}", (top_left[0], top_left[1] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    print(f"Location: {top_left}")
    
    # Display Result
    cv2.imshow("Match Found", haystack_color)
    
    # Wait for key press to close
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print(f"No match found. Highest confidence was only: {max_val:.4f}")