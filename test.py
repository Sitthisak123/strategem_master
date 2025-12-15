import cv2
import numpy as np
import pyautogui

# Load the main image and the template
# pyautogui.sleep(2)
img = cv2.imread('./img/silo.png', cv2.IMREAD_GRAYSCALE)
screenshot = pyautogui.screenshot()

template2 = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)[30:,30:550]
template = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2GRAY)[30:,30:550]
assert img is not None, "Source image not found"
assert template is not None, "Template image not found"

# Get the dimensions of the template
tH, tW = template.shape[:2]

# Use a specific matching method (e.g., Normalized Cross-Correlation)
# Other methods include cv2.TM_SQDIFF, cv2.TM_CCOEFF, etc.
method = cv2.TM_CCOEFF_NORMED
result = cv2.matchTemplate(img, template, method)

# Find the location of the best match
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
if(max_val < 0.8):
    print("No match found: ",max_val)
    exit()
# For TM_CCOEFF_NORMED, the highest value is the best match
top_left = max_loc
bottom_right = (top_left[0] + 68, top_left[1] + 68)

# Draw a rectangle around the detected object in the original (color) image
# You would need to reload the image in color or convert the grayscale to BGR if you want a color rectangle
img_color = template2
cv2.rectangle(img_color, top_left, bottom_right, (0, 255, 0), 2)
print(top_left)
print(f"Match location: {top_left}, Match value: {max_val}")
# Display the result (optional)
cv2.imshow("Match Found", img_color)
cv2.waitKey(0)
cv2.destroyAllWindows()
