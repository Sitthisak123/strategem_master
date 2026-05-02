import cv2
import os
from src.utils.strategem_detection import new_edge_features

# หมายเหตุ: เปลี่ยนรหัส 421231 หรือชื่อไฟล์ ให้ตรงกับไฟล์ในโฟลเดอร์ img/ ของคุณจริงๆ
k9_path = "./img/421231.png"        # รหัส K-9 (เปลี่ยนให้ตรงกับที่มี)
rover_path = "./img/421233.png"     # รหัส Rover 

os.makedirs("./output/debug", exist_ok=True)

if os.path.exists(rover_path):
    img_rover = cv2.imread(rover_path)
    cv2.imwrite("./output/debug/TEMPLATE_EDGE_Rover.png", new_edge_features(img_rover))
    print("Saved Rover Edge Template!")
else:
    print(f"Not found: {rover_path}")

if os.path.exists(k9_path):
    img_k9 = cv2.imread(k9_path)
    cv2.imwrite("./output/debug/TEMPLATE_EDGE_K9.png", new_edge_features(img_k9))
    print("Saved K-9 Edge Template!")
else:
    print(f"Not found: {k9_path}")