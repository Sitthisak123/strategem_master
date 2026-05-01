import os
import time
import cv2
import numpy as np

def apply_scharr_operator(img):
    """
    Step 5: ใช้ Scharr Operator แทน Sobel เพื่อหา Edge 
    (มีความไวต่อการเปลี่ยนแปลงของแสงขอบวัตถุมากกว่า Sobel)
    """
    # คำนวณ Gradient แนวนอน (X) และแนวตั้ง (Y)
    scharr_x = cv2.Scharr(img, cv2.CV_64F, 1, 0)
    scharr_y = cv2.Scharr(img, cv2.CV_64F, 0, 1)
    
    # หา Magnitude (ความเข้มของเส้นขอบ)
    magnitude = cv2.magnitude(scharr_x, scharr_y)
    
    # Normalize ให้อยู่ในช่วง 0-255 (uint8)
    if np.max(magnitude) > 0:
        magnitude = np.uint8(255 * magnitude / np.max(magnitude))
    else:
        magnitude = np.uint8(magnitude)
        
    # ใช้ Otsu's Threshold เพื่อให้เส้นขอบที่ได้เป็น Binary (ขาว-ดำ) คมชัด
    _, scharr_binary = cv2.threshold(magnitude, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return scharr_binary


def enhanced_edge_features(img, sigma=0.33):
    """
    ฟังก์ชันหลักที่รวม 5 เทคนิคเพื่อสร้าง Edge Map ที่ทนทานต่อแสงและ Noise
    """
    # ตรวจสอบว่าเป็นภาพสีหรือไม่ ถ้าใช่ให้แปลงเป็น Grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # --- Step 1: Edge-Preserving Filters ---
    # ใช้ Bilateral Filter แทน Gaussian Blur ลบ Noise พื้นหลังแต่ขอบสัญลักษณ์ยังคมกริบ
    filtered = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)

    # --- Step 2: Auto-Tuning Threshold (สมการ Median) ---
    # หาค่าความสว่างตรงกลางของภาพเพื่อตั้งเกณฑ์ Canny แบบไดนามิก
    v = np.median(filtered)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    
    # ป้องกันกรณีที่ภาพสีโทนเดียวกันหมดจน lower/upper ชนกัน
    if upper <= lower:
        lower, upper = 30, 90

    # --- Step 3: ปรับสมการคำนวณภายใน Canny ---
    # apertureSize=5: ให้มองหาขอบภาพในสเกลที่ใหญ่ขึ้น ลดความยิบย่อย
    # L2gradient=True: ใช้ทฤษฎีบทพีทาโกรัสคำนวณ Gradient ให้แม่นยำขึ้น
    canny_edges = cv2.Canny(filtered, lower, upper, apertureSize=5, L2gradient=True)

    # --- Step 5 (ต่อ): ผสาน Canny เข้ากับ Scharr Operator ---
    scharr_edges = apply_scharr_operator(filtered)
    
    # Hybrid Gradient: นำเส้นขอบจากทั้งสองวิธีมารวมกัน (Bitwise OR)
    hybrid_edges = cv2.bitwise_or(canny_edges, scharr_edges)

    # --- Step 4: ซ่อมแซมเส้นขอบด้วย Morphological Operations ---
    # ใช้ Closing เพื่อถมช่องโหว่เล็กๆ ของเส้นขอบที่อาจขาดตอนให้เชื่อมกันเป็นเส้นทึบ
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed_edges = cv2.morphologyEx(hybrid_edges, cv2.MORPH_CLOSE, kernel)

    return closed_edges


def enhanced_canny_template_match(icon_region, template_img, confidence_threshold=0.4):
    """
    ฟังก์ชันสำหรับทำ Template Matching โดยใช้โครงสร้าง Edge แบบใหม่ที่แม่นยำขึ้น
    """
    # แปลงทั้งภาพ Search และภาพ Template ให้เป็นโครงร่างเส้นขอบขั้นสูง
    screen_edges = enhanced_edge_features(icon_region)
    icon_edges = enhanced_edge_features(template_img)
    
    ih, iw = icon_edges.shape

    # --- MULTI-SCALE MATCHING ---
    best_val = -1
    best_loc = None
    best_scale = None
    best_size = None

    # คำนวณสเกลเริ่มต้น
    fit_scale = min(screen_edges.shape[0] / ih, screen_edges.shape[1] / iw)
    min_scale = max(0.25, fit_scale * 0.65)
    max_scale = min(2.5, max(fit_scale * 1.35, min_scale + 0.05))
    scales = np.linspace(min_scale, max_scale, 14)
    
    for scale in scales:
        resized = cv2.resize(
            icon_edges,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_LINEAR
        )

        h, w = resized.shape

        # ข้ามถ้า Template ใหญ่กว่าภาพที่ค้นหา
        if h > screen_edges.shape[0] or w > screen_edges.shape[1]:
            continue

        # ทำ Template Matching
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
            
            # Early termination หากเจอแมตช์ที่คะแนนสูงมาก (ความแม่นยำเกิน 85%)
            if best_val > 0.85:
                break

    return {
        'match': best_val >= confidence_threshold,
        'location': best_loc,
        'scale': best_scale,
        'score': best_val,
        'size': best_size
    }