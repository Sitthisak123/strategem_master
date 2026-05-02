import time
import csv
import os
from collections import namedtuple

import cv2
import numpy as np

from src.utils.cannyEdgeImplement import apply_scharr_operator
from src.utils.screen_regions import get_hud_region, scale_pixels

# weights สำหรับการคำนวณ Hybrid Score (ปรับได้ตามความเหมาะสม) sum ต้องเท่ากับ 1.0
HYBRID_EDGE_WEIGHT = 0.40
HYBRID_GRAY_WEIGHT = 0.40
HYBRID_HASH_WEIGHT = 0.20

MATCH_THRESHOLD = 0.5 #this will be overridden by main.py for more strict matching
HYBRID_MATCH_MARGIN = 0.005 #this will be overridden by main.py for more strict matching
PHASH_BITS = 256
MIN_ICON_SIZE = 30
MAX_ICON_SIZE = 100
MIN_ICON_AREA = 1800
ICON_PADDING = 1
DEFAULT_STRATEGEM_NAMES = {
    "resupply",
    "reinforce",
    "sos beacon",
    "eagle rearm",
}

DetectionAssets = namedtuple(
    "DetectionAssets",
    ["strategems_by_code", "templates", "template_features"],
)


def is_default_strategem(strategem):
    return strategem.get("name", "").casefold() in DEFAULT_STRATEGEM_NAMES


def load_detection_assets(csv_file="./src/strategems.csv", img_dir="./img"):
    strategems_by_code = {}
    templates = {}

    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = row["Code"]
            display_name = (row.get("Name") or row.get("OriginalName") or "").strip()
            original_name = (row.get("OriginalName") or display_name).strip()
            strategems_by_code[code] = {
                "index": row["Index"],
                "name": display_name,
                "original_name": original_name,
                "code": code,
                "key": code,
            }

            img_path = os.path.join(img_dir, f"{code}.png")
            if os.path.exists(img_path):
                template_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if template_img is not None:
                    templates[code] = template_img

    return DetectionAssets(
        strategems_by_code=strategems_by_code,
        templates=templates,
        template_features=build_template_features(templates),
    )


def ensure_gray(img):
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def normalize_lightness(img):
    # หากภาพเป็นขาวดำอยู่แล้ว (2D array) ให้คืนค่าเดิมกลับไป
    if img.ndim == 2:
        lightness = img
    else:
        # 1. ดึงความสว่าง (Lightness) จาก LAB color space 
        # (ใช้เพื่อดูรายละเอียดสีขาวด้านใน เช่น ลูกกระสุน)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]

        # 2. ดึงความอิ่มตัวสี (Saturation) จาก HSV color space
        # (ใช้เพื่อดึงกรอบสี แดง/เขียว/ฟ้า ให้หลุดออกมาจากท้องฟ้า/เมฆสีขาว)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        s_channel = hsv[:, :, 1]

        # 3. นำทั้งสองส่วนมารวมกัน (Blend)
        # ให้ความสำคัญกับความสว่าง (70%) และกรอบสี (30%)
        # ทำให้ขอบกรอบแข็งแรงขึ้นในขณะที่รายละเอียดสัญลักษณ์ด้านในไม่หายไป
        blended = cv2.addWeighted(l_channel, 0.7, s_channel, 0.3, 0)
        lightness = blended

    # นำไปปรับ Contrast และลด Noise ตามปกติ
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    balanced = clahe.apply(lightness)
    return cv2.bilateralFilter(balanced, d=7, sigmaColor=60, sigmaSpace=60)


def auto_canny(img, sigma=0.20): 
    median = float(np.median(img))
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    if upper <= lower:
        lower, upper = 30, 90
    return cv2.Canny(img, lower, upper)


def close_edges(edges):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)


def preprocess_contours_old(img):
    gray = ensure_gray(img)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(gray, 50, 150)


def preprocess_contours_new(img):
    # ==========================================
    # 🌟 1. Upscale Resolution (เพิ่มความละเอียด 2 เท่า)
    # ==========================================
    scale_factor = 2.0
    high_res = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    
    # ==========================================
    # 🌟 2. ดึงมิติแสงและสี + รวมร่างแบบสมดุล
    # ==========================================
    lab = cv2.cvtColor(high_res, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(high_res, cv2.COLOR_BGR2HSV)
    
    l_channel = lab[:, :, 0] # มิติความสว่าง (ดีสำหรับสัญลักษณ์ข้างใน)
    s_channel = hsv[:, :, 1] # มิติความสดสี (ดีสำหรับกรอบไอคอน)
    
    # ผสมแสง 80% และสี 20% (ดึงขอบให้ชัดโดยไม่ดึง Noise สีมาเยอะเกิน)
    blended = cv2.addWeighted(l_channel, 0.8, s_channel, 0.2, 0)
    
    # ==========================================
    # 🌟 3. อัด Contrast เบาๆ + ลบ Noise อัจฉริยะ (สำคัญมาก!)
    # ==========================================
    # ลด clipLimit ลงมาเหลือ 1.5 ไม่ให้ขุด Noise ฉากหลังมามากเกินไป
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    balanced = clahe.apply(blended)
    
    # 🔥 ทีเด็ด: Bilateral Filter (เบลอพื้นผิวให้เรียบ แต่รักษาความคมของเส้นขอบไอคอนไว้)
    smoothed = cv2.bilateralFilter(balanced, d=9, sigmaColor=75, sigmaSpace=75)
    
    # ==========================================
    # 🌟 4. หาเส้นขอบ Canny + ถมรอยแหว่ง
    # ==========================================
    # Auto-Canny แบบปรับจูนให้รับกับภาพที่ผ่าน Bilateral Filter แล้ว
    median = float(np.median(smoothed))
    sigma = 0.25
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    if upper <= lower:
        lower, upper = 30, 90
        
    edges = cv2.Canny(smoothed, lower, upper)
    
    # ถมเส้นขอบให้เชื่อมต่อกัน
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # ==========================================
    # 🌟 5. ย่อภาพกลับเป็นขนาดเดิม
    # ==========================================
    final_edges = cv2.resize(closed_edges, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
    _, final_edges = cv2.threshold(final_edges, 50, 255, cv2.THRESH_BINARY)
    
    return final_edges


def old_edge_features(img):
    gray = ensure_gray(img)
    sobel = ensure_uint8(apply_scharr_operator(gray))
    canny = cv2.Canny(gray, 50, 150)
    return cv2.bitwise_or(sobel, canny)


def new_edge_features(img):
    # 1. ปรับแสง UI ก่อน เพื่อไม่ให้ความมืดของฉากหลังมารบกวน
    normalized = normalize_lightness(img)
    
    # 2. ส่งภาพที่ปรับแสงแล้ว ไปทำ Edge Detection สไตล์ PineTools
    return pinetools_laplacian_edge(normalized)

def pinetools_laplacian_edge(img):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    laplacian_kernel = np.array([
        [-1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1],
        [-1, -1, 24, -1, -1],
        [-1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1]
    ], dtype=np.float32)
    
    lap_result = cv2.filter2D(gray, cv2.CV_64F, laplacian_kernel)
    abs_lap = np.absolute(lap_result)
    clipped_lap = np.clip(abs_lap, 0, 255).astype(np.uint8)
    
    # กลับสีเป็นพื้นขาว เส้นขอบดำ
    return cv2.bitwise_not(clipped_lap)

def ensure_uint8(img):
    if img.dtype == np.uint8:
        return img
    return np.uint8(np.clip(img, 0, 255))


# เปลี่ยน hash_size จาก 8 เป็น 16
def image_phash(img, hash_size=16, highfreq_factor=4): 
    gray = normalize_lightness(img)
    size = hash_size * highfreq_factor
    resized = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(resized))
    low_freq = dct[:hash_size, :hash_size]
    median = np.median(low_freq[1:, 1:])

    value = 0
    for bit in low_freq.flatten() > median:
        value = (value << 1) | int(bit)
    return value


def hamming_distance(left, right):
    return int(left ^ right).bit_count()


def build_template_features(templates):
    features = {}
    for code, template in templates.items():
        normalized = normalize_lightness(template)
        features[code] = {
            "gray": template,
            "normalized": normalized,
            "edges": new_edge_features(template),
            "hash": image_phash(template),
        }
    return features


def detect_icon_boxes_old(hud_img):
    return _detect_icon_boxes(
        hud_img=hud_img,
        edges=preprocess_contours_old(hud_img),
        min_icon_size=MIN_ICON_SIZE,
        max_icon_size=MAX_ICON_SIZE,
        min_icon_area=MIN_ICON_AREA,
        icon_padding=ICON_PADDING,
        retrieval_mode=cv2.RETR_EXTERNAL,
        max_x_ratio=0.3,
    )


def detect_icon_boxes_new(hud_img, scale=1.0):
    return _detect_icon_boxes(
        hud_img=hud_img,
        edges=preprocess_contours_new(hud_img),
        min_icon_size=scale_pixels(MIN_ICON_SIZE, scale, minimum=12),
        max_icon_size=scale_pixels(MAX_ICON_SIZE, scale, minimum=13),
        min_icon_area=max(250, int(round(MIN_ICON_AREA * scale * scale))),
        icon_padding=scale_pixels(ICON_PADDING, scale, minimum=1),
        retrieval_mode=cv2.RETR_LIST,
        max_x_ratio=0.16,
    )


def _detect_icon_boxes(
    hud_img,
    edges,
    min_icon_size,
    max_icon_size,
    min_icon_area,
    icon_padding,
    retrieval_mode, # ไม่ได้ใช้แล้ว แต่รับมาเพื่อให้ function signature ไม่พัง
    max_x_ratio,
):
    # 🌟 1. เปลี่ยนโหมดเป็น RETR_TREE เพื่อดึง "ลำดับชั้น (Hierarchy)"
    cnts, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    max_icon_x = hud_img.shape[1] * max_x_ratio

    # เช็คก่อนว่ามีกล่องไหม
    if hierarchy is not None:
        # วนลูปจับคู่พิกัดกล่อง (c) กับสถานะลำดับชั้น (h)
        # โครงสร้าง h คือ [Next, Previous, First_Child, Parent]
        for c, h in zip(cnts, hierarchy[0]):
            x, y, w, box_h = cv2.boundingRect(c)
            aspect = w / float(box_h) if box_h != 0 else 0
            area = w * box_h

            # ข้ามกล่องที่อยู่ผิดโซน (ไม่ได้อยู่ฝั่งซ้ายของจอ)
            if x >= max_icon_x:
                continue

            child_idx = h[2]  # index ของลูกตัวแรก (ถ้ามีค่า != -1 แปลว่านี่คือกล่องแม่)
            parent_idx = h[3] # index ของกล่องแม่ (ถ้ามีค่า != -1 แปลว่านี่คือกล่องลูก)

            # 🌟 2. ลอจิกเจาะกล่อง: เลือกเฉพาะ "กล่องที่ใช่"
            
            # เงื่อนไขกล่องปกติ/กล่องแม่ที่ขนาดเป๊ะ
            is_valid_size = (0.7 < aspect < 1.3) and (min_icon_area < area) and (min_icon_size <= w <= max_icon_size)
            
            # ถ้าขนาดมันได้มาตรฐาน ก็ถือว่าเป็นผู้ท้าชิงได้เลย
            if is_valid_size:
                candidates.append((x, y, w, box_h, area))
            
            # 🌟 3. ทีเด็ด: เจอกล่องยักษ์ที่ขนาดไม่ผ่าน แต่มีกล่องลูกอยู่ข้างใน!
            elif child_idx != -1: 
                # (สมมติว่าเป็นกล่องพุ่มไม้ยักษ์ไซส์ 176)
                # เราไม่เอากล่องแม่ แต่เรา "แอบดู" ว่าลูกของมันล่ะ ขนาดพอดี 67x67 ไหม?
                # หมายเหตุ: ในทางปฏิบัติ OpenCV จะคืนค่าลูกๆ ทั้งหมดมาในลูปนี้อยู่แล้ว 
                # และลูกๆ เหล่านั้นจะถูกจับเข้าเงื่อนไข is_valid_size ด้านบนเองโดยอัตโนมัติ!
                # (แปลว่าเราไม่ต้องเขียนโค้ดมุดเข้าไปเอาลูกเลย RETR_TREE ทำหน้าที่ขุดลูกออกมาให้หมดแล้ว)
                pass 

    expected_area = ((min_icon_size + max_icon_size) / 2.0) ** 2

    # เก็บเฉพาะพิกัดกล่อง โดยใช้ Smart NMS 
    # (ตอนนี้จะได้เฉพาะกล่องลูกขนาดเป๊ะๆ 67x67 มาแข่งกันเท่านั้น กล่องแม่ยักษ์จะโดนคัดออกไปตั้งแต่ด่านแรกแล้ว)
    boxes = []
    for x, y, w, box_h, _area in remove_overlapping_boxes(candidates, expected_area):
        y_start = max(0, y - icon_padding)
        x_start = max(0, x - icon_padding)
        y_end = min(hud_img.shape[0], y + box_h + icon_padding)
        x_end = min(hud_img.shape[1], x + w + icon_padding)
        
        boxes.append({
            "y": y_start,
            "x": x_start,
            "w": x_end - x_start,
            "h": y_end - y_start
        })

    # =============== DEBUG LOG: เริ่มต้น ===============
    print(f"\n--- DEBUG: Canny เจอทั้งหมด {len(boxes)} กล่อง ---")
    for i, b in enumerate(boxes):
        print(f"  Box {i+1}: y={b['y']}, x={b['x']}, w={b['w']}, h={b['h']}")

    # ==========================================
    # 🛡️ 3 RULES PIPELINE
    # ==========================================
    if len(boxes) >= 2:
        # --- Rule 1: Similar Size Rule ---
        median_w = np.median([b["w"] for b in boxes])
        median_h = np.median([b["h"] for b in boxes])
        size_tolerance = 0.30  
        
        print(f"\n--- DEBUG: [Rule 1] เทียบขนาด (Median W:{median_w:.1f}, H:{median_h:.1f}) ---")
        size_filtered = []
        for b in boxes:
            w_diff = abs(b["w"] - median_w) / median_w
            h_diff = abs(b["h"] - median_h) / median_h
            if w_diff <= size_tolerance and h_diff <= size_tolerance:
                size_filtered.append(b)
            else:
                print(f"  [ตกอบ Rule 1] กล่องที่ y={b['y']} ขนาดเพี้ยน (w_diff={w_diff:.2f}, h_diff={h_diff:.2f})")
        boxes = size_filtered

    if len(boxes) >= 2:
        # --- Rule 2: Alignment Rule ---
        median_left = np.median([b["x"] for b in boxes])
        median_right = np.median([b["x"] + b["w"] for b in boxes])
        alignment_tolerance = 12

        print(f"\n--- DEBUG: [Rule 2] เทียบแนวตั้ง (Median Left:{median_left:.1f}, Right:{median_right:.1f}) ---")
        aligned_boxes = []
        for b in boxes:
            is_left_aligned = abs(b["x"] - median_left) <= alignment_tolerance
            is_right_aligned = abs((b["x"] + b["w"]) - median_right) <= alignment_tolerance
            if is_left_aligned or is_right_aligned:
                aligned_boxes.append(b)
            else:
                print(f"  [ตกขอบ Rule 2] กล่องที่ y={b['y']} เบี้ยว! (x={b['x']}, right={b['x']+b['w']})")
        boxes = aligned_boxes

    if len(boxes) >= 2:
        # --- Rule 3: Vertical Gap & Cluster Rule ---
        boxes = sorted(boxes, key=lambda b: b["y"])
        median_h = np.median([b["h"] for b in boxes])
        max_gap = median_h * 2.5 
        
        print(f"\n--- DEBUG: [Rule 3] แบ่งกลุ่ม (Max Gap:{max_gap:.1f}) ---")
        clusters = []
        current_cluster = [boxes[0]]
        
        for i in range(1, len(boxes)):
            dist = boxes[i]["y"] - boxes[i-1]["y"]
            if dist <= max_gap:
                current_cluster.append(boxes[i])
            else:
                clusters.append(current_cluster)
                print(f"  [โดนหั่นกลุ่ม] ช่องว่างระหว่าง y={boxes[i-1]['y']} กับ y={boxes[i]['y']} คือ {dist}")
                current_cluster = [boxes[i]]
        clusters.append(current_cluster)
        
        best_cluster = max(clusters, key=len)
        print(f"  [สรุป] มีทั้งหมด {len(clusters)} กลุ่ม เลือกกลุ่มที่มีขนาด {len(best_cluster)} กล่อง")
        boxes = best_cluster

    # ==========================================
    # 🌟 อัปเกรดท่าไม้ตาย: จัดทรงกล่อง + จัดแถว Y (Grid Alignment)
    # ==========================================
    boxes = sorted(boxes, key=lambda b: b["y"]) # เรียงจากบนลงล่างก่อน
    
    final_median_x = int(np.median([b["x"] for b in boxes]))
    # final_median_w = int(np.median([b["w"] for b in boxes]))
    final_median_h = int(np.median([b["h"] for b in boxes]))
    
    # คำนวณระยะห่างระหว่างกล่อง (Step) ที่ควรจะเป็น
    if len(boxes) >= 2:
        gaps = [boxes[i]["y"] - boxes[i-1]["y"] for i in range(1, len(boxes))]
        median_step = int(np.median(gaps))
        
        # ปรับตำแหน่ง Y ของทุกกล่องให้ห่างเท่าๆ กัน โดยยึดกล่องแรกเป็นหลัก
        start_y = boxes[0]["y"]
        for i, b in enumerate(boxes):
            b["y"] = start_y + (i * median_step)
            b["x"] = final_median_x
            b["w"] = final_median_h
            b["h"] = final_median_h
        print(f"\n--- DEBUG: จัดแถวใหม่ Step Y:{median_step}, X:{final_median_x}, W:{final_median_h}, H:{final_median_h} ---")
    else:
        # กรณีมีกล่องเดียว แค่ดัดขนาดเฉยๆ
        for b in boxes:
            b["x"] = final_median_x
            b["w"] = final_median_h
            b["h"] = final_median_h

    # ==========================================
    final_result = []
    for b in boxes:
        icon_region = hud_img[b["y"]:b["y"]+b["h"], b["x"]:b["x"]+b["w"]]
        if icon_region.size > 0:
            final_result.append((b["y"], b["x"], b["w"], b["h"], ensure_gray(icon_region)))

    return sorted(final_result, key=lambda i: i[0])


def remove_overlapping_boxes(candidates, expected_area, overlap_threshold=0.45):
    selected = []
    for candidate in sorted(candidates, key=lambda item: abs(item[4] - expected_area)):
        if any(overlap_ratio(candidate, existing) > overlap_threshold for existing in selected):
            continue
        selected.append(candidate)
    return selected


def overlap_ratio(left_box, right_box):
    lx, ly, lw, lh, _left_area = left_box
    rx, ry, rw, rh, _right_area = right_box

    inter_left = max(lx, rx)
    inter_top = max(ly, ry)
    inter_right = min(lx + lw, rx + rw)
    inter_bottom = min(ly + lh, ry + rh)

    inter_w = max(0, inter_right - inter_left)
    inter_h = max(0, inter_bottom - inter_top)
    inter_area = inter_w * inter_h
    smaller_area = min(lw * lh, rw * rh)

    if smaller_area == 0:
        return 0.0
    return inter_area / smaller_area


def match_edges(search_edges, template_edges, scales):
    best_val = -1.0
    best_loc = None
    best_scale = None
    best_size = None

    for scale in scales:
        resized = cv2.resize(
            template_edges,
            None,
            fx=float(scale),
            fy=float(scale),
            interpolation=cv2.INTER_LINEAR,
        )

        h, w = resized.shape
        if h > search_edges.shape[0] or w > search_edges.shape[1]:
            continue

        result = cv2.matchTemplate(search_edges, resized, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if not np.isfinite(max_val):
            continue

        if max_val > best_val:
            best_val = float(max_val)
            best_loc = max_loc
            best_scale = float(scale)
            best_size = (w, h)

            if best_val > 0.85:
                break

    return {
        "location": best_loc,
        "scale": best_scale,
        "score": best_val,
        "size": best_size,
    }


def fit_template_scales(search_img, template_img):
    ih, iw = template_img.shape[:2]
    fit_scale = min(search_img.shape[0] / ih, search_img.shape[1] / iw)
    min_scale = max(0.25, fit_scale * 0.65)
    max_scale = min(2.5, max(fit_scale * 1.35, min_scale + 0.05))
    return np.linspace(min_scale, max_scale, 14)


def old_template_match(icon_region, template):
    search_edges = old_edge_features(icon_region)
    template_edges = old_edge_features(template)
    scales = np.linspace(0.7, 1.8, 12)
    return match_edges(search_edges, template_edges, scales)


def new_template_match(icon_region, template_feature):
    search_edges = new_edge_features(icon_region)
    template_edges = template_feature["edges"]
    scales = fit_template_scales(search_edges, template_edges)
    return match_edges(search_edges, template_edges, scales)


def build_query_features(icon_region):
    normalized = normalize_lightness(icon_region)
    return {
        "normalized": normalized,
        "edges": new_edge_features(icon_region),
        "hash": image_phash(icon_region),
    }


def hash_similarity(query_hash, template_hash):
    distance = hamming_distance(query_hash, template_hash)
    return max(0.0, 1.0 - (distance / float(PHASH_BITS)))


def clamp_match_score(score):
    return max(0.0, float(score)) if np.isfinite(score) else 0.0


def hybrid_template_match(query_features, template_feature):
    scales = fit_template_scales(query_features["edges"], template_feature["edges"])
    edge_result = match_edges(query_features["edges"], template_feature["edges"], scales)
    gray_result = match_edges(
        query_features["normalized"],
        template_feature["normalized"],
        scales,
    )
    edge_score = clamp_match_score(edge_result["score"])
    gray_score = clamp_match_score(gray_result["score"])
    phash_score = hash_similarity(query_features["hash"], template_feature["hash"])
    score = (
        (edge_score * HYBRID_EDGE_WEIGHT)
        + (gray_score * HYBRID_GRAY_WEIGHT)
        + (phash_score * HYBRID_HASH_WEIGHT)
    )

    return {
        "location": edge_result["location"],
        "scale": edge_result["scale"],
        "score": score,
        "size": edge_result["size"],
        "edge_score": edge_score,
        "gray_score": gray_score,
        "hash_score": phash_score,
    }


def shortlist_by_query_hash(query_hash, template_features, available_codes, limit=15):
    if limit is None or limit <= 0 or limit >= len(available_codes):
        return list(available_codes)

    ranked = sorted(
        available_codes,
        key=lambda code: hamming_distance(query_hash, template_features[code]["hash"]),
    )
    return ranked[:limit]


def shortlist_by_hash(icon_region, template_features, available_codes, limit=15):
    return shortlist_by_query_hash(
        image_phash(icon_region),
        template_features,
        available_codes,
        limit,
    )


def detect_strategems_old(
    screenshot,
    assets,
    match_threshold=MATCH_THRESHOLD,
    include_defaults=False,
    max_results=4,
    skip_first=2,
    verbose=False,
):
    hud_img = screenshot[30:800, 30:600]
    icon_boxes = detect_icon_boxes_old(hud_img)
    return _match_detected_boxes(
        icon_boxes=icon_boxes[skip_first:] if len(icon_boxes) > skip_first else [],
        assets=assets,
        match_threshold=match_threshold,
        include_defaults=include_defaults,
        max_results=max_results,
        mode="old",
        verbose=verbose,
    )


def detect_strategems_new(
    screenshot,
    assets,
    match_threshold=MATCH_THRESHOLD,
    include_defaults=False,
    max_results=4,
    skip_first=2,
    phash_candidates=15,
    verbose=False,
):
    hud_img, hud_region = get_hud_region(screenshot)
    icon_boxes = detect_icon_boxes_new(hud_img, hud_region.scale)
    return _match_detected_boxes(
        icon_boxes=icon_boxes[skip_first:] if len(icon_boxes) > skip_first else [],
        assets=assets,
        match_threshold=match_threshold,
        include_defaults=include_defaults,
        max_results=max_results,
        mode="new",
        phash_candidates=phash_candidates,
        verbose=verbose,
    )


def detect_strategems_hybrid(
    screenshot,
    assets,
    match_threshold=MATCH_THRESHOLD,
    include_defaults=False,
    max_results=4,
    skip_first=2,
    phash_candidates=15,
    hybrid_margin=HYBRID_MATCH_MARGIN,
    verbose=False,
):
    hud_img, hud_region = get_hud_region(screenshot)
    icon_boxes = detect_icon_boxes_new(hud_img, hud_region.scale)
    return _match_detected_boxes(
        icon_boxes=icon_boxes[skip_first:] if len(icon_boxes) > skip_first else [],
        assets=assets,
        match_threshold=match_threshold,
        include_defaults=include_defaults,
        max_results=max_results,
        mode="hybrid",
        phash_candidates=phash_candidates,
        hybrid_margin=hybrid_margin,
        verbose=verbose,
    )


def _match_detected_boxes(
    icon_boxes,
    assets,
    match_threshold,
    include_defaults,
    max_results,
    mode,
    phash_candidates=None,
    hybrid_margin=0.0,
    verbose=False,
):
    detected = []
    matched_codes = set()
    
    # โฟลเดอร์สำหรับเซฟรูประหว่างทำงาน
    debug_dir = "./output/debug"
    os.makedirs(debug_dir, exist_ok=True)

    for idx, (_y, _x, _w, _h, icon_region) in enumerate(icon_boxes, start=1):
        available_codes = [
            code for code in assets.templates
            if code not in matched_codes and code in assets.strategems_by_code
        ]
        query_features = None

        if mode == "hybrid":
            query_features = build_query_features(icon_region)
            available_codes = shortlist_by_query_hash(
                query_features["hash"],
                assets.template_features,
                available_codes,
                limit=phash_candidates,
            )
        elif mode == "new":
            available_codes = shortlist_by_hash(
                icon_region,
                assets.template_features,
                available_codes,
                limit=phash_candidates,
            )

        best_score = match_threshold
        second_score = -1.0
        best_entry = None
        best_extra = {}

        for code in available_codes:
            if mode == "old":
                result = old_template_match(icon_region, assets.templates[code])
            elif mode == "hybrid":
                result = hybrid_template_match(query_features, assets.template_features[code])
            else:
                result = new_template_match(icon_region, assets.template_features[code])

            if result["score"] > best_score:
                second_score = best_score
                best_score = result["score"]
                best_entry = assets.strategems_by_code[code]
                best_extra = result
            elif result["score"] > second_score:
                second_score = result["score"]

        # -------------------------------------------------------------
        # ระบบ Save ภาพแบบ Replace (ลบของเก่าทิ้ง เซฟของใหม่ทับตามหมายเลข Slot)
        # 1. ค้นหาและลบภาพเก่าของ Slot นี้ทิ้งก่อน
        for existing_file in os.listdir(debug_dir):
            if existing_file.startswith(f"slot_{idx}_"):
                try:
                    os.remove(os.path.join(debug_dir, existing_file))
                except Exception:
                    pass

        # 2. เซฟภาพใหม่ด้วยชื่อที่ระบุสถานะชัดเจน
        if not best_entry:
            mismatch_filename = os.path.join(debug_dir, f"slot_{idx}_mismatch.png")
            cv2.imwrite(mismatch_filename, icon_region)
            if verbose:
                print(f"[{mode}] icon {idx}: no match (saved to {mismatch_filename})")
            continue

        margin = best_score - second_score
        if mode == "hybrid" and second_score >= match_threshold and margin < hybrid_margin:
            ambiguous_filename = os.path.join(debug_dir, f"slot_{idx}_ambiguous_{best_entry['code']}.png")
            cv2.imwrite(ambiguous_filename, icon_region)
            if verbose:
                print(
                    f"[{mode}] icon {idx}: ambiguous {best_entry['name']} "
                    f"({best_score:.3f}, margin {margin:.3f}) (saved to {ambiguous_filename})"
                )
            continue

        # กรณี Match สำเร็จ
        match_filename = os.path.join(debug_dir, f"slot_{idx}_match_{best_entry['code']}.png")
        cv2.imwrite(match_filename, icon_region)
        # -------------------------------------------------------------

        matched_codes.add(best_entry["code"])
        if not include_defaults and is_default_strategem(best_entry):
            if verbose:
                print(f"[{mode}] icon {idx}: skipped default {best_entry['name']} ({best_score:.3f})")
            continue

        entry = best_entry.copy()
        entry["confidence"] = best_score
        entry["detector"] = mode
        entry["match_scale"] = best_extra.get("scale")
        if mode == "hybrid":
            entry["edge_score"] = best_extra.get("edge_score")
            entry["gray_score"] = best_extra.get("gray_score")
            entry["hash_score"] = best_extra.get("hash_score")
            entry["match_margin"] = margin
        detected.append(entry)

        if verbose:
            extra = ""
            if mode == "hybrid":
                extra = (
                    f" edge={entry['edge_score']:.3f}"
                    f" gray={entry['gray_score']:.3f}"
                    f" hash={entry['hash_score']:.3f}"
                    f" margin={entry['match_margin']:.3f}"
                )
            print(f"[{mode}] icon {idx}: matched {entry['name']} ({best_score:.3f}){extra}")

    if max_results is None or max_results <= 0:
        return detected
    return detected[-max_results:] if len(detected) > max_results else detected
