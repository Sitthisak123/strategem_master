import cv2
import numpy as np


def apply_sobel_feldman(img, ksize=3):
    """
    Apply Sobel-Feldman edge detection.
    
    Args:
        img: Input grayscale image
        ksize: Kernel size (must be odd, default 3)
    
    Returns:
        Sobel edge map
    """
    # Apply Sobel operator in x and y directions
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)
    
    # Compute magnitude of gradients
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    # Normalize to 0-255 range
    magnitude = np.uint8(255 * magnitude / np.max(magnitude)) if np.max(magnitude) > 0 else magnitude
    
    return magnitude


def canny_edge_detection(icon_region, source_img, threshold_low=50, threshold_high=150, confidence_threshold=0.4):
    """
    Detect and find an icon using Canny edge detection combined with Sobel-Feldman method.
    
    Args:
        icon_region: numpy array of the region to search in (grayscale)
        source_img: numpy array of the template image to match (grayscale)
        threshold_low: Lower threshold for Canny edge detection
        threshold_high: Upper threshold for Canny edge detection
        confidence_threshold: Minimum confidence score to consider a match valid
    
    Returns:
        dict: Contains 'match' (bool), 'location' (tuple), 'scale' (float), 'score' (float), 'size' (tuple)
    """
    # --- PREPARE TEMPLATE (ICON TO FIND) ---
    # Apply Sobel-Feldman edge detection
    icon_sobel = apply_sobel_feldman(source_img)
    # Combine with Canny for robust edge detection
    icon_canny = cv2.Canny(source_img, threshold_low, threshold_high)
    icon_edges = cv2.bitwise_or(icon_sobel, icon_canny)
    
    ih, iw = icon_edges.shape

    # --- PREPARE SEARCH IMAGE ---
    # Apply Sobel-Feldman edge detection
    region_sobel = apply_sobel_feldman(icon_region)
    # Combine with Canny for robust edge detection
    region_canny = cv2.Canny(icon_region, threshold_low, threshold_high)
    screen_edges = cv2.bitwise_or(region_sobel, region_canny)

    # --- MULTI-SCALE MATCHING (OPTIMIZED) ---
    best_val = -1
    best_loc = None
    best_scale = None
    best_size = None

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
            
            # Early termination for very good matches
            if best_val > 0.85:
                break

    # --- RETURN RESULT ---
    return {
        'match': best_val >= confidence_threshold,
        'location': best_loc,
        'scale': best_scale,
        'score': best_val,
        'size': best_size
    }
