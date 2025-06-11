import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import time
from matplotlib.path import Path
import os

def fillPoly(mask, verts, color):
    verts = np.array(verts[0])
    y, x = np.mgrid[0:mask.shape[0], 0:mask.shape[1]]
    points = np.vstack((x.flatten(), y.flatten())).T
    path = Path(verts)
    grid = path.contains_points(points)
    mask_flat = mask.flatten()
    mask_flat[grid] = color
    mask[:] = mask_flat.reshape(mask.shape)

def bitwiseAnd(image, mask):
    return image * (mask // 255)

def processing_region(image):
    if len(image.shape) > 2:
        height, width, channels = image.shape
    else:
        height, width = image.shape
        channels = 1

    vertices = np.array([
        [0, height - 1],
        [width / 2, 45],
        [width - 1, height - 1]
    ], dtype=np.int32)

    if channels > 1:
        mask = np.zeros((height, width, channels), dtype=np.uint8)
        temp_mask = np.zeros((height, width), dtype=np.uint8)
        # cv2.fillPoly(temp_mask, [vertices], 255)
        fillPoly(temp_mask, [vertices], 255)
        
        for c in range(channels):
            mask[:,:,c] = temp_mask
    else:
        mask = np.zeros((height, width), dtype=np.uint8)
        # cv2.fillPoly(mask, [vertices], 255)
        fillPoly(mask, [vertices], 255)

    # masked_image = cv2.bitwise_and(image, mask)
    masked_image = bitwiseAnd(image, mask)
    return masked_image

def bgrToGray(region):
    if len(region.shape) == 3:
        return np.dot(region[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
    else:
        return region

def inRange(src, low, up):
    if len(src.shape) == 2:
        mask = np.zeros_like(src, dtype=np.uint8)
        mask[(src >= low) & (src <= up)] = 255
        return mask
    else:
        mask = np.ones(src.shape[:2], dtype=np.uint8) * 255
        for c in range(src.shape[2]):
            mask = mask & ((src[:,:,c] >= low if isinstance(low, int) else low[c]) & 
                           (src[:,:,c] <= up if isinstance(up, int) else up[c]))
        return mask.astype(np.uint8) * 255

def drawLine(img, pt1, pt2, col, thicc):
    #Bresenham (SSSLLLLLLLLOOOOOOOOOWWWWWWWWw)
    x1, y1 = pt1
    x2, y2 = pt2
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    x, y = x1, y1
    sx = 1 if x2 > x1 else -1
    sy = 1 if y2 > y1 else -1
    if dx > dy:
        err = dx / 2.0
        while x != x2:
            for t in range(-thicc//2, thicc//2 + 1):
                for s in range(-thicc//2, thicc//2 + 1):
                    xi = x + t
                    yi = y + s
                    if 0 <= xi < img.shape[1] and 0 <= yi < img.shape[0]:
                        img[yi, xi] = col
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y2:
            for t in range(-thicc//2, thicc//2 + 1):
                for s in range(-thicc//2, thicc//2 + 1):
                    xi = x + t
                    yi = y + s
                    if 0 <= xi < img.shape[1] and 0 <= yi < img.shape[0]:
                        img[yi, xi] = col
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy

    for t in range(-thicc//2, thicc//2 + 1):
        for s in range(-thicc//2, thicc//2 + 1):
            xi = x2 + t
            yi = y2 + s
            if 0 <= xi < img.shape[1] and 0 <= yi < img.shape[0]:
                img[yi, xi] = col

def HoughLinesP(image, rho, theta, threshold, lines=None, minLineLength=0, maxLineGap=0):
    y_idxs, x_idxs = np.nonzero(image)
    height, width = image.shape
    max_dist = int(np.hypot(height, width))
    thetas = np.arange(0, np.pi, theta)
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)
    accumulator = np.zeros((2 * max_dist, num_thetas), dtype=np.uint64)

    for x, y in zip(x_idxs, y_idxs):
        for t_idx in range(num_thetas):
            rho_val = int(round(x * cos_t[t_idx] + y * sin_t[t_idx])) + max_dist
            accumulator[rho_val, t_idx] += 1

    peaks = np.argwhere(accumulator > threshold)
    detected_lines = []

    for rho_idx, theta_idx in peaks:
        rho_val = rho_idx - max_dist
        theta_val = thetas[theta_idx]
        points = []
        for x, y in zip(x_idxs, y_idxs):
            r = int(round(x * np.cos(theta_val) + y * np.sin(theta_val)))
            if abs(r - rho_val) <= 1:
                points.append((x, y))
        if len(points) < 2:
            continue
        points = sorted(points, key=lambda p: (p[0], p[1]))
        seg_start = points[0]
        seg_end = points[0]
        for i in range(1, len(points)):
            if np.hypot(points[i][0] - seg_end[0], points[i][1] - seg_end[1]) <= maxLineGap + 1:
                seg_end = points[i]
            else:
                if np.hypot(seg_end[0] - seg_start[0], seg_end[1] - seg_start[1]) >= minLineLength:
                    detected_lines.append([[seg_start[0], seg_start[1], seg_end[0], seg_end[1]]])
                seg_start = points[i]
                seg_end = points[i]
        if np.hypot(seg_end[0] - seg_start[0], seg_end[1] - seg_start[1]) >= minLineLength:
            detected_lines.append([[seg_start[0], seg_start[1], seg_end[0], seg_end[1]]])

    return np.array(detected_lines, dtype=np.int32) if detected_lines else None

def process_image(image):
    region = processing_region(image)
    gray = bgrToGray(region)
    mask_white = inRange(gray, 200, 255)
    image_masked = bitwiseAnd(gray, mask_white)
    lines = HoughLinesP(image_masked, 2, np.pi/180, 100, None, minLineLength=40, maxLineGap=5)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            drawLine(image_masked, (x1, y1), (x2, y2), 255, 5)
    return image_masked

img_dir = "./car_pictures/320_180/"
img_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith(".png")]

for image_path in img_files:
    img = mpimg.imread(image_path)
    if img.shape[-1] == 4:
        img = (img[..., :3] * 255).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    img = img[..., ::-1]
    plt.imshow(process_image(img))
    plt.show()
