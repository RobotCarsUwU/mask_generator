import cv2
import matplotlib.pyplot as plt
import numpy as np
import time

img = cv2.imread("./car_pictures/320_180/frame_08521.png")
vid = cv2.VideoCapture("./car_videos/320_180.mp4")

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
        cv2.fillPoly(temp_mask, [vertices], 255)
        for c in range(channels):
            mask[:,:,c] = temp_mask
    else:
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [vertices], 255)

    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def process_image(image):
    region = processing_region(image)
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    blured = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blured, 100, 200)
    img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask_white = cv2.inRange(gray, 200, 255)
    image_masked = cv2.bitwise_and(gray, mask_white)
    lines = cv2.HoughLinesP(image_masked, 2, np.pi/180, 100, None, minLineLength=40, maxLineGap=5)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]  # Access the first element of the line array
            cv2.line(image_masked, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return image_masked

fps = 30
frame_delay = int(1000 / fps)

while vid.isOpened():
    start_time = time.time()
    ret, frame = vid.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    cv2.imshow("Video Frame", process_image(frame))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    elapsed_time = time.time() - start_time
    sleep_time = max(0, (1 / fps) - elapsed_time)
    time.sleep(sleep_time)

vid.release()
cv2.destroyAllWindows()

# cv2.imwrite("./output.png", process_image(img))
