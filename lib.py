import cv2
import numpy as np

cv2.namedWindow("高斯", cv2.WINDOW_NORMAL)
cv2.namedWindow("邊緣", cv2.WINDOW_NORMAL)
cv2.namedWindow("二值", cv2.WINDOW_NORMAL)
cv2.resizeWindow("高斯", 340, 200)  # 寬 600，高 400
cv2.resizeWindow("邊緣", 340, 200)
cv2.resizeWindow("二值", 340, 200)

def filter(image):
    canva1 = np.zeros((200, 340), np.uint8)
    canva2 = np.zeros((200, 340), np.uint8)
    canva3 = np.zeros((200, 340), np.uint8)
    """
    用來取出虹膜的範圍
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 轉成灰階
    gray_image = cv2.GaussianBlur(gray_image, (7, 15), 0)  # 高斯模糊消除雜訊
    h, w= gray_image.shape
    canva1[100-h//2:100+(h-h//2), 170-w//2:170+(w-w//2)] = gray_image

    """
    影像二值化，將顏色較深的區域做區隔***第二個參數為二值化分界，需視情況調整***
    """
    _, binaryimage = cv2.threshold(gray_image, 30, 255, cv2.THRESH_BINARY_INV)
    binaryimage = cv2.erode(binaryimage, np.ones((10, 10)))  # 侵蝕
    binaryimage = cv2.dilate(binaryimage, np.ones((10, 10)))  # 擴張，此兩步驟平滑化二值化影像的邊緣
    h, w= binaryimage.shape
    canva2[100-h//2:100+(h-h//2), 170-w//2:170+(w-w//2)] = binaryimage
    # 將邊緣取出
    contours, _ = cv2.findContours(binaryimage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 將取出的邊緣依範圍大小做排序，因為可能會取到一些雜訊的邊緣，我們只要最大的那個範圍，也就是瞳孔的邊緣
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

    """
    找圈圈法
    """
    邊緣 = cv2.Canny(gray_image, 120, 90)
    h, w= 邊緣.shape
    canva3[100-h//2:100+(h-h//2), 170-w//2:170+(w-w//2)] = 邊緣
    circles = cv2.HoughCircles(
        邊緣,
        cv2.HOUGH_GRADIENT,
        1,
        50,
        param1=100,
        param2=8,
        minRadius=0,
        maxRadius=15,
    )

    cv2.imshow("高斯", canva1)
    cv2.imshow("邊緣", canva3)
    cv2.imshow("二值", canva2)
    # 設定視窗大小

    return circles, contours
