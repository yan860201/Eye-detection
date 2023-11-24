import cv2
import numpy as np

# 輸入來源`,這邊使用示範影片`
cap = cv2.VideoCapture("eye_recording.flv")


def midpoint(p1, p2):
    return int((p1 + p2) / 2)


while True:
    ret, frame = cap.read()

    if ret is False:  # 如果沒輸入訊號就中斷
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 轉成灰階
    gray_frame = cv2.GaussianBlur(gray_frame, (7, 7), 0)  # 高斯模糊消除雜訊

    _, threshold = cv2.threshold(
        gray_frame, 5, 255, cv2.THRESH_BINARY_INV
    )  # 影像二值化，將顏色較深的區域做區隔

    threshold = cv2.dilate(threshold, np.ones((7, 7)))  # 擴張
    # cv2.imshow("擴張", threshold)
    threshold = cv2.erode(threshold, np.ones((8, 8)))  # 侵蝕，此兩步驟平滑化二值化影像的邊緣
    # cv2.imshow("侵蝕", threshold)
    contours, _ = cv2.findContours(
        threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )  # 將邊緣取出

    # 將取出的邊緣依範圍大小做排序，因為可能會取到一些雜訊的邊緣，我們只要最大的那個範圍，也就是瞳孔的邊緣
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    for cnt in contours:
        # (x, y, w, h) = cv2.boundingRect(cnt)
        cv2.drawContours(frame, [cnt], -1, (0, 0, 255), 2)  # 將範圍畫出來
        break  # 只取最大的範圍

    cv2.imshow("frame", frame)
    key = cv2.waitKey(30)
    # 按Esc可結束
    if key == 27:
        break

cv2.destroyAllWindows()
