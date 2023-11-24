import cv2
import numpy as np
import dlib

# 輸入來源，輸入攝影機訊號
cap = cv2.VideoCapture(1)

# 臉部節點預測器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def midpoint(p1, p2):
    return int((p1 + p2) / 2)


while True:
    ret, frame = cap.read()

    if ret is False:  # 如果沒輸入訊號就中斷
        break

    # 於影像中識別人臉
    faces = detector(frame)

    # 若有識別到人臉
    if faces:
        for face in faces:
            # 預測識別到的臉部節點
            landmarks = predictor(frame, face)

            # 從影像中用預測的節點擷取右眼的範圍
            roi = frame[
                int((landmarks.part(19).y + landmarks.part(37).y) / 2) : landmarks.part(
                    30
                ).y
                - 30,
                landmarks.part(17).x : landmarks.part(26).x,
            ]
            rows, cols, _ = roi.shape
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  # 轉成灰階
            gray_roi = cv2.GaussianBlur(gray_roi, (7, 7), 0)  # 高斯模糊消除雜訊

            _, threshold = cv2.threshold(
                gray_roi, 20, 255, cv2.THRESH_BINARY_INV
            )  # 影像二值化，將顏色較深的區域做區隔***第二個參數為二值化分界，需視情況調整***
            threshold = cv2.dilate(threshold, np.ones((7, 7)))  # 擴張
            threshold = cv2.erode(threshold, np.ones((8, 8)))  # 侵蝕，此兩步驟平滑化二值化影像的邊緣
            contours, _ = cv2.findContours(
                threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )  # 將邊緣取出

            # 將取出的邊緣依範圍大小做排序，因為可能會取到一些雜訊的邊緣，我們只要最大的那個範圍，也就是瞳孔的邊緣
            contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
            i = 0
            for cnt in contours:
                cv2.drawContours(roi, [cnt], -1, (0, 0, 255), 2)  # 將範圍畫出來

                # 以下為取出邊緣的bounding box並標出十字
                # (x, y, w, h) = cv2.boundingRect(cnt)
                # cv2.rectangle(
                #     roi,
                #     (x + int(w / 4), y + int(h / 4)),
                #     (x + int(3 * w / 4), y + int(3 * h / 4)),
                #     (255, 0, 0),
                #     2,
                # )
                # cv2.line(
                #     roi, (x + int(w / 2), 0), (x + int(w / 2), rows), (0, 255, 0), 2
                # )
                # cv2.line(
                #     roi, (0, y + int(h / 2)), (cols, y + int(h / 2)), (0, 255, 0), 2
                # )
                if i == 1:
                    break  # 取完左右兩眼就結束
                i += 1
            # 將處理好的的影像再放回原始影像中
            frame[
                int((landmarks.part(19).y + landmarks.part(37).y) / 2) : landmarks.part(
                    30
                ).y
                - 30,
                landmarks.part(17).x : landmarks.part(26).x,
            ] = roi
            roi = frame
            break
    else:
        roi = frame

    cv2.imshow("Roi", roi)
    key = cv2.waitKey(30)
    # 按Esc可結束
    if key == 27:
        break

cv2.destroyAllWindows()
