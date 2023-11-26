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


def filter(roi):
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  # 轉成灰階
    gray_roi = cv2.GaussianBlur(gray_roi, (7, 7), 0)  # 高斯模糊消除雜訊

    _, threshold = cv2.threshold(
        gray_roi, 40, 255, cv2.THRESH_BINARY_INV
    )  # 影像二值化，將顏色較深的區域做區隔***第二個參數為二值化分界，需視情況調整***
    threshold = cv2.dilate(threshold, np.ones((10, 10)))  # 擴張
    cv2.imshow("擴張", threshold)
    threshold = cv2.erode(threshold, np.ones((6, 6)))  # 侵蝕，此兩步驟平滑化二值化影像的邊緣
    cv2.imshow("侵蝕", threshold)
    contours, _ = cv2.findContours(
        threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )  # 將邊緣取出

    # 將取出的邊緣依範圍大小做排序，因為可能會取到一些雜訊的邊緣，我們只要最大的那個範圍，也就是瞳孔的邊緣
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    return contours


while True:
    ret, frame = cap.read()

    if ret is False:  # 如果沒輸入訊號就中斷
        break

    # 於影像中識別人臉
    faces = detector(frame)

    # 若有識別到人臉
    if faces:
        print(len(faces))
        for face in faces:
            # 預測識別到的臉部節點
            landmarks = predictor(frame, face)

            # 從影像中用預測的節點擷取右眼的範圍
            roi_R = frame[
                int((landmarks.part(19).y)) + 10 : int((landmarks.part(28).y)),
                int((landmarks.part(36).x)) : int((landmarks.part(39).x)),
            ]
            rows, cols, _ = roi_R.shape
            print("R", roi_R.shape)
            if rows > 0 and cols > 0:
                contours = filter(roi_R)
                for cnt in contours:
                    cv2.drawContours(roi_R, [cnt], -1, (0, 0, 255), 2)  # 將範圍畫出來

                    # 以下為取出邊緣的bounding box並標出十字
                    (x, y, w, h) = cv2.boundingRect(cnt)
                    # cv2.rectangle(
                    #     roi,
                    #     (x + int(w / 4), y + int(h / 4)),
                    #     (x + int(3 * w / 4), y + int(3 * h / 4)),
                    #     (255, 0, 0),
                    #     2,
                    # )
                    cv2.line(
                        roi_R,
                        (x + int(w / 2), y),
                        (x + int(w / 2), y + h),
                        (0, 255, 0),
                        1,
                    )
                    cv2.line(
                        roi_R,
                        (x, y + int(h / 2)),
                        (x + w, y + int(h / 2)),
                        (0, 255, 0),
                        1,
                    )
                    frame[
                        int((landmarks.part(19).y)) + 10 : int((landmarks.part(28).y)),
                        int((landmarks.part(36).x)) : int((landmarks.part(39).x)),
                    ] = roi_R
                    break  # 取完最大的就結束
            else:
                pass

            roi_L = frame[
                int((landmarks.part(24).y)) + 10 : int((landmarks.part(28).y)),
                int((landmarks.part(42).x)) : int((landmarks.part(45).x)),
            ]
            rows, cols, _ = roi_L.shape
            print("L", roi_L.shape)
            if rows > 0 and cols > 0:
                contours = filter(roi_L)
                for cnt in contours:
                    cv2.drawContours(roi_L, [cnt], -1, (0, 0, 255), 2)  # 將範圍畫出來

                    # 以下為取出邊緣的bounding box並標出十字
                    (x, y, w, h) = cv2.boundingRect(cnt)
                    # cv2.rectangle(
                    #     roi,
                    #     (x + int(w / 4), y + int(h / 4)),
                    #     (x + int(3 * w / 4), y + int(3 * h / 4)),
                    #     (255, 0, 0),
                    #     2,
                    # )
                    cv2.line(
                        roi_L,
                        (x + int(w / 2), y),
                        (x + int(w / 2), y + h),
                        (0, 255, 0),
                        1,
                    )
                    cv2.line(
                        roi_L,
                        (x, y + int(h / 2)),
                        (x + w, y + int(h / 2)),
                        (0, 255, 0),
                        1,
                    )
                    frame[
                        int((landmarks.part(24).y)) + 10 : int((landmarks.part(28).y)),
                        int((landmarks.part(42).x)) : int((landmarks.part(45).x)),
                    ] = roi_L
                    break  # 取完最大的就結束
            else:
                pass
            break
    else:
        pass

    cv2.imshow("frame", frame)
    key = cv2.waitKey(30)
    # 按Esc可結束
    if key == 27:
        break

cv2.destroyAllWindows()
