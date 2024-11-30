import cv2
import numpy as np
import dlib
import datetime
from lib import *

# 輸入來源，輸入攝影機訊號
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# 臉部節點預測器
face_detector = dlib.get_frontal_face_detector()
landmarks_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    success, frame = cap.read()

    if not success:  # 如果沒輸入訊號就中斷
        break

    # 於影像中識別人臉
    frame = cv2.flip(frame, 1)
    faces = face_detector(frame)

    # 若有識別到人臉
    if faces:
        for face in faces:
            # 預測識別到的臉部節點
            landmarks = landmarks_predictor(frame, face)

            # 從影像中用預測的節點擷取右眼的範圍
            """
            右眼
            """
            右眼左邊界 = int((landmarks.part(36).x)) - 10
            右眼右邊界 = int((landmarks.part(39).x)) + 10
            右眼上邊界 = int((landmarks.part(19).y)) + 17
            右眼下邊界 = int((landmarks.part(28).y))

            roi_R = frame[右眼上邊界:右眼下邊界, 右眼左邊界:右眼右邊界]
            rows, cols, _ = roi_R.shape

            if rows > 5 and cols > 5:  # 避免邊界取值錯誤時程式出錯
                circle, contours = filter(roi_R)
                """
                畫圈
                """
                try:
                    det_circle = np.uint16(np.around(circle))
                    for x_R, y_R, r in det_circle[0, :]:
                        # x座標, y座標, 半徑
                        cv2.circle(roi_R, (x_R, y_R), r, (0, 255, 0), 2)
                except:
                    print("noReye")

                """
                畫十字
                """
                if len(contours) == 0:
                    pass

                else:
                    for contour in contours:
                        # cv2.drawContours(roi_R, [contour], -1, (0, 0, 255), 2)  # 將範圍畫出來

                        # 以下為標出十字
                        (x, y, w, h) = cv2.boundingRect(contour)
                        cv2.line(
                            roi_R,
                            (x + w // 2, y),
                            (x + w // 2, y + h),
                            (0, 0, 255),
                            1,
                        )
                        cv2.line(
                            roi_R,
                            (x, y + h // 2),
                            (x + w, y + h // 2),
                            (0, 0, 255),
                            1,
                        )

                        frame[右眼上邊界:右眼下邊界, 右眼左邊界:右眼右邊界] = roi_R

                        break  # 取完最大的就結束

            """
            左眼
            """
            左眼左邊界 = int((landmarks.part(42).x)) - 10
            左眼右邊界 = int((landmarks.part(45).x)) + 10
            左眼上邊界 = int(landmarks.part(24).y) + 17
            左眼下邊界 = int((landmarks.part(28).y))

            roi_L = frame[左眼上邊界:左眼下邊界, 左眼左邊界:左眼右邊界]
            rows, cols, _ = roi_L.shape

            if rows > 5 and cols > 5:  # 避免邊界取值錯誤時程式出錯
                circle, contours = filter(roi_L)
                """
                畫圈
                """
                try:
                    det_circle = np.uint16(np.around(circle))
                    for x_L, y_L, r in det_circle[0, :]:
                        cv2.circle(roi_L, (x_L, y_L), r, (0, 255, 0), 2)
                except:
                    print("noLeye")

                """
                畫十字
                """
                if len(contours) == 0:
                    x_L方法一, y_L方法一 = 0, 0

                else:
                    for contour in contours:
                        # cv2.drawContours(roi_L, [contour], -1, (0, 0, 255), 2)  # 將範圍畫出來

                        # 以下為標出十字
                        (x, y, w, h) = cv2.boundingRect(contour)
                        cv2.line(
                            roi_L,
                            (x + w // 2, y),
                            (x + w // 2, y + h),
                            (0, 0, 255),
                            1,
                        )
                        cv2.line(
                            roi_L,
                            (x, y + h // 2),
                            (x + w, y + h // 2),
                            (0, 0, 255),
                            1,
                        )

                        frame[左眼上邊界:左眼下邊界, 左眼左邊界:左眼右邊界] = roi_L

                        break  # 取完最大的就結束
            else:
                x_L, y_L = 0, 0
                x_L方法一, y_L方法一 = 0, 0
            break

    cv2.imshow("frame", frame)
    key = cv2.waitKey(1)
    # 按Esc可結束
    if key == 27:
        break

cv2.destroyAllWindows()