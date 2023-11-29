import cv2
import numpy as np
import dlib
import datetime

# 輸入來源，輸入攝影機訊號
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# 臉部節點預測器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def filter(roi):
    '''
    用來取出虹膜的範圍
    '''
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  # 轉成灰階
    gray_roi = cv2.GaussianBlur(gray_roi, (7, 15), 0)  # 高斯模糊消除雜訊

    '''
    影像二值化，將顏色較深的區域做區隔***第二個參數為二值化分界，需視情況調整***
    '''
    _, threshold = cv2.threshold(
        gray_roi, 110, 255, cv2.THRESH_BINARY_INV
    )
    threshold = cv2.erode(threshold, np.ones((10, 10)))  # 侵蝕
    threshold = cv2.dilate(threshold, np.ones((10, 10)))  # 擴張，此兩步驟平滑化二值化影像的邊緣
    # 將邊緣取出
    contours, _ = cv2.findContours(
        threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    # 將取出的邊緣依範圍大小做排序，因為可能會取到一些雜訊的邊緣，我們只要最大的那個範圍，也就是瞳孔的邊緣
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

    '''
    找圈圈法
    '''
    邊緣 = cv2.Canny(gray_roi, 120, 90)
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

    cv2.imshow("高斯", gray_roi)
    cv2.imshow("邊緣", 邊緣)
    cv2.imshow("二值", threshold)
    return circles, contours


方法一紀錄 = []
方法二紀錄 = []

while True:
    success, frame = cap.read()

    if not success:  # 如果沒輸入訊號就中斷
        break

    # 於影像中識別人臉
    faces = detector(frame)

    # 若有識別到人臉
    if faces:
        for face in faces:
            # 預測識別到的臉部節點
            landmarks = predictor(frame, face)

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

            if rows > 5 and cols > 5: #避免邊界取值錯誤時程式出錯
                circle, contours = filter(roi_R)
                '''
                畫圈
                '''
                try:
                    det_circle = np.uint16(np.around(circle))
                    for x_R, y_R, r in det_circle[0, :]:
                        # cv2.circle(roi_R, (x_R, y_R), r, (0, 255, 0), 2)
                        x_R = x_R + 右眼左邊界
                        y_R = y_R + 右眼上邊界
                except:
                    print("noReye")
                    x_R, y_R = 0, 0

                '''
                畫十字
                '''
                if len(contours) == 0:
                    x_R方法一, y_R方法一 = 0, 0

                else:
                    for cnt in contours:
                        # cv2.drawContours(roi_R, [cnt], -1, (0, 0, 255), 2)  # 將範圍畫出來

                        # 以下為標出十字
                        (x, y, w, h) = cv2.boundingRect(cnt)
                        cv2.line(
                            roi_R,
                            (x + int(w / 2), y),
                            (x + int(w / 2), y + h),
                            (0, 255, 0),
                            2,
                        )
                        cv2.line(
                            roi_R,
                            (x, y + int(h / 2)),
                            (x + w, y + int(h / 2)),
                            (0, 255, 0),
                            2,
                        )

                        frame[右眼上邊界:右眼下邊界, 右眼左邊界:右眼右邊界] = roi_R

                        x_R方法一 = (x + int(w / 2)) + 右眼左邊界
                        y_R方法一 = (y + int(h / 2)) + 右眼上邊界
                        break  # 取完最大的就結束
            else:
                x_R, y_R = 0, 0
                x_R方法一, y_R方法一 = 0, 0

            """
            左眼
            """
            左眼左邊界 = int((landmarks.part(42).x)) - 10
            左眼右邊界 = int((landmarks.part(45).x)) + 10
            左眼上邊界 = int(landmarks.part(24).y) + 17
            左眼下邊界 = int((landmarks.part(28).y))

            roi_L = frame[左眼上邊界:左眼下邊界, 左眼左邊界:左眼右邊界]
            rows, cols, _ = roi_L.shape

            if rows > 5 and cols > 5: #避免邊界取值錯誤時程式出錯
                circle, contours = filter(roi_L)
                '''
                畫圈
                '''
                try:
                    det_circle = np.uint16(np.around(circle))
                    for x_L, y_L, r in det_circle[0, :]:
                        # cv2.circle(roi_L, (x_L, y_L), r, (0, 255, 0), 2)
                        x_L = x_L + 左眼左邊界
                        y_L = y_L + 左眼上邊界
                except:
                    print("noLeye")
                    x_L, y_L = 0, 0

                '''
                畫十字
                '''
                if len(contours) == 0:
                    x_L方法一, y_L方法一 = 0, 0

                else:
                    for cnt in contours:
                        # cv2.drawContours(roi_L, [cnt], -1, (0, 0, 255), 2)  # 將範圍畫出來

                        # 以下為標出十字
                        (x, y, w, h) = cv2.boundingRect(cnt)
                        cv2.line(
                            roi_L,
                            (x + int(w / 2), y),
                            (x + int(w / 2), y + h),
                            (0, 255, 0),
                            2,
                        )
                        cv2.line(
                            roi_L,
                            (x, y + int(h / 2)),
                            (x + w, y + int(h / 2)),
                            (0, 255, 0),
                            2,
                        )

                        frame[左眼上邊界:左眼下邊界, 左眼左邊界:左眼右邊界] = roi_L

                        x_L方法一 = (x + int(w / 2)) + 左眼左邊界
                        y_L方法一 = (y + int(h / 2)) + 左眼上邊界
                        break  # 取完最大的就結束
            else:
                x_L, y_L = 0, 0
                x_L方法一, y_L方法一 = 0, 0
            break
    else:
        x_R, y_R, x_L, y_L = 0, 0, 0, 0
        x_R方法一, y_R方法一, x_L方法一, y_L方法一 = 0, 0, 0, 0

    方法一紀錄.append([x_R方法一, y_R方法一, x_L方法一, y_L方法一])
    方法二紀錄.append([x_R, y_R, x_L, y_L])

    cv2.imshow("frame", frame)
    key = cv2.waitKey(30)
    # 按Esc可結束
    if key == 27:
        import pandas as pd
        # date = datetime.datetime.now()
        # date = date.strftime("%m%d%X")
        # date = date.split(":")
        # print(date)
        # df方法一 = pd.DataFrame(方法一紀錄)
        # df方法二 = pd.DataFrame(方法二紀錄)
        # df方法一.to_csv(f"座標/{date[0] + date[1] + date[2]}方法一.csv", index=None)
        # df方法二.to_csv(f"座標/{date[0] + date[1] + date[2]}方法二.csv", index=None)
        break

cv2.destroyAllWindows()
