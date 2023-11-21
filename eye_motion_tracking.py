import cv2
import numpy as np
import dlib

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def midpoint(p1, p2):
    return int((p1 + p2) / 2)


while True:
    ret, frame = cap.read()
    if ret is False:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray_frame)

    # print(len(faces))

    if faces:
        for face in faces:
            landmarks = predictor(gray_frame, face)
            roi = frame[
                int((landmarks.part(19).y + landmarks.part(37).y) / 2) : landmarks.part(
                    30
                ).y
                - 30,
                landmarks.part(17).x : landmarks.part(27).x,
            ]
            rows, cols, _ = roi.shape
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray_roi = cv2.GaussianBlur(gray_roi, (7, 7), 0)

            _, threshold = cv2.threshold(gray_roi, 25, 255, cv2.THRESH_BINARY_INV)
            threshold = cv2.dilate(threshold, np.ones((7, 7)))
            threshold = cv2.erode(threshold, np.ones((8, 8)))
            contours, _ = cv2.findContours(
                threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

            for cnt in contours:
                (x, y, w, h) = cv2.boundingRect(cnt)

                # cv2.drawContours(roi, [cnt], -1, (0, 0, 255), 3)
                cv2.rectangle(
                    roi,
                    (x + int(w / 4), y + int(h / 4)),
                    (x + int(3 * w / 4), y + int(3 * h / 4)),
                    (255, 0, 0),
                    2,
                )
                cv2.line(
                    roi, (x + int(w / 2), 0), (x + int(w / 2), rows), (0, 255, 0), 2
                )
                cv2.line(
                    roi, (0, y + int(h / 2)), (cols, y + int(h / 2)), (0, 255, 0), 2
                )
                break
            frame[
                int((landmarks.part(19).y + landmarks.part(37).y) / 2) : landmarks.part(
                    30
                ).y
                - 30,
                landmarks.part(17).x : landmarks.part(27).x,
            ] = roi
            roi = frame
    else:
        roi = frame

    cv2.imshow("Roi", roi)
    key = cv2.waitKey(30)
    if key == 27:
        break

cv2.destroyAllWindows()
