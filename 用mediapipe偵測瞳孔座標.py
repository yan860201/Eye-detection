import mediapipe as mp
from mediapipe import solutions
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyautogui


def draw_landmarks_on_image(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected faces to visualize.
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        # Draw the face landmarks.
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x, y=landmark.y, z=landmark.z
                )
                for landmark in face_landmarks
            ]
        )

        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style(),
        )
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style(),
        )
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style(),
        )

    return annotated_image

def get_eye_coordinate(rgb_image, detection_result, h, w):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    a = []
    b = []
    # Loop through the detected faces to visualize.
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        # Draw the face landmarks.
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x, y=landmark.y, z=landmark.z
                )
                for landmark in face_landmarks
            ]
        )

        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style(),
        )
        
        for i in [468, 473]:
            point = face_landmarks[i]
            x = int(point.x * w)
            y = int(point.y * h)
            a.append(point.x * w)
            b.append(point.y * h)
            cv2.circle(annotated_image, (x, y), 3, (255, 255, 0), -1)

    xpoint = sum(a)/2
    ypoint = sum(b)/2
    print(xpoint, ypoint)

    return annotated_image, xpoint, ypoint

model_path = "face_landmarker_v2_with_blendshapes.task"

FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# 設定臉部節點標註器選項
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    output_face_blendshapes=True
    # 還有其他選項參照https://developers.google.com/mediapipe/solutions/vision/face_landmarker/python#configuration_options
)

# 用設定好的選項建立臉部節點標註器
with FaceLandmarker.create_from_options(options) as landmarker:

    滑鼠座標紀錄 = []
    眼睛座標紀錄 = []

    frame_width = 1280
    frame_height = 720
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    #frame_timestamp要用的的數據
    fps = cap.get(cv2.CAP_PROP_FPS)
    timer = 0
    while True:
        success, frame = cap.read()
        frame_timestamp = int((1000 // fps) * timer)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        face_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp)
        annotated_image, x, y = get_eye_coordinate(mp_image.numpy_view(), face_landmarker_result, frame_height, frame_width)
        cv2.imshow("show", annotated_image)

        #做紀錄
        滑鼠座標紀錄.append(pyautogui.position())
        眼睛座標紀錄.append([x, y])

        timer += 1
        key = cv2.waitKey(20)
        if key == 27:
            import pandas as pd
            import datetime

            date = datetime.datetime.now()
            date = date.strftime("%m%d%X")
            date = date.split(":")
            print(date)
            df滑鼠座標 = pd.DataFrame(滑鼠座標紀錄)
            df眼睛座標 = pd.DataFrame(眼睛座標紀錄)
            df = pd.concat([df眼睛座標, df滑鼠座標], axis=1)
            df.to_csv(f"座標/{date[0] + date[1] + date[2]}.csv", index=None)
            break

    cap.release()
    cv2.destroyAllWindows()