import pyautogui
import time
import cv2

# 螢幕尺寸
size = pyautogui.size()

滑鼠座標 = []

while True:
    # 滑鼠座標
    position = pyautogui.position()

    print(position)
    # 延遲迴圈
    time.sleep(0.25)
