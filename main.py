import cv2
import numpy as np
import os
from deepsort_module import deepsort_start
from forward_module import forward_looking
from ultralytics import YOLO
import func_road
import warning_person

video_path = 'full_vid.mp4'

cap = cv2.VideoCapture(video_path)
model = YOLO('best_yolo8n.pt')

count = 0

while True:
    ret, frame = cap.read()
    yolo_frame = model(frame, verbose=False)

    # 딥소트
    deep_frame = deepsort_start(yolo_frame, frame)  # yolo, frame

    # 로드워닝
    count += 1
    if count % 60 == 0:
        print(func_road.road(frame))
    w_p_frame = warning_person.w_p(yolo_frame, deep_frame)

    # 전방주시
    # forward_looking(0, deep_frame)  # mode, frame

    cv2.imshow('video', w_p_frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
