import cv2
import numpy as np
import os
from deepsort_module import deepsort_start
from forward_module import forward_looking
from ultralytics import YOLO
import func_road
import warning_person
import time
from ffpyplayer.player import MediaPlayer

video_path = os.path.join('videos_in', 'test_1.mp4')
model_path = os.path.join('models', 'best_yolo8n.pt')

cap = cv2.VideoCapture(video_path)
model = YOLO(model_path)

count = 0

# 오디오캡쳐
audio = MediaPlayer(video_path)

# 시간
start_time = time.time()

while True:
    ret, frame = cap.read()
    yolo_frame = model(frame, verbose=False)

    # 오디오

    audio_frame, val = audio.get_frame()  # val은 재생시간 1.0을 지키기 위해 기다리는 시간

    # 딥소트
    deep_frame = deepsort_start(yolo_frame, frame)  # yolo, frame

    # 로드워닝
    count += 1
    if count % 60 == 0:
        print(func_road.road(frame))
    w_p_frame = warning_person.w_p(yolo_frame, deep_frame)

    # 전방주시
    # forward_looking(0, deep_frame)  # mode, frame

    w, h = audio_frame.get_size()
    img = np.asarray(audio_frame.to_bytearray()[0]).reshape(h, w, 3)

    cv2.imshow('video', img)

    elapsed = (time.time()-start_time) * 1000  # ms
    # print(elapsed)
    play_time = int(cap.get(cv2.CAP_PROP_POS_MSEC))*0.001  # 재생 위치 ms
    # print(play_time)
    sleep = max(1, int(elapsed - play_time))
    # print(sleep)
    # print('pt:', play_time)
    # print('audio:', audio.get_pts())
    # print('diff', audio.get_pts()-play_time)
    # time.sleep(val)
    if cv2.waitKey(1) == ord('q'):
        break

audio.close_player()
cap.release()
cv2.destroyAllWindows()
