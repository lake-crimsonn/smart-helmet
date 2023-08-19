from ffpyplayer.player import MediaPlayer
import cv2
import numpy as np
import time
import os
from ultralytics import YOLO
from deepsort_module import deepsort_start
import func_road
import warning_person

model_path = os.path.join('models', 'best_yolo8n.pt')
video_path = os.path.join('videos_in', 'test_1.mp4')

player = MediaPlayer(video_path)
# player.set_size(852, 480)  # resize it
# # player.set_size(400, 300)

start_time = time.time()
frame_time = start_time + 0

model = YOLO(model_path)

count = 0

while True:
    current_time = time.time()

    # check if it is time to get next frame
    if current_time >= frame_time:

        # get next frame
        frame, val = player.get_frame()

        if val != 'eof' and frame is not None:
            image, pts = frame
            w, h = image.get_size()

            # convert to array width, height
            img = np.asarray(image.to_bytearray()[0]).reshape(h, w, 3)

            # convert RGB to BGR because `cv2` need it to display it
            frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            yolo_frame = model(frame, verbose=False)

            # 딥소트
            deep_frame = deepsort_start(yolo_frame, frame)  # yolo, frame

            # 로드워닝
            count += 1
            if count % 60 == 0:
                print(func_road.road(frame))
            w_p_frame = warning_person.w_p(yolo_frame, deep_frame)

            cv2.imshow('video', w_p_frame)

            frame_time = start_time + pts

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cv2.destroyAllWindows()
player.close_player()
