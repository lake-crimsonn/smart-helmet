import os
import random
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort


def deepsort_start(yolo_frame, frame):

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # 라이브러리 충돌 방지

    tracker = DeepSort(max_age=3)

    detection_threshold = 0.5  # 스코어 임계값

    results = yolo_frame

    def switch(key):
        colors = [(243, 168, 188), (160, 249, 175),
                  (158, 231, 245), (255, 241, 166)]
        track_name = {"0": ["car", colors[0]], "1": ["bus", colors[0]], "2": ["motorcycle", colors[0]],
                      "3": ["unknown", colors[0]], "4": ["pedes", colors[1]], "5": ["bicycle", colors[1]], "6": ["red", colors[2]],
                      "7": ["yellow", colors[2]], "8": ["green", colors[2]], "9": ["stopline", colors[3]], "10": ["crosswalk", colors[3]]}.get(key, ["what", colors[0]])
        return track_name

    # 프레임마다 모델이 읽어오는 정보 사용하기 위해서 언랩
    for result in results:
        detections = []
        for r in result.boxes.data.tolist():  # 박스에 대한 정보, 텐서가 아니라 리스트로 사용하기
            x1, y1, x2, y2, score, class_id = r  # coco데이터셋의 class_id
            w = x2-x1
            h = y2-y1
            class_id = int(class_id)
            if score > detection_threshold:  # 스코어가 0.5 이상인 객체만 딥소트 적용
                detections.append([[x1, y1, w, h], score, class_id])

        tracks = tracker.update_tracks(
            detections, frame=frame)  # 딥소트 알고리즘 적용

        for track in tracks:
            # track_id = track.track_id  # 딥소트 알고리즘이 적용된 오브젝트의 아이디
            ltrb = track.to_ltrb()  # left-top, right-bottom
            track_class = track.det_class

            track_name, color = switch(str(track_class))

            # x1,y1,x2,y2
            cv2.rectangle(frame, (int(ltrb[0]), int(ltrb[1])), (int(
                ltrb[2]), int(ltrb[3])), color, 2)

            cv2.putText(frame, track_name, (
                int(ltrb[0]), int(ltrb[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    return frame
