import cv2


def w_p(yolo_image, frame):
    y, x = frame.shape[0], frame.shape[1]
    cv2.rectangle(frame, (2 * x // 5, 0), (3 * x // 5, y),
                  (255, 255, 255, 255), 2)  # 디택할 영상에 사각형 그리기
    for result in yolo_image:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:  # 객체 하나하나에 접근
            print(result.names[int(box.cls[0])])  # 그 접근한 객체 하나하나의 이름
            if result.names[int(box.cls[0])] == 'Pedestrian_Pedestrian':  # 만약에 사람이라면
                r = box.xyxy[0].astype(int)  # 그 디택한 신호등의 사각형 위치
                # print(r)
                # r = [x1,y1,x2,y2]
                if (2 * x // 5 <= r[0] <= 3 * x // 5) and (2 * x // 5 <= r[2] <= 3 * x // 5):
                    cv2.rectangle(frame, r[:2], r[2:],
                                  (0, 0, 255, 255), 2)  # 사람 하얀색 그리기
                else:
                    cv2.rectangle(frame, r[:2], r[2:],
                                  (255, 255, 255, 255), 2)  # 사람 빨강색 그리기
    return frame
