import cv2
import tensorflow as tf
from forward_data.pspunet import pspunet
from forward_data.display import create_mask
import numpy as np

skip_frame_count = 0


def forward_looking(mode, frame):
    global skip_frame_count

    skip_frame_count += 1

    if 0 < skip_frame_count % 10:
        return

    gpus = tf.config.experimental.list_physical_devices('GPU')

    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=500)])
        except RuntimeError as e:
            print(e)

    MODE = mode
    IMG_WIDTH = 480
    IMG_HEIGHT = 272
    n_classes = 7

    blue_count = 0
    red_count = 0

    model = pspunet((IMG_HEIGHT, IMG_WIDTH, 3), n_classes)
    model.load_weights("models\\forward_model.h5")

    def region_of_interest(img):

        vertices = [[
            (IMG_WIDTH*0.3, IMG_HEIGHT),
            (IMG_WIDTH*0.3, 0),
            (IMG_WIDTH*0.7, 0),
            (IMG_WIDTH*0.7, IMG_HEIGHT)
        ]]

        mask = np.zeros_like(img)
        match_mask_color = (255, 255, 255)
        mask = cv2.fillPoly(mask, np.array(
            vertices, dtype=np.int32), match_mask_color)
        masked_image = cv2.bitwise_and(img, mask)

        return masked_image

    frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame = frame[tf.newaxis, ...]
    frame = frame/255

    pre = model.predict(frame)
    pre = create_mask(pre).numpy()
    frame2 = frame / 2

    frame2[0][(pre == 1).all(axis=2)] = [0, 0, 255]  # bikelane, side asphalt
    frame2[0][(pre == 2).all(axis=2)] = [255, 0, 0]  # caution
    frame2[0][(pre == 3).all(axis=2)] = [255, 0, 0]  # crosswalk
    frame2[0][(pre == 4).all(axis=2)] = [255, 0, 0]  # guide
    frame2[0][(pre == 5).all(axis=2)] = [0, 0, 255]  # roadway
    frame2[0][(pre == 6).all(axis=2)] = [255, 0, 0]  # sidewalk

    frame2 = frame2.squeeze()

    roi = region_of_interest(frame2)

    roi_cropped = roi[0:int(IMG_HEIGHT),
                      int(IMG_WIDTH*0.3):int(IMG_WIDTH*0.7)]  # y, x

    frame2 = frame2 * 255
    blue_count = 0
    red_count = 0

    for x in roi_cropped:
        for y in x:
            if (y == [255, 0, 0]).all():
                blue_count += 1
            elif (y == [0, 0, 255]).all():
                red_count += 1

    if (MODE == 0 and red_count >= blue_count) or (MODE == 1 and blue_count >= red_count):
        # print("blue_count:", blue_count)
        # print("red_count:", red_count)
        print("앞을 보라")

    cv2.imshow('video', roi_cropped)
