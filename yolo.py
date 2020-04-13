import colorsys
from timeit import default_timer as timer

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image

class YOLO(object):
    def __init__(self):
        self.model_image_size = (288, 512)
        self.score = 0.3
        self.iou = 0.5
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self._generate()

    def _get_class(self):
        with open('./model_data/nuscenes_classes.txt') as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        with open('./model_data/nuscenes_anchors.txt') as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def _generate(self):
        weights_path = './model_data/yolo.h5'
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)

        is_tiny_version = num_anchors==6
        if is_tiny_version:
            self.yolo_model = tiny_yolo_body(Input(shape=(None, None, 3)), num_anchors // 2, num_classes)
        else:
            self.yolo_model = yolo_body(Input(shape=(None, None, 3)), num_anchors // 3, num_classes)
        self.yolo_model.load_weights(weights_path)

        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

        self.input_image_shape = K.placeholder(shape=(2,))
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_single_image(self, image, test_set_mode=False):
        start = timer()

        boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        image_data = np.array(boxed_image, dtype='float32')

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        font = ImageFont.truetype(font='.DejaVuSans.ttf', size=11)
        thickness = 2

        detection_results = []
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            if test_set_mode:
                detection_results.append([left, top, right, bottom, c, score])
            else:
                if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])

                for i in range(thickness):
                    draw.rectangle(
                        [left + i, top + i, right - i, bottom - i],
                        outline=self.colors[c])
                draw.rectangle(
                    [tuple(text_origin), tuple(text_origin + label_size)],
                    fill=self.colors[c])
                draw.text(text_origin, predicted_class, fill=(0, 0, 0), font=font)
                del draw

        end = timer()
        print(end - start)
        return detection_results if test_set_mode else image

    def close_session(self):
        self.sess.close()

def detect_image(yolo):
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error!')
            continue
        else:
            image = yolo.detect_single_image(image)
            image.show()
    yolo.close_session()

def detect_video(yolo, video_path, output_path=""):
    import cv2
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC = cv2.VideoWriter_fourcc(*'XVID')
    video_fps = vid.get(cv2.CAP_PROP_FPS)
    video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if output_path:
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    frame_count = 0
    start = timer()
    while True:
        return_value, frame = vid.read()
        frame_count += 1
        if not return_value:
            break
        image = Image.fromarray(frame)
        image = yolo.detect_single_image(image)
        result = np.asarray(image)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        if output_path:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    end = timer()
    print("FPS:", frame_count / (end - start()))
    yolo.close_session()

def detect_test_set(yolo, test_path, output_path=""):
    with open(test_path) as f:
        test_lines = f.readlines()
    with open(output_path, 'w') as f:
        for line in test_lines:
            path = line.split()[0]
            image = Image.open(path)
            results = yolo.detect_single_image(image, True)
            f.write(path)
            for result in results:
                f.write(' %d,%d,%d,%d,%d,%f' % tuple(result))
            f.write('\n')
    yolo.close_session()
