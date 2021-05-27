import numpy as np
import cv2 as cv
from pathlib import Path
import os

HAARCASCADES = Path(os.path.dirname(__file__)).resolve() / 'haarcascades'

detectors = None


def init_detectors():
    return {
        'face': cv.CascadeClassifier(str(HAARCASCADES / 'haarcascade_frontalface_default.xml')),
        'eye': cv.CascadeClassifier(str(HAARCASCADES / 'haarcascade_eye.xml')),
        'nose': cv.CascadeClassifier(str(HAARCASCADES / 'nose.xml')),
        'mouth': cv.CascadeClassifier(str(HAARCASCADES / 'mouth.xml'))
    }


def load_detector(detector: str):
    global detectors

    if detectors is None:
        detectors = init_detectors()

    return detectors[detector]

#
# def area_diff_percent(pos_vec1: tuple, pos_vec2: tuple):
#     m = np.array((pos_vec1, pos_vec2))
#     area_diff = 0
#
#     x_dist = np.min(np.array((m[0, 0] + m[0, 2], m[1, 0] + m[1, 2]))) - np.max(np.array((m[0, 0], m[1, 0])))
#     y_dist = np.min(np.array((m[0, 1] + m[0, 3], m[1, 1] + m[1, 3]))) - np.max(np.array((m[0, 1], m[1, 1])))
#     if x_dist > 0 and y_dist > 0:
#         area_diff = x_dist * y_dist
#
#     return np.max(np.array((area_diff / (m[0, 2] * m[0, 3]), area_diff / (m[1, 2] * m[1, 3]))))


def detect(image,
           detector="face",
           scale_factor=1.1, min_neighbors=5, min_size=(30, 30),
           unsharp_mask=True, edge_enhance=True):
    img = image.copy()
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    if unsharp_mask:
        gaussian_blurred = cv.GaussianBlur(gray, (3, 3), 5.0)
        gray = cv.addWeighted(gray, 2, gaussian_blurred, -1, 0)

    if edge_enhance:
        edge_img = cv.Canny(gray, 200, 350)
        gray = cv.addWeighted(gray, 1, edge_img, 0.07, 0)

    face_detector = load_detector(detector)
    rects = face_detector.detectMultiScale(gray,
                                           scaleFactor=scale_factor,
                                           minNeighbors=min_neighbors,
                                           minSize=min_size,
                                           flags=cv.CASCADE_SCALE_IMAGE)

    yield from [tuple(rect) for rect in rects]
