import base64
import numpy as np
import cv2 as cv
import uuid
from pathlib import Path
import os


tmp_root = Path('/tmp/faceval-python/django/cv/').resolve()


def get_uuid():
    return str(uuid.uuid1())


def decode_base64(base64_str: str):
    return cv.imdecode(np.frombuffer(base64.b64decode(base64_str), dtype=np.uint8), flags=cv.IMREAD_COLOR)


def decode_binary(binary_img, ext):
    file_path = tmp_root / f"{get_uuid()}.{ext}"

    if not tmp_root.exists():
        os.makedirs(tmp_root)

    with open(file_path, 'wb+') as f:
        for chunk in binary_img.chunks():
            f.write(chunk)

    img = cv.imread(str(file_path))
    os.remove(file_path)
    return img
