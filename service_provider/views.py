import http.client
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

from .image_decoder import decode_base64, decode_binary
from .rest.handlers import RestRequestHandler
from .cv.face_detect import detect
from .forms import UploadFileForm, UploadFileFormBase64


def detect_faces(img, detector='face', pick_count=None, pick_method="first"):
    faces = [*detect(img, detector)]
    face_count = len(faces)

    if pick_count is not None:
        if pick_method == "last":
            faces = faces[-min(pick_count, face_count):]
        elif pick_method == "first":
            faces = faces[:min(pick_count, face_count)]
        else:
            raise NotImplementedError()

    for face in faces:
        pos = tuple(face)

        yield {
            "start_x": int(pos[0]),
            "start_y": int(pos[1]),
            "length_x": int(pos[2]),
            "length_y": int(pos[3]),
        }


class ImageDetectViewSet(RestRequestHandler):

    def post(self, request, *args, **kwargs):
        use_base64 = request.POST.get('use_base64')
        if use_base64 is not None and use_base64 == 'yes':
            form = UploadFileFormBase64(request.POST)
            use_base64 = True
        else:
            form = UploadFileForm(request.POST, request.FILES)
            use_base64 = False

        if not form.is_valid():
            return HttpResponse(status=http.client.BAD_REQUEST)

        img = decode_binary(form.cleaned_data['bimg'], form.cleaned_data['ext'], use_base64)

        faces = [*detect_faces(img, 'face')]
        face_count = len(faces)

        return JsonResponse({
            'face': faces,
            'eye': [*detect_faces(img, detector='eye', pick_count=face_count * 2, pick_method='first')],
            'nose': [*detect_faces(img, detector='nose', pick_count=face_count, pick_method='first')],
            'mouth': [*detect_faces(img, detector='mouth', pick_count=face_count, pick_method='last')],
        }, safe=False)

    @csrf_exempt
    def rest_view(self, request, *args, **kwargs):
        return super().rest_view(request, *args, **kwargs)
