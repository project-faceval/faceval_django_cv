from django import forms


class UploadFileForm(forms.Form):
    ext = forms.CharField()
    bimg = forms.ImageField()


class UploadFileFormBase64(forms.Form):
    ext = forms.CharField()
    bimg = forms.CharField()
