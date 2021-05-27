from django import forms


class UploadFileForm(forms.Form):
    ext = forms.CharField()
    bimg = forms.ImageField()
