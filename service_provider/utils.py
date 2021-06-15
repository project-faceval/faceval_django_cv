import json


def json_request_compat(request, method="GET"):
    if request.content_type == 'application/json':
        return json.loads(request.body)

    if method == "POST":
        return request.POST
    else:
        return request.GET
