# 6. Add request ID middleware
# middleware.py
import uuid
from flask import request, g

def request_id_middleware():
    request_id = request.headers.get('X-Request-ID')
    if not request_id:
        request_id = str(uuid.uuid4())
    g.request_id = request_id
    return request_id