import io
import json
from urllib.parse import urlparse, parse_qs
from http.server import BaseHTTPRequestHandler, HTTPServer

def serve(interface: str, pipe):

    class Handler(BaseHTTPRequestHandler):

        def gen(self, params):
            image = pipe(params).images[0]
            img_io = io.BytesIO()
            image.save(img_io, 'JPEG')
            img_io.seek(0)
            self.send_response(200)
            self.send_header('Content-Type', 'image/jpeg')
            self.end_headers()
            self.wfile.write(img_io.getvalue())

        def do_POST(self):
            content_length = int(self.headers['Content-Length'])
            params = json.loads(self.rfile.read(content_length).decode('utf-8'))
            self.gen(params)

        def do_GET(self):
            parsed_url = urlparse(self.path)
            params = {k: int(v[0]) if v[0].isdigit() else v[0] for k, v in parse_qs(parsed_url.query).items()}
            self.gen(params)

    host, port = interface.split(':') if ':' in interface else ('127.0.0.1', interface)

    server = HTTPServer((host, int(port)), Handler)
    server.serve_forever()
