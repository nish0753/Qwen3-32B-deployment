import http.server
import urllib.request
import urllib.error
import os
from urllib.parse import urlparse, parse_qs

PORT = 3000
UPSTREAM = os.environ.get('UPSTREAM_URL', 'http://10.43.188.45:8000')
AUTH_KEY = os.environ.get('UI_AUTH_KEY', '')


class ProxyHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
        super().end_headers()

    def _check_key(self, from_query=False):
        if not AUTH_KEY:
            return True
        if from_query:
            qs = parse_qs(urlparse(self.path).query)
            return qs.get('key', [''])[0] == AUTH_KEY
        return self.headers.get('X-Auth-Key') == AUTH_KEY

    def _deny(self):
        self.send_response(403)
        self.send_header('Content-Type', 'text/plain')
        self.end_headers()
        self.wfile.write(b'Forbidden: missing or invalid key')

    def do_GET(self):
        if self.path.startswith('/v1/'):
            if not self._check_key():
                return self._deny()
            return self._proxy('GET')
        if self.path == '/' or self.path.startswith('/?'):
            self.path = '/chat_ui.html' + self.path[1:]
        return super().do_GET()

    def do_POST(self):
        if self.path.startswith('/v1/'):
            if not self._check_key():
                return self._deny()
            return self._proxy('POST')
        return self._deny()

    def _proxy(self, method):
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length) if length else None
        req = urllib.request.Request(UPSTREAM + self.path, data=body, method=method)
        req.add_header('Content-Type', self.headers.get('Content-Type', 'application/json'))
        try:
            with urllib.request.urlopen(req) as upstream:
                self.send_response(upstream.status)
                self.send_header('Content-Type', upstream.headers.get('Content-Type', 'application/json'))
                self.send_header('Cache-Control', 'no-cache')
                self.end_headers()
                while True:
                    chunk = upstream.read(512)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            self.end_headers()
            self.wfile.write(e.read())
        except Exception as e:
            self.send_response(502)
            self.end_headers()
            self.wfile.write(str(e).encode())


if __name__ == '__main__':
    if AUTH_KEY:
        print(f'Auth key active: {AUTH_KEY}')
    server = http.server.ThreadingHTTPServer(('0.0.0.0', PORT), ProxyHandler)
    server.serve_forever()
