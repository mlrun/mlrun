from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from mlrun.runtimes.function import fake_nuclio_context


def example_function(context, event):
    print(event.body)
    return '{"aa": 5}'


class Handler(BaseHTTPRequestHandler):

    def do_call(self):
        print(f'got {self.command} request to {self.path}')
        body = self.rfile.read(
            int(self.headers['Content-Length']))

        context, event = fake_nuclio_context(
            body, headers=self.headers)

        resp = self.handler_function(context, event)

        if isinstance(resp, str):
            resp = bytes(resp.encode())

        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(resp)
        #self.wfile.close()
        #return

    def do_GET(self):
        print('path:', self.path)
        print('headers:', self.headers)
        print('command:', self.command)
        print('request:', self.request)
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(b'Hello world')

    def do_POST(self):
        self.do_call()

    def do_PUT(self):
        self.do_call()


class ThreadingSimpleServer(ThreadingMixIn, HTTPServer):
    pass


def create_function(handler, port):
    def func_wrap(self, context, event):
        return handler(context, event)

    CustomHandler = Handler
    CustomHandler.handler_function = func_wrap

    server = ThreadingSimpleServer(('0.0.0.0', port), CustomHandler)
    server.serve_forever()
