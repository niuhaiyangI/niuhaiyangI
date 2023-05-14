import webbrowser
from wsgiref.simple_server import make_server


def demo_app(environ, start_response):
    path = environ['PATH_INFO']
    print(environ)
    status_code = '200 OK'  # 默认状态码是200
    if path == '/':
        response = '欢迎'
    elif path == '/text':
        response = ''
    elif path == '/demo':

        response = 'demo'
    else:
        response = '页面走丢了！'  # 如果页面出问题了，
        status_code = '404 Not Found'

    start_response(status_code, [('Content-Type', 'text/html;charset=utf8')])
    return [response.encode('utf8')]

if __name__ == '__main__':
    httpd = make_server('10.50.137.11', 8080, demo_app)

    sa = httpd.socket.getsockname()
    print('Serving HTTP on', sa[0], 'port', sa[1], '...')
    # httpd.handle_request()
    httpd.serve_forever()
