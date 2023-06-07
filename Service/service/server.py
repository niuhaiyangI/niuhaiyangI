
#coding=utf-8
from http.server import BaseHTTPRequestHandler
import cgi
import cv2
from Camera.divide import  divide
from urllib.parse import quote

class   PostHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={'REQUEST_METHOD':'POST',
                     'CONTENT_TYPE':self.headers['Content-Type']
                     }
        )
        # self.send_response(200)
        # self.end_headers()
        # self.wfile.write('Client: {} '.format(str(self.client_address)).encode() )
        # self.wfile.write('User-agent: {}'.format(str(self.headers['user-agent'])).encode() )
        # self.wfile.write('Path: %s'.format(self.path).encode())
        # self.wfile.write('Form data:'.encode())
        for field in form.keys():
            field_item = form[field]
            print(type(field_item))
            filename = field_item.filename
            file=field_item.file
            print(type(filename))
            filevalue  = field_item.value
            filesize = len(filevalue)#文件大小(字节)
            print (len(filevalue))
            f=open('copy'+filename,'wb')
            f.write(filevalue)
            # with open('copy'+filename,'wb') as f:
            #     f.write(filevalue)
            camera = cv2.VideoCapture('copy'+filename)
            # camera = cv2.VideoCapture(match_path)
            if camera.isOpened():
                print("开始")
            else:
                camera.open(f)
                if camera.isOpened():
                    print("成功")
            div = divide(camera)
            # div.show_red_channel()
            restr=div.match()
        print(restr)
        self.send_response(200, quote(restr.encode('utf-8')))
        # self.send_response(200,(restr).encode('utf-8'))
        self.end_headers()
        return
if __name__=='__main__':
    from http.server import HTTPServer
    sever = HTTPServer(('10.50.137.11',8080),PostHandler)
    print ('Listening : ip = %s' % str('10.50.137.11'))
    print ('Listening : port = %d' % 8080)
    print ('HttpServer Starting , use <Ctrl-C> to stop')
    sever.serve_forever()
