# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import cv2
from Camera.finger_cover import video
from Camera.divide import divide
from service.service import demo_app
from wsgiref.simple_server import make_server


video_path="D:\\毕业设计\\niuhaiyangI\\实验素材\\profile\\3.mp4"
# video_path="D:\\毕业设计\\niuhaiyangI\\实验素材\\8.mp4"


# match_path="D:\\毕业设计\\niuhaiyangI\\实验素材\\12.mp4"
match_path="D:\\毕业设计\\niuhaiyangI\\实验素材\\test\\xxz.mp4"




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # httpd = make_server('127.0.0.1', 8080, demo_app)
    #
    # sa = httpd.socket.getsockname()
    # print('Serving HTTP on', sa[0], 'port', sa[1], '...')
    #
    # httpd.serve_forever()

    # camera = cv2.VideoCapture(video_path)
    # # camera = cv2.VideoCapture(match_path)
    # if camera.isOpened():
    #     print("开始")
    # else:
    #     camera.open(video)
    #     if camera.isOpened():
    #         print("成功")
    # div=divide(camera)
    # div.show_red_channel()
    # div.profile_feature("Niu Haiyang")
    # # div.match()
    # # video=video(camera)
    # # video.run()

    camera = cv2.VideoCapture(match_path)
    if camera.isOpened():
        print("开始")
    else:
        camera.open(video)
        if camera.isOpened():
            print("成功")
    div=divide(camera)
    div.show_red_channel()
    # div.profile_feature("Niu Haiyang")
    div.match()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
