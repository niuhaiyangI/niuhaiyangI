# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import cv2
from Camera.finger_cover import video
from Camera.divide import divide
from service.service import demo_app
from wsgiref.simple_server import make_server
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"]=["SimHei"]
plt.rcParams["axes.unicode_minus"]=False

# video_path="D:\\毕业设计\\niuhaiyangI\\实验素材\\profile\\1fps60.mp4"
video_path="D:\\毕业设计\\niuhaiyangI\\实验素材\\profile\\30fps1080p.mp4"


# match_path="D:\\毕业设计\\niuhaiyangI\\实验素材\\11.mp4"
# match_path="D:\\毕业设计\\niuhaiyangI\\实验素材\\test\\5.mp4"
# match_path="D:\\毕业设计\\niuhaiyangI\\实验素材\\test\\pos\\4.mp4"
match_path="D:\\毕业设计\\niuhaiyangI\\实验素材\\test\\neg\\wyh.mp4"

# match_path="D:\\毕业设计\\niuhaiyangI\\实验素材\\11.mp4"



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # httpd = make_server('127.0.0.1', 8080, demo_app)
    #
    # sa = httpd.socket.getsockname()
    # print('Serving HTTP on', sa[0], 'port', sa[1], '...')
    #
    # httpd.serve_forever()

    camera = cv2.VideoCapture(video_path)
    # camera = cv2.VideoCapture(match_path)
    if camera.isOpened():
        print("开始")
    else:
        camera.open(video)
        if camera.isOpened():
            print("成功")
    div=divide(camera)
    div.show_red_channel()
    div.profile_feature("Niu Haiyang")
    # div.match()
    # video=video(camera)
    # video.run()

    # camera = cv2.VideoCapture(match_path)
    # if camera.isOpened():
    #     print("开始")
    # else:
    #     camera.open(video)
    #     if camera.isOpened():
    #         print("成功")
    # div=divide(camera)
    # div.show_red_channel()
    # # div.profile_feature("Niu Haiyang")
    # div.match()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
