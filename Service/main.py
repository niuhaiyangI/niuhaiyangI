# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import cv2
from Camera.finger_cover import video




video_path="D:\\毕业设计\\实验素材\\7.mp4"







# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    camera = cv2.VideoCapture(video_path)
    if camera.isOpened():
        print("开始")
    else:
        camera.open(video)
        if camera.isOpened():
            print("成功")
    video=video(camera)
    video.run()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
