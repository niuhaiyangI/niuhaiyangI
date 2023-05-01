import os
import torch
import cv2
from torchvision import transforms
import numpy as np
from PIL import Image, ImageDraw

outfile="out_over.txt"
out_file=open(outfile,'w+')
class video:
    def __init__(self,cam):
        self.camera=cam
        self.threshold=0.7
        self.percent=0.85
        self.bin_div=8
        self.list=[]

    ##顺序为BGR及[B,G,R]对应[:,:,0],[:,:,1],[:,:,2]
    def red_capture(self,img):
        img_tensor = torch.asarray(np.array(img),dtype=torch.int).cuda()
        pr_tensor=img_tensor[:,:,2]/(img_tensor[:,:,0]+img_tensor[:,:,1]+img_tensor[:,:,2]).cuda()
        cmp_torch=torch.tensor([[self.threshold]*pr_tensor.shape[1]]*pr_tensor.shape[0]).cuda()
        cmp=torch.ge(pr_tensor,cmp_torch).cuda()
        red_over=cmp.sum()/(img.shape[0]*img.shape[1])
        t_img=img_tensor.clone()
        t_img[:,:,0]=img_tensor[:,:,2]
        t_img[:,:,2]=img_tensor[:,:,0]
        t_img=t_img.cpu()
        if red_over>=self.percent:
            # 透明背景，RGBA值为：(0, 0, 0, 0)
            imgP = Image.new('RGB', (img.shape[0], img.shape[1]), (0, 0, 0))
            # img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
            # 填充像素
            print(img)
            print(type(img))
            img = Image.fromarray(t_img.numpy(),"RGB")
            img.show()
            print(img)
            imgP.putdata(img,scale=1.0)
            # 显示图片
            imgP.show()
            # 保存图片
            out_file.write("1")
            self.list.append(1)
            return True
        else:
            # img = Image.fromarray(t_img.numpy(), "RGB")
            # print(t_img.shape)
            # img.show()

            out_file.write("0")
            self.list.append(0)
            return False


    def score(self):
        times = 0
        while True:
            times = times + 1
            res, img = self.camera.read()
            if not res:
                print("over")
                print(self.list)
                break
            self.red_capture(img)



    def run(self):
        times=0
        while True:
            times=times+1
            res,img=self.camera.read()
            if not res:
                print("over")
                print(self.list)
                break
            self.red_capture(img)


video_path="D:\\毕业设计\\niuhaiyangI\\实验素材\\half_coverAndCover1.mp4"
# video_path="D:\\毕业设计\\niuhaiyangI\\实验素材\\8.mp4"




# Press the green button in the gutter to run the script.
if __name__ == '__main__':


    camera = cv2.VideoCapture(video_path)
    # camera = cv2.VideoCapture(match_path)
    if camera.isOpened():
        print("开始")
    else:
        camera.open(video)
        if camera.isOpened():
            print("成功")
    v=video(camera)
    v.run()
    # div.match()
    # video=video(camera)
    # video.run()