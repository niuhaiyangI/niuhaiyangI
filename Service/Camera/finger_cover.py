import os
import torch
from torchvision import transforms
import numpy as np
outfile="out_over.txt"
out_file=open(outfile,'w+')
class video:
    def __init__(self,cam):
        self.camera=cam
        self.threshold=0.65
        self.percent=0.95
        self.bin_div=8
        self.list=[]

    ##顺序为BGR及[B,G,R]对应[:,:,0],[:,:,1],[:,:,2]
    def red_capture(self,img):
        img_tensor = torch.asarray(np.array(img),dtype=torch.int).cuda()
        pr_tensor=img_tensor[:,:,2]/(img_tensor[:,:,0]+img_tensor[:,:,1]+img_tensor[:,:,2]).cuda()
        cmp_torch=torch.tensor([[self.threshold]*pr_tensor.shape[1]]*pr_tensor.shape[0]).cuda()
        cmp=torch.ge(pr_tensor,cmp_torch).cuda()
        red_over=cmp.sum()/(img.shape[0]*img.shape[1])
        if red_over>=self.percent:
            out_file.write("1")
            self.list.append(1)
            return True
        else:
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