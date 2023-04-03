import os
import torch
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
# plt.ion()
import cv2
from scipy import signal

# 获取数据帧数量
# frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
# 获取帧率
# fps = videoCapture.get(cv2.CAP_PROP_FPS)
from Camera.differentiation import SLOT


class divide:
    def __init__(self, cam):
        print("SLOT devide starting....")
        self.heart_pump_seconds = 0.8  ##心脏跳动时间
        self.heart_pump_max = 1.1
        self.heart_pump_min = 0.5
        self.camera = cam
        self.frames_num = int(self.camera.get(cv2.CAP_PROP_FRAME_COUNT))
        # print(self.frames_num)
        self.fps = int(self.camera.get(cv2.CAP_PROP_FPS))
        self.window = int(self.heart_pump_seconds * self.fps)
        self.window_max = int(self.heart_pump_max * self.fps)
        self.window_min = int(self.heart_pump_min * self.fps)
        # print(self.fps)
        self.img_list = []
        times = 0
        while True:
            times = times + 1
            res, img = self.camera.read()
            if not res:
                break
            self.img_list.append(img)
        self.pixel_num=self.img_list[0].shape[0]*self.img_list[0].shape[1]
        self.red_average = torch.zeros(self.frames_num)
        self.cal_red()
        self.slots_list,self.slots_size,self.score_average = self.get_slots()
        self.real_pump_time=(self.frames_num/self.fps)/self.slots_size
        self.print()

    def cal_red(self):
        for i in range(len(self.img_list)):
            img_tensor = torch.asarray(np.array(self.img_list[i]), dtype=torch.int).cuda()
            self.red_average[i] = img_tensor[:,:,2].sum() / (img_tensor.shape[0] * img_tensor.shape[1])
            # print(self.red_average[i])

    def show_red_channel(self):
        peak,_=signal.find_peaks(-self.red_average)
        plt.close()
        x = range(self.frames_num)
        plt.plot( peak,self.red_average[peak],"x", color='r', label='red_average')
        plt.plot(x,self.red_average,"--",color='g')
        plt.show()

    def print(self):
        print("心脏跳动时间(单位  秒):" + str(self.heart_pump_seconds))
        print("心脏跳动分帧窗口大小（心脏跳动一次所需帧数）:" + str(self.window))
        print("帧数:" + str(self.frames_num))
        print("帧率(单位  帧/秒):" + str(self.fps))
        print("心脏跳动次数:" + str(self.slots_size))
        print("心脏实际跳动时间(单位：秒):" + str(self.real_pump_time))
        print("评估参数score:" + str(self.score_average))
        print("pixel点数:" + str(self.pixel_num))
        print('Overdown')

    # def get_slot(self, index):
    #     if index >=self.frames_num:
    #         return self.frames_num
    #     slot_average = -self.red_average[index:index + self.window_max]
    #     if index == 0:
    #         return slot_average.argmax()
    #     s_peak = signal.find_peaks(slot_average)
    #     if len(s_peak[0]):
    #         return s_peak[0][0] + index
    #     else:
    #         return self.get_slot(index+self.window_min)

    # def get_slot(self,index):
    #     if index >=self.frames_num:
    #         return self.frames_num
    #     slot_average = self.red_average[index:index + self.window_max]
    #     if slot_average.argmin() is not None:
    #         return slot_average.argmin()+index
    #     else:
    #         return index+1


    def get_slots(self):
        s_list = []
        DN_list=[]
        peak, _ = signal.find_peaks(-self.red_average)
        times=0
        for i in peak:
            # print("this is peak")
            # print(i)
            if times%2==0:
                s_list.append(i)
            else:
                DN_list.append(i)
            times=times+1
        slots = []
        score_sum = 0
        for i in range(len(s_list) - 1):
            average = self.red_average[s_list[i]:s_list[i + 1] + 1]
            temp_list = []
            for j in range(s_list[i], s_list[i + 1] + 1):
                # print(str(j)+" "+str(s_list[i])+" "+str(s_list[i+1])+" ")
                temp_list.append(self.img_list[j])
            slot = SLOT(s_list[i + 1] - s_list[i] + 1, temp_list, average, self.fps)
            slot.DN_index=DN_list[i]
            # slot.show()
            slot.get_Systolic_DiastolicFeature()
            if slot.Hz >= 0.3 and slot.Hz <= 10.0:
                score_sum = score_sum + slot.score
                slots.append(slot)
        return slots, len(s_list) - 1, score_sum / (len(s_list) - 1)

    # def get_slots(self):
    #     peak,_=signal.find_peaks(-self.red_average)
    #     index = 0
    #     cont=0
    #     s_list = []
    #     index = self.get_slot(index)
    #     while index < self.frames_num:
    #         s_list.append(index)
    #         # print(str(index) + '均值为：' + str(self.red_average[index]))
    #         temp = self.get_slot(index+1)
    #         while (temp-index)<self.window_min:
    #             temp=self.get_slot(temp+1)
    #         index=temp
    #     slots = []
    #     score_sum=0
    #     for i in range(len(s_list) - 1):
    #         average = self.red_average[s_list[i]:s_list[i + 1] + 1]
    #         temp_list = []
    #         for j in range(s_list[i], s_list[i+1] + 1):
    #             temp_list.append(self.img_list[j])
    #         slot = SLOT(s_list[i + 1] - s_list[i] + 1, temp_list, average,self.fps)
    #         # slot.show()
    #         # slot.get_Systolic_DiastolicFeature()
    #         if slot.Hz>=0.3 and slot.Hz<=10.0:
    #             score_sum=score_sum+slot.score
    #             slots.append(slot)
    #     return slots,len(s_list) - 1,score_sum/(len(s_list) - 1)

    def get_slot(self, index):
        slot_average = -self.red_average[index:index + self.window_max]
        if index == 0:
            return slot_average.argmax()
        s_peak = signal.find_peaks(slot_average)
        if len(s_peak[0]):
            return s_peak[0][0] + index
        else:
            return index
    #
    # def get_slots(self):
    #     index = 0
    #     s_list = []
    #     index = self.get_slot(index)
    #     while index < self.frames_num:
    #         s_list.append(index)
    #         # print(str(index) + '均值为：' + str(self.red_average[index]))
    #         index = self.get_slot(index + self.window_min)
    #     slots = []
    #     score_sum=0
    #     for i in range(len(s_list) - 1):
    #         average = self.red_average[s_list[i]:s_list[i + 1] + 1]
    #         temp_list = []
    #         for j in range(s_list[i], s_list[i+1] + 1):
    #             temp_list.append(self.img_list[j])
    #         slot = SLOT(s_list[i + 1] - s_list[i] + 1, temp_list, average,self.fps)
    #         # slot.show()
    #         if slot.Hz>=0.3 and slot.Hz<=10.0:
    #             score_sum=score_sum+slot.score
    #             slots.append(slot)
    #     return slots,len(s_list) - 1,score_sum/(len(s_list) - 1)