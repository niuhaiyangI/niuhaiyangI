import math
import os
import pickle

import torch
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
# plt.ion()
import cv2
from scipy import signal
import pandas as pd
import json

# 获取数据帧数量
# frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
# 获取帧率
# fps = videoCapture.get(cv2.CAP_PROP_FPS)
from Camera.differentiation import SLOT


class divide:
    def __init__(self, cam):
        print("SLOT devide starting....")
        self.path='./Profile/User'
        self.distance=4
        self.rolling_window=2*self.distance+1
        self.heart_pump_seconds = 0.8  ##心脏跳动时间
        self.heart_pump_max = 1.3
        self.heart_pump_min = 0.3
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
        self.peak,_=signal.find_peaks(-self.red_average)
        self.slots_list,self.slots_size,self.score_average,self.heart_pump_frames,self.divide,self.W_c = self.get_slots()
        self.real_pump_time=((self.heart_pump_frames)/self.fps)/self.slots_size
        self.print()

    def cal_red(self):
        for i in range(len(self.img_list)):
            img_tensor = torch.asarray(np.array(self.img_list[i]), dtype=torch.int).cuda()
            self.red_average[i] = img_tensor[:,:,2].sum() / (img_tensor.shape[0] * img_tensor.shape[1])
        # b, a = signal.butter(10, [2 * (0.3 / 30), 2 * (10.0 / 30)], btype='bandpass')
        # b, a = signal.butter(10, [0.3,10.0], btype='bandpass',fs=self.fps)
        # temp=signal.filtfilt(b,a,self.red_average)
        # plt.close()
        # plt.plot(range(self.frames_num),temp,color='b')
        # # plt.plot(range(self.frames_num),self.red_average,color='r')
        # plt.show()
            # print(self.red_average[i])

    def show_red_channel(self):
        peak,_=signal.find_peaks(-self.red_average)
        plt.close()
        x = range(self.frames_num)
        Series = pd.Series(self.red_average)
        rol=Series.rolling(window=self.rolling_window).mean()
        peak2,_=signal.find_peaks(-rol)
        s_list = []
        for i in peak2:
            s_list.append(self._get_closest_point(i))
        plt.plot(s_list, self.red_average[s_list], "x", color='r', label='red_average')
        # plt.plot(peak, self.red_average[peak], "x", color='r', label='red_average')
        # plt.plot(range(self.frames_num-self.distance), rol[self.distance:self.frames_num], color='b')
        # plt.plot( range(self.frames_num-self.window_min), rol[self.window_min-1:-1], color='b')
        plt.plot(x,self.red_average,color='g')
        b, a = signal.butter(10, [0.3,10.0], btype='bandpass',fs=self.fps)
        temp=signal.filtfilt(b,a,self.red_average)
        plt.close()
        plt.plot(x,self.red_average,color='b')
        # plt.plot(self.peak,self.red_average[self.peak],"x",color="black")
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


    def _get_closest_point(self,index):
        index_temp=index-self.distance
        temp=int(self.window_min/2)
        for i in range(index_temp-temp,index_temp+temp+1):
            if i in self.peak:
                return i
        return index

    def get_slots(self):
        W_c=[]
        pump_frames=0
        Series = pd.Series(self.red_average)
        rol = Series.rolling(window=self.rolling_window).mean()
        peak, _ = signal.find_peaks(-rol)
        s_list = []
        for i in peak:
            s_list.append(self._get_closest_point(i))
        slots = []
        score_sum = 0
        for i in range(len(s_list) - 1):
            average = self.red_average[s_list[i]:s_list[i + 1] + 1]
            temp_list = []
            for j in range(s_list[i], s_list[i + 1] + 1):

                temp_list.append(self.img_list[j])
            slot = SLOT(s_list[i + 1] - s_list[i] + 1, temp_list, average, self.fps)
            if slot.slot_size >= self.window_min and slot.slot_size <= self.window_max:
                pump_frames=pump_frames+slot.slot_size
                score_sum = score_sum + slot.score
                slots.append(slot)
                W_c=W_c+slot.W_c.tolist()
        W_c=torch.tensor(W_c).cuda()
        return slots, len(slots) , score_sum / len(slots),pump_frames,s_list,W_c

    #带通滤波
    def band_pass(self):
        b, a = signal.butter(10, [0.3, 10], btype='bandpass',fs=self.fps)
        # b, a = signal.butter(10, 0.3*2, btype='highpass',fs=30)
        r = signal.filtfilt(b, a, self.W_c[:,2].tolist(),padtype='odd')
        g = signal.filtfilt(b, a, self.W_c[:, 1].tolist(),padtype='odd')
        b = signal.filtfilt(b, a, self.W_c[:, 0].tolist(),padtype='odd')
        self.W_c[:,2]=torch.tensor(r.copy()).cuda()
        self.W_c[:, 1] = torch.tensor(g.copy()).cuda()
        self.W_c[:, 0] = torch.tensor(b.copy()).cuda()
        index=0
        for slot in self.slots_list:
            slot.W_c=self.W_c[index:index+slot.slot_size]
            index=index+slot.slot_size
            # slot.show_Wc()

    def high_pass1(self):
        T_Wc=torch.zeros([self.W_c.shape[0],3])
        b, a = signal.butter(10, 1, btype='highpass', fs=self.fps)
        r = signal.filtfilt(b, a, self.W_c[:, 2].tolist(),padtype='odd')
        g = signal.filtfilt(b, a, self.W_c[:, 1].tolist(),padtype='odd')
        b = signal.filtfilt(b, a, self.W_c[:, 0].tolist(),padtype='odd')
        T_Wc[:, 2] = torch.tensor(r.copy()).cuda()
        T_Wc[:, 1] = torch.tensor(g.copy()).cuda()
        T_Wc[:, 0] = torch.tensor(b.copy()).cuda()
        index = 0
        for slot in self.slots_list:
            slot.W_c1 = T_Wc[index:index + slot.slot_size]
            index = index + slot.slot_size
            # slot.show_Wc1()

    def high_pass2(self):
        T_Wc = torch.zeros([self.W_c.shape[0], 3])
        b, a = signal.butter(10, 2, btype='highpass', fs=self.fps)
        r = signal.filtfilt(b, a, self.W_c[:, 2].tolist(),padtype='odd')
        g = signal.filtfilt(b, a, self.W_c[:, 1].tolist(),padtype='odd')
        b = signal.filtfilt(b, a, self.W_c[:, 0].tolist(),padtype='odd')
        T_Wc[:, 2] = torch.tensor(r.copy()).cuda()
        T_Wc[:, 1] = torch.tensor(g.copy()).cuda()
        T_Wc[:, 0] = torch.tensor(b.copy()).cuda()
        index = 0
        for slot in self.slots_list:
            slot.W_c2 = T_Wc[index:index + slot.slot_size]
            index = index + slot.slot_size
            # slot.show_Wc2()

    # def profile_feature(self,Username):
    #     threshold=0.9
    #     self.band_pass()
    #     self.high_pass1()
    #     self.high_pass2()
    #     feature_list=[]
    #     cont=1
    #     for slot in self.slots_list:
    #         if cont>70:
    #             break
    #         s_features=slot.Get_Features()
    #         feature_list.append(s_features)
    #         cont=cont+1
    #
    #     A=torch.tensor(feature_list)
    #     A=A.transpose(0,1)
    #     print(A)
    #     print(A.shape)
    #     U,S,Vh=torch.linalg.svd(A)
    #     print(U)
    #     print(U.shape)
    #     T=torch.mm(U,A)
    #     print("T<0")
    #     print((T<0).sum())
    #     print(T.shape)
    #     print((T[0:1].sum(dim=0)/T.sum(dim=0)).max())
    #     T_norm=T.norm(p=2,dim=1,keepdim=False)
    #     print(T_norm)
    #     print(T_norm.shape)
    #     a,index=torch.sort(T_norm,descending=True)
    #     print(a)
    #     print(index)
    #     sum=0.0
    #     k=1
    #     for i in index:
    #         sum=sum+(T_norm[i]/T_norm.sum())
    #         if sum>=threshold:
    #             print(sum)
    #             print(k)
    #             break
    #         k=k+1
    #     id,_=torch.sort(index[:k])
    #     print(id)
    #     print(id.shape)
    #     T_list=T.tolist()
    #     f_vector=[]
    #     for i in id:
    #         f_vector.append(T_list[i])
    #     f_vector=torch.tensor(f_vector).transpose(0,1)
    #     print(f_vector.shape)
    #     torch.save(U,os.path.join(self.path,'U.pt'))
    #     torch.save(f_vector,os.path.join(self.path,'f_vector.pt'))
    #     torch.save(id,os.path.join(self.path,'id.pt'))
    #     profile_dic={'Username':Username,'k':k}
    #     np.save(os.path.join(self.path,'profile_dic.npy'),profile_dic,allow_pickle=True)


    def profile_feature(self,Username):
        threshold=0.9
        self.band_pass()
        self.high_pass1()
        self.high_pass2()
        feature_list=[]
        cont=1
        for slot in self.slots_list:
            if cont>70:
                break
            s_features=slot.Get_Features()
            feature_list.append(s_features)
            cont=cont+1

        A=torch.tensor(feature_list).cuda()
        A=A.transpose(0,1)
        print(A)
        print(A.shape)
        U,S,Vh=torch.linalg.svd(A)
        print(U)
        print(U.shape)
        T=torch.mm(U,A).cuda()
        print("T<0")
        print((T<0).sum())
        print(T.shape)
        print((T[0:1].sum(dim=0)/T.sum(dim=0)).max())

        T_var=torch.var(T,dim=1,unbiased=False,keepdim=False)
        print("T_var")
        print(T_var)
        print(T_var.shape)
        a,index=torch.sort(T_var,descending=True)
        print(a)
        print(index)
        sum=0.0
        k=1
        cont=2
        for i in index:
            cont=cont-1
            if cont<0:
                sum=sum+(T_var[i]/T_var.sum())
            if sum>=threshold:
                print(sum)
                print(k)
                break
            k=k+1
        id,_=torch.sort(index[2:k])
        print(id)
        print(id.shape)
        T_list=T.tolist()
        f_vector=[]
        for i in id:
            f_vector.append(T_list[i])
        f_vector=torch.tensor(f_vector).transpose(0,1)
        print(f_vector.shape)
        torch.save(U,os.path.join(self.path,'U.pt'))
        torch.save(f_vector,os.path.join(self.path,'f_vector.pt'))
        torch.save(id,os.path.join(self.path,'id.pt'))
        profile_dic={'Username':Username,'k':k}
        np.save(os.path.join(self.path,'profile_dic.npy'),profile_dic,allow_pickle=True)


    def match(self):
        U=torch.load(os.path.join(self.path,'U.pt'))
        f_vector=torch.load(os.path.join(self.path,'f_vector.pt'))
        id=torch.load(os.path.join(self.path,'id.pt'))
        load_dict = np.load(os.path.join(self.path,'profile_dic.npy'),allow_pickle=True).item()
        k=load_dict['k']
        print(f_vector.shape)
        self.band_pass()
        self.high_pass1()
        self.high_pass2()
        dist=[]
        for slot in self.slots_list:
            s_features = slot.Get_Features()
            feature_list=[]
            feature_list.append(s_features)
            A = torch.tensor(feature_list).cuda()
            A = A.transpose(0, 1)
            # print(A)
            # print(A.shape)
            T = torch.mm(U, A)
            T_list = T.tolist()
            s = []
            for i in id:
                s.append(T_list[i])
            s = torch.tensor(s).transpose(0, 1)
            dis=self.dist(f_vector,s)
            print(dis)
            dist.append(dis)
        dist=torch.tensor(dist)
        print("max")
        print(dist.max())
        print("average")
        print(dist.sum()/dist.shape[0])
        print("min")
        print(dist.min())


    def dist(self,f,s):
        s_feature=s.repeat(70,1)
        dist=(f-s_feature).norm(p=2,dim=1,keepdim=False).sum()/70
        return dist

