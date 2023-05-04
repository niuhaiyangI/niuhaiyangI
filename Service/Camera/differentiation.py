import math

import pandas as pd
import torch
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fftpack import fft,ifft
class SLOT:
    def __init__(self, slot_size, s_list, red_average,fps):
        self.gama=5
        self.fps=fps
        self.slot_size=slot_size
        self.W_c1=torch.zeros([self.slot_size,3])
        self.W_c2 = torch.zeros([self.slot_size, 3])
        self.Hz=1.0/(slot_size/self.fps)
        self.heart_time=slot_size/self.fps
        # print(1.0/self.Hz)
        self.slot_list=s_list
        self.red_average=red_average
        self.max_frame=s_list[self.red_average.argmax()]
        self.max_average=self.red_average.max()
        self.min_frame=s_list[self.red_average.argmin()]
        self.min_average=self.red_average.min()
        self.bin_size = 10
        self.bin = [x for x in range(256) if x % int(256 / self.bin_size) == 0]
        self.diff=self.max_frame-self.min_frame
        M_temp=np.zeros(self.diff.shape)
        M_temp[:,:,2]=(self.diff[:,:,2]>self.gama)
        M_temp[:, :, 1] = (self.diff[:, :, 2] > self.gama)
        M_temp[:, :, 0] = (self.diff[:, :, 2] > self.gama)
        self.M=M_temp
        self.score=self._cal_score()
        self.W_c=self._cal_W()
        self.rolling_window=4



    def _cal_score(self):
        score=0
        for i in range(self.bin_size):
            score=score+i*i*(((self.diff[:,:,2]>=self.bin[i])&(self.diff[:,:,2]<self.bin[i+1])).sum()/(self.diff.shape[0]*self.diff.shape[1]))
        return score

    def _cal_W(self):
        W_list=[]
        for img in self.slot_list:
            temp=torch.tensor(img*self.M).cuda().sum(dim=(0,1))/torch.tensor(self.M).cuda().sum(dim=(0,1))
            W_list.append(temp.tolist())
        W_list=torch.tensor(W_list)
        W_list.cuda()
        return W_list

    def show(self):
        plt.close()
        x=range(self.slot_size)
        red_channel = self.W_c[:, 2]
        green_channel = self.W_c[:, 1]
        blue_channel = self.W_c[:, 0]
        plt.plot(x, red_channel.tolist(), color='r', label='red_average')
        plt.plot(x, green_channel.tolist(), color='g', label='green_average')
        plt.plot(x, blue_channel.tolist(), color='b', label='blue_average')
        plt.legend(loc='best')
        # plt.plot(x,self.W_c[:,2], color='r', label='red_average')
        # plt.plot(x, self.W_c[:, 1], color='g', label='green_average')
        # plt.plot(x, self.W_c[:, 0], color='b', label='blue_average')
        plt.show()

    def _change(self,x):
        return (x)/(self.slot_size-1)

    def get_Systolic_DiastolicFeature(self):
        # x = (torch.tensor(range(self.slot_size))) / (self.slot_size - 1)
        red_channel = self.W_c[:, 2]
        green_channel = self.W_c[:, 1]
        blue_channel = self.W_c[:, 0]
        red_channel = (red_channel - red_channel.min()) / (red_channel.max() - red_channel.min())
        green_channel = (green_channel - green_channel.min()) / (green_channel.max() - green_channel.min())
        blue_channel = (blue_channel - blue_channel.min()) / (blue_channel.max() - blue_channel.min())
        red_faeture=self.S_feature_get(red_channel)
        green_faeture = self.S_feature_get(green_channel)
        blue_faeture = self.S_feature_get(blue_channel)
        return red_faeture, green_faeture, blue_faeture






    def get_Non_fiducialFeature1(self):
        x = (torch.tensor(range(self.slot_size))) / (self.slot_size - 1)
        red_channel = self.W_c1[:, 2]
        green_channel = self.W_c1[:, 1]
        blue_channel = self.W_c1[:, 0]
        red_channel = (red_channel - red_channel.min()) / (red_channel.max() - red_channel.min())
        green_channel = (green_channel - green_channel.min()) / (green_channel.max() - green_channel.min())
        blue_channel = (blue_channel - blue_channel.min()) / (blue_channel.max() - blue_channel.min())
        red_faeture=self.N_feature_get(red_channel)
        green_faeture = self.N_feature_get(green_channel)
        blue_faeture = self.N_feature_get(blue_channel)
        return red_faeture,green_faeture,blue_faeture
        # sum = (red_channel + green_channel + blue_channel) / 3
        # check=sum
        # Series = pd.Series(check.tolist())
        # rol = Series.rolling(window=3).mean()
        # peak, _ = signal.find_peaks(check.tolist())
        # peak2, _ = signal.find_peaks((-check).tolist())



    def get_Non_fiducialFeature2(self):
        x = (torch.tensor(range(self.slot_size))) / (self.slot_size - 1)
        red_channel = self.W_c2[:, 2]
        green_channel = self.W_c2[:, 1]
        blue_channel = self.W_c2[:, 0]
        red_channel = (red_channel - red_channel.min()) / (red_channel.max() - red_channel.min())
        green_channel = (green_channel - green_channel.min()) / (green_channel.max() - green_channel.min())
        blue_channel = (blue_channel - blue_channel.min()) / (blue_channel.max() - blue_channel.min())
        red_faeture=self.N_feature_get(red_channel)
        green_faeture = self.N_feature_get(green_channel)
        blue_faeture = self.N_feature_get(blue_channel)
        return red_faeture, green_faeture, blue_faeture

    def show_Wc(self):
        print('show start')
        x = (torch.tensor(range(self.slot_size))) / (self.slot_size - 1)
        red_channel = self.W_c[:, 2]
        green_channel = self.W_c[:, 1]
        blue_channel = self.W_c[:, 0]
        red_channel = (red_channel - red_channel.min()) / (red_channel.max() - red_channel.min())
        green_channel = (green_channel - green_channel.min()) / (green_channel.max() - green_channel.min())
        blue_channel = (blue_channel - blue_channel.min()) / (blue_channel.max() - blue_channel.min())
        # sum=(red_channel+green_channel+blue_channel)/3
        sum=(red_channel+green_channel)/2
        plt.close()
        # plt.plot(x, self.W_c[:,2],color='pink')
        plt.plot(x, red_channel.tolist(), color='r', label='red_average')
        plt.plot(x, green_channel.tolist(), color='g', label='green_average')
        plt.plot(x, blue_channel.tolist(), color='b', label='blue_average')
        plt.legend(loc='best')
        # check=sum
        # # check=red_channel
        # plt.plot(x, check.tolist(), color='black', label='blue_average')
        # Series = pd.Series(check.tolist())
        # rol = Series.rolling(window=3).mean()
        # peak, _ = signal.find_peaks(check.tolist())
        # peak2, _ = signal.find_peaks((-check).tolist())
        # plt.plot(x, rol, color='pink', label='blue_average')
        # plt.plot(x[peak], check[peak].tolist(), "x", color='black',label='blue_average')
        # plt.plot(x[peak2], check[peak2].tolist(), "x", color='pink', label='blue_average')
        plt.show()

    def show_Wc1(self):
        print('show1 start')
        x = (torch.tensor(range(self.slot_size))) / (self.slot_size - 1)
        red_channel = self.W_c1[:, 2]
        green_channel = self.W_c1[:, 1]
        blue_channel = self.W_c1[:, 0]
        red_channel = (red_channel - red_channel.min()) / (red_channel.max() - red_channel.min())
        green_channel = (green_channel - green_channel.min()) / (green_channel.max() - green_channel.min())
        blue_channel = (blue_channel - blue_channel.min()) / (blue_channel.max() - blue_channel.min())
        sum=(red_channel+green_channel+blue_channel)/3
        plt.close()
        # plt.plot(x, self.W_c[:,2],color='pink')
        plt.plot(x, red_channel.tolist(), color='r', label='red_average')
        plt.plot(x, green_channel.tolist(), color='g', label='green_average')
        plt.plot(x, blue_channel.tolist(), color='b', label='blue_average')
        check=sum
        # check=red_channel
        plt.plot(x, check.tolist(), color='black', label='blue_average')
        Series = pd.Series(check.tolist())
        rol = Series.rolling(window=3).mean()
        peak, _ = signal.find_peaks(check.tolist())
        peak2, _ = signal.find_peaks((-check).tolist())
        plt.plot(x, rol, color='pink', label='blue_average')
        plt.plot(x[peak], check[peak].tolist(), "x", color='black',label='blue_average')
        plt.plot(x[peak2], check[peak2].tolist(), "x", color='pink', label='blue_average')
        plt.show()

    def show_Wc2(self):
        print('show2 start')
        x = (torch.tensor(range(self.slot_size))) / (self.slot_size - 1)
        red_channel = self.W_c2[:, 2]
        green_channel = self.W_c2[:, 1]
        blue_channel = self.W_c2[:, 0]
        red_channel = (red_channel - red_channel.min()) / (red_channel.max() - red_channel.min())
        green_channel = (green_channel - green_channel.min()) / (green_channel.max() - green_channel.min())
        blue_channel = (blue_channel - blue_channel.min()) / (blue_channel.max() - blue_channel.min())
        sum=(red_channel+green_channel+blue_channel)/3
        plt.close()
        # plt.plot(x, self.W_c[:,2],color='pink')
        plt.plot(x, red_channel.tolist(), color='r', label='red_average')
        plt.plot(x, green_channel.tolist(), color='g', label='green_average')
        plt.plot(x, blue_channel.tolist(), color='b', label='blue_average')
        check=sum
        # check=red_channel
        plt.plot(x, check.tolist(), color='black', label='blue_average')
        Series = pd.Series(check.tolist())
        rol = Series.rolling(window=3).mean()
        peak, _ = signal.find_peaks(check.tolist())
        peak2, _ = signal.find_peaks((-check).tolist())
        plt.plot(x, rol, color='pink', label='blue_average')
        plt.plot(x[peak], check[peak].tolist(), "x", color='black',label='blue_average')
        plt.plot(x[peak2], check[peak2].tolist(), "x", color='pink', label='blue_average')
        plt.show()


    def S_feature_get(self,input,red=False):
        x = (torch.tensor(range(self.slot_size))) / (self.slot_size - 1)
        band1=[0.1,0.3]
        default1=0.2
        band2=[0.21,0.45]
        default2 = 0.35
        band3=[0.6,1.0]
        default3 = 0.8
        check = input.tolist()
        check_neg=(-input).tolist()
        Series = pd.Series(check)
        rol = Series.rolling(window=3).mean()
        peak, _ = signal.find_peaks(check)
        peak2, _ = signal.find_peaks(check_neg)
        DP_x=self._get_default(default1)
        DP_t=list([x for x in peak if band1[0] <= x/(self.slot_size - 1) <= band1[1]])
        if len(DP_t):
            DP_x=DP_t[0]
        band2[0] = (DP_x + 1) / (self.slot_size - 1)



        DN_x = self._get_default(default2)
        DN_t = list([x for x in peak2 if band2[0] <= x/(self.slot_size - 1) <= band2[1]])
        if len(DN_t):
            DN_x = DN_t[0]


        SP_x = self._get_default(default3)
        SP_t = list([x for x in peak if band3[0] <= x / (self.slot_size - 1) <= band3[1]])
        if len(SP_t):
            SP_x = SP_t[0]

        h1=check[DP_x]
        h2=check[DN_x]
        t1=DP_x/ (self.slot_size - 1)
        t2=(DN_x/ (self.slot_size - 1))-(DP_x/ (self.slot_size - 1))
        t3=(SP_x/ (self.slot_size - 1))-(DN_x/ (self.slot_size - 1))
        t4=1.0-(SP_x/ (self.slot_size - 1))
        # if t1<=0 or math.isinf(t1) or math.isnan(t1):
        #     print("t1")
        #     print(t1)
        # if t2<=0 or math.isinf(t2) or math.isnan(t2):
        #     print(DN_x)
        #     print(DP_x)
        #     print(band2[0])
        #     print((DP_x+1)/(self.slot_size - 1))
        #     print(DP_x/(self.slot_size - 1))
        #     print(self.slot_size)
        #     print("t2")
        #     print(t2)
        # if t3<=0 or math.isinf(t3) or math.isnan(t3):
        #     print("t3")
        #     print(t3)
        # if t4<=0 or math.isinf(t4) or math.isnan(t4):
        #     print("t4")
        #     print(t4)

        s1=abs(h1/t1)
        s2 = abs(h2 / t2)
        s3 = abs(1.0 / t3)
        s4 = abs(check[self.slot_size-1] / t4)
        if red:
            plt.close()
            plt.plot(x,check,color='b',label='red_average')
            plt.axvline(x=0, ymin=0, ymax=check[0], c="k", ls="--", lw=2)
            plt.axvline(x=1.0, ymin=0, ymax=check[-1], c="k", ls="--", lw=2)
            plt.scatter(DP_x/(self.slot_size - 1), check[DP_x], color='r', s=50)  # 在最大值点上绘制一个红色的圆点
            plt.text(DP_x/(self.slot_size - 1), check[DP_x]+0.01,'DP')
            plt.axvline(x=DP_x/(self.slot_size - 1), ymin=0, ymax=check[DP_x], c="k", ls="--", lw=2)
            plt.scatter(DN_x/(self.slot_size - 1), check[DN_x]+0.01, color='r', s=50)  # 在最大值点上绘制一个红色的圆点
            plt.text(DN_x/(self.slot_size - 1), check[DN_x],'DN')
            plt.axvline(x=DN_x / (self.slot_size - 1), ymin=0, ymax=check[DN_x], c="k", ls="--", lw=2)
            plt.scatter(SP_x/(self.slot_size - 1), check[SP_x]+0.01, color='r', s=50)  # 在最大值点上绘制一个红色的圆点
            plt.text(SP_x/(self.slot_size - 1), check[SP_x],'SP')
            plt.axvline(x=SP_x / (self.slot_size - 1), ymin=0, ymax=check[SP_x], c="k", ls="--", lw=2)
            plt.axhline(y=0,xmin=0,xmax=DP_x/ (self.slot_size - 1),c='k',ls='solid',lw=2)
            plt.axhline(y=0, xmin=DP_x / (self.slot_size - 1), xmax=DN_x / (self.slot_size - 1), c='k', ls='solid', lw=2)
            plt.axhline(y=0, xmin=DN_x / (self.slot_size - 1), xmax=SP_x / (self.slot_size - 1), c='k', ls='solid', lw=2)
            plt.axhline(y=0, xmin=SP_x / (self.slot_size - 1), xmax=1.0, c='k', ls='solid', lw=2)
            plt.text(DP_x/ (self.slot_size - 1), 0.01, 't1')
            plt.text(DN_x/ (self.slot_size - 1), 0.01, 't2')
            plt.text(SP_x/ (self.slot_size - 1), 0.01, 't3')
            plt.text(1.0, 0.01, 't4')
            plt.legend(loc='best')
            plt.show()
        return h1,h2,t1,t2,t3,t4,s1,s2,s3,s4


    def N_feature_get(self,input,red=False):
        band1 = [0.1, 0.3]
        default1 = 0.2
        band2 = [0.21, 0.45]
        default2 = 0.3
        band3 = [0.31, 0.6]
        default3 = 0.5
        band4 = [0.5, 0.8]
        default4 = 0.7
        band5 = [0.7, 1.0]
        default5 = 0.9
        # band1 = [0.1, 0.3]
        # default1 = 0.2
        # band2 = [0.4, 0.6]
        # default2 = 0.3
        # band3 = [0.6, 0.8]
        # default3 = 0.5
        # band4 = [0.7, 0.8]
        # default4 = 0.7
        # band5 = [0.7, 1.0]
        # default5 = 0.9
        check = input.tolist()
        check_neg=(-input).tolist()
        Series = pd.Series(check)
        rol = Series.rolling(window=3).mean()
        peak, _ = signal.find_peaks(check)
        peak2, _ = signal.find_peaks(check_neg)

        y1_x = self._get_default(default1)
        y1_t = list([x for x in peak if band1[0] <= x / (self.slot_size - 1) <= band1[1]])
        if len(y1_t):
            y1_x = y1_t[0]
            band2[0]=(y1_x+1)/(self.slot_size - 1)

        y2_x = self._get_default(default2)
        y2_t = list([x for x in peak2 if band2[0] <= x / (self.slot_size - 1) <= band2[1]])
        if len(y2_t):
            y2_x = y2_t[0]

        y3_x = self._get_default(default3)
        y3_t = list([x for x in peak if band3[0] <= x / (self.slot_size - 1) <= band3[1]])
        if len(y3_t):
            y3_x = y3_t[0]

        y4_x = self._get_default(default4)
        y4_t = list([x for x in peak2 if band4[0] <= x / (self.slot_size - 1) <= band4[1]])
        if len(y4_t):
            y4_x = y4_t[0]

        y5_x = self._get_default(default5)
        y5_t = list([x for x in peak if band5[0] <= x / (self.slot_size - 1) <= band5[1]])
        if len(y5_t):
            y5_x = y5_t[0]

        x1=(y2_x/ (self.slot_size - 1))-(y1_x/ (self.slot_size - 1))
        x3=(y4_x/ (self.slot_size - 1))-(y3_x/ (self.slot_size - 1))
        x5=x3=1.0-(y5_x/ (self.slot_size - 1))
        if x1<0 :
            print("x1")
            print(x1)
        if x3<0:
            print("x3")
            print(x3)
        if x5<0:
            print("x5")
            print(x5)

        d12=abs(check[y1_x]-check[y2_x])
        d34 = abs(check[y3_x] - check[y4_x])
        d5 = check[y5_x]

        if red:
            plt.close()
            x = (torch.tensor(range(self.slot_size))) / (self.slot_size - 1)
            plt.plot(x,check,color='b',label='red_average')
            plt.axvline(x=0, ymin=0, ymax=check[0], c="k", ls="--", lw=2)
            plt.axvline(x=1.0, ymin=0, ymax=check[-1], c="k", ls="--", lw=2)
            plt.scatter(y1_x/(self.slot_size - 1), check[y1_x], color='r', s=50)  # 在最大值点上绘制一个红色的圆点
            plt.text(y1_x/(self.slot_size - 1), check[y1_x]+0.01,'y1')
            plt.axvline(x=y1_x/(self.slot_size - 1), ymin=0, ymax=check[y1_x], c="k", ls="--", lw=2)
            plt.scatter(y2_x/(self.slot_size - 1), check[y2_x]+0.01, color='r', s=50)  # 在最大值点上绘制一个红色的圆点
            plt.text(y2_x/(self.slot_size - 1), check[y2_x],'y2')
            plt.axvline(x=y2_x / (self.slot_size - 1), ymin=0, ymax=check[y2_x], c="k", ls="--", lw=2)
            plt.scatter(y3_x/(self.slot_size - 1), check[y3_x]+0.01, color='r', s=50)  # 在最大值点上绘制一个红色的圆点
            plt.text(y3_x/(self.slot_size - 1), check[y3_x],'y3')
            plt.axvline(x=y3_x / (self.slot_size - 1), ymin=0, ymax=check[y3_x], c="k", ls="--", lw=2)
            plt.scatter(y4_x / (self.slot_size - 1), check[y4_x] + 0.01, color='r', s=50)  # 在最大值点上绘制一个红色的圆点
            plt.text(y4_x / (self.slot_size - 1), check[y4_x], 'y4')
            plt.axvline(x=y4_x / (self.slot_size - 1), ymin=0, ymax=check[y4_x], c="k", ls="--", lw=2)
            plt.scatter(y5_x / (self.slot_size - 1), check[y5_x] + 0.01, color='r', s=50)  # 在最大值点上绘制一个红色的圆点
            plt.text(y5_x / (self.slot_size - 1), check[y5_x], 'y5')
            plt.axvline(x=y5_x / (self.slot_size - 1), ymin=0, ymax=check[y5_x], c="k", ls="--", lw=2)
            plt.axhline(y=0, xmin=y1_x / (self.slot_size - 1), xmax=y2_x / (self.slot_size - 1), c='k', ls='solid', lw=1)
            plt.axhline(y=0, xmin=y2_x / (self.slot_size - 1), xmax=y3_x / (self.slot_size - 1), c='k', ls='solid', lw=1)
            plt.axhline(y=0, xmin=y3_x / (self.slot_size - 1), xmax=y4_x / (self.slot_size - 1), c='k', ls='solid', lw=1)
            plt.axhline(y=0, xmin=y4_x / (self.slot_size - 1), xmax=y5_x / (self.slot_size - 1), c='k', ls='solid',
                        lw=1)
            plt.axhline(y=0, xmin=y5_x / (self.slot_size - 1), xmax=1.0, c='k', ls='solid',
                        lw=1)
            plt.text(y2_x/ (self.slot_size - 1), -0.04, 'x1')
            plt.text(y3_x/ (self.slot_size - 1), -0.04, 'x2')
            plt.text(y4_x / (self.slot_size - 1), -0.04, 'x3')
            plt.text(y5_x / (self.slot_size - 1), -0.04, 'x4')
            plt.text(1.0, -0.04, 'x5')
            plt.legend(loc='best')
            plt.show()

        return x1,x3,x5,d12,d34,d5





    def _get_default(self,default):
        x = (torch.tensor(range(self.slot_size))) / (self.slot_size - 1)
        temp=0
        min=self.fps*100
        for i in range(len(x)):
            dis=abs(x[i]-default)
            if dis<min:
                min=dis
                temp=i
        return temp

    def Get_Features(self):
        #获取66个features
        # print("features")
        features=[]
        feature_list=self.get_Systolic_DiastolicFeature()
        # print("get_Systolic_DiastolicFeature")
        for feature in feature_list:
            for f in feature:
                if f<0 or math.isinf(f) or math.isnan(f):
                    try:
                        raise ValueError
                    except:
                        raise
                features.append(f)

        # print("get_Non_fiducialFeature1")
        feature_list = self.get_Non_fiducialFeature1()
        for feature in feature_list:
            for f in feature:
                if f<0 or math.isinf(f) or math.isnan(f):
                    try:
                        raise ValueError
                    except:
                        raise
                features.append(f)

        # print("get_Non_fiducialFeature2")
        feature_list = self.get_Non_fiducialFeature2()
        for feature in feature_list:
            for f in feature:
                if f<0 or math.isinf(f) or math.isnan(f):
                    try:
                        raise ValueError
                    except:
                        raise
                features.append(f)
        return features
