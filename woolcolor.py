import pickle
import os
from PIL import Image
import numpy as np

# 羊毛 35
class WoolColor():
    # 默认参数(手动取色器获取的)
    colors = {
        'white'     : [0, np.array([[[232,232,232]]])],  # 白
        'orange'    : [1, np.array([[[228,116,24]]])],   # 橙
        'magenta'   : [2, np.array([[[164,51,154]]])],   # 品红
        'light_blue': [3, np.array([[[60,174,207]]])],   # 淡蓝
        'yellow'    : [4, np.array([[[235,193,45]]])],   # 黄色
        'lime'      : [5, np.array([[[123,189,34]]])],   # 酸橙
        'pink'      : [6, np.array([[[221,122,153]]])],  # 粉
        'grey'      : [7, np.array([[[56,61,65]]])],     # 灰
        'light_grey': [8, np.array([[[132,132,125]]])],  # 亮灰
        'cyan'      : [9, np.array([[[20,138,141]]])],   # 青
        'purple'    : [10, np.array([[[114,39,160]]])],  # 紫
        'blue'      : [11, np.array([[[51,54,148]]])],   # 蓝
        'brown'     : [12, np.array([[[103,64,36]]])],   # 棕
        'green'     : [13, np.array([[[82,107,24]]])],   # 绿
        'red'       : [14, np.array([[[146,34,31]]])],   # 红
        'black'     : [15, np.array([[[0,0,0]]])]        # 黑
    }

    # 初始化羊毛参数
    def __init__(self):
        if os.path.isfile('woolcolors.pkl'):
            with open('woolcolors.pkl', 'rb') as f:
                self.colors = pickle.load(f)
                print('从颜色文件读取数据完毕')
            return
        
        if os.path.exists('wool_picture'):
            for cls in self.colors:
                path = 'wool_picture/' + cls + '.png'
                if os.path.isfile(path):
                    img = np.asarray(Image.open(path))[:,:,:3]
                    self.colors[cls][1] = img.sum(0).sum(0) / (img.shape[0] * img.shape[1])
                    print(cls + '被重新计算')
            with open('woolcolors.pkl', 'wb') as f:
                pickle.dump(self.colors, f)
            print('更新颜色文件完毕')
            return
        
        print('使用默认颜色参数')


    # HSV颜色追踪
    def hsv_track(
        self,
        pix : np.ndarray
    ) -> str:
        if pix[2] <= 46:
            return 'black'
        
        if pix[1] <= 43 and pix[2] <= 220:
            return 'grey'
        if pix[1] <= 30 and 221 <= pix[2]:
            return 'white'
        
        if pix[0] <= 10 or 176 <= pix[0]:
            return 'red'
        if pix[0] <= 25:
            return 'orange'
        if pix[0] <= 34:
            return 'yellow'
        if pix[0] <= 77:
            return 'green'
        if pix[0] <= 99:
            return 'cyan'
        if pix[0] <= 124:
            return 'blue'
        if pix[0] <= 155:
            return 'purple'
        if pix[0] <= 200:
            return 'pink'
        return 'magenta'


    # 颜色距离(欧几里得距离)
    def rgb_track(
        self,
        rgb : np.ndarray
    ) -> str:
        clrs = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.int8)
        minarr = np.zeros((rgb.shape[0], rgb.shape[1])) + 195075 # 255*255*3

        for i in self.colors:
            sqarr = ((rgb - self.colors[i][1])**2).sum(axis=-1)
            cmp = sqarr < minarr
            minarr[cmp] = sqarr[cmp]
            clrs[cmp] = self.colors[i][0]

        return clrs
    
    
    # 人眼距离
    def lab_track(
            self,
            rgb : np.ndarray
        ) -> np.ndarray:
        clrs = np.zeros([rgb.shape[0], rgb.shape[1]], dtype=np.int8)
        minarr = np.zeros([rgb.shape[0], rgb.shape[1]]) + 585225 # 255*255*9

        for i in self.colors:
            r_ = rgb[:, :, 0] + self.colors[i][1][0]
            delta_sq = (rgb - self.colors[i][1])**2
            tmp = (2+r_/256)*delta_sq[:,:,0] + \
                4*delta_sq[:,:,1] + (2+(255-r_)/256)*delta_sq[:,:,2]
            cmp = tmp < minarr
            minarr[cmp] = tmp[cmp]
            clrs[cmp] = self.colors[i][0]

        return clrs