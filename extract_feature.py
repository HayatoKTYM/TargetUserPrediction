import pandas as pd
import numpy as np
from collections import Counter
import cv2
from sklearn import preprocessing
mm = preprocessing.MinMaxScaler()

import warnings
warnings.filterwarnings('ignore')


class Extract_feature(object):
    """
    datasetの構築
    @param: audio_path audioファイルのpath
    @param: label_path labelファイルのpath
    @param: frame 過去何フレームの特徴量を入力にするか

    @return: audio_A Aの音響特徴量/ shape=(None,frame,10)
             image   画像特徴量/    shape=(None,32,96,1)
             robot_state 1フレーム前の[target出力]推定結果/ shape=(None,)
             y1 Aの発話区間ラベル shape=(None,)
             y2 Bの
             y3 視線状態のラベル shape=(None,)
             y targetラベル

    """
    def __init__(self,
                audio_path=None,
                label_path=None,
                frame=10):

        self.frame = frame

        self.audio_A = []      #Aの音響特徴量
        self.audio_B = []      #Bの音響特徴量
        self.image  = []       #目画像特徴量
        self.robot_state  = [] #ロボットの過去の顔向き

        self.y1 = [] #Aの発話有無出力
        self.y2 = [] #Bの発話有無出力
        self.y3 = [] #視線状態出力
        self.y = [] #ロボットの顔向きラベル
        self.y_act = [] #ロボットの行動type

        self.df_label = pd.read_csv(label_path)
        self.df_audio = pd.read_csv(audio_path)
        self.max_frame = len(self.df_label) * 10 #フレーム同期用
        self.get_feature()
        self.get_Audio_feature()


    def get_Audio_feature(self):
        """
        音響特徴量[10fps]datasetの構築
        """
        feature_A,feature_B = mm.fit_transform(self.df_audio.iloc[:,8:50].values[:self.max_frame]),\
                              mm.fit_transform(self.df_audio.iloc[:,50:92].values[:self.max_frame])

        self.audio_A = [feature_A[i:i+10] for i in range(0,self.max_frame,10)]#[self.frame-1:-1]
        self.audio_B = [feature_B[i:i+10] for i in range(0,self.max_frame,10)]#[self.frame-1:-1]
        self.audio_A = [self.audio_A[i:i+self.frame//10] for i in range(len(self.audio_B)-self.frame//10)]
        self.audio_A = np.reshape(self.audio_A,(len(self.audio_A),-1,42))
        self.audio_B = [self.audio_B[i:i+self.frame//10] for i in range(len(self.audio_B)-self.frame//10)]
        self.audio_B = np.reshape(self.audio_B,(len(self.audio_B),-1,42))

    def get_feature(self):
        """
        [100fps]datasetの構築
        """
        y_cam = self.df_label['gaze'].values
        paths = self.df_label['path'].values
        X_img = self.get_image(paths)
        self.image.extend(X_img[self.frame//10-1:-1])
        self.y3.extend(y_cam[self.frame//10-1:-1])##

        y_a = self.df_label.iloc[:,6:8].values
        self.y1.extend([a[0] for a in y_a[self.frame//10-1:-1]])##
        self.y2.extend([a[1] for a in y_a[self.frame//10-1:-1]])##

        self.df_label = self.df_label['target']
        self.df_label = self.df_label.map(lambda x: 0 if x == "A" else 1)
        self.y.extend(self.df_label.values[self.frame//10:])
        self.robot_state.extend(self.df_label.values[self.frame//10-1:-1])



    def get_image(self,paths):
        #pathを受け取って画像を返す
        img_feature = []
        for path in paths:
            x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if x is None:
                x = np.array([0]*32*96)
            x = x.reshape(32,96,1)
            img_feature.append(x / 255.0)
        return img_feature


"""
LLD特徴量を使うならこちら
"""
class Extract_LLDfeature(object):
    """
    datasetの構築
    @param: audio_path audioファイルのpath
    @param: label_path labelファイルのpath
    @param: frame 過去何フレームの特徴量を入力にするか

    @return: audio_A Aの音響特徴量/ shape=(None,frame,10)
             image   画像特徴量/    shape=(None,32,96,1)
             robot_state 1フレーム前の[target出力]推定結果/ shape=(None,)
             y1 Aの発話区間ラベル shape=(None,)
             y2 Bの
             y3 視線状態のラベル shape=(None,)
             y targetラベル

    """
    def __init__(self,
                audio_path=None,
                label_path=None,
                frame=10):

        self.frame = frame

        self.audio_A = []      #Aの音響特徴量
        self.audio_B = []      #Bの音響特徴量
        self.image  = []       #目画像特徴量
        self.robot_state  = [] #ロボットの過去の顔向き

        self.y1 = [] #Aの発話有無出力
        self.y2 = [] #Bの発話有無出力
        self.y3 = [] #視線状態出力
        self.y = [] #ロボットの顔向きラベル
        self.y_act = [] #ロボットの行動type

        self.df_label = pd.read_csv(label_path)
        self.df_audio = pd.read_csv(audio_path)
        self.max_frame = len(self.df_audio) // 10 #フレーム同期用
        self.get_feature()
        self.get_Audio_feature()


    def get_Audio_feature(self):
        """
        音響特徴量[10fps]datasetの構築
        """
        feature_A,feature_B = \
                mm.fit_transform(self.df_audio.iloc[:,:114].values),\
                mm.fit_transform(self.df_audio.iloc[:,114:].values)

        self.audio_A = [feature_A[i*10:(i+1)*10] for i in range(0,len(self.df_audio)//10)]#[self.frame//10-1:-1]#[self.frame-1:-1]
        self.audio_B = [feature_B[i*10:(i+1)*10] for i in range(0,len(self.df_audio)//10)]#[self.frame//10-1:-1]#[self.frame-1:-1]
        self.audio_A = [self.audio_A[i:i+self.frame//10] for i in range(len(self.audio_B)-self.frame//10)]
        self.audio_A = np.reshape(self.audio_A,(len(self.audio_A),-1,114))
        self.audio_B = [self.audio_B[i:i+self.frame//10] for i in range(len(self.audio_B)-self.frame//10)]
        self.audio_B = np.reshape(self.audio_B,(len(self.audio_B),-1,114))

    def get_feature(self):
        """
        [100fps]datasetの構築
        """
        y_cam = self.df_label['gaze'].values[:self.max_frame]
        y_cam = [1 if i == 2 else i for i in y_cam]
        paths = self.df_label['path'].values[:self.max_frame]
        X_img = self.get_image(paths)
        self.image.extend(X_img[self.frame//10-1:self.max_frame-1])
        self.y3.extend(y_cam[self.frame//10-1:self.max_frame-1])##

        y_a = self.df_label.iloc[:,6:8].values[:self.max_frame]
        self.y1.extend([a[0] for a in y_a[self.frame//10-1:-1]])##
        self.y2.extend([a[1] for a in y_a[self.frame//10-1:-1]])##

        self.df_label = self.df_label['target']
        self.df_label = self.df_label.map(lambda x: 0 if x == "A" else 1)
        #print(len(self.df_label.values[self.frame//10:self.max_frame]))
        self.y.extend(self.df_label.values[self.frame//10:self.max_frame])
        self.robot_state.extend(self.df_label.values[self.frame//10-1:self.max_frame-1])



    def get_image(self,paths):
        #pathを受け取って画像を返す
        img_feature = []
        for path in paths:
            x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if x is None:
                x = np.array([0]*32*96)
            x = x.reshape(32,96,1)
            img_feature.append(x / 255.0)
        return img_feature

#確認用
if __name__ == '__main__':
    from glob import glob
    label_files = sorted(glob('/mnt/aoni02/katayama/dataset/feature/*csv'))
    audio_files = sorted(glob('/mnt/aoni02/katayama/dataset/audio_feature/*csv'))
    #label_files = sorted(glob('/Users/hayato/Desktop/drive-download-20190111T234012Z-001/feature/*csv'))
    #audio_files = sorted(glob('/Users/hayato/Desktop/drive-download-20190111T234012Z-001/audio_feature/*csv'))
    lld_files = sorted(glob('/mnt/aoni02/katayama/dataset/LLD/*csv'))
    assert len(label_files) == len(audio_files), print('AssertionError :check the file_path')
    assert len(label_files) != 0, print('AssertionError :check the file_path')

    for i in range(len(label_files)):
        ex = Extract_LLDfeature( audio_path = lld_files[i],
                                 label_path = label_files[i],
                                 frame = 20)
        assert len(ex.audio_A) == len(ex.image), print('AssertionError :check the file_path')
        #print(np.shape(ex.audio_A))
        #print(np.shape(ex.image))
        print(i+1)
