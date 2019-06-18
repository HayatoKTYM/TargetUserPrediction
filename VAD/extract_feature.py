from sklearn import preprocessing
mm = preprocessing.MinMaxScaler()
mm = preprocessing.StandardScaler()
import warnings
warnings.filterwarnings('ignore')


class Extract_feature(object):
    def __init__(self,
                 lld_path=None,
                 label_path=None,
                 frame=1):

        self.frame = frame
        self.audio_A = []         #Aの音響特徴量
        self.audio_B = []         #Bの音響特徴量
        self.y1 = [] #Aの発話有無出力
        self.y2 = [] #Bの発話有無出力

        self.df = pd.read_csv(label_path)
        self.df_lld = pd.read_csv(lld_path)
        self.max_frame = len(self.df_lld) // 10
        self.get_imgfeature()
        self.get_feature()


    def get_feature(self):

        feature_A,feature_B = \
                self.df_lld.iloc[:,:114].values,self.df_lld.iloc[:,114:].values

        self.audio_A = [feature_A[i*10:(i+1)*10] for i in range(0,len(self.df_lld)//10)]#[self.frame-1:-1]
        self.audio_B = [feature_B[i*10:(i+1)*10] for i in range(0,len(self.df_lld)//10)]#[self.frame-1:-1]
        self.audio_A = [self.audio_A[i:i+self.frame//10] for i in range(len(self.audio_B)-self.frame//10)]
        self.audio_A = np.reshape(self.audio_A,(len(self.audio_A),-1,114))
        self.audio_B = [self.audio_B[i:i+self.frame//10] for i in range(len(self.audio_B)-self.frame//10)]
        self.audio_B = np.reshape(self.audio_B,(len(self.audio_B),-1,114))

    def get_imgfeature(self,mitigation=True):
        y_a = self.df.iloc[:,6:8].values[:self.max_frame]
        self.y1.extend([a[0] for a in y_a[self.frame//10:]])
        self.y2.extend([a[1] for a in y_a[self.frame//10:]])
