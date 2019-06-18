import numpy as np
from glob import glob
import pandas as pd
from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import classification_report, precision_recall_fscore_support,\
    confusion_matrix,accuracy_score
np.random.seed(0)

from make_gazemodel import *
import keras
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(
        visible_device_list="0", # specify GPU number
        allow_growth=True
    )
)
set_session(tf.Session(config=config))
graph = tf.get_default_graph()


def extract_feature(train):
    df   = pd.read_csv(train+"/yes_eye_data.csv",header=None).sample(frac=1,random_state=100)#読み込んで順番shuffle
    X = df.iloc[:, 2:].values.astype(np.float32).reshape(len(df),32,96,1)
    y = df.iloc[:, 1].astype(np.int32) .values
    df   = pd.read_csv(train+"/no_eye_data.csv",header=None).sample(frac=1,random_state=100)#読み込んで順番shuffle
    X_ = df.iloc[:, 2:].values.astype(np.float32).reshape(len(df),32,96,1)
    y_ = df.iloc[:, 1].astype(np.int32) .values
    X = np.vstack((X,X_))
    y = np.append(y,y_)
    return X / 255 ,y



if __name__ == '__main__':
    time_step = 10
    Acc = []
    files = sorted(glob('/mnt/aoni02/katayama/short_project/proken2018_A/input/*'))
    for i in range(23):
        try:
            if i == 0:
                x,y = extract_feature(files[i])
                X = np.array(x)
                y3 = np.array(y)

            elif i == 20:
                x,y = extract_feature(files[i])
                X_val = np.array(x)
                y3_val = np.array(y)

            elif i > 20:
                x,y = extract_feature(files[i])
                X_val = np.append(X_val,x,axis=0)
                y3_val = np.append(y3_val,y,axis=0)

            else:
                x,y = extract_feature(files[i])
                X = np.append(X,x,axis=0)
                y3 = np.append(y3,y,axis=0)
        except Exception as e:
            print(e)
            print(files[i])

    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',patience=5,verbose=2)
    model_save = keras.callbacks.ModelCheckpoint(filepath = 'gaze_model/weights.{epoch:02d}.h5',monitor='val_loss')
    model = CNN()
    hist = model.fit(X,y3,
                     batch_size=128,epochs=50,verbose = 2,
                     validation_data=(X_val, y3_val),
                     class_weight='balanced',
                     callbacks=[early_stopping,model_save])

    for i in range(23,25):#6:
        if i == 23:
            x,y = extract_feature(files[i])
            X = np.array(x)
            y3 = np.array(y)
        else:
            x,y = extract_feature(files[i])
            X = np.append(X,x,axis=0)
            y3 = np.append(y3,y,axis=0)

    x = model.predict(X).reshape(-1)
    print(model.evaluate(X,y3)[1])
    y_pred = [0 if p < 0.5 else 1 for p in x]
    print(classification_report(y3,y_pred))
