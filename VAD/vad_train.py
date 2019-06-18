import numpy as np
from glob import glob
import pandas as pd
from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import classification_report, precision_recall_fscore_support,\
    confusion_matrix,accuracy_score
np.random.seed(0)

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

from extract_feature import *
from make_vadmodel import *

import pandas as pd

if __name__ == '__main__':
    # LLD dataset の構築
    path = '/mnt/aoni02/katayama/dataset/LLD/*csv'
    lld_a = sorted(glob(path))
    time_step = 10
    Acc = []

    for i in range(40):
        if i == 10: continue
        if i in [30,31,32,33,34]: continue

        ex = Extract_feature(lld_path=lld_a[i],
                             label_path=files[i],
                             frame=time_step)

        if i == 0:
                X_A = np.array(ex.audio_A)
                X_B = np.array(ex.audio_B)
                y1 = np.array(ex.y1)
                y2 = np.array(ex.y2)
                X = np.array(ex.audio_A)
                y = np.array(ex.y1)

        elif i == 35:
                X_A_val = np.array(ex.audio_A)
                X_B_val = np.array(ex.audio_B)
                y1_val = np.array(ex.y1)
                y2_val = np.array(ex.y2)
                X_val = np.array(ex.audio_A)
                y_val = np.array(ex.y1)

        elif i > 35:
                X_A_val = np.append(X_A_val,ex.audio_A,axis=0)
                X_B_val = np.append(X_B_val,ex.audio_B,axis=0)
                y1_val = np.append(y1_val,ex.y1,axis=0)
                y2_val = np.append(y2_val,ex.y2,axis=0)
                X_val = np.append(X_val,ex.audio_A,axis=0)
                X_val = np.append(X_val,ex.audio_B,axis=0)
                y_val = np.append(y_val,ex.y1,axis=0)
                y_val = np.append(y_val,ex.y2,axis=0)

        else:
                X_A = np.append(X_A,ex.audio_A,axis=0)
                X_B = np.append(X_B,ex.audio_B,axis=0)
                y1 = np.append(y1,ex.y1,axis=0)
                y2 = np.append(y2,ex.y2,axis=0)
                X = np.append(X,ex.audio_A,axis=0)
                X = np.append(X,ex.audio_B,axis=0)
                y = np.append(y,ex.y1,axis=0)
                y = np.append(y,ex.y2,axis=0)

    print('making dataset')
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',patience=10,verbose=2)
    model_save = keras.callbacks.ModelCheckpoint(filepath = 'vad_model/weights.{epoch:02d}.h5',monitor='val_loss')
    model = VAD_lldmodel(time_step=time_step)

    hist = model.fit(X,y,
                     batch_size=256,epochs=50,verbose = 2,
                     validation_data=(X_val, y_val),
                     #validation_data=(X_A_val, y1_val),
                     class_weight=['balanced'],
                     callbacks=[early_stopping,model_save])

    for i in range(30,35):#6:
        ex = Extract_feature(lld_path=lld_a[i],
                             label_path=files[i],
                             frame=time_step)
        if i == 30:
                X_A = np.array(ex.audio_A)
                X_B = np.array(ex.audio_B)
                y1 = np.array(ex.y1)
                y2 = np.array(ex.y2)
                X = np.array(ex.audio_A)
                y = np.array(ex.y1)
        else:
                X_A = np.append(X_A,ex.audio_A,axis=0)
                X_B = np.append(X_B,ex.audio_B,axis=0)
                y1 = np.append(y1,ex.y1,axis=0)
                y2 = np.append(y2,ex.y2,axis=0)
                X = np.append(X,ex.audio_A,axis=0)
                X = np.append(X,ex.audio_B,axis=0)
                y = np.append(y,ex.y1,axis=0)
                y = np.append(y,ex.y2,axis=0)

    x = model.predict(X).reshape(-1)
    #plt.figure(figsize=(25,5))
    #plt.plot(x[:300],color="r")
    #plt.plot(y[:300],color="c")
    print(model.evaluate(X,y)[1])
    y_pred = [0 if p < 0.5 else 1 for p in x]
    print(classification_report(y,y_pred))
