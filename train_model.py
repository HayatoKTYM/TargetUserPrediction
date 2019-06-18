import keras

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(
        visible_device_list="0", # specify GPU number
        allow_growth=True
    )
)
set_session(tf.Session(config=config))

from extract_feature import Extract_feature,Extract_LLDfeature
from make_model import *


if __name__ == '__main__':
    #100fpsの音響特徴量ファイル
    audio_files = sorted(glob('/mnt/aoni02/katayama/dataset/audio_feature/*csv'))
    lld_files = sorted(glob('/mnt/aoni02/katayama/dataset/LLD/*csv'))
    #10fpsのその他特徴量ファイル
    files = sorted(glob('/mnt/aoni02/katayama/dataset/feature/*csv'))
    time_step = 10
    for i,file in enumerate(files[:30]):
        ex = Extract_LLDfeature(audio_path=lld_files[i],
                                label_path=files[i],
                                frame=time_step)
        if i == 10: continue
        if i == 0:
            X_A = np.array(ex.audio_A)
            X_B = np.array(ex.audio_B)
            X_image = np.array(ex.image)
            y = np.array(ex.y)
            y1 = np.array(ex.y1)
            y2 = np.array(ex.y2)
            y3 = np.array(ex.y3)
            X_robot = np.array(ex.robot_state)
        else:
            X_A = np.append(X_A,ex.audio_A,axis=0)
            X_B = np.append(X_B,ex.audio_B,axis=0)
            X_image = np.append(X_image,ex.image,axis=0)
            y = np.append(y,ex.y,axis=0)
            y1 = np.append(y1,ex.y1,axis=0)
            y2 = np.append(y2,ex.y2,axis=0)
            y3 = np.append(y3,ex.y3,axis=0)
            X_robot = np.append(X_robot,ex.robot_state,axis=0)

    print('made dataset..')
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',patience=5,verbose=2)
    model_save = keras.callbacks.ModelCheckpoint(filepath = 'target_model/weights.{epoch:02d}.h5',monitor='val_loss')
    model = multitask_model(time_step = time_step)
    hist = model.fit([X_A,\
                      X_B,\
                      X_image,\
                      X_robot,\
                     ],
                     [y,y1,y2,y3],
                     batch_size=128,epochs=30,
                     verbose = 1,
                     validation_split=0.25,
                     callbacks=[early_stopping,model_save])

    print('training finish')

    for i,file in enumerate(files[30:35]):#6:
        ex = Extract_LLDfeature(audio_path=lld_files[i],
                                label_path=files[i],
                                frame=time_step)
        if i == 30:
                X_A = np.array(ex.audio_A)
                X_B = np.array(ex.audio_B)
                X_image = np.array(ex.image)
                y = np.array(ex.y)
                y1 = np.array(ex.y1)
                y2 = np.array(ex.y2)
                y3 = np.array(ex.y3)
                X_robot = np.array(ex.robot_state)
        else:
                X_A = np.append(X_A,ex.audio_A,axis=0)
                X_B = np.append(X_B,ex.audio_B,axis=0)
                X_image = np.append(X_image,ex.image,axis=0)
                y = np.append(y,ex.y,axis=0)
                y1 = np.append(y1,ex.y1,axis=0)
                y2 = np.append(y2,ex.y2,axis=0)
                y3 = np.append(y3,ex.y3,axis=0)
                X_robot = np.append(X_robot,ex.robot_state,axis=0)
    x=[]
    y_pred=[]
    X_robot = X_robot[:1]
    for i in range(len(y)):
        score = model.predict([X_A[i].reshape(1,time_step,114),\
                               X_B[i].reshape(1,time_step,114),\
                               X_image[i].reshape(1,32,96,1),\
                               X_robot[i].reshape(1,1)
                              ])

        x.append(score[0][0])
        if score[0] < 0.5:
            y_pred.append(0)
            X_robot = np.append(X_robot,0)
        else:
            y_pred.append(1)
            X_robot = np.append(X_robot,1)
    plt.plot(x[:1000],color="r")
    plt.plot(y[:1000],color="c")
    print(classification_report(y,y_pred))
    print('accuracy:',accuracy_score(y,y_pred))
