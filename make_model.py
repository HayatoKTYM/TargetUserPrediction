from keras import backend as K
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



def multitask_model(time_step=10):

    def CNN(inputs):
            conv1 = keras.layers.Conv2D(32, 5,
                            activation='relu',
                            name='conv1')(inputs)
            pool1 = keras.layers.MaxPooling2D((2, 2),name='pool1')(conv1)

            conv2 = keras.layers.Conv2D(32, 5,
                           activation='relu',
                           name='conv2')(pool1)
            pool2 = keras.layers.MaxPooling2D((2, 2),name='pool2')(conv2)

            bn = keras.layers.BatchNormalization(name='bn1')(pool2)
            x = keras.layers.Flatten()(bn)
            x = keras.layers.Dense(128,
                      activation='relu',
                      name='dense1')(x)
            output = keras.layers.Dense(1,activation='sigmoid')(x)

            gazemodel = keras.models.Model(inputs=inputs,outputs=output)
            gazemodel.load_weights("/mnt/aoni02/katayama/project/estimate/pretrain/model/gaze/gaze_1114_middle128.h5")
            return gazemodel


    def VAD_model(inputs):
        x = keras.layers.Dense(16,activation='relu',input_shape=(42,))(inputs)
        lstm = keras.layers.LSTM(128)(x)
        x = keras.layers.BatchNormalization()(lstm)
        x = keras.layers.Dense(128,activation='relu')(x)
        output = keras.layers.Dense(1,activation='sigmoid')(x)
        vadmodel = keras.models.Model(inputs=inputs,outputs=output)
        vadmodel.load_weights("/mnt/aoni02/katayama/project/estimate/pretrain/model/audio/frame10/audio_1114_middle128.h5")
        return vadmodel

    def LLD_model(inputs):
        x = keras.layers.Dense(32,activation='relu',kernel_initializer='random_uniform',input_shape=(114,))(inputs)
        x = keras.layers.LSTM(64,return_sequences=False)(x)
        x = keras.layers.BatchNormalization()(x)
        output = keras.layers.Dense(1,activation='sigmoid',kernel_initializer='random_uniform')(x)

        vadmodel = keras.models.Model(inputs=inputs,outputs=output)
        vadmodel.load_weights("/mnt/aoni02/katayama/short_project/proken2018_B/VAD/utterance_lld_final10.h5")
        #for layer in vadmodel.layers:
        #    layer.trainable = False
        return vadmodel

    input_audioA = keras.layers.Input(shape = (time_step,114,),name='audioA' )
    input_audioB = keras.layers.Input(shape = (time_step,114,),name='audioB')
    x = LLD_model(input_audioA)
    x_A = x.layers[3].output
    x = LLD_model(input_audioB)
    x_B = x.layers[3].output

    input_img = keras.layers.Input(shape=(32,96,1),name='image')
    input_r = keras.layers.Input(shape=(1,),name='robot_face')
    img = CNN(input_img)
    x_img = img.layers[7].output
    output1 = keras.layers.Dense(1,activation='sigmoid',name='vadA')(x_A)
    output2 = keras.layers.Dense(1,activation='sigmoid',name='vadB')(x_B)
    output3 = keras.layers.Dense(1,activation='sigmoid',name='gaze')(x_img)

    x = keras.layers.concatenate([x_img,input_r,x_A,x_B])
    x = keras.layers.Dense(256,activation='relu',name='dense2')(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(256,activation='relu',name='dense3')(x)
    x = keras.layers.Dropout(0.5)(x)
    output = keras.layers.Dense(1,activation = 'sigmoid',name='target')(x)

    model = keras.models.Model(inputs=[input_audioA,
                          input_audioB,
                          input_img,
                          input_r,
                         ],
                  outputs = [output,output1,output2,output3])

    optimizer = keras.optimizers.Adam(lr=1e-4,beta_1=0.9,beta_2=0.999)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'],
                  loss_weights = {'target':1,'vadA':0.25,'vadB':0.25,'gaze':0.25}
                 )
    model.summary()
    return model
