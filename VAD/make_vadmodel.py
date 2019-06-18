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

def VAD_model(time_step = 10):
    inputs = keras.layers.Input(shape=(time_step,42))
    lstm = keras.layers.LSTM(64,return_sequences=False)(inputs)
    x = keras.layers.BatchNormalization()(lstm)
    x = keras.layers.Dense(64,activation='relu',kernel_initializer='random_uniform')(x)
    #x = Attention(lstm,32)
    output = keras.layers.Dense(1,activation='sigmoid',kernel_initializer='random_uniform')(x)

    vadmodel = keras.models.Model(inputs=inputs,outputs=output)
    optimizer = keras.optimizers.Adam(lr=1e-4,beta_1=0.9,beta_2=0.999)
    vadmodel.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
    vadmodel.summary()
    return vadmodel

def VAD_lldmodel(time_step = 10):
    inputs = keras.layers.Input(shape=(time_step,114))
    x = keras.layers.Dense(32,activation='relu',kernel_initializer='random_uniform',input_shape=(114,))(inputs)
    x = keras.layers.LSTM(64,return_sequences=False)(x)
    x = keras.layers.BatchNormalization()(x)
    output = keras.layers.Dense(1,activation='sigmoid',kernel_initializer='random_uniform')(x)

    vadmodel = keras.models.Model(inputs=inputs,outputs=output)
    optimizer = keras.optimizers.Adam(lr=1e-4,beta_1=0.9,beta_2=0.999)
    vadmodel.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
    vadmodel.summary()
    return vadmodel

def Attention(x,n_hidden = 64):
    attention = keras.layers.Dense(1,activation='sigmoid',)(x)
    attention = keras.layers.Flatten()(attention)
    attention = keras.layers.Activation('softmax',)(attention)
    attention = keras.layers.RepeatVector(n_hidden,)(attention)
    attention = keras.layers.Permute((2, 1),)(attention)

    feature = keras.layers.Multiply()([x,attention])
    x =  keras.layers.Lambda(lambda a: K.sum(a,axis=-2), output_shape=(n_hidden,))(feature)
    return x
