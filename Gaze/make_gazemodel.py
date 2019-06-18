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


def CNN():
    inputs = keras.layers.Input(shape=(32,96,1))
    conv1 = keras.layers.Conv2D(32, 5,activation='relu',name='conv1')(inputs)
    conv2 = keras.layers.Conv2D(32, 5,activation='relu',name='conv2')(conv1)
    pool1 = keras.layers.MaxPooling2D((2, 2),name='pool1')(conv2)

    conv3 = keras.layers.Conv2D(32, 3,activation='relu',name='conv3')(pool1)
    pool2 = keras.layers.MaxPooling2D((2, 2),name='pool2')(conv3)
    conv4 = keras.layers.Conv2D(32, 3,activation='relu',name='conv4')(conv3)
    pool3 = keras.layers.MaxPooling2D((2, 2),name='pool3')(conv4)

    bn = keras.layers.BatchNormalization(name='bn1')(pool3)
    x = keras.layers.Flatten()(bn)
    x = keras.layers.Dense(1024,activation='relu',name='dense1')(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(128,activation='relu',name='dense2')(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(64,activation='relu',name='dense3')(x)
    x = keras.layers.Dropout(0.5)(x)
    output = keras.layers.Dense(1,activation='sigmoid')(x)

    gazemodel = keras.models.Model(inputs=inputs,outputs=output)
    optimizer = keras.optimizers.Adam(lr=1e-3,beta_1=0.9,beta_2=0.999)
    gazemodel.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
    gazemodel.summary()
    return gazemodel
