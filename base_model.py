import tensorflow as tf
import keras
from keras import layers

def make_base_model(img_height, img_width, num_classes,batch_size=32):
    inputs = layers.Input(shape=(img_height,img_width,3),batch_size=batch_size)
    rescaling = layers.Rescaling(1./255)(inputs)

    conv_1_1 = layers.Conv2D(16, 3, activation='relu',name='conv_1_1')(rescaling)
    conv_1_2 = layers.Conv2D(16, 3, activation='relu',name='conv_1_2')(conv_1_1)
    bn_1 = layers.BatchNormalization()(conv_1_2)
    mp_1 = layers.MaxPooling2D(2)(bn_1)

    conv_2_1 = layers.Conv2D(32, 3, activation='relu',name='conv_2_1')(mp_1)
    conv_2_2 = layers.Conv2D(32, 3, activation='relu',name='conv_2_2')(conv_2_1)
    bn_2 = layers.BatchNormalization()(conv_2_2)
    mp_2 = layers.MaxPooling2D(2)(bn_2)

    conv_3_1 = layers.Conv2D(64, 2, activation='relu',name='conv_3_1')(mp_2)
    conv_3_2 = layers.Conv2D(64, 2, activation='relu',name='conv_3_2')(conv_3_1)
    bn_3 = layers.BatchNormalization()(conv_3_2)

    flatten = layers.Flatten()(bn_3)
    dense_1 = layers.Dense(16, activation='relu', name='dense_1')(flatten)

    dense_out = layers.Dense(num_classes, activation='softmax', name='dense_out')(dense_1)

    base_model = keras.Model(inputs=inputs, outputs=dense_out,name='base_model')
    return base_model