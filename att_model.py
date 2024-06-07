import tensorflow as tf
import keras
from keras import layers


def make_att_model(img_height,img_width, num_classes, batch_size=32):
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

    g_1,c_1 = CnnAttentionLayer(global_shape=16,name='Attention_1')(conv_1_2, dense_1)
    g_2,c_2 = CnnAttentionLayer(global_shape=16,name='Attention_2')(conv_2_2, dense_1)
    g_3,c_3 = CnnAttentionLayer(global_shape=16,name='Attention_3')(conv_3_2, dense_1)

    g_concat = ConcatLayer()([g_1,g_2,g_3])

    dense_out = layers.Dense(num_classes, activation='softmax', name='dense_out')(g_concat)

    att_model = keras.Model(inputs=inputs, outputs=dense_out,name='att_model')
    
    return att_model


class CnnAttentionLayer(keras.layers.Layer):
  def __init__(self, name, global_shape, compat_function=None, out_ci=True):
    super(CnnAttentionLayer, self).__init__(name=name)
    self.compat_function = compat_function
    self.global_shape = global_shape
    self.out_ci = out_ci

  def build(self, input_shape):
    self.kernel = self.add_weight(name="kernel",
                                  shape=[int(input_shape[-1]),
                                         self.global_shape])

  def call(self, local_features, global_features):
    # e.g. convolutional output 49x49x64: n=2401
    # global vector: 16
    # attention mapping kernel shape: 64x16
    # mapped features shape: 49x49x16 => 2401x16; attention score vector shape: 2401x1 
    
    # map the features at each spatial location to the shape of the 
    local_features_mapped = tf.matmul(local_features, self.kernel)
    local_features_flat = tf.reshape(local_features_mapped, 
                                     (
                                       tf.shape(local_features)[0], 
                                       tf.shape(local_features)[-3]*tf.shape(local_features)[-2],
                                       self.global_shape))
    
    assert local_features_flat.shape[-1] == global_features.shape[-1]
    # scalar product for now
    compat_scores = tf.matmul(local_features_flat, tf.expand_dims(global_features,axis=-1)) #should be scalar
    # assert compat_scores.shape == (local_features.shape[0], local_features.shape[-3]*local_features.shape[-2],1)
    cs_softmax = tf.nn.softmax(compat_scores)
    ga = tf.math.reduce_sum(
      tf.reshape(local_features, (
        tf.shape(local_features)[0], 
        tf.shape(local_features)[-3]*tf.shape(local_features)[-2],
        tf.shape(local_features)[-1])
        )*cs_softmax,axis=1)
    if self.out_ci:
      return ga, compat_scores
    else:
      return ga
    
class ConcatLayer(layers.Layer):
    def call(self, x):
        return tf.concat(x, axis=1)
