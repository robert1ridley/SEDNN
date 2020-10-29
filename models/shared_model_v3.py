import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as layers
from custom_layers.zeromasking import ZeroMaskedEntries
from custom_layers.attention import Attention


@tf.custom_gradient
def gradient_reverse(x, lamda=1.0):
    y = tf.identity(x)

    def grad(dy):
        return lamda * -dy, None

    return y, grad


class GradientReversalLayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x, lamda=1.0):
        return gradient_reverse(x, lamda)


class SharedModelV3(keras.Model):
    def __init__(self, vocab_size, maxnum, maxlen, embedding_dim, embedding_weights):
        super(SharedModelV3, self).__init__()

        self.maxnum = maxnum
        self.maxlen = maxlen

        self.embedding = layers.Embedding(output_dim=embedding_dim, input_dim=vocab_size, input_length=maxnum * maxlen,
                                          weights=embedding_weights, mask_zero=True, name='embedding')
        self.x_maskedout = ZeroMaskedEntries(name='x_maskedout')
        self.drop_x = layers.Dropout(0.5, name='drop_x')
        self.reshape_W = layers.Reshape((maxnum, maxlen, embedding_dim), name='reshape_W')
        self.z_cnn = layers.TimeDistributed(layers.Conv1D(100, 5, padding='valid'), name='z_cnn')
        self.avg_zcnn = layers.TimeDistributed(Attention(), name='avg_zcnn')
        self.hz_lstm = layers.LSTM(100, return_sequences=True, name='hz_lstm')
        self.avg_hzlstm = Attention(name='avg_hzlstm')

        self.domain_grad_rev = GradientReversalLayer(name='domain_grad_rev')
        self.domain_fc1 = layers.Dense(100, activation='relu', name='domain_fc1')
        self.domain_fc2 = layers.Dense(2, activation='softmax', name='domain_fc2')

        self.score_fc1 = layers.Dense(100, activation='relu', name='score_fc1')
        self.score_fc2 = layers.Dense(1, activation='sigmoid', name='score_fc2')

    def call(self, x, src=True, lamda=1.0):
        x = self.embedding(x)
        x = self.x_maskedout(x)
        x = self.drop_x(x)
        x = self.reshape_W(x)
        x = self.z_cnn(x)
        x = self.avg_zcnn(x)
        x = self.hz_lstm(x)
        x = self.avg_hzlstm(x)

        domain_grad_rev = self.domain_grad_rev(x, lamda)
        domain_fc1 = self.domain_fc1(domain_grad_rev)
        domain_fc2 = self.domain_fc2(domain_fc1)

        if src:
            score_fc1 = self.score_fc1(x)
            score_fc2 = self.score_fc2(score_fc1)
            return domain_fc2, score_fc2
        else:
            return domain_fc2
