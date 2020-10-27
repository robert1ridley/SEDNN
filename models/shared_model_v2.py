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
    def __init__(self):
        super().__init__()

    def call(self, x, lamda=0.1):
        return gradient_reverse(x, lamda)


class SharedModelV2(keras.Model):
    def __init__(self, vocab_size, maxnum, maxlen, embedding_dim, embedding_weights):
        super(SharedModelV2, self).__init__()

        # feature extractor
        self.emb = layers.Embedding(output_dim=embedding_dim, input_dim=vocab_size, input_length=maxnum * maxlen,
                                    weights=embedding_weights, mask_zero=True)
        self.x_maskedout = ZeroMaskedEntries()
        self.drop_x = layers.Dropout(0.5)
        self.resh_W = layers.Reshape((maxnum, maxlen, embedding_dim))
        self.zcnn = layers.TimeDistributed(layers.Conv1D(100, 5, padding='valid'))
        self.avg_zcnn = layers.TimeDistributed(Attention())
        self.hz_lstm = layers.LSTM(100, return_sequences=True)
        self.avg_hz_lstm = Attention()

        # scorer
        self.y_score = layers.Dense(1, activation='sigmoid')

        # discriminator
        self.grad_rev = GradientReversalLayer()
        self.y_class = layers.Dense(1, activation='sigmoid')

    def call(self, x, lamda=0.1, src=True):
        x = self.emb(x)
        x = self.x_maskedout(x)
        x = self.drop_x(x)
        x = self.resh_W(x)
        x = self.zcnn(x)
        x = self.avg_zcnn(x)
        x = self.hz_lstm(x)
        features = self.avg_hz_lstm(x)
        rev = self.grad_rev(features, lamda)
        y_class = self.y_class(rev)

        if src:
            y_score = self.y_score(features)
            return y_class, y_score
        else:
            return y_class
