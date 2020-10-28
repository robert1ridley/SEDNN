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


class SharedModelV2():
    def __init__(self, vocab_size, maxnum, maxlen, embedding_dim, embedding_weights):
        super(SharedModelV2, self).__init__()

        self.maxnum = maxnum
        self.maxlen = maxlen

        self.feature_extractor = keras.models.Sequential([
            layers.Embedding(output_dim=embedding_dim, input_dim=vocab_size, input_length=maxnum * maxlen,
                             weights=embedding_weights, mask_zero=True),
            ZeroMaskedEntries(),
            layers.Dropout(0.5),
            layers.Reshape((maxnum, maxlen, embedding_dim)),
            layers.TimeDistributed(layers.Conv1D(100, 5, padding='valid')),
            layers.TimeDistributed(Attention()),
            layers.LSTM(100, return_sequences=True),
            Attention()
        ])

        self.scorer = keras.models.Sequential([
            layers.Dense(100, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])

        self.discriminator = keras.models.Sequential([
            GradientReversalLayer(),
            layers.Dense(100, activation='relu'),
            layers.Dense(2),
            layers.Activation('softmax')
        ])

        self.predict_score = keras.models.Sequential([
            self.feature_extractor,
            self.scorer
        ])

        self.predict_domain = keras.models.Sequential([
            self.feature_extractor,
            self.discriminator
        ])
