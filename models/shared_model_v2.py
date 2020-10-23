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
        word_input = layers.Input(shape=(maxnum * maxlen,), dtype='int32', name='word_input')
        x = layers.Embedding(output_dim=embedding_dim, input_dim=vocab_size, input_length=maxnum * maxlen,
                             weights=embedding_weights, mask_zero=True, name='x')(word_input)
        x_maskedout = ZeroMaskedEntries(name='x_maskedout')(x)
        drop_x = layers.Dropout(0.5, name='drop_x')(x_maskedout)
        resh_W = layers.Reshape((maxnum, maxlen, embedding_dim), name='resh_W')(drop_x)
        zcnn = layers.TimeDistributed(layers.Conv1D(100, 5, padding='valid'), name='zcnn')(resh_W)
        avg_zcnn = layers.TimeDistributed(Attention(), name='avg_zcnn')(zcnn)
        hz_lstm = layers.LSTM(100, return_sequences=True, name='hz_lstm')(avg_zcnn)
        avg_hz_lstm = Attention(name='avg_hz_lstm')(hz_lstm)

        # scorer
        y_score = layers.Dense(1, activation='sigmoid', name='y_score')

        # discriminator
        grad_rev = GradientReversalLayer()
        y_class = layers.Dense(1, activation='sigmoid', name='y_class')

        y_score_out = y_score(avg_hz_lstm)
        self.scorer = keras.Model(inputs=word_input, outputs=y_score_out, name="scorer_model")

        y_class_out = y_class(avg_hz_lstm)
        self.discriminator = keras.Model(inputs=word_input, outputs=y_class_out, name="discriminator_model")

        grad_rev_rep = grad_rev(avg_hz_lstm)
        y_class_reversed_out = y_class(grad_rev_rep)
        self.feature_generator = keras.Model(inputs=word_input, outputs=[y_class_reversed_out, y_score_out])
