from tensorflow import keras
import tensorflow.keras.layers as layers
from custom_layers.zeromasking import ZeroMaskedEntries
from custom_layers.attention import Attention


class SharedModel(keras.Model):
    def __init__(self, vocab_size, maxnum, maxlen, embedding_dim, embedding_weights):
        super(SharedModel, self).__init__()
        self.feature_generator = self.feature_generator_model(vocab_size, maxnum, maxlen, embedding_dim, embedding_weights)
        self.scorer = self.scorer_model()
        self.discriminator = self.discriminator_model()
        self.feature_generator.summary()
        self.scorer.summary()
        self.discriminator.summary()
        print("Shared Model built")

    def feature_generator_model(self, vocab_size, maxnum, maxlen, embedding_dim, embedding_weights):
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
        return keras.Model(inputs=word_input, outputs=avg_hz_lstm, name="feature_generator_model")

    def scorer_model(self):
        scorer_input = layers.Input(shape=(100,), name='scorer_input')
        y_score = layers.Dense(1, activation='sigmoid', name='y_score')(scorer_input)
        return keras.Model(inputs=scorer_input, outputs=y_score, name="scorer_model")

    def discriminator_model(self):
        discriminator_input = layers.Input(shape=(100,), name='discriminator_input')
        y_class = layers.Dense(1, name='y_class')(discriminator_input)
        return keras.Model(inputs=discriminator_input, outputs=y_class, name="discriminator_model")
