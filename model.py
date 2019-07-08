

from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint

from keras.layers import Embedding, Input, LSTM, RepeatVector, Dense
from keras.models import Model

from gensim.models import Word2Vec


class NeuralModel:

    def __init__(self,
                 embedding_dim=300,
                 lstm_dims=(100, 100),
                 dense_dims=(100,),
                 bidirectional=True,
                 optimizer='nadam',
                 loss='binary_crossentropy',
                 metrics=('binary_crossentropy',),
                 checkpoint=True,
                 cp_filename='checkpoint/chkp_giraffe.best.hdf5'):

        self.model = None
        self.lstm_dims = lstm_dims
        self.dense_dims = dense_dims
        self.bidirectional = bidirectional
        self.embedding_dim = embedding_dim
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.checkpoint = checkpoint
        self.cp_filename = cp_filename

    def build(self):

        model = Word2Vec.load_word2vec_format('/home/edilson/GoogleNews-vectors-negative300.bin', binary=True)

        embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
        for word, i in word_index.items():
            if word in model:
                embedding_matrix[i] = model[word]
            else:
                embedding_matrix[i] = np.random.rand(1, EMBEDDING_DIM)[0]

        embedding_layer = Embedding(len(word_index) + 1,
                                    EMBEDDING_DIM,
                                    weights=[embedding_matrix],
                                    input_length=MAX_SEQUENCE_LENGTH,
                                    trainable=False)

        inputs = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
        embedded_sequences = embedding_layer(inputs)
        encoded = LSTM(HIDDEN_DIM)(embedded_sequences)

        decoded = RepeatVector(MAX_SEQUENCE_LENGTH)(encoded)
        decoded = LSTM(HIDDEN_DIM)(decoded)

        # decoded = Dropout(0.5)(decoded)
        decoded = Dense(y_train.shape[1], activation='softmax')(decoded)

        sequence_autoencoder = Model(inputs, decoded)

        encoder = Model(inputs, encoded)

        self._compile()
        self._summary()

        return self

    def _compile(self):
        print('Compiling...')
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)

    def _summary(self):
        self.model.summary()

    def save_model(self, filename):
        # serialize model to JSON
        model_json = self.model.to_json()
        with open("%s.json" % filename, "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights("%s.h5" % filename)
        print("Saved model to disk")

    def load_model(self, filename):
        json_file = open('%s.json' % filename, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        # load weights into new model
        self.model.load_weights("%s.h5" % filename)
        print("Loaded model from disk")

    def fit(self, X_input_dic, y, epochs=10, batch_size=16, shuffle=True, stopped=False):
        callbacks_list = []
        if self.checkpoint:
            checkpoint = ModelCheckpoint(self.cp_filename, monitor='val_loss', verbose=1, save_best_only=True,
                                         mode='auto')
            callbacks_list.append(checkpoint)

        if stopped:
            self.model.load_weights(self.cp_filename)

        self.model.fit(X_input_dic, y,
                       epochs=epochs,
                       batch_size=batch_size,
                       shuffle=shuffle,
                       validation_split=0.1,
                       callbacks=callbacks_list)

    def get_model(self):
        return self.model
