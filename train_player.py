from os import listdir
from queue import Queue
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras import backend as K
from keras.engine.topology import Layer

card_norm = {
    'Js': np.asarray([0,0,0,1]),
    'Qs': np.asarray([0,0,1,0]),
    'Ks': np.asarray([0,1,0,0]),
    'As': np.asarray([1,0,0,0])
}

def get_data():
    #for f in listdir('./play_dataset'):
    #    pass
    f = listdir('./play_dataset')[0]
    
    with open('./play_dataset/' + f) as log:
        data = [l.strip() for l in log.readlines()[4:]]
        me, p1, p2 = data[0].split(':')[5].split('|')
        p1_bluffs = 0
        p2_bluffs = 0
        p1_recent_bluffs = Queue(10)
        p2_recent_bluffs = Queue(10)
        X = []
        y = []
        i = 1
        for l in data[:-1]:
            params = l.split(':')
            cards = params[3].split('|')
            order = params[5].split('|')
            plays = list(params[2])
            me_c = cards[order.index(me)]
            p1_i = order.index(p1)
            p2_i = order.index(p2)
            p1_bluff = 0
            p2_bluff = 0
            if cards[p1_i] == 'Js':
                if plays[p1_i] == 'c':
                    p1_bluff += 0.75
                elif plays[p1_i] == 'r':
                    p1_bluff += 1
            if cards[p1_i] == 'Qs':
                if plays[p1_i] == 'c':
                    p1_bluff += 0.5
                elif plays[p1_i] == 'r':
                    p1_bluff += 0.75
            if cards[p1_i] == 'Ks':
                if plays[p1_i] == 'c':
                    p1_bluff += 0.125
                if plays[p1_i] == 'r':
                    p1_bluff += 0.25

            if cards[p2_i] == 'Js':
                if plays[p2_i] == 'c':
                    p2_bluff += 0.75
                elif plays[p2_i] == 'r':
                    p2_bluff += 1
            if cards[p2_i] == 'Qs':
                if plays[p2_i] == 'c':
                    p2_bluff += 0.5
                elif plays[p2_i] == 'r':
                    p2_bluff += 0.75
            if cards[p2_i] == 'Ks':
                if plays[p2_i] == 'c':
                    p2_bluff += 0.125
                if plays[p2_i] == 'r':
                    p2_bluff += 0.25

            p1_bluffs += p1_bluff
            p2_bluffs += p2_bluff
            p1_recent_bluffs.put(p1_bluff)
            p2_recent_bluffs.put(p2_bluff)
            if i > 9:
                p1_recent_bluffs.get()
                p2_recent_bluffs.get()
            p1_recent_bluff_rate = sum(list(p1_recent_bluffs.queue))/10
            p2_recent_bluff_rate = sum(list(p2_recent_bluffs.queue))/10
            #print(np.append(card_norm[me_c], (p1_recent_bluff_rate, p2_recent_bluff_rate, p1_bluffs/i, p2_bluffs/i)).tolist())
            X.append(np.append(card_norm[me_c], (p1_recent_bluff_rate, p2_recent_bluff_rate, p1_bluffs/i, p2_bluffs/i)).tolist())
            y.append((card_norm[cards[p1_i]], card_norm[cards[p2_i]]))
            i += 1
    return X, y

class Round(Layer):

    def __init__(self, **kwargs):
        super(Round, self).__init__(**kwargs)

    def get_output(self, train=False):
        X = self.get_input(train)
        return K.round(X)

    def get_config(self):
        config = {"name": self.__class__.__name__}
        base_config = super(Round, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
class Rounder(Layer):

	def __init__(self, **kwargs):
		super(Rounder, self).__init__(**kwargs)
		self.output_dim = (2,)

	def get_output_shape_for(self, input_shape):
		return (input_shape[0], input_shape[1])

	def call(self, x, mask=None):
		x1 = K.round(x)
		return x1

def train_model():
    X, y = get_data()
    #y = to_categorical(y)
    X = np.asarray([x[:6] for x in X])
    y = np.asarray([y[0] for y in y])
    X_train = X[:int(len(X)*0.8)]
    y_train = y[:int(len(y)*0.8)]
    X_test = X[int(len(X)*0.8)+1:]
    y_test = y[int(len(y)*0.8)+1:]
    #print(X_train[0:5])
    #print(y_train[0])
    def custom_activation(x):
    #    print(type((K.sigmoid(x) * 4) + 1) == type(K.round((K.sigmoid(x) * 4) + 1)))
        return ((K.softsign(x) * 2.5) + 2.5)
    def loss(y_pred, y_true):
        #return K.categorical_crossentropy(y_true, K.round(y_pred))
        return K.mean(K.square(K.round(y_pred) - y_true), axis=-1)
    model = Sequential()
    model.add(Dense(1024, activation='relu', input_shape=(6,)))
    for i in range(1):
        model.add(Dense(1024, activation='relu'))
    #for i in range(30):
    #    model.add(Dense(1024, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    #model.add(Dense(4, activation=custom_activation))
    #model.add(Rounder())
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x=X_train, y=y_train, batch_size=32, epochs=50, verbose=1)
    print(model.evaluate(x=X_test, y=y_test))
    predictions = model.predict(np.asarray(X_train))
    for i,prediction in enumerate(predictions):
        print(X_test[i], [(pred) for pred in prediction], y_test[i])

train_model()

