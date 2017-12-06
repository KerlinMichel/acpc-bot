from os import listdir
from queue import Queue
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Input
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
    num = 0
    for f in listdir('./play_dataset'):
        num += 1
        if num > 10:
            break
    #f = listdir('./play_dataset')[0]
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

def train_model():
    X, y = get_data()
    #y = to_categorical(y)
    X = np.asarray([x for x in X])
    y1 = np.asarray([y[0] for y in y])
    y2 = np.asarray([y[1] for y in y])
    X_train = X[:int(len(X)*0.8)]
    y1_train = y1[:int(len(y)*0.8)]
    y2_train = y2[:int(len(y)*0.8)]
    X_test = X[int(len(X)*0.8)+1:]
    y1_test = y1[int(len(y)*0.8)+1:]
    y2_test = y2[int(len(y)*0.8)+1:]
    #print(X_train[0:5])
    #print(y_train[0])
    def custom_activation(x):
    #    print(type((K.sigmoid(x) * 4) + 1) == type(K.round((K.sigmoid(x) * 4) + 1)))
        return ((K.softsign(x) * 2.5) + 2.5)
    def loss(y_pred, y_true):
        #return K.categorical_crossentropy(y_true, K.round(y_pred))
        return K.mean(K.square(K.round(y_pred) - y_true), axis=-1)
    model = Sequential()
    input_layer = Input(shape=(8,))
    #model.add(input_layer)
    #for i in range(1):
        #model.add(Dense(1024, activation='relu'))
    act = 'relu'
    hidden = Dense(1024, activation=act)(input_layer)
    #hidden = Dense(1024, activation=act)(hidden)
    #model.add(hidden(input_layer))
    #for i in range(30):
    #    model.add(Dense(1024, activation='relu'))
    output1 = Dense(4, activation='softmax')(hidden)
    output2 = Dense(4, activation='softmax')(hidden)
    #model.add(Dense(4, activation=custom_activation))
    #model.add(Rounder())
    model = Model(inputs=input_layer, outputs=[output1, output2])
    model.compile(optimizer='adam', loss=['categorical_crossentropy', 'categorical_crossentropy'], metrics=['accuracy'])
    history = model.fit(x=X_train, y=[y1_train, y2_train], batch_size=32, epochs=50, verbose=1)
    return model
    #print(model.evaluate(x=X_test, y=[y1_test, y2_test]))
    #[predictions1, predictions2] = model.predict(np.asarray(X_train[:10]))
    #for i,prediction in enumerate(predictions1):
    #    print(X_test[i][:4], prediction, y1_test[i], predictions2[i], y2_test[i])

