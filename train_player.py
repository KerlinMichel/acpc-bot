from os import listdir
from queue import Queue
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
 
from keras.preprocessing import sequence
from keras.utils import np_utils

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
        states = []
        i = 1
        for l in data[:10]:
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
                if plays[p1_i] == 'r':
                    p1_bluff += 1
            if cards[p1_i] == 'Qs':
                if plays[p1_i] == 'c':
                    p1_bluff += 0.5
                if plays[p1_i] == 'r':
                    p1_bluff += 0.75

            if cards[p2_i] == 'Js':
                if plays[p2_i] == 'c':
                    p2_bluff += 0.75
                if plays[p2_i] == 'r':
                    p2_bluff += 1
            if cards[p2_i] == 'Qs':
                if plays[p2_i] == 'c':
                    p2_bluff += 0.5
                if plays[p2_i] == 'r':
                    p2_bluff += 0.75
            p1_bluffs += p1_bluff
            p2_bluffs += p2_bluff
            p1_recent_bluffs.put(p1_bluff)
            p2_recent_bluffs.put(p2_bluff)
            p1_recent_bluff_rate = sum(list(p1_recent_bluffs.queue))/10
            p2_recent_bluff_rate = sum(list(p2_recent_bluffs.queue))/10
            states.append(((me_c, p1_recent_bluff_rate, p2_recent_bluff_rate, p1_bluffs/i, p2_bluffs), (cards[p1_i], cards[p2_i])))
            i += 1
    return states

data = get_data()
print(data[:5])

def train_model():
    pass
