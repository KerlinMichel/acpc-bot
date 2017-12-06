from socket import socket, AF_INET, SOCK_STREAM
from sys import argv
from queue import Queue
from train_player import train_model, card_norm
import numpy as np
import operator

sock = socket(AF_INET, SOCK_STREAM)

sock.connect((argv[1], int(argv[2])))

sock.send(b'VERSION:2.0.0\r\n')

def parse_line(line):
    params = line.split(b':')
    return int(params[1]), params[2], params[3].decode(), params[-1].decode().strip()

def isShowDown(cards):
    return len([c for c in cards.split('|') if c]) == 3

i = 0
h = 1
p1_bluff = 0
p2_bluff = 0
p1_rbluff = Queue(10)
p2_rbluff = Queue(10)
model = train_model()
while True:
    line = sock.recv(4096)
    if not line:
        print('Done playing')
        break
    line = line.split()[-1]
    params = parse_line(line)
    if p1_rbluff.full():
        p1_rbluff.get()
    if p2_rbluff.full():
        p2_rbluff.get()
    if isShowDown(params[3]) or not line:
        plays = list(params[2])
        if len(plays) < 3:
            i = 0
            continue
        cards = params[3].split('|')
        del cards[params[0]]
        p1i = (params[0] + 1) % 2
        if cards[0] == 'Js':
            if plays[p1i] == 'c':
                p1_bluff += 0.75
            elif plays[p1i] == 'r':
                p1_bluff += 1
        if cards[0] == 'Qs':
            if plays[p1i] == 'c':
                p1_bluff += 0.5
            elif plays[p1i] == 'r':
                p1_bluff += 0.75
        if cards[0] == 'Ks':
            if plays[p1i] == 'c':
                p1_bluff += 0.125
            if plays[p1i] == 'r':
                p1_bluff += 0.25
        p1_bluff += p1_bluff
        p1_rbluff.put(p1_bluff)
        i = 0
        continue
    #print('params', params)
    if i % 3 == params[0] or True:
       m = 0
       if 'r' in params[2]:
           moves = ['f', 'r']
       else:
           moves = ['c', 'r']
       mycard = params[3].split('|')[0]
       if mycard:
           p1rb = sum(list(p1_rbluff.queue))/10
           p2rb = sum(list(p2_rbluff.queue))/10
           X = np.append(card_norm[mycard], [p1rb, p1_bluff/h, p2rb, p2_bluff/h])
           pred = model.predict(np.asarray([X])) 
           #print(pred[0].tolist()[0], pred[1].tolist()[0])
           index, value = max(enumerate(pred[0].tolist()[0]), key=operator.itemgetter(1))
           if card_norm[mycard].tolist().index(1) < index:
               me = 1 
       sock.send(line + b':' + moves[1].encode() + b'\r\n')
    i += 1
    h += 1
