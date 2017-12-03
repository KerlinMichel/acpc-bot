from socket import socket, AF_INET, SOCK_STREAM
from sys import argv

sock = socket(AF_INET, SOCK_STREAM)

sock.connect(('localhost', int(argv[1])))
print('connect')

sock.send(b'VERSION:2.0.0\r\n')

def parse_line(line):
    params = line.split(b':')
    print('params', params)
    return int(params[1]), params[2], params[3].decode(), params[-1].decode().strip()

def isShowDown(cards):
    return len([c for c in cards.split('|') if c]) == 3

i = 0
while True:
    line = sock.recv(4096)
    params = parse_line(line)
    if isShowDown(params[3]) or not line:
        i = 0
        continue
    if i % 3 == params[0]:
       if 'r' in params[2]:
           move = 'c'
       else:
           move = 'r'
       sock.send(line[:-2] + b':' + move.encode() + b'\r\n')
       print('send', line[:-2] + b':' + move.encode() + b'\r\n')
    i += 1
