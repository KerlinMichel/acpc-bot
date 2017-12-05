from os import listdir

def get_data():
    #for f in listdir('./play_dataset'):
    #    print(f)
    f = listdir('./play_dataset')[0]
    with open('./play_dataset/' + f) as log:
        data = [l.strip() for l in log.readlines()[4:]]
        print(data)
    return data

data = get_data()
