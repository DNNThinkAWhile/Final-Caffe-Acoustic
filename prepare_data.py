import lmdb
import numpy as np
import caffe
import sys

MAP_FILE = 'MLDS_HW1_RELEASE_v1/phones/48_idx_chr.map'

class Datum:
    def __init__(self, spchid='', x=np.array([]), y=-1):
        self.spchid = spchid
        self.x = x
        self.y = y

def main(argv):
    if len(argv) < 4:
        print 'cmd: prepare_data.py train.ark train.lab mylmdb'
        exit(-1)

    filein = argv[1]
    dbname = argv[3]
    label_file = argv[2]
  
  
    spchid_phone_map, d_index_phone, d_phone_index, d_phone_alphabet  = read_map(label_file, MAP_FILE)
    
    data = read_data(filein, spchid_phone_map)
    write_db(data, dbname)

def read_map(file_label, file_48_idx):
    d_speechid_index = {}
    d_index_phone = {}
    d_phone_index = {}
    d_phone_alphabet = {}
    with open(file_48_idx, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            # map phonemes(sil, aa,..) to alphabets(a, b, c,..)
            d_phone_alphabet[tokens[0]] = tokens[2]
            # map phonemes(sil, aa,..) to their indices(int), 0:47
            d_phone_index[tokens[0]] = tokens[1]
            # map index(int), 0:47 to phonemes(sil, aa,..)
            d_index_phone[tokens[1]] = tokens[0]

    # map speech_id(like "maeb0_si1411_1") to their phonemes' indices, 1943->48
    with open(file_label, 'r') as f:
        for line in f:
            tokens = line.strip().split(',')
            d_speechid_index[tokens[0]] = d_phone_index[tokens[1]]
    return d_speechid_index, d_index_phone, d_phone_index, d_phone_alphabet


def read_data(fn, spchid_phone_map):
    FEAT_CNT = 69 + 39
    
    k = 1     # channels 
    h = 1
    w = FEAT_CNT
    
    #x = np.zeros((n, k, h, w), type=np.float32)
    #y = np.zeros(n, dtype=np.int64)
    xs = []
    ys = []
    data = []

    with open(fn, 'r') as f:
        lines = f.readlines()
        for l in lines:
            spchid, x = parseline(l)
            y = int(spchid_phone_map[spchid])
            data.append(Datum(spchid, x, y))
    return data


def parseline(line_str):
    l = line_str.strip().split()
    x = np.array([float(feat) for feat in l[1:]], dtype=np.float32)
    spchid = l[0]
    return spchid, x

def write_db(data, db):
    MAP_SIZE = 1024*1024*1024*1024
    env = lmdb.open(db, map_size=MAP_SIZE)
    with env.begin(write=True) as txn:
        for mydatum in data:
            datum = build_datum(mydatum)
            str_id = mydatum.spchid
            txn.put(str_id, datum.SerializeToString())

def build_datum(mydatum):
    datum = caffe.proto.caffe_pb2.Datum()
    datum.channels = 1
    datum.height = 1
    datum.width = len(mydatum.x)
    datum.label = mydatum.y
    datum.data = mydatum.x.tobytes()
    return datum



if __name__ == '__main__':
    main(sys.argv)







