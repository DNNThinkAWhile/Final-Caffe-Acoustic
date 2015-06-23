import numpy as np
import caffe
import sys
from prepare_data import Datum
import prepare_data

MODEL_FILE = 'acoustic_predict.prototxt'
PRETRAINED = '_iter_180000.caffemodel'
FEAT_NUM = 69 + 39
INPUT_FILE = 'MLDS_HW1_RELEASE_v1/mfccfbank/train.normalized.cv.test.ark'
BATCH = 256
LABEL_FILE = 'MLDS_HW1_RELEASE_v1/label/train.lab'
MAP_FILE = 'MLDS_HW1_RELEASE_v1/phones/48_idx_chr.map'

def main(argv):
    print 'start!!!'

    spchid_phone_map, d_index_phone, d_phone_index, d_phone_alphabet  = prepare_data.read_map(LABEL_FILE, MAP_FILE)
    
    caffe.set_mode_gpu()
    net = caffe.Classifier(MODEL_FILE, PRETRAINED,raw_scale=1, image_dims=(1, FEAT_NUM))

    
    
    #input : iterable : (H x W x K)
    (spids, xs)  = read_data(INPUT_FILE, count=2048)
    ys = [int(spchid_phone_map[spid]) for spid in spids]
    sum_pred = 0
    sum_correct = 0
    for i in xrange(len(ys) / BATCH):
        ys_part = ys[i*BATCH: i*BATCH + BATCH]
        xs_part = xs[i*BATCH: i*BATCH + BATCH]
        pred = net.predict(xs_part)
        #print 'prediction shape:', pred[0].shape
        #print 'pred class:', pred[0].argmax()
        sum_pred += BATCH
        sum_correct += \
                sum([1 if prob.argmax() == y else 0 \
                        for (prob, y) in zip(pred, ys_part)])
        print 'ys_part:',ys_part
        print 'pred:', [prob.argmax() for prob in pred]

    print 'predicted:', sum_pred
    print 'correct:', sum_correct
    print 'acc:', float(sum_correct) / float(sum_pred)
    

def read_data(in_file, count=-1):
    lines = []
    with open(in_file, 'r') as f:
        if count < 0:
            lines = f.readlines()
        else:
            for i in xrange(count):
                lines.append(f.readline())
    xs = []
    spids = []
    for l in lines:
        l = l.strip().split()
        x = [float(ft) for ft in l[1:]]
        spids.append(l[0])
        xs.append(np.array(x).reshape((1,FEAT_NUM,1)))
    return spids, xs




if __name__=='__main__':
    main(sys.argv)
