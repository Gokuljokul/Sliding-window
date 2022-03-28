import sys
import json
import numpy as np
from tflearn.data_utils import to_categorical
from model import model

def train(fname, out_fname):
    f = open(fname)
    planesnet = json.load(f)
    f.close()


    X = np.array(planesnet['data']) / 255.
    X = X.reshape([-1,3,20,20]).transpose([0,2,3,1])
    Y = np.array(planesnet['labels'])
    Y = to_categorical(Y, 2)

    model.fit(X, Y, n_epoch=50, shuffle=True, validation_set=.2, show_metric=True, batch_size=128, run_id='planesnet')

    model.save(out_fname)


if __name__ == "__main__":

    # Train using input file
    train(sys.argv[1], sys.argv[2])
