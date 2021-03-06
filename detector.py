import sys
import os
import numpy as np
from PIL import Image
from scipy import ndimage
from model import model

def detector(model_fname, in_fname, out_fname=None):
 
    model.load(model_fname)

    im = Image.open(in_fname)
    arr = np.array(im)[:,:,0:3]
    shape = arr.shape

    if not out_fname: 
        out_fname = os.path.splitext(in_fname)[0] + '_detection.png'

    detections = np.zeros((shape[0], shape[1]), dtype='uint8')
    output = np.copy(arr)

    step = 2
    win = 20

    print('Processing...')
    for i in range(0, shape[0]-win, step):
        print('row %1.0f of %1.0f' % (i, (shape[0]-win-1)))

        for j in range(0, shape[1]-win, step):

            chip = arr[i:i+win,j:j+win,:] 
            
            prediction = model.predict_label([chip / 255.])[0][0]

            if prediction == 1:
                detections[i+int(win/2), j+int(win/2)] = 1

    dilation = ndimage.binary_dilation(detections, structure=np.ones((3,3)))
    labels, n_labels = ndimage.label(dilation)
    center_mass = ndimage.center_of_mass(dilation, labels, np.arange(n_labels)+1)

    if type(center_mass) == tuple: center_mass = [center_mass]
    for i, j in center_mass:
        i = int(i - win/2)
        j = int(j - win/2)
        
        output[i:i+win, j:j+2, 0:3] = [255,0,255]
        output[i:i+win, j+win-2:j+win, 0:3] = [255,0,255]
        output[i:i+2, j:j+win, 0:3] = [255,0,255]
        output[i+win-2:i+win, j:j+win, 0:3] = [255,0,255]

    outIm = Image.fromarray(output)
    outIm.save(out_fname)


if __name__ == "__main__":

    if len(sys.argv) == 3:
        detector(sys.argv[1], sys.argv[2])
    else:
        detector(sys.argv[1], sys.argv[2], sys.argv[3])