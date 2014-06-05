
# coding: utf-8
import numpy as np
from runner import Runner
from mlcin.prototypes.rps import RandomPrototypeSelection
from mlcin.utils.keel import load_dataset
from mlcin.utils.graphics import plot_and_save




if __name__ == '__main__':

    # creating prototype generation object
    rps = RandomPrototypeSelection()

    datasets = ['banana', 'normal', 'normal_multimodal']
    for dataset in datasets:
        X_orig, y_orig = load_dataset('datasets/artificial/' + dataset + '.data')
        y_orig = np.asarray(y_orig, dtype=int)
        
        rps.fit(X_orig, y_orig)
        X, y = rps.reduce_data()

        print dataset + '\treduction: %.2f' % (1.0 - float(y.shape[0])/len(y_orig))
        plot_and_save(X_orig, y_orig, title='ORIGINAL', filename='images/ORIG_' + dataset + '.png')
        plot_and_save(X, y, title='RSP', filename='images/RSP_' + dataset + '.png')
    

