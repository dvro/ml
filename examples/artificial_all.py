
# coding: utf-8
import numpy as np
from runner import Runner
from mlcin.prototypes.rsp1 import ReductionBySpacePartitioning1
from mlcin.utils.keel import load_dataset
from mlcin.utils.graphics import plot_and_save
from mlcin.prototypes.rsp2 import ReductionBySpacePartitioning2
from mlcin.prototypes.rsp3 import ReductionBySpacePartitioning3
from mlcin.prototypes.cnn import CNN
from mlcin.prototypes.enn import ENN
from mlcin.prototypes.renn import RENN
from mlcin.prototypes.sgp import SGP
from mlcin.prototypes.tomek_links import TomekLinks





if __name__ == '__main__':


    datasets = ['banana', 'normal', 'normal_multimodal']
    for dataset in datasets:
        X_orig, y_orig = load_dataset('datasets/artificial/' + dataset + '.data')
        y_orig = np.asarray(y_orig, dtype=int)
        
        plot_and_save(X_orig, y_orig, title='ORIGINAL', filename='images/ORIG_' + dataset + '.png')
            # creating prototype generation object
            
        for mode in range(1,3):
            title = ""
            if mode == 1:
                rps = ReductionBySpacePartitioning1(b=30)
                title = "ReductionBySpacePartitioning 1 - b =30"
            elif mode == 2:
                rps = ReductionBySpacePartitioning2(b=30)
                title = "ReductionBySpacePartitioning 2 - b =30"
            elif mode == 3:
                rps = ReductionBySpacePartitioning3()  
                title = "ReductionBySpacePartitioning  3"
            elif mode == 4:
                rps = CNN()
                title = "CNN"
            elif mode == 5:
                rps = ENN()
                title = "ENN"
            elif mode == 6:
                rps = RENN()
                title = "RENN"
            elif mode == 7:
                rps = SGP()
                title = "SGP"
            else:
                rps = TomekLinks()
                title = "Tomek Links"
                
            r = rps.fit(X_orig, y_orig)
        
            if (mode < 4):
                r.reduce_data()

            
            X, y = rps.get_prototypes()

            print "Algoritmo ", mode, dataset + '\treduction: %.2f' % (1.0 - float(y.shape[0])/len(y_orig))
            plot_and_save(X, y, title=title, filename='images/RSP_b=30' + dataset + "Algoritmo " + str(mode) + '.png')
    

