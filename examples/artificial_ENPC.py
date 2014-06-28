
# coding: utf-8
import numpy as np
import os
from mlcin.prototypes.ENPC import ENPC
from mlcin.utils.keel import load_dataset
from mlcin.utils.graphics import plot_and_save




if __name__ == '__main__':

    datasets = ['banana', 'normal', 'normal_multimodal']
    
    iterations = [ 2, 10, 50, 100, 150, 200 ]
    
    for dataset in datasets:
        
        X_orig, y_orig = load_dataset('datasets/artificial/' + dataset + '.data')
        y_orig         = np.asarray(y_orig, dtype=int)
        
        
        for iteration in iterations:

            enpc = ENPC( X_orig, y_orig, 3, iteration )
            enpc.run_ENPC()
            X, y = enpc.getResult()
    
    
            path = '../images'
            if not os.path.exists( path ):
                os.mkdir( path )
    
            path = '../images/' + dataset
            if not os.path.exists( path ):
                os.mkdir( path )
    
            path = '../images/' + dataset + '/iterations ' + str( iteration )
            if not os.path.exists( path ):
                os.mkdir( path )
    
            f = open( path + '/reduction.txt', 'w' )
            f.write( dataset + '\treduction: %.2f' % ( 1.0 - float(y.shape[0])/len(y_orig) ) )
            f.close()
            
            plot_and_save(X_orig, y_orig, title='ORIGINAL', filename=path + '/ORIG_' + dataset + '.png')
            plot_and_save(X     , y     , title='ENPC'    , filename=path + '/ENPC_' + dataset + '.png')

