# coding: utf-8
import numpy as np
from time import time
from runner import Runner
from mlcin.prototypes.rsp1 import ReductionBySpacePartitioning1
from mlcin.prototypes.rsp2 import ReductionBySpacePartitioning2
from mlcin.prototypes.rsp3 import ReductionBySpacePartitioning3
from mlcin.prototypes.cnn import CNN
from mlcin.prototypes.enn import ENN
from mlcin.prototypes.renn import RENN
from mlcin.prototypes.sgp import SGP
from mlcin.prototypes.tomek_links import TomekLinks

class RunnerRPS(Runner):

    def __init__(self, folds=5, normalize=True, prefix=None, module=None,mode=1,b=1):
        self.folds = 5
        self.normalize = normalize
        self.prefix = prefix
        self.module = module
        self.datasets = []
        self.output_buffer = ''
        self.mode = mode
        self.b = b
        
    def get_prototypes(self, X, y):
        if self.mode == 1:
            rps = ReductionBySpacePartitioning1(b=self.b)
        elif self.mode == 2:
            rps = ReductionBySpacePartitioning2(b=self.b)
        elif self.mode == 3:
            rps = ReductionBySpacePartitioning3()  
        elif self.mode == 4:
            rps = CNN()
        elif self.mode == 5:
            rps = ENN()
        elif self.mode == 6:
            rps = RENN()
        elif self.mode == 7:
            rps = SGP()
        else:
            rps = TomekLinks()
            
        r = rps.fit(X, y)
        
        if (self.mode < 4):
            r.reduce_data()
        #print rps.get_prototypes()
        return rps.get_prototypes()

if __name__ == '__main__':
    modulos = ['regular10']
    mode = 1
    b = 10
    
    for i in range(4,8):
        mode = i
        for modulo in modulos:
            if (mode == 1 or mode == 2):
                for j in range(5,6):
                    b = j
                    if( modulo == 'regular10' ):
                    
                        runner = RunnerRPS( folds=9, normalize=True, prefix='datasets', module=modulo,mode=mode,b=b )
                        
                        datasets =            ['glass'   , 'image_segmentation', 'ionosphere'   , 'iris'  ]
                        #datasets = datasets + ['liver'   , 'pendigits'         , 'pima_diabetes', 'sonar' ]
                        #datasets = datasets + ['spambase', 'vehicle'           , 'vowel'        , 'wine'  , 'yeast' ]
                        
                        
                    elif( modulo == 'imbalanced' ):
                    
                        runner = RunnerRPS( folds=5, normalize=True, prefix='datasets', module=modulo,mode=mode,b=b )
                    
                        datasets =            ['glass1', 'ecoli-0_vs_1', 'iris0'           , 'glass0'          ]
                        datasets = datasets + ['ecoli1', 'new-thyroid2', 'new-thyroid1'    , 'ecoli2'          ]
                        datasets = datasets + ['glass6', 'glass2'      , 'shuttle-c2-vs-c4', 'glass-0-1-6_vs_5']
                
                    
                    print "Modulo: ",modulo
                    print "Algoritmo RSP",mode
                    print "Numero de iteracoes: ",b
                    print "Datasets: ",datasets
                    runner.set_datasets(datasets)
                
                    beginTime = time()
                    runner.run()
                    endTime = time()
                    print "\nTempo: ",(endTime - beginTime), " Segundos\n"
                
                    output = 'dataset\tGen. Accuracy\tMaj. Accuracy\tMin. Accuracy\t'
                    output = output + 'AUC. Accuracy\tData Reduction\n'
                    print output + runner.get_output_buffer()
            else:
                #Don't have the parameter b
                if( modulo == 'regular10' ):
                
                    runner = RunnerRPS( folds=9, normalize=True, prefix='datasets', module=modulo,mode=mode,b=b )
                    
                    datasets =            ['glass'   , 'image_segmentation', 'ionosphere'   , 'iris'  ]
                    #datasets = datasets + ['liver'   , 'pendigits'         , 'pima_diabetes', 'sonar' ]
                    #datasets = datasets + ['spambase', 'vehicle'           , 'vowel'        , 'wine'  , 'yeast' ]
                    
                    
                elif( modulo == 'imbalanced' ):
                
                    runner = RunnerRPS( folds=5, normalize=True, prefix='datasets', module=modulo,mode=mode,b=b )
                
                    datasets =            ['glass1', 'ecoli-0_vs_1', 'iris0'           , 'glass0'          ]
                    datasets = datasets + ['ecoli1', 'new-thyroid2', 'new-thyroid1'    , 'ecoli2'          ]
                    datasets = datasets + ['glass6', 'glass2'      , 'shuttle-c2-vs-c4', 'glass-0-1-6_vs_5']
            
                
                print "Modulo: ",modulo
                print "Algoritmo RSP",mode
                print "Numero de iteracoes: ",b
                print "Datasets: ",datasets
                runner.set_datasets(datasets)
            
                beginTime = time()
                runner.run()
                endTime = time()
                print "\nTempo: ",(endTime - beginTime), " Segundos\n"
            
                output = 'dataset\tGen. Accuracy\tMaj. Accuracy\tMin. Accuracy\t'
                output = output + 'AUC. Accuracy\tData Reduction\n'
                print output + runner.get_output_buffer()