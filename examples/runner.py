# coding: utf-8
import numpy as np
import mlcin.utils.keel as keel
from mlcin.utils.metrics import evaluate
from mlcin.utils import data


class Runner(object):

    def __init__(self, folds=5, normalize=True, prefix=None, module=None):
        self.folds = folds
        self.normalize = normalize
        self.prefix = prefix
        self.module = module
        self.datasets = []
        self.output_buffer = ''

    def set_datasets(self, datasets):
        self.datasets = datasets

    def get_prototypes(self, X, y):
        return X, y

    def print_results(self, dataset, results):
        results = np.asarray(results)
        mean = np.mean(results, axis=0)
        stdv = np.std(results, axis=0)

        output = dataset + '\t'
        output = output + '%.2f\t%.2f\t' % (mean[0], stdv[0])
        output = output + '%.2f\t%.2f\t' % (mean[1], stdv[1])
        output = output + '%.2f\t%.2f\t' % (mean[2], stdv[2])
        output = output + '%.2f\t%.2f\t' % (mean[3], stdv[3])
        output = output + '%.2f\t%.2f\t' % (mean[4], stdv[4])

        self.output_buffer = self.output_buffer + output + '\n'
        # print output

    def run(self):
        for index, dataset in enumerate(self.datasets):
            results = []
            for fold in range(0, self.folds + 1): # Para base balanceada
            #for fold in range(1, self.folds + 1): # Para base desbalanceada
                X_tra, y_tra = keel.load_keel_dataset(
                    dataset, fold, 'tra', self.prefix, self.module, fold_count=self.folds)
                X_tst, y_tst = keel.load_keel_dataset(
                    dataset, fold, 'tst', self.prefix, self.module, fold_count=self.folds)

                if self.normalize:
                    mx, mn, rg = data.normalize_args(X_tra)
                    X_tra = data.normalize(X_tra, mx, mn, rg)
                    X_tst = data.normalize(X_tst, mx, mn, rg)

                X_ptt, y_ptt = self.get_prototypes(X_tra, y_tra)
                fold_results = evaluate(
                    X_ptt,
                    y_ptt,
                    X_tst,
                    y_tst,
                    k=1,
                    positive_label=1)
                fold_results = list(fold_results) + [
                    1.0 - float(len(y_ptt)) / len(y_tra)]
                results = results + [fold_results]

            self.print_results(str(index + 1), results)

    def get_output_buffer(self):
        return self.output_buffer


if __name__ == '__main__':
    
    modulo = 'regular10'
    
    if( modulo == 'regular10' ):
    
        runner = Runner( folds=9, normalize=True, prefix='datasets', module=modulo )

        datasets =            ['glass'   , 'image_segmentation', 'ionosphere'   , 'iris'  ]
        datasets = datasets + ['liver'   , 'pendigits'         , 'pima_diabetes', 'sonar' ]
        datasets = datasets + ['spambase', 'vehicle'           , 'vowel'        , 'wine'  , 'yeast' ]
        
    elif( modulo == 'imbalanced' ):
    
        runner = Runner( folds=5, normalize=True, prefix='datasets', module=modulo )
    
        datasets =            ['glass1', 'ecoli-0_vs_1', 'iris0'           , 'glass0'          ]
        datasets = datasets + ['ecoli1', 'new-thyroid2', 'new-thyroid1'    , 'ecoli2'          ]
        datasets = datasets + ['glass6', 'glass2'      , 'shuttle-c2-vs-c4', 'glass-0-1-6_vs_5']


    runner.set_datasets(datasets)

    runner.run()

    output = 'dataset\tGen. Accuracy\tMaj. Accuracy\tMin. Accuracy\t'
    output = output + 'AUC. Accuracy\tData Reduction\n'
    print output + runner.get_output_buffer()
