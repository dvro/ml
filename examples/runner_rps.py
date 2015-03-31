# coding: utf-8
import numpy as np
from runner import Runner
from mlcin.prototypes.rps import RandomPrototypeSelection


class RunnerRPS(Runner):

    def get_prototypes(self, X, y):
        rps = RandomPrototypeSelection()
        rps.fit(X, y).reduce_data()
        print rps.get_prototypes()
        return rps.get_prototypes()

if __name__ == '__main__':
    runner = RunnerRPS(
        folds=5,
        normalize=True,
        prefix='datasets',
        module='imbalanced')

    datasets = ['glass1', 'ecoli-0_vs_1', 'iris0', 'glass0']
    datasets = datasets + ['ecoli1', 'new-thyroid2', 'new-thyroid1', 'ecoli2']
    datasets = datasets + [
        'glass6',
        'glass2',
        'shuttle-c2-vs-c4',
        'glass-0-1-6_vs_5']
    runner.set_datasets(datasets)

    runner.run()

    output = 'dataset\tGen. Accuracy\tMaj. Accuracy\tMin. Accuracy\t'
    output = output + 'AUC. Accuracy\tData Reduction\n'
    print output + runner.get_output_buffer()
