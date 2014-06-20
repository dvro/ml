import numpy as np


def load_keel_dataset(dataset, fold, kind, prefix, module, fold_count=5):
    if kind not in ['tra', 'tst']:
        print 'ERROR: kind must be in [\'tra\', \'tst\']'
        return np.asarray([]), np.asarray([])

    if module not in ['imbalanced', 'regular10']:
        print 'ERROR: module must be in [\'imbalanced\', \'regular10\']'
        return np.asarray([]), np.asarray([])

    path = prefix + '/' + module + '/' + dataset
    if module == 'imbalanced':
        path = path + '/' + dataset + '-' + \
            str(fold_count) + '-' + str(fold) + kind + '.dat'
    elif module == 'regular10':
        path = path + '/' + \
            ('training' if kind == 'tra' else 'test') + '_' + str(fold)

    return load_dataset(path)


def load_dataset(string, separator=","):
    try:
        f = open(string, "r")
        s = [line for line in f]
        f.close()
    except:
        print 'ERROR LOADING DATABASE ' + string
        s = []

    s = filter(lambda e: e[0] != '@', s)
    s = [v.strip().split(separator) for v in s]
    X = np.asarray([[float(e) for e in v[:-1]] for v in s])

    d = {'positive': 2, 'negative': 1}
    y = np.asarray([d[v[-1].strip()] if v[-1].strip() in d else v[-1].strip()
                   for v in s])

    return X, y
