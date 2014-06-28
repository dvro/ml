import matplotlib.pyplot as plt


def plot_and_save(X, y, x_lim=(-10,10), y_lim=(-10,10), markers=['bs', 'ro', 'k^', 'yo'], title='TITLE', filename='filename.png'):

    plt.title(title)
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    plt.xlabel('feature 1')
    plt.ylabel('feature 2')

    for lbl, mkr in zip(set(y), markers):
        mask = y == lbl
        _X = X[mask]
        
        plt.plot(_X[:,0], _X[:,1], mkr)

    plt.savefig(filename)
    plt.clf() 



