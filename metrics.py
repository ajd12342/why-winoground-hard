import numpy as np

def compute_metrics(x):
    # From AVLnet code
    sx = np.sort(-x, axis=1)
    d = np.diag(-x)
    d = d[:, np.newaxis]
    ind = sx - d
    ind = np.where(ind == 0)
    ind = ind[1]
    metrics = {}
    test_set_size = x.shape[0]
    metrics['R1'] = float(np.sum(ind == 0)) / test_set_size
    metrics['R2'] = float(np.sum(ind == 0)) / test_set_size
    metrics['R5'] = float(np.sum(ind < 5)) / test_set_size
    metrics['R10'] = float(np.sum(ind < 10)) / test_set_size
    metrics['MR'] = np.median(ind) + 1
    print('Recall @ 1', metrics['R1'])
    print('Recall @ 2', float(np.sum(ind < 2)) / test_set_size)
    print('Recall @ 5', metrics['R5'])
    print('Recall @ 10', metrics['R10'])
    print('Median R', metrics['MR'])
    return metrics, ind
