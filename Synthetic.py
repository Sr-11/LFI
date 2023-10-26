path = '/Users/patrik/tmp'
if __name__=='__main__':
    from random import *
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from numba import jit
    import scipy
    k = 100
    eps = 3e-1
    p0 = [1/k + (2*(i%2)-1)*eps/k for i in range(k)]
    p1 = [1/k - (2*(i%2)-1)*eps/k for i in range(k)]
    n_exp = 1e4
    n_exp=int(n_exp)
    grid = np.logspace(1,4,40)
    probs = np.zeros(shape=(len(grid), len(grid)))
    for a,n in enumerate(grid):
        for b,m in enumerate(grid):
            n=int(n)
            m=int(m)
            print(a,b)
            for _ in range(n_exp):
                x_draws = np.random.multinomial(n,p0)/n
                y_draws = np.random.multinomial(n,p1)/n
                z_draws = np.random.multinomial(m,p0)/m
                T = sum((x-z)**2 for (x,z) in zip(x_draws, z_draws)) - sum((y-z)**2 for (y,z) in zip(y_draws, z_draws))
                probs[a,b] += T > 0
            probs[a,b] /= n_exp
            if probs[a,b]==0: break
    probs
    num_ticks = 10
    ticks = np.linspace(0, len(grid) - 1, num_ticks, dtype=int)
    ticklabels = ['1e1','2e1','4e1','1e2','2e2','4e2','1e3','2e3','4e3','1e4']
    df3_smooth = scipy.ndimage.filters.gaussian_filter(probs, sigma=4)
    ax = sns.heatmap(df3_smooth, square=True)
    ax.set_yticks(ticks)
    ax.set_xticks(ticks)
    ax.set_yticklabels(ticklabels, size=12)
    ax.set_xticklabels(ticklabels, size=12)
    ax.set_title(“Type-I error”, size=15)
    ax.invert_yaxis()
    ax.set_xlabel('m', size=15)
    ax.set_ylabel('n', size=15)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)
    plt.savefig(path + '/theoretical_trade_off.png', bbox_inches='tight')