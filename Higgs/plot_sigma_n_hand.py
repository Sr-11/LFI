import matplotlib.pyplot as plt
import numpy as np

# n_list =    [1000,  5000, 10000, 30000,  100000, 300000, 600000,  1000000, 1500000, 2000000, 2500000]#,3000000]
# pval_list = [0.832, 0.94, 1.15,  1.761,  1.1,    2.34,   2.9854,  4.177,   4.97,    4.49,    4.588]#  3.9225]
# pval_var =  [0.935, 1.49, 0.58,  1.1,    1.2,    0.51,   0.6589,  1.025,   0.88,    0.968,   0.4556]#  0.479]

n_list =    [100000, 400000, 700000,  1000000, 1300000, 1600000]#,3000000]
pval_list = [1.1,    2.34,   2.9854,  4.177,   4.97,    4.49,   ]#  3.9225]

n_list = np.array(n_list)
pval_list = np.array(pval_list)
pval_var = np.array(pval_var)**0.5

plt.plot(n_list, pval_list)
plt.errorbar(n_list, pval_list, yerr=pval_var)
plt.xlabel('n')
plt.ylabel('p-value/σ')
plt.savefig('pval_n.png')
plt.clf()

plt.plot(np.log(n_list)/np.log(10), pval_list)
plt.errorbar(np.log(n_list)/np.log(10), pval_list, yerr=pval_var)
plt.xlabel('lg(n)')
plt.ylabel('p-value/σ')
plt.savefig('pval_lg(n).png')
