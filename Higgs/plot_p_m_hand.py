# import re
# cProfile.run('re.compile("LFI")', filename='result.out')
# import pstats
# # 创建 Stats 对象
# p = pstats.Stats('result.out')
# # 按照运行时间和函数名进行排序
# # p.strip_dirs().sort_stats("cumulative", "name").print_stats(0.5)
# p.strip_dirs().sort_stats("cumulative", "name").print_stats(30)
# # 按照函数名排序，只打印前 3 行函数的信息, 参数还可为小数, 表示前百分之几的函数信息
# # 如果想知道有哪些函数调用了 ccc
# # p.print_callers(0.5, "ccc")
# # 查看 ccc() 函数中调用了哪些函数
# # p.print_callees("ccc")
# import snakeviz
# #!snakeviz result.out

# from scipy.stats import norm
# norm.cdf(norm.ppf(1-2.866515719235352e-07))

# from statistics import NormalDist
# NormalDist(mu=0, sigma=1).inv_cdf(1-3e-7)
# """
# >>> NormalDist(mu=0, sigma=1).inv_cdf(1-1e-7)
# 5.199337582290662
# >>> NormalDist(mu=0, sigma=1).inv_cdf(1-2e-7)
# 5.068957749712318
# >>> NormalDist(mu=0, sigma=1).inv_cdf(1-3e-7)
# 4.991217139937878
# >>> 
# """

import matplotlib.pyplot as plt

ps = [-0.13377257,  0.51514362,  1.16726525,  1.68319534  ,2.34236005 , 2.33427755,
  3.17973069 , 3.360711 ,   3.93915874,  4.23414262 , 4.8926314  , 5.38715209,
  5.82080613 , 6.10696337 , 6.62039536,  6.92670253  ,7.21955963  ,7.7952545,
  8.09667744 , 8.62055556]
xs = [0, 10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190]
# set y ticks
plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
plt.plot(xs, ps, label='no pi')

plt.savefig('test.png')