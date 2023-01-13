import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import sys
import pickle

def main(title):
    t_list = ['var001_FF','var001_TF']
    t_list = ['FF','TF','FT','TT']
    t_list = ['FF','TF']
    for title in t_list:
        try:
            with open('./data/PARAMETERS_'+title, 'rb') as f:
                params = parameters = pkl.load(f)
        except:
            continue
        n=parameters['n']
        result= np.load('./data/LFI_tst_'+title+str(n)+'.npy')
        assert len(result) == 2
        for i in range(2):
            pl=[]
            for l in range(len(params['m_list'])):
                m=params['m_list'][l]
                print(result.shape)
                arr=result[i,:,l]
                pl.append(1-np.mean(arr))
            plt.plot(params['m_list'],pl, label=title+',Z~'+str(i))
            print(pl)
            plt.legend()
        plt.xlabel('m')
        plt.ylabel('Probability of success')
        plt.title('LFI from TST on '+params['gen_fun'])
    plt.savefig('./data/LFI_tst_'+'all_'+title+str(n)+'.png')
    plt.show()
    print('----------printed plot at ----------')
    print('./data/LFI_tst_'+'all_'+title+str(n)+'.png')
        

