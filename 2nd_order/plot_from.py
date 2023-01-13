import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import sys
import pickle
def plot(result, params, title=''):
    '''result: (2, trial, m)'''
    n = params['n']
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
    plt.savefig('./data/LFI_tst_'+title+str(n)+'.png')
    plt.show()
    print('----------printed plot at ----------')
    print('./data/LFI_tst_'+title+str(n)+'.png')
    return 



if __name__ == '__main__':
    try:
        title=sys.argv[1]
    except:
        print("Warning: No title given, using default")
        print('Please use specified titles for saving data')
        title='untitled_run'
    with open('./data/PARAMETERS_'+title, 'rb') as f:
        parameters = pkl.load(f)
    print(parameters)
    n=parameters['n']
    result= np.load('./data/LFI_tst_'+title+str(n)+'.npy')
    plot(result, parameters)
