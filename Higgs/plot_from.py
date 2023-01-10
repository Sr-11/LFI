import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import sys

def plot(result, params):
    '''result: (2, trial, m)'''
    assert len(result) == 2
    for i in range(2):
        pl=[]
        for l in range(len(params['m_list'])):
            m=params['m_list'][l]
            print(result.shape)
            arr=result[i][:][l]
            pl.append(np.mean(arr))
        plt.plot(pl)
        print(pl)
        plt.legend('Type '+str(i))
    plt.xlabel('m')
    plt.ylabel('Probability of success')
    plt.title('LFI from TST on '+params['gen_fun'])
    plt.show()

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
