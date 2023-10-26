import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import sys
def load_diffusion_cifar_32():
    diffusion = np.load("../Diffusion/ddpm_generated_images2.npy").transpose(0,3,1,2)
    cifar10 = np.load('../data/cifar_data.npy')
    dataset_P = diffusion.reshape(diffusion.shape[0], -1)
    dataset_Q = cifar10.reshape(cifar10.shape[0], -1)
    return dataset_P, dataset_Q
def ims(img):
    return np.transpose(img.reshape(3, 32, 32), (1, 2, 0))/256

DP, DQ = load_diffusion_cifar_32()
DP1 = DP[:100, :]
DQ1 = DQ[:100, :]

plt.figure(figsize=(3, 3)) # specifying the overall grid size

for i in range(64):
    plt.subplot(8,8,i+1)    # the number of images in the grid is 5*5 (25)
    plt.imshow(ims(DP1[i]))
    plt.axis('off')
plt.savefig('Diffusion_visual.pdf')
plt.show()

plt.figure(figsize=(3, 3)) # specifying the overall grid size

for i in range(64):
    plt.subplot(8,8,i+1)    # the number of images in the grid is 5*5 (25)
    plt.imshow(ims(DQ1[i]))
    plt.axis('off')
plt.savefig('CIFAR_visual.pdf')
plt.show()

