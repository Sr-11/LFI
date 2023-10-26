from diffusers import DDPMPipeline, DDIMPipeline, PNDMPipeline
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
from IPython.display import clear_output
model_id = "google/ddpm-cifar10-32"
ddpm = DDPMPipeline.from_pretrained(model_id)  # you can replace DDPMPipeline with DDIMPipeline or PNDMPipeline for faster inference
# number of samples 
n = 10000
for i in range(n):
    clear_output(wait=True)
    print('{}-th sample'.format(i))
    ddpm_output = ddpm()
    image = np.asarray(ddpm_output.images[0], dtype=float)[np.newaxis, :, :, :]
    # save images array
    samples = np.load("ddpm_generated_images.npy")
    np.save("ddpm_generated_images.npy", np.concatenate((samples,image)))
samples = np.load("ddpm_generated_images.npy")
print(samples.shape)
plt.imshow(samples[0]/255)