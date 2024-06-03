import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from custom_module import gray_to_spin_2d, get_topological_number_2d

# load_examples
examples = [Image.open("examples/2d/" + path, 'r') for path in os.listdir("examples/2d")]
examples = [-np.array(x).astype(np.float32)[..., :1] * 2. / 255. + 1. for x in examples]

# convert the gray image to spin
spins = gray_to_spin_2d(examples)

# compute the skyrmion number
topological_numbers = get_topological_number_2d(spins)

fig, axes = plt.subplots(2, len(examples),  figsize=(len(examples) * 1.5,4))
for i in range(len(examples)):
    axes[0][i].imshow(examples[i])
    axes[0][i].axis('off')
    axes[0][i].set_title(os.listdir("examples")[i])
    axes[1][i].imshow(spin2rgb(outputs[i]))
    axes[1][i].axis('off')
    axes[1][i].set_title("n={:0.2f}".format(topological_numbers[i]), y=-0.2)
plt.tight_layout()
plt.show()