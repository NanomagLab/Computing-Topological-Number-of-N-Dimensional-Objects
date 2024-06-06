import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import tensorflow as tf
from three_dimension import gray_to_spin_3d, get_solid_angle_density_3d
def normalize(v, axis=-1):
    norm = np.linalg.norm(v, ord=2, axis=axis, keepdims=True)
    return norm, np.nan_to_num(v/norm)
def spin2rgb(X):
    def hsv2rgb(hsv):
        hsv = np.asarray(hsv)
        if hsv.shape[-1] != 3: raise ValueError("Last dimension of input array must be 3; " "shape {shp} was found.".format(shp=hsv.shape))
        in_shape = hsv.shape
        hsv = np.array(hsv, copy=False, dtype=np.promote_types(hsv.dtype, np.float32), ndmin=2)

        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        r, g, b = np.empty_like(h), np.empty_like(h), np.empty_like(h)

        i = (h * 6.0).astype(int)
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))

        idx = i % 6 == 0
        r[idx], g[idx], b[idx] = v[idx], t[idx], p[idx]

        idx = i == 1
        r[idx], g[idx], b[idx] = q[idx], v[idx], p[idx]

        idx = i == 2
        r[idx], g[idx], b[idx] = p[idx], v[idx], t[idx]

        idx = i == 3
        r[idx], g[idx], b[idx] = p[idx], q[idx], v[idx]

        idx = i == 4
        r[idx], g[idx], b[idx] = t[idx], p[idx], v[idx]

        idx = i == 5
        r[idx], g[idx], b[idx] = v[idx], p[idx], q[idx]

        idx = s == 0
        r[idx], g[idx], b[idx] = v[idx], v[idx], v[idx]

        rgb = np.stack([r, g, b], axis=-1)
        return rgb.reshape(in_shape)

    norm, normed_X = normalize(X)
    norm = np.clip(norm, 0, 1)
    X = norm * normed_X
    sxmap, symap, szmap = np.split(X, 3, axis=-1)
    szmap = 0.5 * szmap + (norm / 2.)
    H = np.clip(-np.arctan2(sxmap, -symap) / (2 * np.pi) + 0.5, 0, 1)
    S = np.clip(2 * np.minimum(szmap, norm - szmap), 0, norm)
    V = np.clip(2 * np.minimum(norm, szmap + norm / 2.) - 1.5 * norm + 0.5, 0.5 - 0.5 * norm, 0.5 + 0.5 * norm)
    img = np.concatenate((H, S, V), axis=-1)
    for i, map in enumerate(img): img[i] = hsv2rgb(map)
    return img
# load_examples
examples = [np.load("examples/3d/" + path) for path in os.listdir("examples/3d")]
examples = np.expand_dims(examples,axis=-1)

# convert the gray image to spin
spins = [gray_to_spin_3d(example) for example in examples]
# compute the skyrmion number
solid_angle_densities = [get_solid_angle_density_3d(spin) for spin in spins]
topological_numbers = [tf.reduce_sum(solid_angle_density, axis=(0, 1, 2)) / (2*np.pi**2) for solid_angle_density in solid_angle_densities]

fig, axes = plt.subplots(3, len(examples)*2,  figsize=(len(examples) * 4, 6))

axis=[" \nviews of xy plane", " \nviews of xz plane", " \nviews of yz plane"]

for i in range(len(examples)):
    axes[0][i*2].imshow(examples[i][:,:,100,-1])
    axes[0][i*2].axis('off')
    axes[0][i*2].set_title(os.listdir("examples/3d")[i]+axis[0])

    axes[0][2*i+1].imshow(spin2rgb(spins[i][:,:,100,:3]))
    axes[0][2*i+1].axis('off')
    axes[0][2*i+1].set_title("n={:0.2f}".format(topological_numbers[i]), y=-0.2)

    axes[1][i*2].imshow(examples[i][:,100,:,-1])
    axes[1][i*2].axis('off')
    axes[1][i*2].set_title(os.listdir("examples/3d")[i]+axis[1])

    axes[1][2*i+1].imshow(spin2rgb(spins[i][:,100,:,:3]))
    axes[1][2*i+1].axis('off')
    axes[1][2*i+1].set_title("n={:0.2f}".format(topological_numbers[i]), y=-0.2)

    axes[2][i*2].imshow(examples[i][100,:,:,-1])
    axes[2][i*2].axis('off')
    axes[2][i*2].set_title(os.listdir("examples/3d")[i]+axis[2])

    axes[2][2*i+1].imshow(spin2rgb(spins[i][100,:,:,:3]))
    axes[2][2*i+1].axis('off')
    axes[2][2*i+1].set_title("n={:0.2f}".format(topological_numbers[i]), y=-0.2)

plt.tight_layout()
plt.show()