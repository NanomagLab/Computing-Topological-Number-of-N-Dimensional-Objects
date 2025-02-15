This repository was developed using Python 3.9, matplotlib 3.8.4, numpy 1.26.4, and tensorflow 2.10.1. While these specific versions were used during development, the code is expected to be compatible with other versions as well.

To compute the topological numbers of two-dimensional objects, run "example_2d.py" after downloading the "custom_modules" and "examples" directories; this script processes 2D data in the "examples/2d" folder and generates output images like "disk.png", "circle.png", and "square.png". Similarly, running "example_3d.py" computes the topological numbers for three-dimensional objects using data found in the "examples/3d" folder (including "solid torus.npy", "ball.npy", and "Sphere" data); note that "3d_data_generation.py" in the "examples" directory is provided as an example for generating 3D data. The "custom_modules" directory contains all the functions necessary to run these examples, and it also includes "arbitrary_dimension.py", a standalone script capable of computing the topological number for N-dimensional objects—with built-in functions that allow you to view results for structures such as a 4D ball, S³, the Klein bottle, and RP² without requiring any additional code.

If you encounter out-of-memory issues or very long runtimes when using get_solid_angle_fine, you can lower its parameters such as size or ratio to reduce the up-sampled grid dimensions. Conversely, if you need maximum precision (and have sufficient memory), increasing ratio will produce more accurate—but more resource-intensive—results. By adjusting these variables, you can strike a balance between speed, memory usage, and numerical accuracy for N-dimensional objects.
