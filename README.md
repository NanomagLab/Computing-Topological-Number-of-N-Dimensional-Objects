python 3.9\
matplotlib 3.8.4\
numpy 1.26.4\
tensorflow 2.10.1

run "example_2d.py" to compute topological numbers of two-dimensional objects\
run "example_3d.py" to compute topological numbers of three-dimensional objects

Download the "custom_modules" and "examples" directories; running "example_2d.py" will execute the code that computes the Euler number for 2D object data located in the "examples/2d" folder—specifically, it generates the output images "disk.png", "circle.png", and "square.png"—and running "example_3d.py" will execute the code that computes the Euler number for 3D object data found in the "examples/3d" folder, which includes "solid torus.npy", "ball.npy", and "Sphere" data; note that the file "3d_data_generation.py" in the "examples" directory is provided as an example for generating 3D data, while the "custom_modules" directory contains the functions necessary to run "example_2d.py" and "example_3d.py", and it also includes "arbitrary_dimension.py", which is a standalone script capable of computing the topological number for N-dimensional objects—this script comes with built-in functions that allow you to view results for structures such as a 4D ball, S³, the Klein bottle, and RP² without requiring any additional code.
