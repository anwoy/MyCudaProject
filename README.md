# CUDA Black Hole Simulator

A high-performance gravitational lensing simulator written in CUDA C++. This project renders the visual distortion of a Schwarzschild black hole.

## About the Project
* **General Relativistic Ray Tracing:** Simulates the bending of light using the geodesic equations.
* **GPU Accelerated:** Leverages CUDA kernels for parallel processing of millions of light rays.
* **High-Fidelity Output:** Generates PNG sequences of black hole visuals.

## Build and Run
### Prerequisites
* NVIDIA GPU
* CUDA Toolkit
* C++ Compiler

### Instructions
Simply run the `.build.sh` from the `build` folder, then run the resulting `a.out` executable, tested on Ubuntu 20.04
You need to have `starmap.png` and `disk3d.txt` files in the `data` folder before running the executable. Run `create_disk.py` script to create the Perlin noise map `disk3d.txt` for the accretion disk.

## Credits and Third-Party Assets
* **Lodepng**: Used for PNG image encoding. Developed by Lode Vandevenne. Licensed under the [Zlib License](https://opensource.org/licenses/Zlib).
* **NASA Scientific Visualization Studio**: The background starfield/Milky Way textures are courtesy of [NASA/Goddard Space Flight Center](https://svs.gsfc.nasa.gov/4851/).

## License
This project is licensed under the **GNU General Public License v3 (GPL v3)**. 

**Contact Info:** Please direct any queries to **anwoy.rkl@gmail.com**.
