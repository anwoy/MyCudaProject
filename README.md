# CUDA Black Hole Simulator
<img src="assets/output.gif" alt="Alt text" width="500" height="300" />

A gravitational lensing simulator written in CUDA C++. This project renders the visual distortion of a Schwarzschild black hole.

<span style="font-size: 25px;">Watch the summary and final render on [youtube](https://youtu.be/BUqQJPbZieQ).</span>

## About the Project
* Simulates the bending of light using the geodesic equations.
* Solves $\frac{\partial^2 u}{\partial \phi^2}
=\frac{3GM}{C^2} u^2 - u$ using RK4 method, where $u = \frac{1}{r}$, and $r$ and $\phi$ are polar coordinates in 2D. This generates the path taken by a ray under the influence of the black hole.
* Leverages CUDA kernels for parallel processing of millions of light rays.
* Generates PNG sequences of black hole visuals.

## Build and Run
### Instructions
* You need to run a couple of python scripts before running the renderer. Install python packages using `requirements.txt`.
* Run `create_disk.py` script to create the Perlin noise map `data/disk3d.txt` for the accretion disk.
* Download the [starmap_2020_4k.exr](https://svs.gsfc.nasa.gov/vis/a000000/a004800/a004851/starmap_2020_4k.exr) file and place it in the `data` folder.
* Run `starmap_preprocess.py` to create the `data/starmap.png` file.
* Run the `.build.sh` script from the `build` folder, then run the resulting `a.out` executable. You need to have `starmap.png` and `disk3d.txt` files in the `data` folder (which were created in the previos steps) before running the executable.
* The rendered frames will be located in the `data` folder.

### Tested on
* GPU: NVIDIA Tesla T4
* OS: Ubuntu 24.04 LTS
* NVIDIA Driver: 570.211.01
* CUDA Toolkit: 12.6 Update 2 (nvcc 12.6.85)
* Compilers: NVCC 12.6.85 / GCC 13.3.0
* Build Tools: CMake / GNU Make

## Credits and Third-Party Assets
* Lodepng: Used for PNG image encoding. Developed by Lode Vandevenne. Licensed under the [Zlib License](https://opensource.org/licenses/Zlib).
* NASA Scientific Visualization Studio: The background starfield/Milky Way textures are courtesy of [NASA/Goddard Space Flight Center](https://svs.gsfc.nasa.gov/4851/).

## License
This project is licensed under the GNU General Public License v3 (GPL v3).
