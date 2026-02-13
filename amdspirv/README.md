# what

This is a set of changes to Intel's Triton fork to allow the AMD compiler backend to output SPIRV for use in Vulkan drivers. It is not fully complete
but is enough to demonstrate the concept. It is very much a "not-production-ready" prototype, and is not supported.

# setup

It is highly recommended to install these in a Docker container environment.
Fedora 43 (fedora:43) was used during development and testing

If you are using a Fedora 43 container, you can use `install_deps.sh` to install some python and other niceties.

Install [Intel Deep Learning Essentials](
https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html?packages=dl-essentials&dl-lin=offline&dl-essentials-os=linux)

Install the [nightly Triton/Torch wheels](https://github.com/intel/intel-xpu-backend-for-triton/actions/workflows/nightly-wheels.yml) from Intel's fork

Install the ["ze_headers"](https://github.com/oneapi-src/level-zero) which are needed to compile Triton with Intel support.
Clone the repo at the link, follow the Linux build instructions. The "install" phase should place the headers and static lib in `/usr/local/`. You may have to add them to your `*_PATH` variables.

# compiling

Use the script at `scripts/compile-triton.sh` to recompile

# running

Use `python amdspirv/tk.py` to run the compile of the test kernel to CUDA, AMD SPIRV, and Intel SPIRV. You can comment out any of these target in the file if you wish.
