# ECE1782GPU_Project

Repo for ECE1782 GPU Project

## Project organization

The source code for all layers and helper functions in the neural network is in the `src/` folder.

The source code for testing the CUDA implementations is in `testbench/`.

Compiled binary files are generated in `bin/`.

## Instructions

To compile all files of the `CNN` and test benches run `make all`.

To execute the all test bench files run `make test`.

Single test benches can be executed by running their binary files in `bin/`. For example: `./bin/convolutionMulti.out`.
