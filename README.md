# ism-rir-dataset
Code for the generation of a dataset of synthetic RIRs with image-source method 
## Getting started 
To install the required packages using conda environments open the terminal at the repo directory and run the following command
```
conda env create -f ism_dataset.yml
```
The main code is `ism_dataset.py`  

For each RIR, the room geometry is sampled from a set of 3 triplets of room dimensions (small, medium, and large).  

The absorption coefficents of the walls is sampled from a dictionary of meatrials. You can find more materials at the [pyroomacoustics repo](https://github.com/LCAV/pyroomacoustics/blob/df8af24c88a87b5d51c6123087cd3cd2d361286a/pyroomacoustics/data/materials.json#L4).
