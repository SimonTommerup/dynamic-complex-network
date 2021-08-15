# Learning Dynamic Embeddings for Complex Networks
This repository contains the code for creating the MSc thesis Learning Dynamic Embeddings for Complex Networks.

The models developed are available in the model folder.

- smallnet_sqdist uses a common bias term and the squared euclidean distance
- smallnet_eucliddist uses a common bias term and the euclidean distance
- smallnet_node_specific_bias uses node specific bias terms and the euclidean distance

The algorithms for simulating data are in the simulation folder.

- nhpp.py simulates data from a model using the euclidean distance
- nhpp_mod.py simulates data from a model using the squared euclidean distance.

The used data sets are in the data folder.

The results generating scripts
are available in the results folder, and with a little changing of paths and rearranging of files it should hopefully be possible to run these
scripts without too many complications.
