# Gratin

##### Graphs on Trajectories for Inference

Gratin is an analysis tool for stochastic trajectories, based on graph neural networks.

### Model description

First, each trajectory is turned into a graph, in which positions are nodes, and edges are drawn between them following a pattern based on their time difference. Then, features computed from normalized positions are attached to nodes : cumulated distance covered since origin, distance to origin, maximal step size since origin... These graphs are then passed as input to a graph convolution module (graph neural network), which outputs, for each trajectory, a latent representation in a high-dimensional space. This fixed-size latent vector is then passed as input to task-specific modules, which can predict the anomalous exponent or the random walk type. Several output modules can be trained at the same time, using the same graph convolution module, by summing task-specific losses. The model can receive trajectories of any size as inputs. The high-dimensional latent representation of trajectories can be projected down to a 2D space for visualisation and provides interesting insights regarding the information extracted by the model (see details in the paper).

##### References : 

Hippolyte Verdier, Maxime Duval, François Laurent, Alhassan Cassé,  Christian Vestergaard, et al.. Learning physical properties of anomalous random walks using graph neural networks. 2021. : https://hal-pasteur.archives-ouvertes.fr/pasteur-03150190v1

