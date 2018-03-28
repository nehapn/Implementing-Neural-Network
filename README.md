# Implementing-Neural-Network
This project aims at implementing a generic neural network in C++. A deep network consists of many layers with each layer consisting of neurons.
The neural network implementation is in the main.cpp file. Any architecture can be set for the network by changing the topmost line in the dataset
text files. For example if the topmost line is in the manner "topology: 13 3 3 3" it means that the neural network is 3 layers deep with input layer
having 3 neurons, layer next to it having 3 neurons, the layer next 3 and the output layer having 3 neurons. The input layers' dimension is 
determined by the dimension of the input data, whereas the number of neurons in the output layer is the number of classes in the dataset.
The number of hidden layers as well as the number of neurons in each of them can be any arbitrary number. I have also tried to test the 
effect of the type of activation function on the performance of my network. In order to change the activation function use the desired
activation function's class object in the main.cpp file. 
