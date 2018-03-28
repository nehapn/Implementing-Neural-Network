# Implementing-Neural-Network
This project aims at implementing a generic neural network in C++. A deep network consists of many layers with each layer consisting of neurons.
The neural network implementation is in the main.cpp file. Any architecture can be set for the network by changing the topmost line in the dataset
text files. For example if the topmost line is in the manner "topology: 13 3 3 3" it means that the neural network is 3 layers deep with input layer
having 3 neurons, layer next to it having 3 neurons, the layer next 3 and the output layer having 3 neurons. The input layers' dimension is 
determined by the dimension of the input data, whereas the number of neurons in the output layer is the number of classes in the dataset.
The number of hidden layers as well as the number of neurons in each of them can be any arbitrary number. I have also tried to test the 
effect of the type of activation function on the performance of my network. In order to change the activation function use the desired
activation function's class object in the main.cpp file. Apart from this one can also set the number of iterations and the batch size for training. 1 iteration refers to the training and update of the network parameters for 1 batch. 

To get results run main.cpp file and compile with gcc c++11 or above compiler. Different training parameters can be changed in the following manner:
1) Training dataset - In main.cpp file "main" function line 299
2) Number of iterations - In main.cpp file "main" function line 307
3) Batch size - In main.cpp file "main" function line 308
4) Activation function - In main.cpp file "main" function line 327 and line 342 change the second parameter of the feed forward and back                            prop functions to tan_h, leaky_relu or sigmoid.
