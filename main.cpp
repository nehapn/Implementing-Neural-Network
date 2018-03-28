#include <vector>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>
#include <random>
#include <tuple>
#include "sigmoid.h"
#include "tanh.h"
#include "leaky_relu.h"
#include "training_data.cpp"

using namespace std;
using sample = vector<tuple<vector<double>, vector<double>>>;

struct Connection
{
	double weight;
	double deltaWeight;
};

class Neuron;

typedef vector<Neuron> Layer;

// ****************** class Neuron ******************

class Neuron
{
public:
	Neuron(unsigned numOutputs, unsigned myIndex);
	void setOutputVal(double val) { m_outputVal = val; }
	double getOutputVal(void) const { return m_outputVal; }
	void feedForward(const Layer &prevLayer, func &f);
	void calcOutputGradients(double targetVals, func &f);
	void calcHiddenGradients(const Layer &nextLayer, func &f);
	void updateInputWeights(Layer &prevLayer);
private:
	static double eta; // [0.0...1.0] overall net training rate
	static double alpha; // [0.0...n] multiplier of last weight change [momentum]
	static double transferFunction(double x, func &f);
	static double transferFunctionDerivative(double x, func &f);
	// randomWeight: 0 - 1
	static double randomWeight(void) { return rand() / double(RAND_MAX); }
	double sumDOW(const Layer &nextLayer) const;
	double m_outputVal;
	vector<Connection> m_outputWeights;
	unsigned m_myIndex;
	double m_gradient;
};

double Neuron::eta = 0.15; // overall net learning rate
double Neuron::alpha = 0.5; // momentum, multiplier of last deltaWeight, [0.0..n]


void Neuron::updateInputWeights(Layer &prevLayer)
{
	// The weights to be updated are in the Connection container
	// in the nuerons in the preceding layer

	for(unsigned n = 0; n < prevLayer.size(); ++n)
	{
		Neuron &neuron = prevLayer[n];
		double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;

		double newDeltaWeight =
				// Individual input, magnified by the gradient and train rate:
				eta
				* neuron.getOutputVal()
				* m_gradient
				// Also add momentum = a fraction of the previous delta weight
				+ alpha
				* oldDeltaWeight;
		neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
		neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
	}
}
double Neuron::sumDOW(const Layer &nextLayer) const
{
	double sum = 0.0;

	// Sum our contributions of the errors at the nodes we feed

	for (unsigned n = 0; n < nextLayer.size() - 1; ++n)
	{
		sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
	}

	return sum;
}

void Neuron::calcHiddenGradients(const Layer &nextLayer, func &f)
{
	double dow = sumDOW(nextLayer);
	m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal, f);
}
void Neuron::calcOutputGradients(double targetVals, func &f)
{
	double delta = targetVals - m_outputVal;
	m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal, f);
}

double Neuron::transferFunction(double x, func &f)
{
	// tanh - output range [-1.0..1.0]
	return f(x);
}

double Neuron::transferFunctionDerivative(double x, func &f)
{
	// tanh derivative
	return f.grad(x);
}

void Neuron::feedForward(const Layer &prevLayer, func &f)
{
	double sum = 0.0;

	// Sum the previous layer's outputs (which are our inputs)
    // Include the bias node from the previous layer.

	for(unsigned n = 0 ; n < prevLayer.size(); ++n)
	{
		sum += prevLayer[n].getOutputVal() *
				 prevLayer[n].m_outputWeights[m_myIndex].weight;
	}

	m_outputVal = Neuron::transferFunction(sum, f);
}

Neuron::Neuron(unsigned numOutputs, unsigned myIndex)
{
	for(unsigned c = 0; c < numOutputs; ++c){
		m_outputWeights.push_back(Connection());
		m_outputWeights.back().weight = randomWeight();
	}

	m_myIndex = myIndex;
}
// ****************** class Net ******************
class Net
{
public:
	Net(const vector<unsigned> &topology);
	void feedForward(const vector<double> &inputVals, func &f);
	void backProp(const vector<double> &targetVals, func &f);
	void getResults(vector<double> &resultVals) const;
	double getRecentAverageError(void) const { return m_recentAverageError; }

private:
	vector<Layer> m_layers; //m_layers[layerNum][neuronNum]
	double m_error;
	double m_recentAverageError;
	static double m_recentAverageSmoothingFactor;
};

double Net::m_recentAverageSmoothingFactor = 100.0; // Number of training samples to average over

void Net::getResults(vector<double> &resultVals) const
{
	resultVals.clear();

	for(unsigned n = 0; n < m_layers.back().size() - 1; ++n)
	{
		resultVals.push_back(m_layers.back()[n].getOutputVal());
	}
}

void Net::backProp(const std::vector<double> &targetVals, func &f)
{
	// Calculate overall net error (RMS of output neuron errors)
	Layer &outputLayer = m_layers.back();
	m_error = 0.0;

	for(unsigned n = 0; n < outputLayer.size() - 1; ++n)
	{
		double delta = targetVals[n] - outputLayer[n].getOutputVal();
		m_error += delta *delta;
	}
	m_error /= outputLayer.size() - 1; // get average error squared
	m_error = sqrt(m_error); // RMS
	// Implement a recent average measurement:
	m_recentAverageError =
			(m_recentAverageError * m_recentAverageSmoothingFactor + m_error)
			/ (m_recentAverageSmoothingFactor + 1.0);
	// Calculate output layer gradients
	for(unsigned n = 0; n < outputLayer.size() - 1; ++n)
	{
		outputLayer[n].calcOutputGradients(targetVals[n], f);
	}
	// Calculate gradients on hidden layers
	for(unsigned layerNum = m_layers.size() - 2; layerNum > 0; --layerNum)
	{
		Layer &hiddenLayer = m_layers[layerNum];
		Layer &nextLayer = m_layers[layerNum + 1];

		for(unsigned n = 0; n < hiddenLayer.size(); ++n)
		{
			hiddenLayer[n].calcHiddenGradients(nextLayer, f);
		}
	}
	// For all layers from outputs to first hidden layer, update connection weights
	for(unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum)
	{
		Layer &layer = m_layers[layerNum];
		Layer &prevLayer = m_layers[layerNum - 1];

		for(unsigned n = 0; n < layer.size() - 1; ++n)
		{
			layer[n].updateInputWeights(prevLayer);
		}
	}
}

void Net::feedForward(const vector<double> &inputVals, func &f)
{
	// Check the num of inputVals euqal to neuron num expect bias
	assert(inputVals.size() == m_layers[0].size() - 1);

	// Assign {latch} the input values into the input neurons
	for(unsigned i = 0; i < inputVals.size(); ++i){
		m_layers[0][i].setOutputVal(inputVals[i]);
	}

	// Forward propagate
	for(unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum){
		Layer &prevLayer = m_layers[layerNum - 1];
		for(unsigned n = 0; n < m_layers[layerNum].size() - 1; ++n){
			m_layers[layerNum][n].feedForward(prevLayer, f);
		}
	}
}
Net::Net(const vector<unsigned> &topology)
{
	unsigned numLayers = topology.size();
	for(unsigned layerNum = 0; layerNum < numLayers; ++layerNum){
		m_layers.push_back(Layer());
		// numOutputs of layer[i] is the numInputs of layer[i+1]
		// numOutputs of last layer is 0
		unsigned numOutputs = layerNum == topology.size() - 1 ? 0 :topology[layerNum + 1];

		// We have made a new Layer, now fill it ith neurons, and
		// add a bias neuron to the layer:
		for(unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum){
			m_layers.back().push_back(Neuron(numOutputs, neuronNum));
			//cout << "Made a Neuron!" << endl;
		}

		// Force the bias node's output value to 1.0. It's the last neuron created above
		m_layers.back().back().setOutputVal(1.0);
	}
}

void showVectorVals(string label, vector<double> &v)
{
	cout << label << " ";
	for(unsigned i = 0; i < v.size(); ++i)
	{
		cout << v[i] << " ";
	}
	cout << endl;
}
void softmax(vector<double> &in)
{
    double max_el = *max_element(in.begin(), in.end());
    double sum = 0;
    for(auto it = in.begin(); it!=in.end(); it++)
    {
        *it = exp(*it - max_el);
        sum = sum + *it;
    }
    for(auto it = in.begin(); it!=in.end(); it++)
    {
        *it = *it/sum;
    }
}

sample get_batch(sample inputs, int batch_size)
{
    sample b;
    random_device rd;
    mt19937 g(rd());
    shuffle(inputs.begin(), inputs.end(), g);
    for(int i = 0; i <= batch_size; i++)
    {
        b.push_back(inputs[i]);
    }
    return b;
}
int main()
{
    Sigmoid sigmoid;
    Tanh tan_h;
    leakyRelu leaky_relu;
	TrainingData trainData("wine.txt");
	vector<unsigned> topology;
	trainData.getTopology(topology);
	Net myNet(topology);

	vector<double> inputVals, targetVals, resultVals;
	sample inputs, batch;
	int trainingPass = 0;
	int num_iter = 2;
	int batch_size = 10;
	inputs.clear();
	while(!trainData.isEof())
    {
        trainData.getNextInputs(inputVals);
        trainData.getTargetOutputs(targetVals);
        //inputs.clear();
        inputs.push_back(make_tuple(inputVals, targetVals));
    }

    for(int i = 0; i<num_iter; i++)
    {
    cout<<"\n Iteration "<<i+1<<endl;
    batch = get_batch(inputs, batch_size);

	for(auto it = batch.begin(); it != batch.end(); it++)
	{
	    // Forward propagation
		showVectorVals("Inputs: ", get<0>(*it));
		myNet.feedForward(get<0>(*it), sigmoid);

		// Collect the net's actual results:
		myNet.getResults(resultVals);
		showVectorVals("Scores: ", resultVals);

		// Probabilities for each class
		softmax(resultVals);
		showVectorVals("Probs: ", resultVals);


		// Train the net what the outputs should have been:
		showVectorVals("Targets:", get<1>(*it));
		assert(get<1>(*it).size() == topology.back());

        myNet.backProp(get<1>(*it), sigmoid);

		// Result Class
		int res_class = distance(resultVals.begin(), max_element(resultVals.begin(), resultVals.end()));
		cout<<"\n Result Class: "<< res_class;

        // True Class
        int true_class = distance(get<1>(*it).begin(), max_element(get<1>(*it).begin(), get<1>(*it).end()));
		cout<<"\n True Class: "<< true_class;

		cout << "\n Net recent average error: "
		     << myNet.getRecentAverageError() << endl;
	}
    }

	cout << endl << "Done" << endl;
	cout << "Number of iterations = "<<num_iter<<" Batch_size = "<<batch_size<< endl;
	cout << "Topology ";
	for (auto i = topology.begin(); i != topology.end(); ++i)
    {
      std::cout << *i << ' ';
    }
}

