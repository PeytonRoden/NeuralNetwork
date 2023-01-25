import neuron
import numpy as np

class neuralLayer:


    def __init__(self, num_inputs, layer_size):
        self.layer = []
        self.layer_size = layer_size
        self.weights = []
        self.input = np.empty(num_inputs)
        self.outputs = np.empty(layer_size)
        self.weights = []
        self.num_inputs = num_inputs


        for i in range(0,layer_size):
            newNeuron = neuron.neuron(0.0, num_inputs)

            self.layer.append(newNeuron)


    #get outputs
    def get_outputs(self):
        for i in range(0, len(self.layer)):
            self.outputs[i] = self.activation(np.dot(self.layer[i].weights, self.inputs) + self.layer[i].bias)
            #self.outputs[i] = self.relu_forward(np.dot(self.layer[i].weights, self.inputs) + self.layer[i].bias)
            #self.outputs[i] = self.relu(np.dot(self.layer[i].weights, self.inputs) + self.layer[i].bias)
        
        return self.outputs


    def feed_forward(self, inputs:list):
        self.inputs = np.array(inputs)

        if(len(inputs) != self.num_inputs):
            print("problem with inputs length at layer level")

        return self.get_outputs()

    #get input values, used for first layer
    def get_inputs(self):
        return self.inputs

    def get_layer(self):
        return self.layer
    
    def get_weights(self):

        self.weights = []
        
        for i in range(0, self.layer_size):
            #print(i)
            self.weights.append(self.layer[i].weights)

        return self.weights
            
    #pass in list of lists to be weights for our neurons
    def set_weights(self, weights: list):

        if(len(weights) != self.num_inputs):
            print("error, number of weights arrays incorrect")
            return
        if(len(weights[0]) != len(self.layer[0].get_weights())):
            print("errro, number of weights incorrect")
            return

        for i in range(0, len(self.layer)):
            self.layer[i].set_weights( weights[i])

    def set_biases(self):
        pass

    #sigmoid
    def sigmoid(self, x):
        return(1/(1 + np.exp(-x)))

    def relu_forward(self, inputs):
        return np.maximum(0, inputs)

    def activation(self, inputs):
        return np.maximum(0, inputs)

    def relu_backward(self, x):
        pass




class outputLayer(neuralLayer):
    def __init__(self, num_inputs, layer_size):
        super().__init__(num_inputs,layer_size)

    #override activaiton funciton with softmax for last layer
    def activation(self, x):
        # we don't want to havw np.exp(1000) becasue that would overflow
        # we cap the max value to be 1, by inputing -inf - 0
        return np.exp(x- np.max(self.inputs)) 


    def derivative(self, x):
        return 0

    def get_distribution(self):

        if(np.count_nonzero(self.outputs) == 0):
            norm_values = np.zeros(len(self.outputs))
            return norm_values

        norm_base = sum(self.outputs)
        norm_values = self.outputs/norm_base
        return norm_values
