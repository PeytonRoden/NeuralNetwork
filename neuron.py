import numpy as np

class neuron:

    def __init__(self, bias: float, num_inputs):
        self.bias = bias
        self.weights = np.empty(num_inputs)
        self.output=0.0

    #get output of one specific neuron
    def get_output(self, inputs: list):
        self.output = 0.0
        for i in range(0, len(inputs)):
            self.output += (inputs[i]*self.weights[i])

        self.output += self.bias

        return self.sigmoid(self.output)

    def sigmoid(self, x):
        return(1/(1 + np.exp(-x)))

    def generateRandWeights(self):
        self.weights = (self.sigmoid(np.random.randn(len(self.weights))) * 2) - 1

    
    def set_weights(self, weights: list):
        if(len(weights) != len(self.weights)):
            print("error setting weights in weights array length")
            return

        self.weights = np.array(weights)

    def get_weights(self):
        return self.weights

    def set_bias(self, bias: float):
        self.bias = bias