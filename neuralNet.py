import neuralLayer
import neuron
import numpy as np

class neuralNet:

    #pass in input size, array of hidden layer sizes, output size
    def __init__(self, num_inputs:int, hidden_layers: list, num_outputs: int):

        self.num_inputs = num_inputs
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs

        #hold the neural layer objects in the list
        self.neuralNetwork = []
        self.num_hidden_layers = len(hidden_layers)

        #hold weights in this list to, for outputting purposes
        #calculations are done with weights in neuron objects
        self.weights = []

        input_layer = neuralLayer.neuralLayer(num_inputs, num_inputs)
        self.neuralNetwork.append(input_layer)

        for i in range(0, len(hidden_layers)):

            #for first hidden layer number of inputs is number of inputs for total netowrk
            if(i == 0):
                hidden_layer = neuralLayer.neuralLayer(num_inputs, hidden_layers[i])
                self.neuralNetwork.append(hidden_layer)
            else:
                hidden_layer = neuralLayer.neuralLayer(hidden_layers[i-1], hidden_layers[i])
                self.neuralNetwork.append(hidden_layer)

        output_layer = neuralLayer.outputLayer(hidden_layers[len(hidden_layers)-1], num_outputs)
        self.neuralNetwork.append( output_layer)




    def generate_weights(self):
        #generate random weights for all neurons
        #no need to generate weigths for input layer
        #generate weights for hidden layers and output layer
        #loop thorugh all layers and hiddne layers
        for i in range(1,len(self.neuralNetwork)):
            #each layer has neurons, each neuron has an array of weights associated with it
            #loop through each neurron in each layer
            for neuron in self.neuralNetwork[i].get_layer():
                #each nueron has function to generate random weights for itself
                neuron.generateRandWeights()



    def get_weights(self):
        self.weights = []
        for i in range(1,len(self.neuralNetwork)):
            self.weights.append( self.neuralNetwork[i].get_weights())
        return self.weights



    def get_output(self, inputs: list):

        if(len(inputs) != self.num_inputs):
            print("must match number of inputs to be same as number of neurons in first layer")
            return

        #feed the inputs to the neural network
        for i in range(1, len(self.neuralNetwork)):
            inputs = self.neuralNetwork[i].feed_forward(inputs)


        self.outputs = inputs
        return self.outputs

    def get_output_distribution(self):
        self.output_distribution = self.neuralNetwork[-1].get_distribution()
        return self.output_distribution

    def loss(self, target):
        #categorical cross-entropy loss

        #target will look like [0,1,0]
        #outputs will look like [0.01,0.98,0.01]

        if(len(target) != len(self.output_distribution)):
            print("target array doesn't match up to otuputdistro")
            return

        #clip distribution so that we don't do log(0)
        clipped_distribution = np.clip(self.output_distribution, 1e-7, 1 - 1e-7)

        #do multiplication and get -log(0.98) as answer
        self.loss_value = -np.log(np.dot(target, clipped_distribution))

        return self.loss_value





def main():
    neural = neuralNet(3,[20,20],2)
    neural.generate_weights()
    print(neural.get_output(list(np.random.randn(3))))
    print(neural.get_output_distribution())
    print(neural.loss([1,0]))


if __name__ == "__main__":
    main()



