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


    def loss(self, target):
        #MSE loss

        self.target= target

        if(len(target) != len(self.outputs)):
            print("target array doesn't match up to otuputdistro")
            return


        self.loss_value = self.neuralNetwork[-1].loss(target)

        return self.loss_value
    
    def adjust_weights(self, learning_rate):
        self.neuralNetwork[-1].update_weights(learning_rate)
    
    def adjust_weights(self, learning_rate):

        for i in range( len(self.neuralNetwork) - 1, 0 , -1):
            self.neuralNetwork[i].update_weights(learning_rate)
        

    def get_gradients(self):
        
        for i in range( len(self.neuralNetwork) - 1, 0 , -1):
            #(activation(w1*i1 + w2*i2 + bias) - target)^2

            #last layer has different activation function
            if(i == len(self.neuralNetwork) - 1):
                #loss function portion
                d_loss = -(2 * (self.neuralNetwork[i].outputs - self.target))

                #calculate derivateve for activation function
                #activation function is linear so it is 1
                d_activation = 1


                #derivative for weights
                #it will just be the inputs into that layer
                d_weights = self.neuralNetwork[i].inputs

                gradient = d_weights * d_loss

                self.neuralNetwork[i].set_gradient(gradient)

                #print(self.gradient)

            else:

                for j in range(0, len(self.neuralNetwork[i].get_layer())):

                    d_relu = 0
                    #calculate derivateve for activation function
                    #activation function is relu
                    if(self.neuralNetwork[i].get_outputs()[j] <= 0 ):
                        d_relu = 0
                    else:
                        d_relu = 1


                    #derivative for weights
                    #it will just be the inputs into that layer
                    d_weights = self.neuralNetwork[i].inputs

                    #get the gradient for each neuron
                    gradient = d_weights * d_relu

                    #set the gradient for the neuron
                    self.neuralNetwork[i].get_layer()[j].set_gradient(gradient)


    def get_loss(self):
        return self.loss_value






def main():

    inputs = [0.33, 0.11]

    neural = neuralNet(2,[10,10],1)
    neural.generate_weights()
    print(neural.get_output(inputs))
    print(neural.loss([0.5]))

    for i in range(0,1000):
        neural.get_output(inputs)
        print("LOSS",neural.loss([0.5]))
        neural.get_gradients()
        neural.adjust_weights(1 * (1/np.sqrt(i+1)) * np.sqrt(np.log(neural.get_loss() + 1)) )

    print(neural.get_output(inputs))
    print(neural.loss([0.5]))


if __name__ == "__main__":
    main()



