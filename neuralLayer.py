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
    def calc_outputs(self):
        for i in range(0, len(self.layer)):
            self.outputs[i] = self.activation(np.dot(self.layer[i].weights, self.inputs) + self.layer[i].bias)
            #self.outputs[i] = self.relu_forward(np.dot(self.layer[i].weights, self.inputs) + self.layer[i].bias)
            #self.outputs[i] = self.relu(np.dot(self.layer[i].weights, self.inputs) + self.layer[i].bias)
        
        return self.outputs


    def feed_forward(self, inputs:list):
        self.inputs = np.array(inputs)

        if(len(inputs) != self.num_inputs):
            print("problem with inputs length at layer level")

        return self.calc_outputs()

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

    def activation(self, inputs):
        return np.maximum(0, inputs)

    def backward(self, x):
        pass

    def get_layer(self):
        return self.layer

    def get_outputs(self):
        return self.outputs


    def update_weights(self, learning_rate):

        for i in range(0, self.layer_size):
            #last output is only one neuron so you can use layer[0] to update weights
            self.layer[i].weights = self.layer[i].weights + (learning_rate * self.layer[i].get_gradient())




class outputLayer(neuralLayer):
    def __init__(self, num_inputs, layer_size):
        super().__init__(num_inputs,layer_size)

    #override activaiton funciton with linear activation funciton
    def activation(self, input):
        self.output = input
        return self.output


    def backward(self):

        #(activation(w1*i1 + w2*i2 + bias) - target)^2
        
        #loss function portion
        d_loss = -(2 * (self.output - self.target))


        #calculate derivateve for activation function
        #activation function is linear so it is 1
        d_activation = 1


        #derivative for weights
        #it will just be the input


        return 0


    def set_gradient(self, gradient : list):
        self.layer[0].set_gradient(gradient)



    def loss(self, target):

        #store target to do back prop
        self.target = target


        self.loss_value = (self.output - target) ** 2
        return self.loss_value


    def update_weights(self, learning_rate):

        #last output is only one neuron so you can use layer[0] to update weights
        self.layer[0].weights = self.layer[0].weights + (learning_rate * self.layer[0].get_gradient())

        

