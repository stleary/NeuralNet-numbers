import scipy.special
import numpy
import scipy.misc
import scipy.ndimage
import tensorflow as tf
from tensorflow import keras
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#import imageio
#from matplotlib import pyplot



class NeuralNet:
    """
    This is a simple neural network that trains itself to classify data via supervised learning, and can then be
    queried with as yet unclassified data.

    The network consists of 3 layers: input, hidden, and output. The number of nodes for each layer
    is configurable.

    The network utilizes feedforward via weighting of signals between node layers, and backpropagation that adjusts
    the weights in order to correct errors detected while  learning. The input-to-hidden weights and the
    hidden-to-output weights are where the network stores what its learnings.

    Input nodes simply take the user input and deliver it to the hidden layer nodes as a signal.

    Hidden layer nodes receive signals from all of the nodes in the input layer, after the signals have been
    conditioned by weights.  The initial weight values are randomly assigned from a normal distribution with
    std dev (1 / sqrt(3)), or (-0.5, +0.5).
    Each hidden node produces output by applying a sigmoid function to the input and emitting a resulting signal to
    all of the nodes of the output layer. The sigmoid function is used because it produces a nonlinear smooth output
    between 0 and 1. A second reason for using the sigmoid is that it is easily differentiable, which is
    used to fine tune the weights during training. A third reason is that it is a nonlinear squashing function.

    Output layer nodes receive signals from all of the nodes in the hidden layer, after the signals have been
    conditioned by weights. Each output node applies the sigmoid function and emits a signal which becomes the output.

    The network operates by first training it with a large amount of labeled data, where each input is accompanied
    by the expected output. After training is completed, query() and back_query() operations may be performed.
    query() allows you to provide the network with previously unseen data in order to get a response.
    back_query() allows you to ask the network how it came up with some given output.

    INITIALIZATION PARAMETERS:
    Input nodes should be equal to the number of inputs.
        Ex: For 28x28 px images, use 784 input nodes
    Output nodes should be equal to the number of expected results.
        Ex: When identifying single digits, use 10 output nodes
    Hidden nodes are how the network allows can be tuned for performance. Start with a number between input and output node counts.
        Ex: Given 784 inputs and 10 outputs, start with 100 hidden nodes
    Learning rate can be tuned for performance. Start with a number less than one.
        Ex: 0.3

    TRAINING OPERATION
    Training should be performed before query. Call the train() method repeatedly, each time passing in
    a sample input (a list of input values, one for each input node) and expected output (a list of output
    values, one for each output node).
    For example, the MNIST numeric database contains 60,000 images of numbers.
    Input and output values should be normalized to (0,1), excluding 0.0 and 1.0.

    Training consists of processing the input data to produce an output, just as in a query(). The delta
    between the actual and expected output for each output node is back-propagated to the hidden node
    output weights via gradient descent. That is, think of the error as being a function of all of the
    different hidden layer signals and weights. Gradient descent adjusts the weight values by using partial
    derivatives to follow the slope (the gradient) downwards (the descent) to minimize the error delta.
    Thus the weights are adjusted according to how much they contributed to the error. Back propagation
    is also applied to the weights between the input and hidden layers, using the same principle but
    a slightly different formula.


    QUERY OPERATION
    Inputs are combined with per-node weights to all of the hidden nodes. Hidden nodes apply a sigmoid
    function to smoooth the output signal, which is then combined with per-node weights and sent to the
    output nodes. Output nodes again apply the sigmoid function, which is returned to the caller.


    During training
    """

    def __init__(self, input_nodes: int, hidden_nodes: int, output_nodes: int, learning_rate: float):
        """
        Initializes the net based on the requested number of nodes and the learning rate
        :param input_nodes: number of input nodes
        :param hidden_nodes: number of hidden nodes
        :param output_nodes: number of output nodes
        :param learning_rate: learning rate between (0,1) exclusive
        """

        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate

        self.activation_function = lambda x: scipy.special.expit(x)
        self.inverse_activation_function = lambda x: scipy.special.logit(x)

        # Creates the weight matrices. Range of weights is a normal distribution centered at 0,
        # with std dev +-1 / square root of number of incoming nodes. For 3 nodes, that
        # will be about +- .6

        # when multiplying wih * raw input signal = weighted input signal to hidden nodes
        # row i=node weights for hidden node i
        # col j=input node selector
        self.weights_input_hidden = numpy.random.normal(0.0, pow(self.input_nodes, -0.5),
                                                       (self.hidden_nodes, self.input_nodes))
        # when multiplying who * raw hidden signal = weighted input signal to output nodes
        # row i=node weights for output node i
        # col j=output node selector
        self.weights_hidden_output = numpy.random.normal(0.0, pow(self.hidden_nodes, -0.5),
                                                        (self.output_nodes, self.hidden_nodes))

    def train(self, inputs, targets):
        """
        Train the network with input and target values
        :param inputs: the inputs to be evaluated
        :param targets: the expected outputs
        """
        inputs_transpose = numpy.array(inputs, ndmin=2).T
        targets_transpose = numpy.array(targets, ndmin=2).T

        hidden_input = numpy.dot(self.weights_input_hidden, inputs_transpose)
        hidden_output = self.activation_function(hidden_input)

        final_input = numpy.dot(self.weights_hidden_output, hidden_output)
        final_output = self.activation_function(final_input)

        # refine the weights between hidden and final
        output_errors = targets_transpose - final_output
        # refine the weights between input and hidden
        hidden_errors = numpy.dot(self.weights_hidden_output.T, output_errors)

        # perform the hidden-final refinement
        self.weights_hidden_output = \
            self.weights_hidden_output + \
            self.learning_rate * \
            numpy.dot((output_errors * final_output * (1.0 - final_output)), numpy.transpose(hidden_output))

        # perform the input-hidden refinement
        self.weights_input_hidden = \
            self.weights_input_hidden + \
            self.learning_rate * \
            numpy.dot((hidden_errors * hidden_output * (1.0 - hidden_output)), numpy.transpose(inputs_transpose))


    def query(self, inputs):
        """
        query the trained network with some inputs
        :param inputs: array of input values
        :return: list of outputs
        """

        hidden_input = numpy.dot(self.weights_input_hidden, inputs)
        #print('hidden input: ' + str(hidden_input))
        hidden_output = self.activation_function(hidden_input)
        #print('hidden output: ' + str(hidden_output))

        final_input = numpy.dot(self.weights_hidden_output, hidden_output)
        #print('final input: ' + str(final_input))
        final_output = self.activation_function(final_input)
        #print('final output: ' + str(final_output))

        return final_output

    # backquery the neural network
    # we'll use the same termnimology to each item,
    # eg target are the values at the right of the network, albeit used as input
    # eg hidden_output is the signal to the right of the middle nodes
    def backquery(self, targets_list):
        # transpose the targets list to a vertical array
        final_outputs = numpy.array(targets_list, ndmin=2).T

        # calculate the signal into the final output layer
        final_inputs = self.inverse_activation_function(final_outputs)

        # calculate the signal out of the hidden layer
        hidden_outputs = numpy.dot(self.weights_hidden_output.T, final_inputs)
        # scale them back to 0.01 to .99
        hidden_outputs -= numpy.min(hidden_outputs)
        hidden_outputs /= numpy.max(hidden_outputs)
        hidden_outputs *= 0.98
        hidden_outputs += 0.01

        # calculate the signal into the hidden layer
        hidden_inputs = self.inverse_activation_function(hidden_outputs)

        # calculate the signal out of the input layer
        inputs = numpy.dot(self.weights_input_hidden.T, hidden_inputs)
        # scale them back to 0.01 to .99
        inputs -= numpy.min(inputs)
        inputs /= numpy.max(inputs)
        inputs *= 0.98
        inputs += 0.01

        return inputs



def main():

    # 1 node for every cell of the 28x28 image
    input_nodes = 784
    # arbitrary, should be a less than input_nodes to summarize results,
    # but enough to find patterns
    hidden_nodes = 100
    # 1 node for every expected matching number (0-9)
    output_nodes = 10
    # small moves, don't jump around too much when correcting errors
    #learning_rate = 0.2 # suitable for mnist numbers
    learning_rate = 0.05


    neural_net = NeuralNet(input_nodes=input_nodes,
                    output_nodes=output_nodes,
                    hidden_nodes=hidden_nodes,
                    learning_rate=learning_rate)

    #training_data_file = open("mnist_train.csv", "r")
    training_data_file = open("fashion-mnist_train.csv", "r")
    # only fashion mnist has column headers, read and discard
    columns = training_data_file.readline()
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    # train the network!
    print("total training lines: " + str(len(training_data_list)))

    # run the training this many times
    for r in range(1):
        print('starting epoch ' + str(r))
        for line in training_data_list:
            # the mnist dataset is a csv file with integer greyscale color values of 0-255.
            # each line is a 28x28 image
            # Normalize this to .001-1.0
            # the first digit is the actual number, which should be skipped
            all_values = line.split(',')
            scaled_input = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            output_nodes = 10
            targets = numpy.zeros(output_nodes) + 0.01
            targets[int(all_values[0])] = 0.99
            neural_net.train(scaled_input, targets)

            """
            Uncomment this code if you want to improve your test data via small rotations
            """
            # scaled_input_plus_10 = \
            #     scipy.ndimage.interpolation.rotate(scaled_input.reshape(28,28), 10, cval=0.01, order=1, reshape=False)
            # scaled_input_minus_10 = \
            #     scipy.ndimage.interpolation.rotate(scaled_input.reshape(28,28), -10, cval=0.01, order=1, reshape=False)
            # neural_net.train(scaled_input_plus_10.reshape(784), targets)
            # neural_net.train(scaled_input_minus_10.reshape(784), targets)

    # test the network!
    #test_data_file = open("mnist_test.csv", "r")
    test_data_file = open("fashion-mnist_test.csv", "r")
    # only fashion mnist has columns, read and discard
    columns = test_data_file.readline()
    test_data_list = test_data_file.readlines()
    test_data_file.close()
    correct_lines = 0
    print("total test lines: " + str(len(test_data_list)))
    for line in test_data_list:
        all_values = line.split(',')
        output = neural_net.query(numpy.asfarray(all_values[1:]) / 255.0 * .99) + 0.01
        correct_label = int(all_values[0])
        label = numpy.argmax(output)
        if label == correct_label:
            correct_lines += 1
    correct = round( correct_lines/len(test_data_list), 2)
    correct = str(int(correct * 100))
    print('correct: {} percent'.format(correct))

    """
    Uncomment this code if you want to try your own hand-written examples. 
    Draw in Paintbrush, resize/resample in FastStone to a 28x28 pixel image.
    """
    # img_array3a = scipy.misc.imread("3arr.png", flatten=True)
    # img_data = 255.0 - img_array3a.reshape(784)
    # img_data = (img_data / 255.0 * 0.99) + 0.01
    # output = neural_net.query(img_data)
    # label = numpy.argmax(output)
    # print('3a.png identified as: ' + str(label))
    # img_array3b = scipy.misc.imread("3brr.png", flatten=True)
    # img_data = 255.0 - img_array3b.reshape(784)
    # img_data = (img_data / 255.0 * 0.99) + 0.01
    # output = neural_net.query(img_data)
    # label = numpy.argmax(output)
    # print('3b.png identified as: ' + str(label))
    # img_array4 = scipy.misc.imread("4rr.png", flatten=True)
    # img_data = 255.0 - img_array4.reshape(784)
    # img_data = (img_data / 255.0 * 0.99) + 0.01
    # output = neural_net.query(img_data)
    # label = numpy.argmax(output)
    # print('4.png identified as: ' + str(label))
    # img_array6 = scipy.misc.imread("6rr.png", flatten=True)
    # img_data = 255.0 - img_array6.reshape(784)
    # img_data = (img_data / 255.0 * 0.99) + 0.01
    # output = neural_net.query(img_data)
    # label = numpy.argmax(output)
    # print('6.png identified as: ' + str(label))


    """
    Uncomment this code if you want to see what the neural net thinks a number looks like
    """
    # # run the network backwards, given a label, see what image it produces
    # # label to test
    # label = 3
    # # create the output signals for this label
    # targets = numpy.zeros(output_nodes) + 0.01
    # # all_values[0] is the target label for this record
    # targets[label] = 0.99
    # # get image data
    # image_data = neural_net.backquery(targets)
    # # plot image data
    # image_array = numpy.asfarray(image_data.reshape(28, 28))
    # pyplot.matshow(image_array, cmap='Greys')
    # pyplot.show()

main()