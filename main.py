from NeuralNet import NeuralNet
from matplotlib import pyplot
import numpy
import scipy.misc
import scipy.ndimage
#import imageio


def main():

    # 1 node for every cell of the 28x28 image
    input_nodes = 784
    # arbitrary, should be a less than input_nodes to summarize results,
    # but enough to find patterns
    hidden_nodes = 100
    # 1 node for every expected matching number (0-9)
    output_nodes = 10
    # small moves, don't jump around too much when correcting errors
    learning_rate = 0.2

    neural_net = NeuralNet(input_nodes=input_nodes,
                    output_nodes=output_nodes,
                    hidden_nodes=hidden_nodes,
                    learning_rate=learning_rate)

    training_data_file = open("mnist_train.csv", "r")
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
    test_data_file = open("mnist_test.csv", "r")
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
    print('correct: {} percent'.format( str( round( correct_lines/len(test_data_list), 3) * 100)))

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
    # run the network backwards, given a label, see what image it produces
    # label to test
    label = 3
    # create the output signals for this label
    targets = numpy.zeros(output_nodes) + 0.01
    # all_values[0] is the target label for this record
    targets[label] = 0.99
    # get image data
    image_data = neural_net.backquery(targets)
    # plot image data
    image_array = numpy.asfarray(image_data.reshape(28, 28))
    pyplot.matshow(image_array, cmap='Greys')
    pyplot.show()

main()