import csv
import numpy as np
import matplotlib as mp


number_of_epochs = 10
number_of_outputs = 10
dimensions_including_bias = 785
accuracy_ofEach_epoch = np.zeros((number_of_epochs, 1), dtype=float)
weight_matrix = np.random.uniform(-0.05, 0.05, (number_of_outputs, dimensions_including_bias))
learning_rate = 0.001  # 0.001 0.1

def get_trainingData(filename):
    global weight_matrix
    with open(filename, "rt") as fin:
        correctPredictValue = 0
        cin = csv.reader(fin)
        cin_array = list(cin)
        training_matrix = np.array(cin_array, dtype=float)
        confusion_matrix = np.zeros((number_of_outputs,number_of_outputs), dtype=int)
        for each_input in training_matrix:
            input_vector = each_input.copy()
            dimension = len(input_vector)
            input_vector = np.reshape(input_vector,(1,dimension))
            target_output = each_input[0]
            predicted_output = 0
            input_vector = input_vector/255.0
            input_vector[0][0] = 1
            target_vector = np.zeros((number_of_outputs,1), dtype=float)
            target_vector[int(target_output)] = 1 #Bcz target_output = 5 would translate to index 4 since array starts @ index 0
            outputs_y = np.dot(weight_matrix, np.transpose(input_vector))
            predicted_output = outputs_y.argmax()
            # print("Target output = {} \n Predicted output = {} ".format(target_output,predicted_output))
            if target_vector[predicted_output] != 1:
            #     #print("Correct prediction")
            #     # correctPredictValue = correctPredictValue + 1
            # else:
                outputs_y = np.where(outputs_y > 0, 1.0, 0.0)
                weight_matrix = weight_matrix - learning_rate*np.dot(outputs_y - target_vector,input_vector)
                # print("weight matrix = {}".format(weight_matrix))
            confusion_matrix[int(target_output)][int(predicted_output)] = confusion_matrix[int(target_output)][int(predicted_output)] + 1
        sum_diagnol = 0.0
        for i in range(10):
            sum_diagnol = sum_diagnol + confusion_matrix[i][i]
        print(confusion_matrix)
        accuracy = (sum_diagnol/confusion_matrix.sum())*100
        # accuracy2 = (correctPredictValue/60000)*100
        print("accuracy = " ,accuracy)
        # print("accuracy2 = " ,accuracy2)


if __name__ == "__main__":
    filename_trainingData = "/Users/anantpathak/OneDrive/PortlandStateUniversity/Year1/Sem2/MachineLearning/Homework/HW1/Q11Prog/Data/mnist_train.csv"
    filename_testData = "/Users/anantpathak/OneDrive/PortlandStateUniversity/Year1/Sem2/MachineLearning/Homework/HW1/Q11Prog/Data/mnist_test.csv"
    for i in range(30):
        print("Training Epoch: {}".format(i))
        get_trainingData(filename_trainingData)
        print("Testing Epoch: {}".format(i))
        get_trainingData(filename_testData)

