from model import *
from get_data_steer import *
import sys
import torch.nn as nn

def predict(input_tensor):
    hidden = rnn.initHidden()
    
    outputs = []
    for i in range(len(input_tensor)):
        output, hidden = rnn(input_tensor[i], hidden)
        outputs.append(output)
    
    return outputs

if __name__ == '__main__':
    rnn = torch.load(sys.argv[2])
    total_error = 0
    sequence_errors = []
    batch_length = int(sys.argv[1])
    data_length, n_train, n_valid, n_test, x_train, x_valid, x_test, y_train, y_valid, y_test = read_data("training_data/norm_aalborg_provided.csv", 1, 22)
    for i in range(int(n_valid / batch_length)):
        y, x = sequentialPair(x_valid, y_valid, batch_length, i)
        predictions = predict(x)
        error_sequence = 0
        for j, pred in enumerate(predictions):
            error = abs(y[j].data[0] - pred[0].data[0])
            print("actual: ", y[j].data[0], " prediction: ", pred[0].data[0])
            print("error:", error)
            error_sequence += error
            total_error += error
        sequence_errors.append(error_sequence)
    per_sequence_error = sum(sequence_errors) / len(sequence_errors) / batch_length * 100
    print("total MSE: ", total_error / int(n_valid / batch_length))
    print("MSE per sequence:", per_sequence_error)
    
