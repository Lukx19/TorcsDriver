from model import *
from get_data import *
import sys
import torch.nn as nn

torch.nn.Module.dump_patches = True
def evaluate(input_tensor):
    hidden = rnn.initHidden()
    
    softmax = nn.Sigmoid()
    outputs = []
    for i in range(len(input_tensor)):
        output, hidden = rnn(input_tensor[i], hidden)
        outputs.append(softmax(output))
    
    return outputs

def predict(sequence, threshold):
    outputs = evaluate(sequence)

    predictions = []
    for output in outputs:
        if output[0].data[0] > threshold:
            prediction = 1
        else:
            prediction = 0
        predictions.append(prediction)

    return predictions

if __name__ == '__main__':
    rnn = torch.load(sys.argv[3])
    correct = 0
    actual = 0
    correct_sequence = []
    batch_length = int(sys.argv[2])
    data_length, n_train, n_valid, n_test, x_train, x_valid, x_test, y_train, y_valid, y_test = read_data("training_data/norm_aalborg_provided.csv", 1, 22, sys.argv[4])
    for i in range(int(n_valid / batch_length)):
        y, x = sequentialPair(x_valid, y_valid, batch_length, i)
        predictions = predict(x, float(sys.argv[1]))
        this_sequence = 0
        for j, pred in enumerate(predictions):
            actual += y[j].data[0]
            if y[j].data[0] == pred:
                correct += 1
                this_sequence += 1
        correct_sequence.append(this_sequence)
    per_sequence_correct = sum(correct_sequence) / len(correct_sequence) / batch_length * 100
    wrong = int(n_valid) - correct
    print("percentage 1 in val set: ", actual / int(n_valid) * 100)
    print("percentage correct: ", correct / int(n_valid) * 100)
    print("percentage wrong: ", wrong / int(n_valid) * 100)
    print("correct per sequence mean:", per_sequence_correct)
    
