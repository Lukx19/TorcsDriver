#! /usr/bin/env python3

from pytocl.main import main
from pytocl.driver import Driver
from rnn_model import RNNModelSteering

if __name__ == '__main__':
    main(Driver(RNNModelSteering("Weights/dim7-steer/", True)))
