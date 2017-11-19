from pytocl.main import main
from driver import Driver
from rnn_model import RNNModelSteering

if __name__ == '__main__':
    main(Driver(
    	RNNModelSteering("TrainedNNs/steer_norm_aalborg_provided_batch-50.pt"),
    	logdata=False))
