from pytocl.protocol import Client
from driver import Driver

import argparse
import signal
import sys
import logging


model = None
model_type = ""


def sigterm_handler(_signo, _stack_frame):
    print('Someone killed me')
    if model_type == 'NEAT' and model is not None:
        model.saveResults()
    sys.exit(0)


signal.signal(signal.SIGINT, sigterm_handler)

signal.signal(signal.SIGTERM, sigterm_handler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Model selection for the client for TORCS racing car simulation with SCRC network server.'
    )
    parser.add_argument(
        '-m',
        '--model',
        help='Choose model [RNN,NEAT]',
        default="RNN"
    )
    parser.add_argument(
        '-f',
        '--model_file',
        help='Filename to the model file',
        default="TrainedNNs/steer_norm_aalborg_provided_batch-50.pt"
    )
    parser.add_argument(
        '-o',
        '--output_file',
        help='Output file with the results',
        default="results.txt"
    )
    parser.add_argument(
        '--hostname',
        help='Racing server host name.',
        default='localhost'
    )
    parser.add_argument(
        '-p',
        '--port',
        help='Port to connect, 3001 - 3010 for clients 1 - 10.',
        type=int,
        default=3001
    )
    parser.add_argument('-v', help='Debug log level.', action='store_true')
    args = parser.parse_args()
    print(args.model)
    model_type = args.model
    if args.model == 'RNN':
        from rnn_model import RNNModelSteering
        model = RNNModelSteering(args.f)
    elif args.model == 'NEAT':
        from neat_model import NeatModel
        model = NeatModel(args.model_file, args.output_file)
    print(args.model,model)
        # switch log level:
    if args.v:
        level = logging.DEBUG
    else:
        level = logging.INFO
    del args.v
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)7s %(name)s %(message)s"
    )
    driver = Driver(model=model, logdata=False)
    print(driver)
    # start client loop:
    client = Client(driver=driver, hostname=args.hostname,port=args.port)
    client.run()
