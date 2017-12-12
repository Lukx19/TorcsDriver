import argparse
import neat
import pickle
from neat.six_util import iteritems, itervalues

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='NEAT algorithm'
    )

    parser.add_argument(
        '-c',
        '--checkpoint',
        help='Checkpoint file',
        type=str,
        default=None
    )
    parser.add_argument(
        '-o',
        '--output',
        help='output pickle dump file',
        type=str,
        default=None
    )

    args, _ = parser.parse_known_args()

    if args.checkpoint is not None:
        print('Loading from ', args.checkpoint)
        pop = neat.checkpoint.Checkpointer.restore_checkpoint(args.checkpoint)
        winner = None
        for g in itervalues(pop.population):
            winner = g
            break

        print(winner)
        print('Saving best net in {}'.format(args.output))
        pickle.dump(neat.nn.recurrent.RecurrentNetwork.create(
            winner, pop.config), open(args.output, "wb"))
