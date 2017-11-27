from __future__ import print_function
from datetime import datetime
import os
import neat
import visualize
import numpy as np
import configparser

# parameters 
filename = "alpine-1.csv"
datacolumn = 2 #0 for accelerate, 1 for brake, 2 for steering
batch_size = 10 
n_xvaria = 22 #number of x variables
start_hidden = 20

# adjust the configuration file for the number of input and hidden nodes
config = configparser.ConfigParser()
config.read("config-neat")
config['DefaultGenome']['num_inputs'] = str(n_xvaria)
config['DefaultGenome']['num_hidden'] = str(start_hidden)


# load data and convert data so that
# x_input is a list of tuples (each of length 22), and target is list of floats. 
data = np.loadtxt(open("data/" + filename, 'r'), delimiter=',', skiprows=1)   
target = []
x_input = []
for row in data:
    temp = row[datacolumn]
    if datacolumn < 2:
        temp = int(temp)
    target.append(float(temp))
    x_input.append(tuple(row[3:]))

# split data into training, test and validation set
data_length = len(x_input)
num_batches = data_length / batch_size
n_train = round(0.8 * data_length)
n_valid = round(0.1 * data_length)
n_test = round(0.1 * data_length)
x_train, x_valid, x_test = x_input[:n_train], x_input[n_train:(n_train+n_valid)], x_input[n_test:]
y_train, y_valid, y_test = target[:n_train], target[n_train:(n_train+n_valid)], target[n_test:]
print(len(x_train))
print(len(y_train))

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 4.0
        net = neat.nn.RecurrentNetwork.create(genome, config)
        for xi, xo in zip(x_train, y_train):
            output = net.activate(xi)
            from_mid = 0 - xi[1] # track position from middle
            # print("from mid: ", from_mid)
            genome.fitness -= (output[0] - xo) ** 2 + from_mid
            # genome.fitness -= (output[0] - xo)


def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(10))

    # Run for up to 200 generations.
    winner = p.run(eval_genomes, 200)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.RecurrentNetwork.create(winner, config)
    for xi, xo in zip(x_train, y_train):
        output = winner_net.activate(xi)
        print("expected output {!r}, got {!r}".format(xo, output))

    # node_names = {-1:'A', -2: 'B', 0:'A XOR B'}
    # visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-9')
    p.run(eval_genomes, 10)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-neat')
    run(config_path)