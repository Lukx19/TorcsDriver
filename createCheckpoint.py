import neat
import argparse
from neat.six_util import iteritems


def addConnection(genom, config, from_node, to_node):
    connection = neat.DefaultGenome.create_connection(
        config, from_node, to_node)
    connection.weight = 1
    connection.enabled = True
    genom.connections[connection.key] = connection


def fitness(genoms, config):
    for i, g in genoms:
        g.fitness = 0


def generate(output, neat_config):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         neat_config)
    pop = neat.population.Population(config)
    for i, genom in list(iteritems(pop.population)):
        addConnection(genom, config.genome_config, -1, 0)
        addConnection(genom, config.genome_config, -2, 1)
        addConnection(genom, config.genome_config, -3, 2)
        # print(str(genom))
    checkpointer = neat.checkpoint.Checkpointer(generation_interval=1,
                                                time_interval_seconds=None, filename_prefix=output)
    pop.add_reporter(checkpointer)
    pop.run(fitness_function=fitness, n=1)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='NEAT algorithm'
    )
    parser.add_argument(
        '-o',
        '--output',
        help='Output file containing checkpoint',
        type=str,
        default="init_checkpoint"
    )
    parser.add_argument(
        '-n',
        '--neat_config',
        help='NEAT configuration file',
        type=str,
        default='neat.conf',
    )
    args, _ = parser.parse_known_args()
    generate(**args.__dict__)
