from __future__ import print_function

import os
import pickle
import os.path
import argparse
import sys
# import simulation
# import datetime
import importlib
import simulate
# from neat import nn, population, statistics
import neat
import visualize
# , neat.visualize

FILE_PATH = os.path.realpath(__file__)
DIR_PATH = os.path.dirname(FILE_PATH)


# sys.path.insert(0, os.path.join(DIR_PATH, '../../'))
def getFitnessFce(path):
    dir, file = os.path.split(os.path.abspath(os.path.splitext(path)[0]))
    sys.path.insert(0, dir)
    mod = importlib.import_module(file)
    return mod.fitness


def defaultFitness(time, timelimit, raced_distance, distance_from_start, damage, offroad_penalty, avg_speed):
    distance = avg_speed * time / timelimit
    print('\tEstimated Distance = ', distance)
    return avg_speed + distance - 300 * offroad_penalty  # - 0.2 * damage


def evalGenomes(genomes, config, evaluate_function=None, cleaner=None, timelimit=None, fitness=None):
    if fitness is not None:
        fitness_fce = getFitnessFce(fitness)
    else:
        fitness_fce = defaultFitness
    print('\nStarting evaluation...\n\n')
    tot = len(genomes)
    # print(genomes)
    # evaluate the genotypes one by one
    for i, (idx, g) in enumerate(genomes):
        print('evaluating', i + 1, '/', tot, '\n')
        net = neat.nn.recurrent.RecurrentNetwork.create(g, config)
        # run the simulation to evaluate the model
        values = evaluate_function(net)[0]
        # print(values, len(values))
        if values is None or len(values) == 0:
            fitness = -100
        else:
            time, raced_distance, distance_from_start, damage, offroad_penalty, avg_speed = values[-1]
            print('\tTotal time = ', time)
            print('\tDistance from start = ', distance_from_start)
            print('\tRaced Distance = ', raced_distance)
            print('\tDamage = ', damage)
            print('\tPenalty = ', offroad_penalty)
            print('\tAvgSpeed = ', avg_speed)
            fitness = fitness_fce(time, timelimit,
                                  raced_distance, distance_from_start,
                                  damage, offroad_penalty, avg_speed)
        print('\tFITNESS =', fitness, '\n')
        g.fitness = fitness

    print('\nfinished evaluation\n\n')

    # if cleaner is not None:

    #     # at the end of the generation, clean the files we don't need anymore

    #     cleaner()


# def get_best_genome(population):

#     best = None
#     for s in population.species:
#         for g in s.members:
#             if best is None or best.fitness is None or (g.fitness is not None and g.fitness > best.fitness):
#                 best = g
#     return best


def run(output_dir, neat_config=None,
        generations=20, port=3001, frequency=None, unstuck=False,
        fitness=None, checkpoint=None, configuration=None, timelimit=None):
    clients = [{
        'port': port,
    }]
    print(str(clients))
    sim = simulate.TorcsFitnessEvaluation(
        torcs_config=configuration, clients=clients, debug_path=output_dir, timelimit=timelimit)

    best_model_file = os.path.join(output_dir, 'best.pickle')

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         neat_config)
    pop = neat.population.Population(config)
    # loading last checkpoint
    if checkpoint is not None:
        print('Loading from ', checkpoint)
        pop = neat.checkpoint.restore_checkpoint(checkpoint)
    # define reporting and checkpointing
    reporter_set = neat.reporting.ReporterSet()
    reporter_set.add(neat.checkpoint.Checkpointer(generation_interval=frequency,
                                                  time_interval_seconds=None, filename_prefix=output_dir + '/checkpoint-'))
    stats = neat.statistics.StatisticsReporter()
    reporter_set.add(stats)
    reporter_set.add(neat.reporting.StdOutReporter(show_species_detail=False))
    pop.add_reporter(reporter_set)

    def fitness_fce(genomes, config): return evalGenomes(
        genomes=genomes,
        config=config,
        evaluate_function=lambda model: sim.evaluate([model]),
        cleaner=None,
        timelimit=timelimit,
        fitness=fitness,)
    winner = pop.run(fitness_function=fitness_fce, n=generations)

    # print('Number of evaluations: {0}'.format(pop.total_evaluations))
    print('Saving best net in {}'.format(best_model_file))
    pickle.dump(neat.nn.recurrent.RecurrentNetwork.create(
        winner, config), open(best_model_file, "wb"))

    # Visualize the winner network and plot/log statistics.

    visualize.draw_net(config, winner, view=False, show_disabled=False,
                       filename=os.path.join(output_dir, "nn_winner"))
    visualize.draw_net(config, winner, view=False, filename=os.path.join(
        output_dir, "nn_winner-enabled"), show_disabled=False)
    # visualize.draw_net(config, winner, view=False, filename=os.path.join(
    #     output_dir, "nn_winner-enabled-pruned"), show_disabled=False, prune_unused=True)

    visualize.plot_stats(stats, filename=os.path.join(
        output_dir, 'avg_fitness.svg'))

    visualize.plot_species(stats, filename=os.path.join(
        output_dir, 'speciation.svg'))
    stats.save_genome_fitness(filename=os.path.join(
        output_dir, 'fitness_history.csv'))

    stats.save_species_count(
        filename=os.path.join(output_dir, 'speciation.csv'))

    stats.save_species_fitness(filename=os.path.join(
        output_dir, 'species_fitness.csv'))


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
        '-g',
        '--generations',
        help='Number of generations to train',
        type=int,
        default=10
    )

    parser.add_argument(
        '-f',
        '--frequency',
        help='How often to store checkpoints',
        type=int,
        default=1
    )

    parser.add_argument(
        '-o',
        '--output_dir',
        help='Directory where to store checkpoint.',
        type=str,
        default="debug"
    )

    parser.add_argument(
        '-p',
        '--port',
        help='Port to use for comunication between server (simulator) and client',
        type=int,
        default=3001
    )

    # parser.add_argument(
    #     '-u',
    #     '--unstuck',
    #     help='Make the drivers automatically try to unstuck',
    #     action='store_true'
    # )

    parser.add_argument(
        '-n',
        '--neat_config',
        help='NEAT configuration file',
        type=str,
        default='neat.conf',
    )

    parser.add_argument(
        '-x',
        '--configuration',
        help='XML configuration file for running the race',
        type=str,
        default='config-torcs/forza.xml'
    )

    parser.add_argument(
        '-e',
        '--fitness',
        help='Python file containing the function for fitness evaluation',
        type=str,
        default=None
    )

    parser.add_argument(
        '-t',
        '--timelimit',
        help='Timelimit for the race',
        type=int,
        default=3
    )
    args, _ = parser.parse_known_args()
    run(**args.__dict__)
