from __future__ import print_function

import os
import pickle
import os.path
import argparse
# import sys
# import simulation
# import datetime
# import importlib
import simulate
# from neat import nn, population, statistics
import neat
# , neat.visualize

FILE_PATH = os.path.realpath(__file__)
DIR_PATH = os.path.dirname(FILE_PATH)


# sys.path.insert(0, os.path.join(DIR_PATH, '../../'))


def evalGenomes(genomes, config, evaluate_function=None, cleaner=None, timelimit=None):

    print('\nStarting evaluation...\n\n')
    tot = len(genomes)
    # print(genomes)
    # evaluate the genotypes one by one
    for i, g in genomes:
        print('evaluating', i + 1, '/', tot, '\n')
        net = neat.nn.recurrent.RecurrentNetwork.create(g, config)
        # run the simulation to evaluate the model
        values = evaluate_function(net)[0]
        # print(values, len(values))
        if values is None or len(values) == 0:
            fitness = -100
        else:
            time, raced_distance, distance_from_start, damage, offroad_penalty, avg_speed = values[-1]
            distance = avg_speed * time / timelimit
            fitness = avg_speed + distance_from_start - \
                300 * offroad_penalty  # - 0.2 * damage

            print('\tDistance = ', distance_from_start)
            print('\tEstimated Distance = ', distance)
            print('\tDamage = ', damage)
            print('\tPenalty = ', offroad_penalty)
            print('\tAvgSpeed = ', avg_speed)
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


def run(output_dir, neat_config=None, generations=20, port=3001, frequency=None, unstuck=False, evaluation=None, checkpoint=None, configuration=None, timelimit=None):
    clients = [{
        'port': port,
    }]
    print(str(clients))
    sim = simulate.TorcsFitnessEvaluation(
        torcs_config=configuration, clients=clients, debug_path=output_dir)

    best_model_file = os.path.join(output_dir, 'best.pickle')

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         neat_config)
    pop = neat.population.Population(config)
    # define reporting and checkpointing
    reporter_set = neat.reporting.ReporterSet()
    reporter_set.add(neat.checkpoint.Checkpointer(generation_interval=frequency,
                                                  time_interval_seconds=None, filename_prefix=output_dir + '/checkpoint-'))
    reporter_set.add(neat.statistics.StatisticsReporter())
    reporter_set.add(neat.reporting.StdOutReporter(show_species_detail=False))
    # loading last checkpoint
    if checkpoint is not None:
        print('Loading from ', checkpoint)
        pop = neat.checkpoint.restore_checkpoint(checkpoint)

    def fitness_fce(genomes, config): return evalGenomes(
        genomes=genomes,
        config=config,
        evaluate_function=lambda model: sim.evaluate([model]),
        cleaner=None,
        timelimit=timelimit)
    winner = pop.run(fitness_function=fitness_fce, n=generations)
    # for g in range(1, generations + 1):
    # if g % frequency == 0:
    #     print('Saving best net in {}'.format(best_model_file))
    #     best_genome = get_best_genome(pop)
    #     pickle.dump(nn.create_recurrent_phenotype(
    #         best_genome), open(best_model_file, "wb"))

    #     # new_checkpoint = os.path.join(
    #     #     output_dir, 'neat_gen_{}.checkpoint'.format(pop.generation))

    #     # print('Storing to ', new_checkpoint)
    #     # pop.save_checkpoint(new_checkpoint)

    #     print('Plotting statistics')
    #     neat.visualize.plot_stats(pop.statistics, filename=os.path.join(
    #         output_dir, 'avg_fitness.svg'))
    #     neat.visualize.plot_species(pop.statistics, filename=os.path.join(
    #         output_dir, 'speciation.svg'))

    #     print('Save network view')
    #     neat.visualize.draw_net(best_genome, view=False,
    #                        filename=os.path.join(
    #                            output_dir, "nn_winner-enabled-pruned.gv"),
    #                        show_disabled=False, prune_unused=True)
    #     neat.visualize.draw_net(best_genome, view=False,
    #                        filename=os.path.join(output_dir, "nn_winner.gv"))
    #     neat.visualize.draw_net(best_genome, view=False, filename=os.path.join(output_dir, "nn_winner-enabled.gv"),
    #                        show_disabled=False)

    print('Number of evaluations: {0}'.format(pop.total_evaluations))
    print('Saving best net in {}'.format(best_model_file))
    pickle.dump(neat.nn.recurrent.RecurrentNetwork.create(
        winner, config), open(best_model_file, "wb"))

    # Visualize the winner network and plot/log statistics.

    neat.visualize.draw_net(winner, view=True, filename=os.path.join(
        output_dir, "nn_winner.gv"))
    neat.visualize.draw_net(winner, view=True, filename=os.path.join(
        output_dir, "nn_winner-enabled.gv"), show_disabled=False)
    neat.visualize.draw_net(winner, view=True, filename=os.path.join(
        output_dir, "nn_winner-enabled-pruned.gv"), show_disabled=False, prune_unused=True)

    neat.visualize.plot_stats(pop.statistics, filename=os.path.join(
        output_dir, 'avg_fitness.svg'))

    neat.visualize.plot_species(pop.statistics, filename=os.path.join(
        output_dir, 'speciation.svg'))

    statistics.save_stats(pop.statistics, filename=os.path.join(
        output_dir, 'fitness_history.csv'))

    statistics.save_species_count(
        pop.statistics, filename=os.path.join(output_dir, 'speciation.csv'))

    statistics.save_species_fitness(
        pop.statistics, filename=os.path.join(output_dir, 'species_fitness.csv'))


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
        default=10
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

    # parser.add_argument(
    #     '-e',
    #     '--evaluation',
    #     help='Python file containing the function for fitness evaluation',
    #     type=str,
    #     default=None
    # )

    parser.add_argument(
        '-t',
        '--timelimit',
        help='Timelimit for the race',
        type=int,
        default=100
    )
    args, _ = parser.parse_known_args()
    run(**args.__dict__)
