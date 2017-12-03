

def fitness(result, timelimit):
    return 10*result['avg_speed'][-1] + result['raced_distance'][-1] - 200 - 800 * result['offroad_penalty'][-1] - 0.1 * result['damage'][-1]
