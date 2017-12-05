

def fitness(result, timelimit):
    return 10*result['avg_speed'][-1] + result['raced_distance'][-1] - 200 - 2000 * result['offroad_penalty'][-1] - 0.1 * result['damage'][-1] - 800*abs(result['distance_from_center'][-1])
