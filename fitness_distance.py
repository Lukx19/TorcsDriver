
def fitness(result, timelimit):
    if result['raced_distance'][-1] < 10:
        return -8000
    return 100*result['avg_speed'][-1] + result['raced_distance'][-1] - 1000 * result['offroad_penalty'][-1]
