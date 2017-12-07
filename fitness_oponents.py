
def fitness(result, timelimit):
    return (5000
            + 100*result['avg_speed'][-1]
            + 100 * result['raced_distance'][-1] / 5000
            - 100 * result['offroad_penalty'][-1]
            - result['car_hit_penalty'][-1])
