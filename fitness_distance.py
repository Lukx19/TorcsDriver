
def fitness(result, timelimit):
    if result['raced_distance'][-1] < 10:
        return 0
    return (2000
            + 300*result['avg_speed'][-1]
            + 100 * result['raced_distance'][-1] / 5000
            - 500 * result['offroad_penalty'][-1])
