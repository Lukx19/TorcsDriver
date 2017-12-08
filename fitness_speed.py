
def fitness(result, timelimit):
    if result['raced_distance'][-1] < 0:
        return 0
    return (500
            + result['raced_distance'][-1] / result['time'][-1]
            - 500 * result['offroad_penalty'][-1])
