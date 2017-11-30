

def fitness(time, timelimit, raced_distance, distance_from_start, damage, offroad_penalty, avg_speed):
    distance = avg_speed * time / timelimit
    print('\tEstimated Distance = ', distance)
    return avg_speed + distance - 300 * offroad_penalty  # - 0.2 * damage
