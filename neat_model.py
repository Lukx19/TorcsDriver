from model import Model
import numpy as np
import pickle
import math


class NeatModel(Model):

    def __init__(self, model_file, result_file,
                 ff_model_file=None, max_speed=300):
        with open(model_file, 'rb') as f:
            self.net = pickle.load(f)
            self.net.reset()

        self.ff_model = None
        if ff_model_file is not None:
            with open(ff_model_file, 'rb') as f:
                self.ff_model = pickle.load(f)

        self.result_file = result_file
        self.predictions = 0
        self.offroad_penalty = 0
        self.cumul_speed = 0
        self._current_lap_time = 0
        self._last_laps_accumulated_time = 0
        self.last_distance_from_start = 0
        self.acc_distance_from_start = 0
        self.laps = -1
        self.cumul_dist_from_center = 0
        self.car_hit_penalty = 0

        self.results = []
        self._max_speed = max_speed
        self.opponents_array = [14, 16, 18, 20, 22]

    # def __del__(self):
    #     """Try to save data before instance is destroyed."""
    #     self.saveResults()

    def saveResults(self):
        with open(self.result_file, 'w') as f:
            if len(self.results) > 0:
                f.write(
                    "time,raced_distance,distance_from_start,damage,"
                    "distance_from_center,angle,offroad_penalty,"
                    "avg_speed,laps,car_hit_penalty\n")
                for r in self.results:
                    f.write("{},{},{},{},{},{},{},{},{},{}\n".format(*r))

    def _projectSpeed(self, speed_x, speed_y, angle):
        velocity = np.angle(speed_x + 1j * speed_y,
                            True), np.sqrt(speed_x ** 2 + speed_y ** 2)
        return velocity[1] * math.cos(math.radians(angle - velocity[0]))

    def _isHittingCar(self, state):
        for i in self.opponents_array:
            # opponent is one meter or less to our car. This means we hit him.
            if state.opponents[i] < 1:
                return True
        return False

    def predict(self, state):
        inputs = self.stateToInput(state)
        if np.isnan(inputs).any():
            print('#####################  NaN inputs')
            return
        output = self.net.activate(inputs)
        self.acceleration = output[0]
        # using left and right steering as a separate output nodes
        self.steering = output[1] - output[2]
        self.breaking = output[3]
        self.predictions += 1
        if self._current_lap_time > state.current_lap_time:
            # we are in the new round
            self.laps += 1
            self._last_laps_accumulated_time += state.last_lap_time
        self._current_lap_time = state.current_lap_time

        # print(state.distance_from_start)
        # filtering start position behind the starting line
        if self.last_distance_from_start == 0 and state.distance_from_start > 1000:
            self.acc_distance_from_start = 0
            distance = 0
        else:
            if abs(self.last_distance_from_start - state.distance_from_start) > 1000:
                # new lap
                # print('newLap')
                self.acc_distance_from_start += self.last_distance_from_start
            else:
                self.last_distance_from_start = state.distance_from_start
            distance = state.distance_from_start
        if abs(state.distance_from_center) > 0.95:
            self.offroad_penalty += 1
        # self.offroad_penalty += (max(0,
        # math.fabs(state.distance_from_center) - 0.95)) ** 2
        projected_speed = self._projectSpeed(
            state.speed_x, state.speed_y, state.angle)

        self.cumul_speed += projected_speed
        self.cumul_dist_from_center += state.distance_from_center

        if self._isHittingCar(state):
            self.car_hit_penalty += 1

        if self.predictions % 100 == 0:
            self.results.append([
                self._current_lap_time + self._last_laps_accumulated_time,
                state.distance_raced,
                self.acc_distance_from_start + distance,
                state.damage,
                self.cumul_dist_from_center / self.predictions,
                state.angle / 180,
                math.sqrt(self.offroad_penalty / self.predictions),
                self.cumul_speed / self.predictions,
                self.laps,
                self.car_hit_penalty,
            ])

    def stateToInput(self, state):
        array = []
        if self.ff_model is not None:
            self.ff_model.predict(state)
            array.append(self.ff_model.getSteering())
            array.append(self.ff_model.getBreak())
            array.append(self.ff_model.getAcceration())
        array.append(state.angle / 180.0)
        array.append(state.speed_x / self._max_speed)
        array.append(state.speed_y / self._max_speed)
        array.append(state.speed_z / 10.0)
        for j in [7, 8, 9, 10, 11, 12]:
            if math.fabs(state.distance_from_center) > 1 or state.distances_from_edge[j] < 0:
                array.append(-1)
            else:
                array.append(state.distances_from_edge[j] / 200.0)
        array.append(state.distance_from_center)
        for j in range(4):
            array.append(state.wheel_velocities[j] / 3000.0)  # /150.0
        array.append(state.z - 0.36)

        for j in self.opponents_array:
            array.append(state.opponents[j] / 200.0)
        return np.array(array)
