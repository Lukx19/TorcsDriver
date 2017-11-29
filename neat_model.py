from model import Model
import numpy as np
import pickle
import math


class NeatModel(Model):
    def __init__(self, model_file, result_file):
        with open(model_file, 'rb') as f:
            self.net = pickle.load(f)
            self.net.reset()
        self.result_file = result_file
        self.predictions = 0
        self.offroad_penalty = 0
        self.cumul_speed = 0
        self._current_lap_time = 0
        self._last_laps_accumulated_time = 0

        self.results = []

    # def __del__(self):
    #     """Try to save data before instance is destroyed."""
    #     self.saveResults()

    def saveResults(self):
        with open(self.result_file, 'w') as f:
            if len(self.results) > 0:
                f.write(
                    "time, raced_distance, distance_from_start, damage, offroad_penalty, avg_speed\n")
                for r in self.results:
                    f.write("{}, {}, {}, {}, {}, {}\n".format(*r))

    def _projectSpeed(self,speed_x, speed_y, angle):
        velocity = np.angle(speed_x + 1j * speed_y, True),np.sqrt(speed_x ** 2 + speed_y ** 2)
        return velocity[1] * math.cos(math.radians(angle - velocity[0]))

    def predict(self, state):
        inputs = self.stateToInput(state)
        if np.isnan(inputs).any():
            print('#####################  NaN inputs')
            return
        output = self.net.activate(inputs)
        self.acceleration = output[0]
        self.steering = output[1]
        self.breaking = output[2]
        self.predictions += 1

        if self._current_lap_time > state.current_lap_time:
            self._last_laps_accumulated_time += state.last_lap_time
        self._current_lap_time = state.current_lap_time

        self.offroad_penalty += (max(0,
                                     math.fabs(state.distance_from_center) - 0.95)) ** 2
        projected_speed = self._projectSpeed(
            state.speed_x, state.speed_y, state.angle)

        self.cumul_speed += projected_speed

        if self.predictions % 100 == 0:
            self.results.append([
                self._current_lap_time + self._last_laps_accumulated_time,
                state.distance_raced,
                state.distance_from_start,
                state.damage,
                math.sqrt(self.offroad_penalty / self.predictions),
                self.cumul_speed / self.predictions,
            ])

    def stateToInput(self, state):
        array = []
        array.append(state.angle / 180.0)
        array.append(state.speed_x / 40.0)
        array.append(state.speed_y / 40.0)
        for j in range(0, 19, 3):
            if math.fabs(state.distance_from_center) > 1 or state.distances_from_edge[j] < 0:
                array.append(-1)
            else:
                array.append(state.distances_from_edge[j] / 200.0)
        array.append(state.distance_from_center)
        for j in range(4):
            array.append(state.wheel_velocities[j] / 3000.0)  # /150.0
        array.append(state.z - 0.36)
        array.append(state.speed_z / 10.0)
        return np.array(array)
