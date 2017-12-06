import logging

from pytocl.analysis import DataLogWriter
from pytocl.car import State, Command
import os.path
import time
import random
import pickle

_logger = logging.getLogger(__name__)


class DriverNeat:

    def __init__(self, model, logdata=True):
        self.data_logger = DataLogWriter() if logdata else None
        self.model = model
        self.stuck_count = 0
        self.unstucking = False

        self.my_state = {'pos': 0, 'steering': 0}

        time.sleep(random.uniform(0, 0.5))
        if os.path.isfile("out1.comm"):
            self.out_comm = "out2.com"
            self.in_comm = "out1.com"
        else:
            self.out_comm = "out1.com"
            self.in_comm = "out2.com"
        with open(self.out_comm, 'w') as f:
            f.write("I am first")

    @property
    def range_finder_angles(self):
        """Iterable of 19 fixed range finder directions [deg].

        The values are used once at startup of the client to set the directions
        of range finders. During regular execution, a 19-valued vector of track
        distances in these directions is returned in ``state.State.tracks``.
        """
        return list(range(-90, 91, 10))

    def on_shutdown(self):
        """
        Server requested driver shutdown.

        Optionally implement this event handler to clean up or write data
        before the application is stopped.
        """
        if self.data_logger:
            self.data_logger.close()
            self.data_logger = None

    def drive(self, carstate: State) -> Command:
        cmd = Command()
        shared_state = self.loadSharedState()
        if shared_state['pos'] < self.my_state['pos']:
            self.model.predict(carstate, shared_state)
        else:
            self.model.predict(carstate, self.my_state)
        self.my_state['pos'] = carstate.race_position
        self.my_state['steering'] = self.model.getSteering()
        self.saveSharedState(self.my_state)

        if self.isStuck(carstate):
            print('stuck')
            self.reverse(carstate, cmd)
            self.unstucking = True
        else:
            # change gear from reverse after unstucking
            if self.unstucking and carstate.gear <= 0:
                print('unstuck')
                carstate.gear = 1
                self.unstucking = False

            cmd.steering = self.model.getSteering()
            self.accelerate(carstate, max(
                0.1, self.model.getAcceleration()), self.model.breaking, cmd)

        if self.data_logger:
            self.data_logger.log(carstate, cmd)
        return cmd

    def shift(self, carstate, command):
        if command.gear >= 0 and command.brake < 0.1 and carstate.rpm > 8000:
            command.gear = min(6, command.gear + 1)

        if carstate.rpm < 2500 and command.gear > 1:
            command.gear = command.gear - 1

        if not command.gear:
            command.gear = carstate.gear or 1

    def accelerate(self, carstate, acceleration, brake, command):
        # if brake > 0.8:
        #     command.brake = 1
        # else:
        if acceleration > 0:
            command.accelerator = acceleration
        else:
            command.brake = -1*acceleration
        command.gear = carstate.gear
        self.shift(carstate, command)

    def isStuck(self, carstate):
        if carstate.speed_x < 2 \
                and abs(carstate.distance_from_center) > 0.95 \
                and abs(carstate.angle) > 15\
                and carstate.angle * carstate.distance_from_center < 0:
            self.stuck_count += 1
        else:
            self.stuck_count = 0
        return self.stuck_count > 100

    def reverse(self, carstate, command):
        command.accelerator = 1.0
        command.gear = -1
        command.brake = 0.0
        command.clutch = 0.0
        command.steering = -1*carstate.angle * 3.14159265359/(180.0 * 0.785398)

    def loadSharedState(self):
        dict_state = None
        try:
            with open(self.in_comm, 'r') as f:
                dict_state = pickle.load(f)
        except:
            dict_state = None
            print("not able to load data from other driver")
        return dict_state

    def saveSharedState(self, dict_state):
        try:
            with open(self.in_comm, 'w') as f:
                pickle.dump(dict_state, f)
        except:
            dict_state = None
            print("not able to write data to other driver")
