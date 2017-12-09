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

        # init vars for driving
        self.data_logger = DataLogWriter() if logdata else None
        self.model = model
        self.stuck_count = 0
        self.unstucking = False
        self.reverse_counter = 0
        self.distance_from_start = 0
        self.last_raced_dist = 0
        self.backwards_count = 0
        self.turning180 = False
        self.turn180_count = 0
        self.position_to_slow = None
        self.counter = 0
        self.start_position = 0
        self.log_counter = 0
        self.first_car = None
        self.last_position = None

        # carstate for sharing
        self.my_state = {'pos': 1, 'steering': 0, 'distance_from_start': None}

        time.sleep(random.uniform(0, 0.5))
        if os.path.isfile("out1.com"):
            self.out_comm = "out2.com"
            self.in_comm = "out1.com"
            with open(self.out_comm, 'wb') as f:
                pickle.dump(self.my_state, f)
        else:
            self.out_comm = "out1.com"
            self.in_comm = "out2.com"
            with open(self.out_comm, 'wb') as f:
                pickle.dump(self.my_state, f)

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

        # update race position
        if carstate.race_position != self.last_position:
            print('car in pos', carstate.race_position)
            self.last_position = carstate.race_position

        # first time drive method gets called save distance from start for reference and save state
        if self.counter == 0:
            self.counter += 1
            self.last_position = carstate.race_position
            self.distance_from_start = carstate.distance_from_start
            self.my_state['pos'] = carstate.race_position
            self.my_state['steering'] = self.model.getSteering()
            self.my_state['distance_from_start'] = None

        # initialize instance of command
        cmd = Command()

        # save carstate and log if crashed
        self.my_state['pos'] = carstate.race_position
        self.my_state['steering'] = self.model.getSteering()
        self.my_state['distance_from_start'] = carstate.distance_from_start
        self.logIf(carstate, self.my_state)
        self.model.predict(carstate, self.my_state)

        # load shared state from other driver every 10
        if carstate.distance_from_start - self.distance_from_start > 10:

            shared_state = self.loadSharedState()

            if shared_state:

                dangerous_distance = shared_state['distance_from_start']

                if shared_state['pos'] >= carstate.race_position:
                    self.first_car = True

                if dangerous_distance:
                    print("Found dangerous place")
                    self.position_to_slow = dangerous_distance

            # update distance from start log
            self.distance_from_start = carstate.distance_from_start

        # check if at dangerous position and slow down if so
        if self.position_to_slow and abs(self.position_to_slow - carstate.distance_from_start) < 5:
                print("dangerous place")

                if self.isStuck(carstate) or self.reverse_counter > 0:
                    print('stuck')
                    if self.reverse_counter == 0:
                        self.reverse_counter = 10
                    self.reverse(carstate, cmd)
                    self.unstucking = True
                    self.reverse_counter -= 1
                    if (abs(carstate.angle) < 15
                        or abs(carstate.distance_from_center) < 0.2):
                        self.reverse_counter = 0

                elif self.isGoingBack(carstate) or self.turn180_count > 0:
                    print("backwards")
                    self.turn180(carstate, cmd)
                    if self.turn180_count == 0:
                        self.turn180_count = 30
                    self.turn180_count -= 1

                    if abs(carstate.angle) < 15:
                        self.turn180_count = 0
                else:
                    if carstate.gear < 0:
                        cmd.gear = 0
                        cmd.brake = 1
                    else:
                        self.unstucking = False
                        cmd.steering = self.model.getSteering()
                        print('Extra braking')

                        # increase braking at dangerous position
                        breaking = self.model.getBreak()
                        breaking += 0.05
                        self.accelerate(carstate, max(0.1, self.model.getAcceleration()), breaking, cmd)

        # if not at dangerous position, drive normally
        else:

            if self.isStuck(carstate) or self.reverse_counter > 0:
                print('stuck')
                if self.reverse_counter == 0:
                    self.reverse_counter = 10
                self.reverse(carstate, cmd)
                self.unstucking = True
                self.reverse_counter -= 1
                if (abs(carstate.angle) < 15
                        or abs(carstate.distance_from_center) < 0.2):
                    self.reverse_counter = 0

            elif self.isGoingBack(carstate) or self.turn180_count > 0:
                print("backwards")
                self.turn180(carstate, cmd)
                if self.turn180_count == 0:
                    self.turn180_count = 30
                self.turn180_count -= 1

                if abs(carstate.angle) < 15:
                    self.turn180_count = 0
            else:
                if carstate.gear < 0:
                    cmd.gear = 0
                    cmd.brake = 1
                else:
                    self.unstucking = False
                    cmd.steering = self.model.getSteering()
                    self.accelerate(carstate, max(0.1, self.model.getAcceleration()), self.model.getBreak(), cmd)

        if self.data_logger:
            self.data_logger.log(carstate, cmd)

        self.last_raced_dist = carstate.distance_raced

        return cmd

    def logIf(self, carstate, dictstate):
        """
        log state as dangerous place if stuck more than N counts
        """

        if self.log_counter > 100:
            self.log_counter = 0
            return self.saveSharedState(dictstate)

        if carstate.distances_from_edge[0] == -1 and carstate.speed_x < 1:
            print("saving dangerous state")
            print(dictstate)
            print(self.log_counter)
            self.log_counter += 1

    def shift(self, carstate, command):
        """
        method for shifting gears
        """
        if command.gear >= 0 and command.brake < 0.1 and carstate.rpm > 8000:
            command.gear = min(6, command.gear + 1)

        if carstate.rpm < 2500 and command.gear > 1:
            command.gear = command.gear - 1

        if not command.gear:
            command.gear = carstate.gear or 1

    def accelerate(self, carstate, acceleration, brake, command):
        """
        method for accelerating
        """
        if brake > 0.4:
            command.brake = 1

        command.accelerator = acceleration
        command.gear = carstate.gear
        self.shift(carstate, command)

    def isStuck(self, carstate):
        """
        method that returns when car is stuck for more than 10 counts
        """
        if carstate.speed_x < 2 \
                and abs(carstate.distance_from_center) > 0.95 \
                and abs(carstate.angle) > 15 \
                and carstate.angle * carstate.distance_from_center < 0:
            self.stuck_count += 1
        else:
            self.stuck_count = 0
        return self.stuck_count > 10

    def isGoingBack(self, carstate):
        """
        method that notices when car is driving in wrong direction
        """
        if (carstate.distance_raced < self.last_raced_dist
                and carstate.gear > 0 and abs(carstate.angle) > 150):
            self.backwards_count += 1
        else:
            self.backwards_count = 0
        return self.backwards_count > 50
        # return False

    def reverse(self, carstate, command):
        """
        method for driving backwards
        """
        command.accelerator = 1.0
        command.gear = -1
        command.brake = 0.0
        command.clutch = 0.0
        if carstate.angle > 0:
            command.steering = -1
        else:
            command.steering = 1

    def turn180(self, carstate, command):
        """
        method for turning around
        """
        command.accelerator = 0.2
        # command.gear = 2
        if carstate.angle > 0:
            command.steering = 1
        else:
            command.steering = -1

    def loadSharedState(self):
        print("loading state")
        if not os.path.isfile(self.in_comm):
            print("no file")
            return None

        try:
            with open(self.in_comm, 'rb') as f:
                dict_state = pickle.load(f)
        except:
            dict_state = None
            print("not able to load data from other driver")
        print(dict_state)
        return dict_state

    def saveSharedState(self, dict_state):
        print("saving state")
        try:
            with open(self.out_comm, 'wb') as f:
                pickle.dump(dict_state, f)
        except:
            print("not able to write data to other driver")
