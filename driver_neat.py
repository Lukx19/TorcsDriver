import logging

from pytocl.analysis import DataLogWriter
from pytocl.car import State, Command

_logger = logging.getLogger(__name__)

class DriverNeat:
    def __init__(self, model,logdata=True):
        self.data_logger = DataLogWriter() if logdata else None
        self.model = model

    @property
    def range_finder_angles(self):
        """Iterable of 19 fixed range finder directions [deg].

        The values are used once at startup of the client to set the directions
        of range finders. During regular execution, a 19-valued vector of track
        distances in these directions is returned in ``state.State.tracks``.
        """
        return -90, -75, -60, -45, -30, -20, -15, -10, -5, 0, 5, 10, 15, 20, \
            30, 45, 60, 75, 90

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
        self.model.predict(carstate)
        cmd.steering = self.model.getSteering()
        self.accelerate(carstate, max(0.1,self.model.getAcceleration()), cmd)

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
    def accelerate(self, carstate, acceleration, command):

        if acceleration > 0:
            command.accelerator = acceleration
        else:
            command.brake = -1*acceleration
        command.gear = carstate.gear
        self.shift(carstate, command)

