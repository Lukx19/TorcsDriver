import abc
class Model:
    steering = 0
    breaking = 0
    acceleration = 1

    def getSteering(self)->float:
        """ return steering value in range [-1,1]"""
        return self.steering

    def getBreak(self)->int:
        """ return breaking value [0,1]"""
        return self.breaking


    def getAcceleration(self)->float:
        """ return Acceleration [0,]"""
        return self.acceleration

    @abc.abstractproperty
    def predict(self,carstate):
        """ calls network and predicts values for breaking,acceleration and steering"""
