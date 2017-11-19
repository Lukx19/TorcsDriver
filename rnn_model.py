from model import Model
import torch
import numpy as np
import functools
import operator

class RNNModelSteering(Model):
	def __init__(self,model_file):
		self.model = torch.load(model_file)
	def predict(self,state):
		data = [state.speed_x,state.distance_from_start,state.angle,state.distances_from_edge]
		data = functools.reduce(operator.concat, data)
		self.steering = self.model(torch.from_numpy(np.array(data)))
