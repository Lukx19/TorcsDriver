from model import Model
import torch
import numpy as np
import functools
import operator
from torch.autograd import Variable

class RNNModelSteering(Model):
	def __init__(self,model_file):
		self.model = torch.load(model_file)
		self.max = 0
		self.min = 0
	def predict(self,state):
		data =  [state.speed_x,state.distance_from_center,state.angle / 180] + list(state.distances_from_edge)
		self.output, self.hidden = self.model(Variable(torch.FloatTensor(data)).view(1,-1), Variable(torch.FloatTensor(torch.zeros(1, 50))))
		print(Variable(torch.FloatTensor(data)).view(1,-1))
		self.steering = self.output.data[0].numpy()[0]
		if self.steering > self.max:
			self.max = self.steering
		if self.steering < self.min:
			self.min = self.steering
		print(self.max)
		print(self.min)

