from model import Model
import torch
import numpy as np
import functools
import operator
from torch.autograd import Variable
import csv
import copy
import torch.nn as nn
import torch.nn.functional as F

class RNNModelSteering(Model):

	def __init__(self, model_file, weights):
		self.max = 0
		self.min = 0
		self.log = []
		self.old = 0
		self.weights = weights
		if weights:
			
			linear_in_hid = np.loadtxt(open(model_file + 'linear_in_hid.out', 'r'))
			linear_in_hid_bias = np.loadtxt(open(model_file + 'linear_in_hid_bias.out', 'r'))

			linear_hid_hid = np.loadtxt(open(model_file + 'linear_hid_hid.out', 'r'))
			linear_hid_hid_bias = np.loadtxt(open(model_file + 'linear_hid_hid_bias.out', 'r'))

			linear_hid_out = np.loadtxt(open(model_file + 'linear_hid_out.out', 'r'))
			linear_hid_out_bias = np.loadtxt(open(model_file + 'linear_hid_out_bias.out', 'r'))

			self.linear_in_hid = Variable(torch.cuda.FloatTensor(linear_in_hid))
			self.linear_hid_hid = Variable(torch.cuda.FloatTensor(linear_hid_hid))
			self.linear_hid_out = Variable(torch.cuda.FloatTensor(linear_hid_out))

			self.linear_in_hid_bias = Variable(torch.cuda.FloatTensor(linear_in_hid_bias))
			self.linear_hid_hid_bias = Variable(torch.cuda.FloatTensor(linear_hid_hid_bias))
			self.linear_hid_out_bias = Variable(torch.cuda.FloatTensor(np.array([linear_hid_out_bias])))
		else:
			self.model = torch.load(model_file)

	def predict(self, state):
		state_copy = copy.deepcopy(state)

		# for nn with new data from teachers
		#data = [state.speed_x, state.speed_y, state.speed_z, state.distance_from_center, state_copy.angle / 180] + [state_copy.distances_from_edge[0]] + [state_copy.distances_from_edge[5]] + [state_copy.distances_from_edge[13]] + [state_copy.distances_from_edge[18]]

		# for nn with 4 sensors evenly spaced
		data =  [state_copy.speed_x, state_copy.distance_from_center, state_copy.angle / 180] + [state_copy.distances_from_edge[0]] + [state_copy.distances_from_edge[5]] + [state_copy.distances_from_edge[13]] + [state_copy.distances_from_edge[18]]

		# for nn with 2 sensors
		#data =  [state_copy.speed_x, state_copy.distance_from_center, state_copy.angle / 180] + [state_copy.distances_from_edge[0]] + [state_copy.distances_from_edge[18]]

		# for nn with all sensors
		#data = [state_copy.speed_x, state_copy.angle / 180, state_copy.distance_from_center] + list(state_copy.distances_from_edge)
		
		if self.weights:
			
			x_input = Variable(torch.cuda.FloatTensor(data))
			output = self.forward(x_input)
			if abs(state.distance_from_center) <= 1:
				prediction = 0.01 * output + 0.99 * self.old
			else:
				prediction = 0.2 * output + 0.8 * (0.0 - state.distance_from_center)
			self.steering = prediction
			self.old = prediction
		else:
			#hidden = self.model.init_hidden()
			x_input = Variable(torch.cuda.FloatTensor(data))
			#output, hidden = self.model(x_input.view(-1, 1, len(data)), hidden)
			output = self.model(x_input.view(-1, len(data)))
			prediction = self.old * 0.99 + output.data[0].cpu().numpy()[0] * 0.01
			self.steering = prediction
			self.old = prediction
	
	def forward(self, x_in):
		hidden = torch.matmul(self.linear_in_hid, x_in) + self.linear_in_hid_bias
		hidden = F.tanh(hidden)
		hidden = torch.matmul(self.linear_hid_hid, hidden) + self.linear_hid_hid_bias
		hidden = F.tanh(hidden)
		output = torch.matmul(self.linear_hid_out, hidden) + self.linear_hid_out_bias
		return output.data[0]
		

