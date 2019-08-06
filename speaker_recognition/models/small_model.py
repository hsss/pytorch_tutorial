import torch
import torch.nn as nn
import torch.nn.functional as F


def initialize_weights(module):
	if isinstance(module, nn.Conv2d):
		nn.init.kaiming_normal_(module.weight.data, mode='fan_out')
	elif isinstance(module, nn.BatchNorm2d):
		module.weight.data.fill_(1)
		module.bias.data.zero_()
	elif isinstance(module, nn.Linear):
		module.bias.data.zero_()

class Network(nn.Module):
	def __init__(self):
		super(Network, self).__init__()
				
		self.conv1 = nn.Conv2d(
			in_channels = 1,
			out_channels = 64,
			kernel_size = (7, 7),
			stride= (2,2) )
		
		self.conv2 = nn.Conv2d(	64, 128, (3, 3), (1,1))
		self.conv3 = nn.Conv2d( 128, 256, 3, 1)
		
		self.fc_code = nn.Linear(in_features = 256, out_features = 128)
		self.fc_output = nn.Linear(128, 1211)
		
		self.apply(initialize_weights)

	def forward(self, x):
	
		feature_map = self.conv1(x)
		activated = F.relu(feature_map)
		compressed = F.max_pool2d(activated, 
							kernel_size = (2,2), 
							stride = (2,2))
	

		x = F.max_pool2d(F.relu(self.conv2(compressed)), 2, 2)
	
		x = F.max_pool2d(F.relu(self.conv3(x)), 2, 2)
	

		x = F.adaptive_avg_pool2d(x, output_size=1)
		x = x.view(x.size(0), -1)
	

		code = self.fc_code(x)
	
		output = self.fc_output(code)
	

		return output, code
