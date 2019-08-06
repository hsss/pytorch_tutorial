#!/usr/bin/env python
import os
import pathlib
import time
import numpy as np
import random

from models.small_model import *
from mfcc_IO import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import zipfile

torch.backends.cudnn.benchmark = True
global_step = 0

def accuracy(output, target, topk=(1, )):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1 / batch_size))
    return res
class VoxCeleb(Dataset):
	def __init__(self, lines, spk_dic, num_frames):
		self.lines = lines
		self.spk_dic = spk_dic
		self.num_frames = num_frames


	def __getitem__(self, index):
		line = self.lines[index]

		tmp = line.strip().split(' ')
		utt = tmp[0] 
			
		pointer = int (tmp[1].split(':')[1])
		src = open(tmp[1].split(':')[0], 'rb')
		src.seek(pointer)
		spec = read_kaldi_mfcc(src)
		src.close()
		
		spk = utt.split('/')[0]
		spk = self.spk_dic[spk]			
		ans = spk
					
		while spec.shape[0] < self.num_frames:
			spec = np.concatenate([spec, spec])
			
		margin = int((spec.shape[0] - self.num_frames)/2)
		
		if margin == 0:
			st_idx = 0
		else:
			st_idx = np.random.randint(0, margin)
			
		ed_idx = st_idx + self.num_frames
		
		data = spec[st_idx:ed_idx,:]
		data = data.reshape((data.shape[0], data.shape[1], 1))

		return np.transpose(data, (2,0,1 )), ans

	def __len__(self):
		return len(self.lines)
		

def zipdir(path, ziph):
	for root, dirs, files in os.walk(path):
		for file in files:
			fn, ext = os.path.splitext(file)
			if ext != '.py': continue

			ziph.write(os.path.join(root, file))

def train(epoch, model, optimizer,  train_criterion, train_loader,  f_results):
		
	device = torch.device('cuda')
	model.train()

	for step, (data, targets) in enumerate(train_loader):	
		data = data.to(device, dtype=torch.float)
		targets = targets.to(device)
		
		optimizer.zero_grad()
					
		outputs, code = model(data)
		loss = nn.CrossEntropyLoss(reduction='mean')(outputs, targets)
	
		loss.backward()
		optimizer.step()
		
		loss_ = loss.item()
		acc = accuracy(outputs, targets)[0].item()
		

		if step % 10 == 0:
			print('Epoch {} Step {}/{} '
						'Loss {:.4f} '
						'Accuracy {:.4f}'.format(
							epoch,
							step,
							len(train_loader),
							loss_,
							acc
						))
			

def main():
	outdir = pathlib.Path('results/small_model')
	outdir.mkdir(exist_ok=True, parents=True)
	
	zipf = zipfile.ZipFile(str(outdir) + '/codes.zip', 'w', zipfile.ZIP_DEFLATED)
	zipdir('./', zipf)
	zipf.close()
	
	f_results = open(str(outdir) + '/results.txt', 'w', buffering = 1)

	seed = 777

	torch.manual_seed(seed)
	np.random.seed(seed)
	random.seed(seed)
	epoch_seeds = np.random.randint(
		np.iinfo(np.int32).max // 2, size=100)

	voxceleb1_all_lines = open('scp/fbank_voxceleb1.scp', 'r').readlines()
	voxceleb1_lines = []

	for line in voxceleb1_all_lines:
		tmp = line.strip().split(' ')
		utt = tmp[0]
		spk = utt.split('/')[0]
		
		if spk[0] == 'E':
			continue
		else:
			voxceleb1_lines.append(line)
	
	spk_dic = {}
	tr_lines = voxceleb1_lines
	for line in tr_lines:
		tmp = line.strip().split(' ')
		utt = tmp[0]
		
		spk = utt.split('/')[0]
		
		if spk not in spk_dic:
			spk_dic[spk] = len(spk_dic)
	print('Number of spks', len(spk_dic))
	
	train_dataset = VoxCeleb(tr_lines, spk_dic, 300)
	train_loader = torch.utils.data.DataLoader(
		train_dataset,
		batch_size=100,
		shuffle=True,
		num_workers=5,
		pin_memory=True,
		drop_last=True
	)
	

	device = torch.device('cuda')
	
	model = Network()
	model.to(device)
	params = model.parameters()
	optimizer = torch.optim.Adam(params)
	
	
	train_criterion = nn.CrossEntropyLoss(reduction='mean')
	for epoch, seed in zip(range(1, 101), epoch_seeds):
		np.random.seed(seed)
		
		train(epoch, model, optimizer, train_criterion, train_loader, f_results)
		torch.save(model.state_dict(), str(outdir) + '/model_%d.wts'%(epoch))

if __name__ == '__main__':
	main()
