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
def getErrorRate(scoreList, threshold=0):

    TRNum = 0    
    FRNum = 0    
    FANum = 0    
    TANum = 0    
    
    for element in scoreList:
        featureType, logLikelihoodRatio = element
        
        if logLikelihoodRatio < threshold:
            if featureType == 'trueSpeaker':
                FRNum += 1
            elif featureType == 'imposter':
                TRNum += 1
       
        else:        
            if featureType == 'trueSpeaker':
                TANum += 1
            elif featureType == 'imposter':
                FANum += 1
                
    imposterNum = TRNum + FANum
    trueSpeakerNum = FRNum + TANum
    
    FARate = float(FANum) / float(imposterNum) * 100.        
    FRRate = float(FRNum) / float(trueSpeakerNum) * 100.     
    
    correctNum = TRNum + TANum
    wrongNum = FANum + FRNum
    
    return FARate, FRRate, correctNum, wrongNum

def calculateEER(scoreList):
    boundary = 0.001            
    repeatNum = 500                
    
    left = -100.
    right = 100.
    
    
    for index in range(repeatNum): #@UnusedVariable
                    
        
        middle = (left + right) / 2.0
        
        FARate, FRRate, correctNum, wrongNum = getErrorRate(scoreList, threshold=middle)
        errorRate = FRRate - FARate
        
        if abs(errorRate) <= boundary:
            return middle
        
        
        if errorRate < 0:
            left = middle
        else:
            right = middle
            
    return middle

class VoxCeleb_test(Dataset):
	def __init__(self, val_lines, val_key):
		self.val_lines = val_lines
		self.val_key = val_key
		

	def __getitem__(self, index):
		line = self.val_lines[index]

		tmp = line.strip().split(' ')
		utt = tmp[0] 
			
		pointer = int (tmp[1].split(':')[1])
		src = open(tmp[1].split(':')[0], 'rb')
		src.seek(pointer)
		spec = read_kaldi_mfcc(src)
		src.close()
		spec = spec.reshape((spec.shape[0], spec.shape[1], 1))
		
		return np.transpose(spec, (2,0,1 )), self.val_key[index]

	def __len__(self):
		return len(self.val_lines)
		

def zipdir(path, ziph):
	for root, dirs, files in os.walk(path):
		for file in files:
			fn, ext = os.path.splitext(file)
			if ext != '.py': continue

			ziph.write(os.path.join(root, file))

def main():
	
	seed = 777

	torch.manual_seed(seed)
	np.random.seed(seed)
	random.seed(seed)
	epoch_seeds = np.random.randint(
		np.iinfo(np.int32).max // 2, size=100)
	
	tst_lines_voxceleb1 = open('scp/voxceleb1_test.txt', 'r').readlines()

	voxceleb1_all_lines = open('scp/fbank_voxceleb1.scp', 'r').readlines()
	voxceleb1_lines = []
	voxceleb1_val_key = []
	for line in voxceleb1_all_lines:
		tmp = line.strip().split(' ')
		utt = tmp[0]
		spk = utt.split('/')[0]
		
		if spk[0] == 'E':
			voxceleb1_lines.append(line)
			voxceleb1_val_key.append(utt)
	
	spk_dic = {}
	tst_lines = voxceleb1_lines
	

	for line in tst_lines:
		tmp = line.strip().split(' ')
		utt = tmp[0]
		
		spk = utt.split('/')[0]
		
		if spk not in spk_dic:
			spk_dic[spk] = len(spk_dic)
			
	print('Number of spks', len(spk_dic))
	
	test_dataset = VoxCeleb_test(tst_lines, voxceleb1_val_key)
	test_loader = torch.utils.data.DataLoader(
		test_dataset,
		batch_size=1,
		num_workers=1,
		shuffle=False,
		pin_memory=True,
		drop_last=False,
	)

	device = torch.device('cuda')
	
	model = Network()
	model.to(device)
	
	state_dict = torch.load('results/small_model/model_20.wts')
	model.load_state_dict(state_dict, strict=True)
	
	model.eval()
	_trial_str = ['imposter', 'trueSpeaker']
	with torch.no_grad():
		
		e_dic = {}
				
		for	data, targets in test_loader:
			data = data.to(device, dtype=torch.float)
			
			
			outputs, code = model(data)
			e_dic[targets[0]] = np.array(code.cpu(), np.float32)[0]
		score_list = []
	
		for line in tst_lines_voxceleb1:
			tmp = line.strip().split(' ')
			score = np.dot(e_dic[tmp[1]], e_dic[tmp[2]])
			score_list.append([_trial_str[int(tmp[0])],score])
		threshold = calculateEER(score_list)
		FARate, FRRate, correctNum, wrongNum = getErrorRate(score_list, threshold = threshold)
		val_eer = np.mean([FARate, FRRate])
		
	print(val_eer)

if __name__ == '__main__':
	main()
