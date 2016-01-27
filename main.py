import numpy as np 
import h5py
import matplotlib.pyplot as plt

f = h5py.File('train_data.mat','r')
refs = f['#refs#']
data = f['data']
cell = f[data[28][0]]

class cell(object):
	def __init__(self,cell):
		self.tgt = cell['tgt'][...][0][0]
		# 1 tgt
		self.id = [cell['id'][...][i][0] for i in range(len(cell['id'][...]))]
		# 2 id
		self.f_eeg = cell['fsample/eeg'][...][0][0]
		self.f_wav = cell['fsample/wav'][...][0][0]
		# 3 fsample
		self.eeg = f[cell['eeg'][...][0][0]][...]
		# 4 eeg
		self.wav = f[cell['wav'][...][0][0]][...]
		# 5 wav
		self.sample = cell['event/eeg/sample'][...][0]
		length = len(cell['event/eeg/value'][...][0])
		self.value = []
		for i in range(length):
			if all(f[cell['event/eeg/value'][...][0][i]][...]) == all([0,0]):
				self.value.append(0.)
			else:
				self.value.append(f[cell['event/eeg/value'][...][0][i]][...][0][0])
		# 6 event
	def var(self):
		debut = a.sample[a.value.index(200.)]
		fin = a.sample[a.value.index(150.0)]
		var = []
		for i in range(len(a.eeg)):
			it = []
			it.append(np.var(a.eeg[i][:debut]))
			it.append(np.var(a.eeg[i][debut:fin]))
			it.append(np.var(a.eeg[i][fin:]))
			var.append(it)
		return var

a = cell(f[data[57][0]])
var = a.var()
print np.shape(var)
for i in var:
	print i
##############################################################################


