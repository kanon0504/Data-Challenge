import numpy as np 
import h5py
import matplotlib.pyplot as plt

f = h5py.File('train_data.mat','r')
ff = h5py.File('test_data.mat','r')
data = f['data']
data1 = ff['data']
fqc_wav = 44100
fqc_eeg = 128.
cut = .05
para = 1.5

class cell(object):
    def __init__(self,cell):
        self.tgt = cell['tgt'][...][0][0]
        # 1 tgt
        self.id = [cell['id'][...][i][0] for i in range(len(cell['id'][...]))]
        # 2 id
        self.f_eeg = cell['fsample/eeg'][...][0][0]
        self.f_wav = cell['fsample/wav'][...][0][0]
        # 3 fsample
        self.wav = f[cell['wav'][...][0][0]][...]
        # 4 wav
        self.sample = cell['event/eeg/sample'][...][0]
        length = len(cell['event/eeg/value'][...][0])
        self.value = []
        for i in range(length):
            if all(f[cell['event/eeg/value'][...][0][i]][...]) == all([0,0]):
                self.value.append(0.)
            else:
                self.value.append(f[cell['event/eeg/value'][...][0][i]][...][0][0])
        # 5 event
        self.eeg = f[cell['eeg'][...][0][0]][...]
        debut = self.sample[self.value.index(200.)]
        fin = self.sample[self.value.index(150.0)]
        self.eeg_act = [i[debut:fin] for i in self.eeg]
        # 6 eeg
    def var(self):
        debut = self.sample[self.value.index(200.)]
        fin = self.sample[self.value.index(150.0)]
        var = []
        for i in range(len(self.eeg)):
            it = []
            it.append(np.var(self.eeg[i][:debut]))
            it.append(np.var(self.eeg[i][debut:fin]))
            it.append(np.var(self.eeg[i][fin:]))
            var.append(it)
        return var

class cell1(object):
    def __init__(self,cell):
        self.id = [cell['id'][...][i][0] for i in range(len(cell['id'][...]))]
        # 2 id
        self.eeg = ff[cell['eeg'][...][0][0]][...]
        # 4 eeg
        self.wav = ff[cell['wav'][...][0][0]][...]
        # 5 wav

cell = [cell(f[data[i][0]]) for i in range(len(data))]
cell1 = [cell1(ff[data1[i][0]]) for i in range(len(data1))]


speech0 = []
speech1 = []
def speech(fqc_wav,cut,cell,side):
    speech = []
    def chunks(l,n):
        for i in xrange(0,len(l),n):
             yield l[i:i+n]
    # Devide list l into n chunks
    for i in cell:
        duration = len(i.wav[side])/fqc_wav
        num_piece = duration/cut
        i_cut = chunks(i.wav[side],int(len(i.wav[side])/num_piece))
        bi = []
        for j in i_cut:
            if np.var(j) >= 0.00013:
                bi.append(1)
            else:
                bi.append(0)
        timing = []
        it = []
        counter = 0
        for j in range(len(bi)-1):
            if bi[j] == 0 and bi[j] != bi[j+1] and counter == 0:
                it.append(j+1)
                counter += 1
            if bi[j] == 1 and bi[j] != bi[j+1] and counter == 1:
                it.append(j)
                timing.append((round(it[0]*cut,2),round(it[1]*cut,2)))
                it = []
                counter = 0
        speech.append(timing)
    return speech

speech0 = speech(fqc_wav,cut,cell,0)
speech1 = speech(fqc_wav,cut,cell,1)


# each element in speech0/1 is a list of tuples with the
# beginning and ending index of a sentence (0:left, 1:right)

################################### select eeg #######################################
var = [cell[i].var() for i in range(len(data))]
# Create a matrix which stores in each line the variance of each eeg
cell_var = []
for i in var:
    temp = []
    for j in i:
        temp.append(j[1]/j[0]+j[2]/j[1])
    cell_var.append(temp)

selected_egg = []
for i in cell_var:
    index = [i.index(j) for j in list(reversed(sorted(i)))]
    selected_egg.append(index[:3])
# Create a matrix which stores in each line the index whose value 
# in its varicance(centralized) is greater than 50
s = []
for i in selected_egg:
    s += i
candidates = set(s)
dic = {}
for i in candidates:
    dic[i] = s.count(i)
selected_egg = []
for i in range(4):
    selected_egg.append(max(dic.iteritems(), key=operator.itemgetter(1))[0])
    del dic[max(dic.iteritems(), key=operator.itemgetter(1))[0]]
################################### select eeg #######################################



#### Define a spike in eeg and, count the number of spikes in a give time range ###
def get_spikes(cell,wav_cut,para,selected_egg):
    ratio = []
    for i in selected_egg:
        eeg = abs(cell.eeg_act[i])
        threshold = para*np.mean(eeg)
        total_spikes = len([i for i in eeg if i>threshold])
        counter = 0.
        for j in wav_cut:
            segment = eeg[j[0]*fqc_eeg:j[1]*fqc_eeg]
            counter += len([i for i in segment if i> para*np.mean(segment)])
        ratio.append(counter/total_spikes)
        counter = 0.
    return ratio


spikes0 = []
spikes1 = []
for i in range(len(cell)):
    spikes0.append(get_spikes(cell[i],speech0[i],para,selected_egg))
    spikes1.append(get_spikes(cell[i],speech1[i],para,selected_egg))
#### Define a spike in eeg and, count the number of spikes in a give time range ###




merge_eeg_0 = []
merge_eeg_1 = []
for i in range(len(selected_egg)):
    if cell[i].tgt == 0:
        merge_eeg_0 += selected_egg[i]
    else:
        merge_eeg_1 += selected_egg[i]
count_eeg_0 = {}
count_eeg_1 = {}
for i in set(merge_eeg_0):
    count_eeg_0[i] = merge_eeg_0.count(i)
for i in set(merge_eeg_1):
    count_eeg_1[i] = merge_eeg_1.count(i)
# Count_eeg counts the occurance of all the eegs who is 
# considered to be relevant to the ongoing of experiment

print count_eeg_0
print count_eeg_1
c0 = []
c1 = []
for i in range(128):
    if i in count_eeg_0:
        c0.append(count_eeg_0[i])
    else:
        c0.append(0)
for i in range(128):
    if i in count_eeg_1:
        c1.append(count_eeg_1[i])
    else:
        c1.append(0)
plt.plot(c0,'g')
plt.plot(c1,'r')
plt.show()
