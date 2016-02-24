import numpy as np 
import h5py
import matplotlib.pyplot as plt

f = h5py.File('train_data.mat','r')
ff = h5py.File('test_data.mat','r')
data = f['data']
data1 = ff['data']

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
    
    # return 2 wavs, the max(abs(eegs)) in a specific period
    # Change the parameters 'begin_wav' and 'duration' to specify the period (unit='seconds') 
    # Set the begin of audio as time 0; default period: from 30s to 40s 
    # the frequence of wavs are transformed to 128 Hz by calculate the variance
    def get_wav_eeg(self, begin_wav=30,duration=10):
        transform_rate = self.f_wav/self.f_eeg
        wav0 = []
        wav1 = []
        for i in range(duration*128):
            begin = int(begin_wav*self.f_wav+i*transform_rate)
            end  = int(begin_wav*self.f_wav+(i+1)*transform_rate)
            wav0.append(np.var(self.wav[0][begin:end])*1000) 
            wav1.append(np.var(self.wav[1][begin:end])*1000) 
            eeg = [self.eeg[i][self.eeg_debut+begin_wav*128:self.eeg_debut+(begin_wav+duration)*128] for i in range(128)] 
        return abs(np.asarray(wav0)),abs(np.asarray(wav1)),np.max(abs(np.asarray(eeg)),axis=0)
   

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


frequency = 44100
cut = 0.05
speech0 = []
speech1 = []
def speech(frequency,cut,cell,side):
    speech = []
    def chunks(l,n):
        for i in xrange(0,len(l),n):
             yield l[i:i+n]
    # Devide list l into n chunks
    for i in cell:
        duration = len(i.wav[side])/frequency
        num_piece = duration/cut
        i_cut = chunks(i.wav[side],int(len(i.wav[side])/num_piece))
        bi = []
        for j in i_cut:
            if np.var(j) >= 0.00002:
                bi.append(1)
            else:
                bi.append(0)
        timing = []
        it = [0]
        counter = 1
        for j in range(len(bi)-1):
            if bi[j] != bi[j+1] and counter < 2:
                it.append(j+1)
                counter += 1
            if bi[j] != bi[j+1] and counter == 2:
                timing.append((it[0]*cut,it[1]*cut))
                it = []
                counter = 0
        speech.append(timing)
    return speech

speech0 = speech(frequency,cut,cell,0)
speech1 = speech(frequency,cut,cell,0)
# each element in speech0/1 is a list of tuples with the
# beginning and ending index of a sentence (0:left, 1:right)

var = [cell[i].var() for i in range(len(data))]
# Create a matrix which stores in each line the variance of each eeg
cell_var = []
for i in var:
    temp = []
    for j in i:
        temp.append(max(abs(j[1]-j[0]),abs(j[1]-j[2])))
    cell_var.append(temp)
selected_egg = [[i for i in range(len(j)) if j[i]> 50] for j in cell_var]
# Create a matrix which stores in each line the index whose value in its varicance(centralized) is greater than 50


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
# Count_eeg counts the occurance of all the eegs who is considered to be relevant to the ongoing of experiment

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

##########################################select feature1, feature2#################################################################
# feature 1 : l2 distance between max(eeg) and wav0 estimated 
# feature 2 : l2 distance between max(eeg) and wav1 estimated by top 30 points,128Hz

feature1=[]
feature2=[]
top_n=30

for k in cell:
    
    wav0,wav1,eeg = k.get_wav_eeg(begin_wav = 10, duration = 40)

    # wav0 approximated by top 30 points
    x0 = np.append(wav0.argsort()[-top_n:][::-1],[0,40*128-1]) 
    y0 = scale(wav0[x0])
    f0 = interp1d(x0, y0)
    
    # wav1 approximated by top 30 points
    x1 = np.append(wav1.argsort()[-top_n:][::-1],[0,40*128-1]) 
    y1 = scale(wav1[x1])
    f1 = interp1d(x1, y1)

    
    x_eeg = np.append(eeg.argsort()[-top_n:][::-1],[0,40*128-1])
    y_eeg = scale(eeg[x_eeg])
    f_eeg = interp1d(x_eeg,y_eeg)
    
    # l2 distance
    x_new=np.linspace(10,40*128-10,1000)
    feature1.append(sum((f0(x_new)-f_eeg(x_new))**2))
    feature2.append(sum((f1(x_new)-f_eeg(x_new))**2))

print 'for first 30 cells (tgt=0)',np.mean(np.asarray(feature1[:30]))
print 'for last 30 cells (tgt=1)',np.mean(np.asarray(feature1[30:]))

plt.figure()
plt.subplot(121)
plt.boxplot(np.asarray(feature1[:30]))
plt.ylim(0,6000)
plt.subplot(122)
plt.boxplot(np.asarray(feature1[30:]))
plt.ylim(0,6000)
plt.show()

print 'for first 30 cells (tgt=0)',np.mean(np.asarray(feature2[:30]))
print 'for last 30 cells (tgt=1)',np.mean(np.asarray(feature2[30:]))

plt.figure()
plt.subplot(121)
plt.boxplot(np.asarray(feature2[:30]))
plt.subplot(122)
plt.boxplot(np.asarray(feature2[30:]))
plt.show()


##########################################select feature1, feature2#################################################################

