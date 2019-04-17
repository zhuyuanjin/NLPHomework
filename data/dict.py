#coding:utf-8

'''
处理数据、分词
'''
import torch
import pickle 
PAD = 0
UNK = 1
BOS = 2
EOS = 3

PAD_WORD = '<blank>' 
UNK_WORD = '<unk> '
BOS_WORD = '<s>'
EOS_WORD = '</s>'
SPA_WORD = ' '

def flatten(l): 
    for el in l:
        if hasattr(el, "__iter__"):
            for sub in flatten(el):
                yield sub
        else:
            yield el

class Dict(object):
    def __init__(self, data=None):
        self.idxToLabel = {}
        self.labelToIdx = {}
        self.frequencies = {}
        # self.lower = lower
        self.special = [] 

        if data != None:
        	if type(data) != str:
        		self.add_specials(data)



    def add(self,utter,idx=None):
    	if idx is not None:
    		self.idxToLabel[idx] = utter
    		self.labelToIdx[utter] = idx
    	else:
    		if utter in self.labelToIdx:
    			idx = self.labelToIdx[utter]
    		else:
    			idx = len(self.idxToLabel)
    			self.idxToLabel[idx] = utter
    			self.labelToIdx[utter] = idx
    	return idx

    def add_special(self,label):
    	idx = self.add(label)
    	self.special.append(idx)	

    def add_specials(self,data):
    	for label in data:
    		self.add_special(label)

    def lookup(self,label,default=1):
    	try:
    		return self.labelToIdx[label]
    	except KeyError:
    		return default

    def convertToIdx(self, labels, unkWord, bosWord=None, eosWord=None):
        vec = []

        if bosWord is not None:
            vec += [self.lookup(bosWord)]

        unk = self.lookup(unkWord)
        vec += [self.lookup(label, default=unk) for label in labels]

        if eosWord is not None:
            vec += [self.lookup(eosWord)]

        vec = [x for x in flatten(vec)]

        # return torch.LongTensor(vec)
        return vec

    def convertToLabel(self,idx,stop=2):
        labels = []
        # print(idx)
        for i in idx:
            # print(i)
            if i == stop:
                break
            else:
                # print(int(i))
                labels.append(self.idxToLabel[int(i)])
        return labels


    def size(self):
    	return len(self.idxToLabel)

    def loadFile(self, filename):
        with open(filename,'rb') as file:
            self.idxToLabel = pickle.load(file)

        for i in self.idxToLabel:
            self.labelToIdx[self.idxToLabel[i]] = i

    # Write entries to a file.
    def writeFile(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.idxToLabel,file)




