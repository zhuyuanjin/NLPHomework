import torch
import pickle 
import argparse
from data.dict import *
from  SCN_Net import *
import gensim
parser = argparse.ArgumentParser(description='preprocess.py')


parser.add_argument('-embedding_size', type = int, default=300)
parser.add_argument('-save_data_dir', default='./save_data') 
parser.add_argument('-embedding_dir',default='./data/embedding/sgns.weibo.word')

opt = parser.parse_args()

def main():

	save_data = torch.load(opt.save_data_dir)
	dicts = save_data['dicts']
	embedding_matrix = torch.randn(dicts.size(),opt.embedding_size)
	print(dicts.size())
	gensim_model = gensim.models.KeyedVectors.load_word2vec_format(opt.embedding_dir)
	for i in range(dicts.size()):
		if dicts.idxToLabel[i] in gensim_model.wv.vocab:
			embedding_matrix[i,:] = gensim_model[dicts.idxToLabel[i]]
		# else:
			# pass
	print(embedding_matrix)
	model = SCN_Net(embedding_matrix)

if __name__ == '__main__':
	main()

