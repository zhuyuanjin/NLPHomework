from preprocess import *
# from SCN import SCN_Net
from torch import nn
import Evaluate
import torch
import gensim 
from SCN_p import SCN_Net
import numpy as np
import numpy
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='train.py')


parser.add_argument('-embedding_size', type = int, default=300)
parser.add_argument('-save_data_dir', default='./tune_data2/save_data') 
parser.add_argument('-is_save_data',default='True')
parser.add_argument('-embedding_dir',default='./data/embedding/sgns.weibo.word')
parser.add_argument('-q_file', default='./data/train/file_q.txt')
parser.add_argument('-r_file', default='./data/train/file_r.txt')
parser.add_argument('-neg_file', default='./data/train/neg_res.txt')
parser.add_argument('-save_valid', default='./save_valid')
parser.add_argument('-valid_ground_truth', default='./data/valid/valid_ground.txt')
parser.add_argument('-save_test', default='./save_test')
parser.add_argument('-store_test', default='./store_result.txt')
parser.add_argument('-resume', default=True)
opt = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#file_path
valid_set = torch.load(opt.save_valid)['valid']
validloader = dataloader.get_loader(valid_set, batch_size=10, shuffle=False, num_workers=2)
labels = np.loadtxt(opt.valid_ground_truth)
def valid():
        all_scores = []
        for mini_batch in validloader:
                query ,response = mini_batch[0],mini_batch[1]
                query = query.long().to(device)
                response = response.long().to(device)
                logit = model(query, response).squeeze()
                scores = F.softmax(logit,0).cpu().detach().numpy()
                all_scores.append(np.argmax(scores[:,1]))
        #all_scores = np.concatenate(all_scores,axis=0)
        print(np.size(all_scores),np.size(labels))
        return Evaluate.ComputeR10_1(all_scores,labels)
test_set = torch.load(opt.save_test)['test']
testloader = dataloader.get_loader(test_set, batch_size=10, shuffle=False, num_workers=2)
def test(n_epoch):
	all_scores = []
	with open(opt.store_test+str(n_epoch),'w') as f:
		for batch in testloader: 	
			query, response =batch[0],batch[1]
			query = query.long().to(device)
			response = response.long().to(device)
			logit = model(query,response).squeeze()
			scores = F.softmax(logit,0).cpu().detach().numpy()
			index = np.argsort(-scores[:,1])
			all_scores.append(np.argsort(-scores[:,1]))		
			for i in range(10):
				f.write(str(index[i]))
				f.write('\n')
			f.write('\n')
		all_scores = np.concatenate(all_scores,axis=0)
	
model_param = {
    # 'embedding_matrix': torch.randn([200004, 300]),
    'rnn_units': 200,
    'word_embedding_size': opt.embedding_size
}

#The super paramters
num_epoch = 100
lr = 1e-4
batch_size = 40
num_workers = 2

if opt.is_save_data is not None:
	save_data = torch.load(opt.save_data_dir)
	dicts = save_data['dicts']	

	embedding_matrix = torch.randn(dicts.size(),opt.embedding_size)
	# print(dicts.size())
	gensim_model = gensim.models.KeyedVectors.load_word2vec_format(opt.embedding_dir)
	print('load the existing embedding from %s' % opt.embedding_dir)
	total = 0
	for i in range(dicts.size()):
		if dicts.idxToLabel[i] in gensim_model.wv.vocab:
			embedding_matrix[i,:] = torch.FloatTensor(gensim_model[dicts.idxToLabel[i]])
			total = total + 1
	print(total)
	trainset = save_data['train']
else:

# gett the dataoder from MingHan's Code
	dicts = initVocabulary('source and target',
	                                  opt.q_file,
	                                  opt.vocab,
	                                  opt.vocab_size)
	trainset = makeData(opt.q_file, opt.r_file, dicts,opt.neg_file)
	embedding_matrix = torch.randn(dicts.size(),opt.embedding_size)



#build the model
# model = SCN_Net(embedding_matrix,**model_param)
model = SCN_Net(embedding_matrix)
if opt.resume:
    print('loding model from. ...')
    checkpoint = torch.load('./checkpoint/param')
    model.load_state_dict(checkpoint['net'])
model = model.to(device)
#print(model.device)
if __name__ == '__main__':
	print('predicting...')
	test(1000)
