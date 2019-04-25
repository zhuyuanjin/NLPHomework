from preprocess import *
# from SCN import SCN_Net
from torch import nn
import Evaluate
import torch
import gensim 
from SCN_tune import SCN_Net
import numpy as np
import numpy
import torch.nn.functional as F
from cal_p1 import cal_p1
parser = argparse.ArgumentParser(description='train.py')


parser.add_argument('-embedding_size', type = int, default=300)
parser.add_argument('-save_data_dir', default='./tune_data/save_data') 
parser.add_argument('-is_save_data',default='True')
parser.add_argument('-embedding_dir',default='./data/embedding/sgns.weibo.word')
parser.add_argument('-q_file', default='./data/train/file_q.txt')
parser.add_argument('-r_file', default='./data/train/file_r.txt')
parser.add_argument('-neg_file', default='./data/train/neg_res.txt')
parser.add_argument('-save_valid', default='./tune_data/save_valid')
parser.add_argument('-valid_ground_truth', default='./data/valid/valid_ground.txt')
parser.add_argument('-save_test', default='./tune_data/save_test')
parser.add_argument('-store_test', default='./store_result.txt')
opt = parser.parse_args()
print(opt.save_test,opt.save_data_dir)
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
test_set = torch.load(opt.save_valid)['valid']
testloader = dataloader.get_loader(test_set, batch_size=10, shuffle=False, num_workers=2)
def test(n_epoch):
	all_scores = []
	with open(opt.store_test+str(n_epoch),'w') as f:
		for batch in validloader: 	
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
	cal_p1(opt.store_test+str(n_epoch),opt.valid_ground_truth)
	
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


trainloader = dataloader.get_loader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


#build the model
# model = SCN_Net(embedding_matrix,**model_param)
model = SCN_Net(embedding_matrix)
model = model.to(device)
#print(model.device)
#model.cuda()
#model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
n_epoch = 0
accu = 0.0
old_accu = 0.0
for epoch in range(num_epoch):
    total_right = 0
    total_b = 0.0
    for batch in trainloader:
        query, response , neg_response = batch
        n_epoch = n_epoch + 1
        query = torch.cat([query,query])
        response = torch.cat([response,neg_response])
        #print(query,response,query.size(),response.size())

        query = query.long().to(device)
#        print(query.size())
        response = response.long().to(device)
        # if use_cuda:
        	# query = query.cuda()
        	# response = response.cuda()

        #print(n_epoch)
        logit = model(query, response)
        #print(logit.size())
        y_true = torch.cat([torch.ones(batch_size),torch.zeros(batch_size)]).long().to(device)
        #print(y_true.size())
        #print(logit.size(),y_true.size())
        loss_fn = nn.CrossEntropyLoss()
        predict = logit.max(1)[1]
        total_right += predict.eq(y_true).sum().item()
        total_b += response.size()[0]
        loss = loss_fn(logit, y_true)
        if n_epoch%50000==0:
            print('training loss %f',loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if n_epoch%50==0:
            accu = valid()
            test(n_epoch)
            if accu > old_accu:
                print('saving......')
                state= {'accu':accu,'net':model.state_dict()}
                torch.save(state,'./checkpoint/tune_paran')
                old_accu = accu
    print('training accu %f',total_right/total_b)





