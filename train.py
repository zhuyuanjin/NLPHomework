from preprocess import *
from SCN import SCN_Net
from torch import nn
import torch
import gensim 

parser = argparse.ArgumentParser(description='train.py')


parser.add_argument('-embedding_size', type = int, default=300)
parser.add_argument('-save_data_dir', default='./save_data') 
parser.add_argument('-is_save_data',default='True')
parser.add_argument('-embedding_dir',default='./data/embedding/sgns.weibo.word')
parser.add_argument('-q_file', default='./data/train/file_q.txt')
parser.add_argument('-r_file', default='./data/train/file_r.txt')
parser.add_argument('-neg_file', default='./data/train/neg_res.txt')

opt = parser.parse_args()

#file_path


model_param = {
    # 'embedding_matrix': torch.randn([200004, 300]),
    'rnn_units': 200,
    'word_embedding_size': opt.embedding_size
}

#The super paramters
num_epoch = 10
lr = 1e-4
batch_size = 4
num_workers = 2

if opt.is_save_data is not None:
	save_data = torch.load(opt.save_data_dir)
	dicts = save_data['dicts']	

	embedding_matrix = torch.randn(dicts.size(),opt.embedding_size)
	# print(dicts.size())
	gensim_model = gensim.models.KeyedVectors.load_word2vec_format(opt.embedding_dir)
	for i in range(dicts.size()):
		if dicts.idxToLabel[i] in gensim_model.wv.vocab:
			embedding_matrix[i,:] = gensim_model[dicts.idxToLabel[i]]
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
model = SCN_Net(embedding_matrix,**model_param)
opt = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(num_epoch):
    for batch in trainloader:
        query, response , neg_response = batch

        query = query.long()
        response = response.long()
        logit = model(query, response,neg_response)

        y_true = torch.ones(batch_size).long()
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logit, y_true)
        print(loss.item())
        opt.zero_grad()
        loss.backward()
        opt.step()
