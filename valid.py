from preprocess import *
from SCN import SCN_Net
from torch import nn
import torch


result = './result.txt'
stact_dict_file = ''

model_param = {
    'embedding_matrix': torch.randn([434511, 200]),
    'rnn_units': 200,
    'word_embedding_size': 200,
    'max_sentence_len':30
}

#The super paramters
num_epoch = 1000
lr = 1e-4
batch_size = 10
num_workers = 0


# gett the dataoder from MingHan's Code
dicts = initVocabulary('source and target',
                                  opt.q_file,
                                  opt.vocab,
                                  opt.vocab_size)
validset = makeData(opt.q_file, opt.r_file, dicts)

validloader = dataloader.get_loader(validset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

#build the model
model = SCN_Net(**model_param)
model.load_state_dict(stact_dict_file)

f = open(result, 'wb')
for batch in validloader:
    query, response = batch
    logit = model(query, response).squeeze()
    _,indices = torch.sort(logit)
    for i in indices:
        f.write(str(i) + '\n')
    f.write('\n')

f.close()

