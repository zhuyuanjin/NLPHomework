from preprocess import *
from SCN import SCN_Net
from torch import nn
import torch


#file_path


model_param = {
    'embedding_matrix': torch.randn([434511, 200]),
    'rnn_units': 200,
    'word_embedding_size': 200,
    'max_sentence_len':30
}

#The super paramters
num_epoch = 1000
lr = 1e-4
batch_size = 4
num_workers = 0


# gett the dataoder from MingHan's Code
dicts = initVocabulary('source and target',
                                  opt.q_file,
                                  opt.vocab,
                                  opt.vocab_size)
trainset = makeData(opt.q_file, opt.r_file, dicts)

trainloader = dataloader.get_loader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


#build the model
model = SCN_Net(**model_param)
opt = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(num_epoch):
    for batch in trainloader:
        query, response = batch

        query = query.long()
        response = response.long()
        logit = model(query, response)

        y_true = torch.ones(batch_size).long()
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logit, y_true)
        print(loss.item())
        opt.zero_grad()
        loss.backward()
        opt.step()

