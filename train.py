from data.dataloader import *
from SCN import SCN_Net
from torch import nn
import torch

embedding_file = ''
dataloader = None
model = SCN_Net(embedding_file)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for batch in dataloader:
    query, response = batch
    utterance = query[:, :-1, :]
    response = query[:, -1, :].squeeze()
    ground_truth = torch.ones(batch_size).
    logit = model(utterance, response)
    loss_fn = nn.NLLLoss()
    loss = loss_fn(logit, ground_truth)
    opt.zero_grad()
    loss.backward()
    opt.step()

