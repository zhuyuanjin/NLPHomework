import torch
import torch.utils.data as torch_data
import os
# import data.utils

class dataset(torch_data.Dataset):

    def __init__(self, query,response):

        self.query = query
        self.response = response

    def __getitem__(self, index):
        return self.query[index], self.response[index]

    def __len__(self):
        return len(self.query)


def load_dataset(path):
    pass

def save_dataset(dataset, path):
    if not os.path.exists(path):
        os.mkdir(path)


def padding(data):
    #data.sort(key=lambda x: len(x[0]), reverse=True)
    query, response = zip(*data)

    q_len = [len(s) for s in query]

    q_pad = torch.zeros(len(query), 50).long()
    for i, s in enumerate(query):
        end = q_len[i]
        q_pad[i, :end] = s[:end]

    r_len = [len(s) for s in response]
    print(max(r_len))
    print(query,q_pad)
    r_pad = torch.zeros(len(response), 50).long()
    for i, s in enumerate(response):
        end = r_len[i]
        r_pad[i, :end] = s[:end]
    #tgt_len = [length-1 for length in tgt_len]

    #return src_pad.t(), src_len, tgt_pad.t(), tgt_len
    return q_pad.t(), r_pad.t()


def get_loader(dataset, batch_size, shuffle, num_workers):

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=padding)
    return data_loader