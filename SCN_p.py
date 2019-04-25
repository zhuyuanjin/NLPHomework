import torch
from torchvision import *
from torch import nn
import torch.nn.functional as F
from torch.nn import init
import pickle



# class Config():
#     def __init__(self):
#         self.max_num_utterance = 10
#         self.negative_samples = 1  # 抽样一个负例
#         self.max_sentence_len = 50
#         self.word_embedding_size = 200
#         self.rnn_units = 200
#         self.total_words = 434511
#         self.batch_size = 40  # 用80可以跑出论文的结果，现在用论文的参数

device = 'cuda' if torch.cuda.is_available() else 'cpu'
class SCN_Net(nn.Module):
    def __init__(self, embedding_matrix,max_num_utterance = 4,negative_samples = 1,max_sentence_len = 30,word_embedding_size = 300,rnn_units = 200):
        # 标准动作
        super(SCN_Net, self).__init__()
        print(f'this is SCN all share Model')
        # 参数设定
        self.max_num_utterance = max_num_utterance
        self.negative_samples = negative_samples
        self.max_sentence_len = max_num_utterance
        self.word_embedding_size = word_embedding_size
        self.rnn_units = rnn_units
        self.embedding_matrix = embedding_matrix
        # self.total_words = total_words
        # batch_size指的是正例的个数，然后从负例数据集中随机抽config.negative_samples个负例，再和utterance组成一个完整的负例
        # self.batch_size = config.batch_size + config.negative_samples * config.batch_size

        # 需要的模块

        # with open(embedding_file, 'rb') as f:
        #     embedding_matrix = pickle.load(f, encoding="bytes")
        #     assert embedding_matrix.shape == (434511, 200)
        self.word_embedding = nn.Embedding.from_pretrained(self.embedding_matrix)
        # self.word_embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.word_embedding.weight.requires_grad = False

        # 论文用的是单向的GRU，而且需要10个用于utterance
        # 这个版本的模型所有的模块都共享，即utterance都用相同的GRU
        self.utterance_GRU = nn.GRU(self.word_embedding_size, self.rnn_units, bidirectional=False, batch_first=True)
        ih_u = (param.data for name, param in self.utterance_GRU.named_parameters() if 'weight_ih' in name)
        hh_u = (param.data for name, param in self.utterance_GRU.named_parameters() if 'weight_hh' in name)
        for k in ih_u:
            nn.init.orthogonal_(k)
        for k in hh_u:
            nn.init.orthogonal_(k)
        # 用于response的GRU
        self.response_GRU = nn.GRU(self.word_embedding_size, self.rnn_units, bidirectional=False, batch_first=True)
        ih_r = (param.data for name, param in self.response_GRU.named_parameters() if 'weight_ih' in name)
        hh_r = (param.data for name, param in self.response_GRU.named_parameters() if 'weight_hh' in name)
        for k in ih_r:
            nn.init.orthogonal_(k)
        for k in hh_r:
            nn.init.orthogonal_(k)
        # 1、初始化参数的方式要注意
        # 2、参数共享的问题要小心
        # 3、gru不共享参数，conv2d和linear共享参数
        # 正因为conv2d和linear共享参数 只需要定义一个就可以了，gru要一个个定义
        self.conv2d = nn.Conv2d(2, 8, kernel_size=(3, 3))
        conv2d_weight = (param.data for name, param in self.conv2d.named_parameters() if "weight" in name)
        for w in conv2d_weight:
            init.kaiming_normal_(w)

        self.pool2d = nn.MaxPool2d((3, 3), stride=(3, 3))

        self.linear = nn.Linear(648, 50)
        linear_weight = (param.data for name, param in self.linear.named_parameters() if "weight" in name)
        for w in linear_weight:
            init.xavier_uniform_(w)

        self.Amatrix = torch.ones((self.rnn_units, self.rnn_units), requires_grad=True)
        init.xavier_uniform_(self.Amatrix)
        self.Amatrix = self.Amatrix.to(device)
        # 最后一层的gru
        self.final_GRU = nn.GRU(50, self.rnn_units, bidirectional=False, batch_first=True)
        ih_f = (param.data for name, param in self.final_GRU.named_parameters() if 'weight_ih' in name)
        hh_f = (param.data for name, param in self.final_GRU.named_parameters() if 'weight_hh' in name)
        for k in ih_f:
            nn.init.orthogonal_(k)
        for k in hh_f:
            nn.init.orthogonal_(k)
        # final_GRU后的linear层
        self.final_linear = nn.Linear(200, 2)
        final_linear_weight = (param.data for name, param in self.final_linear.named_parameters() if "weight" in name)
        for w in final_linear_weight:
            init.xavier_uniform_(w)

    def forward(self, utterance, response):
        '''
            utterance:(self.batch_size, self.max_num_utterance, self.max_sentence_len)
            response:(self.batch_size, self.max_sentence_len)
        '''
        # (batch_size,10,50)-->(batch_size,10,50,200)
        all_utterance_embeddings = self.word_embedding(utterance)
        #print(self.word_embedding.size(),utterance.size(),all_utterance_embeddings.size())
        response_embeddings = self.word_embedding(response)

        # tensorflow:(batch_size,10,50,200)-->分解-->10个array(batch_size,50,200)
        # pytorch:(batch_size,10,50,200)-->(10,batch_size,50,200)
        all_utterance_embeddings = all_utterance_embeddings.permute(1, 0, 2, 3)

        # (batch_size,10)-->(10,batch_size)
        # 在pytorch里面貌似没啥用 这个是只是为了方便tf里面定义dynamic_rnn用的
        # all_utterance_len = all_utterance_len.permute(1, 0)

        # 先处理response的gru
        response_GRU_embeddings, _ = self.response_GRU(response_embeddings)
        response_embeddings = response_embeddings.permute(0, 2, 1)
        response_GRU_embeddings = response_GRU_embeddings.permute(0, 2, 1)
        matching_vectors = []

        for utterance_embeddings in all_utterance_embeddings:
            matrix1 = torch.matmul(utterance_embeddings, response_embeddings)

            utterance_GRU_embeddings, _ = self.utterance_GRU(utterance_embeddings)
            matrix2 = torch.einsum('aij,jk->aik', utterance_GRU_embeddings, self.Amatrix)
            matrix2 = torch.matmul(matrix2, response_GRU_embeddings)

            matrix = torch.stack([matrix1, matrix2], dim=1)
            # matrix:(batch_size,channel,seq_len,embedding_size)
            conv_layer = self.conv2d(matrix)
            # add activate function
            conv_layer = F.relu(conv_layer)
            pooling_layer = self.pool2d(conv_layer)
            # flatten
            pooling_layer = pooling_layer.view(pooling_layer.size(0), -1)
            matching_vector = self.linear(pooling_layer)
            # add activate function
            matching_vector = F.tanh(matching_vector)
            matching_vectors.append(matching_vector)

        _, last_hidden = self.final_GRU(torch.stack(matching_vectors, dim=1))
        last_hidden = torch.squeeze(last_hidden)
        logits = self.final_linear(last_hidden)

        # use CrossEntropyLoss,this loss function would accumulate softmax
        # y_pred = F.softmax(logits)
        y_pred = logits
        return y_pred


    




