import torch
from torchvision import *
from torch import nn
import torch.nn.functional as F
from torch.nn import init


class SCN_Net(nn.modules):
    def __init__(self, embedding_matrix):
        super(SCN_Net, self).__init__()

        ## Super parameters
        self.max_num_utterance = 10
        self.negative_samples = 1     ##What is it?
        self.max_sentence_len = 50
        self.word_embedding_size = 200
        self.rnn_units = 200  ## What is this?
        self.total_words = 434511
        self.batch_size = 40

        ## Input of the
        self.embedding_matrix = embedding_matrix


        ##NetWork Parameters
        self.sentence_GRU = nn.GRU(input_size=self.word_embedding_size, hidden_size=self.rnn_units,)
        self.final_GRU = nn.GRU(input_size=50, hidden_size=self.rnn_units)
        self.fc = nn.Linear(16 * 16 * 8, 50)
        self.embedding = nn.Embedding.from_pretrained(self.embedding_matrix)
        self.conv2d = nn.Conv2d(2, 8, kernel_size=(3, 3))
        self.pool2d = nn.MaxPool2d((3, 3), stride=(3, 3))
        self.fc_final = nn.Linear(200, 2) #TODO: check the parameter

    def forward(self, utterance_ph, response_ph):


        """
        :param x:
        :return:
        """
        A_matrix = torch.ones([self.rnn_units, self.rnn_units])
        init.xavier_normal(A_matrix)
        response_embeddings = self.embedding(response_ph)
        all_utterance_embeddings = self.embedding(utterance_ph.permute(1, 0, 2)) ##make it in the shape of [max_num_utterance, batch_size, max_sentence_len]
        response_GRU_embeddings, _ = self.sentence_GRU(response_embeddings)

        response_embeddings = response_embeddings.permute(0, 2, 1)
        matching_vectors = []
        response_GRU_embeddings = response_GRU_embeddings.permute(0,2,1)
        for utterance_embeddings in all_utterance_embeddings:
            matrix1 = torch.matmul(utterance_embeddings, response_embeddings)
            utterance_GRU_embeddings, _ = self.sentence_GRU(utterance_embeddings)

            matrix2 = torch.einsum('aij, ij->aik', utterance_GRU_embeddings, A_matrix)
            matrix2 = torch.matmul(matrix2, response_GRU_embeddings)
            matrix = torch.stack([matrix1, matrix2], dim=1) ## TODO: check the parameters
            conv_layer = F.relu(self.conv2d(matrix))
            pooling_layer = self.pool2d(conv_layer)
            pooling_layer = pooling_layer.view(pooling_layer.size(0), -1)
            matching_vector = F.tanh(self.fc(pooling_layer))
            matching_vectors.append(matching_vector)
        _, last_hidden = self.final_GRU(torch.stack(matching_vectors, dim=1))
        last_hidden = torch.squeeze(last_hidden)
        logits = self.fc_final(last_hidden)
        y_pred = F.softmax(logits)
        return y_pred







