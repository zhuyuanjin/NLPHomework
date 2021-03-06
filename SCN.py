import torch
from torchvision import *
from torch import nn
import torch.nn.functional as F
from torch.nn import init
import pickle


class SCN_Net(nn.Module):
    def __init__(self, embedding_matrix, rnn_units=200, word_embedding_size=300, max_sentence_len=30):
        super(SCN_Net, self).__init__()

        ## Super parameters
        self.max_sentence_len = max_sentence_len
        self.word_embedding_size = word_embedding_size
        self.rnn_units = rnn_units #200  ## TODO:ckeck the parameter
        self.embedding_matrix = embedding_matrix

        ##NetWork Parameters
        self.sentence_GRU = nn.GRU(input_size=self.word_embedding_size, hidden_size=self.rnn_units, batch_first=True)
        self.utterance_GRU = nn.GRU(input_size=self.word_embedding_size, hidden_size=self.rnn_units, batch_first=True)
        self.final_GRU = nn.GRU(input_size=50, hidden_size=self.rnn_units, batch_first=True)
        self.fc = nn.Linear(int(self.max_sentence_len / 3) ** 2 * 8, 50)
        self.embedding = nn.Embedding.from_pretrained(self.embedding_matrix)
        self.conv2d = nn.Conv2d(2, 8, kernel_size=(3, 3), padding=1)
        self.pool2d = nn.MaxPool2d((3, 3), stride=(3, 3))
        self.fc_final = nn.Linear(200, 2) #TODO: check the parameter

    def forward(self, utterance_ph, response_ph):


        """
        :param utterance_ph: the utterance withe the shape of [batch_size, max_num_utterance, max_sentence_len]
               response_ph: the response with the shape of [batch_size, max_sentence_len]
        :return: the softmax_logit of each response
        """

        # make and initialize the A_matrix

        A_matrix = torch.ones([self.rnn_units, self.rnn_units])   #TODO:check the meaning of the A_matrix
        init.xavier_normal(A_matrix)

        #get the embeddings of the data: response, utterance
        response_embeddings = self.embedding(response_ph)
        all_utterance_embeddings = self.embedding(utterance_ph)

        #Using GRU to capture the features of response
        response_GRU_embeddings, _ = self.sentence_GRU(response_embeddings)
        #print('response_embeddings', response_embeddings.shape)

        #transpose the response embeding vectors to vertical vector to be multiplied with utterance
        response_embeddings = response_embeddings.permute(0, 2, 1)
        response_GRU_embeddings = response_GRU_embeddings.permute(0, 2, 1)


        matching_vectors = []

        for utterance_embeddings in all_utterance_embeddings.permute(1, 0, 2, 3):
            #get the embeddings of each utterance
            utterance_GRU_embeddings, _ = self.sentence_GRU(utterance_embeddings)

            #print('utterance_GRU_embeddings', utterance_GRU_embeddings.shape)

            #get the matrix1 and matrix2 and the matrix
            # print(utterance_embeddings.device,utterance_embeddings.size(),response_embeddings.device,response_embeddings.size())
            matrix1 = torch.matmul(utterance_embeddings, response_embeddings) # the covariance of utterance embedding asn response embedding

            matrix2 = torch.einsum('aij,jk->aik', [utterance_GRU_embeddings, A_matrix])
            matrix2 = torch.matmul(matrix2, response_GRU_embeddings) #the covariance of response_GRU_embeddings and utterance_GRU_embeddings

            matrix = torch.stack([matrix1, matrix2], dim=1) ## TODO: check the parameters

            ## Using the convolution net to capture the features of the matrix
            conv_layer = F.relu(self.conv2d(matrix))
            pooling_layer = self.pool2d(conv_layer)
            pooling_layer = pooling_layer.view(pooling_layer.size(0), -1)
            matching_vector = F.tanh(self.fc(pooling_layer))
            matching_vectors.append(matching_vector)

        _, last_hidden = self.final_GRU(torch.stack(matching_vectors, dim=1)) ## stack all of the matching_vectors to make a sequence
        last_hidden = torch.squeeze(last_hidden)
        logits = self.fc_final(last_hidden)
        y_pred = logits
        return y_pred






