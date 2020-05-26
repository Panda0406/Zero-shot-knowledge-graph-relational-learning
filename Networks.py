# -*- coding: utf-8 -*-
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from modules import *
from args import read_options
from torch.autograd import Variable
from spectral_norm import spectral_norm

class Extractor(nn.Module):
    """
    Matching metric based on KB Embeddings
    """
    def __init__(self, embed_dim, num_symbols, embed=None):
        super(Extractor, self).__init__()
        self.embed_dim = int(embed_dim)
        self.pad_idx = num_symbols
        self.symbol_emb = nn.Embedding(num_symbols + 1, embed_dim, padding_idx=num_symbols)
        self.num_symbols = num_symbols

        self.gcn_w = nn.Linear(self.embed_dim, int(self.embed_dim/2))
        self.gcn_b = nn.Parameter(torch.FloatTensor(self.embed_dim))

        self.fc1 = nn.Linear(self.embed_dim, int(self.embed_dim/2))
        self.fc2 = nn.Linear(self.embed_dim, int(self.embed_dim/2))

        self.dropout = nn.Dropout(0.2)
        self.dropout_e = nn.Dropout(0.2)

        self.symbol_emb.weight.data.copy_(torch.from_numpy(embed))

        self.symbol_emb.weight.requires_grad = False

        d_model = self.embed_dim * 2
        self.support_encoder = SupportEncoder(d_model, 2*d_model, dropout=0.2)
        #self.query_encoder = QueryEncoder(d_model, process_steps)

    def neighbor_encoder(self, connections, num_neighbors):
        '''
        connections: (batch, 200, 2)
        num_neighbors: (batch,)
        '''
        num_neighbors = num_neighbors.unsqueeze(1)
        entities = connections[:,:,1].squeeze(-1)
        ent_embeds = self.dropout(self.symbol_emb(entities)) # (batch, 50, embed_dim)
        concat_embeds = ent_embeds

        out = self.gcn_w(concat_embeds)
        out = torch.sum(out, dim=1) # (batch, embed_dim)
        out = out / num_neighbors
        return out.tanh()

    def entity_encoder(self, entity1, entity2):
        entity1 = self.dropout_e(entity1)
        entity2 = self.dropout_e(entity2)
        entity1 = self.fc1(entity1)
        entity2 = self.fc2(entity2)
        entity = torch.cat((entity1, entity2), dim=-1)
        return entity.tanh() # (batch, embed_dim)


    def forward(self, query, support, query_meta=None, support_meta=None):
        '''
        query: (batch_size, 2)
        support: (few, 2)
        return: (batch_size, )
        '''
        query_left_connections, query_left_degrees, query_right_connections, query_right_degrees = query_meta
        support_left_connections, support_left_degrees, support_right_connections, support_right_degrees = support_meta

        query_e1 = self.symbol_emb(query[:,0]) # (batch, embed_dim)
        query_e2 = self.symbol_emb(query[:,1]) # (batch, embed_dim)
        query_e = self.entity_encoder(query_e1, query_e2)

        support_e1 = self.symbol_emb(support[:,0]) # (batch, embed_dim)
        support_e2 = self.symbol_emb(support[:,1]) # (batch, embed_dim)
        support_e = self.entity_encoder(support_e1, support_e2)

        query_left = self.neighbor_encoder(query_left_connections, query_left_degrees)
        query_right = self.neighbor_encoder(query_right_connections, query_right_degrees)

        support_left = self.neighbor_encoder(support_left_connections, support_left_degrees)
        support_right = self.neighbor_encoder(support_right_connections, support_right_degrees)
        
        query_neighbor = torch.cat((query_left, query_e,  query_right), dim=-1) # tanh
        support_neighbor = torch.cat((support_left, support_e, support_right), dim=-1) # tanh

        support = support_neighbor
        query = query_neighbor

        support_g = self.support_encoder(support) # 1 * 100
        query_g = self.support_encoder(query)

        support_g = torch.mean(support_g, dim=0, keepdim=True)

        # cosine similarity
        matching_scores = torch.matmul(query_g, support_g.t()).squeeze()

        return query_g, matching_scores


class Generator(nn.Module):
    
    def __init__(self, args, input_dim=300, dropout=0.5):
        super(Generator, self).__init__()

        self.noise_dim = args.noise_dim
        self.fc1_dim = 250
        self.ep_dim = 200

        fc1 = nn.Linear(input_dim+self.noise_dim, self.fc1_dim)
        self.fc1 = spectral_norm(fc1)

        fc2 = nn.Linear(self.fc1_dim, self.ep_dim)
        self.fc2 = spectral_norm(fc2)

        self.layer_norm = LayerNormalization(self.ep_dim)

    def forward(self, description, noise):
        x_noise = torch.cat([noise, description], 1)
        x_noise = self.fc1(x_noise)
        false = self.fc2(x_noise)
        false = self.layer_norm(false)

        return false


class Discriminator(nn.Module):
    def __init__(self, dropout=0.3):
        super(Discriminator, self).__init__()

        fc_middle = nn.Linear(200, 200)
        self.fc_middle = spectral_norm(fc_middle)

        fc_TF = nn.Linear(200, 1) # True or False
        self.fc_TF = spectral_norm(fc_TF)

        self.layer_norm = LayerNormalization(200)

    def forward(self, ep_vec, centroid_matrix):

        middle_vec = F.leaky_relu(self.fc_middle(ep_vec))
        middle_vec = self.layer_norm(middle_vec)

        centroid_matrix = F.leaky_relu(self.fc_middle(centroid_matrix))
        centroid_matrix = self.layer_norm(centroid_matrix)

        # determine True or False
        logit_TF = self.fc_TF(middle_vec)

        # determine label
        class_scores = torch.matmul(middle_vec, centroid_matrix.t())

        return middle_vec, logit_TF, class_scores


if __name__ == '__main__':
    args = read_options()

    # Extractor
    print('###  Extractor  ###')
    query = Variable(torch.ones(64,2).long())
    query_meta = [Variable(torch.ones(64,4,2).long()), Variable(torch.ones(64)), Variable(torch.ones(64,4,2).long()), Variable(torch.ones(64))]
    embeddings = np.ones((201, 50))
    extractor = Extractor(50, 200, 51, embeddings)
    query_g, class_vec = extractor(query, query_meta)
    print(query_g.size()) # (64L, 100L)
    print(class_vec.size())


    # Generator
    print('\n###  Generator  ###')
    descriptions = Variable(torch.ones(1063,768)).cuda()
    Generator = Generator(args)
    false, logits = Generator(descriptions)
    print(false.size()) # (1063L, 400L)
    print(logits.size()) # (1063L, 51L)

    # Discriminator
    print('\n###  Discriminator  ###')
    ep_vec = Variable(torch.ones(200,300))
    Discriminator = Discriminator(args)
    logit_TF, logit_class = Discriminator(ep_vec)
    print(logit_TF.size()) # (100L, 2L)
    print(logit_class.size()) # (100L, 51L)
