import datetime
import math
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from aggregator import LocalAggregator, GlobalAggregator
from torch.nn import Module, Parameter
import torch.nn.functional as F


class CombineGraph(Module):
    def __init__(self, opt, num_node, adj_all, num):
        super(CombineGraph, self).__init__()
        self.opt = opt
        self.rate = opt.in_rate
        self.batch_size = opt.batch_size
        self.num_node = num_node
        self.dim = opt.hiddenSize
        self.dropout_local = opt.dropout_local
        self.dropout_global = opt.dropout_global
        self.hop = opt.n_iter
        self.sample_num = opt.n_sample
        self.adj_all = trans_to_cuda(torch.Tensor(adj_all)).long()
        self.num = trans_to_cuda(torch.Tensor(num)).float()
        # Aggregator
        self.local_agg = []
        self.global_agg = []
        for i in range(self.hop):
            agg_local = LocalAggregator(self.dim, self.opt.alpha, dropout=0.0 )
            GlobalGraph = GlobalAggregator(self.dim, opt.dropout_gcn, act=torch.relu)
            self.add_module('agg_local_{}'.format(i), agg_local)
            self.add_module('agg_global_{}'.format(i), GlobalGraph)
            self.local_agg.append(agg_local)
            self.global_agg.append(GlobalGraph)

        self.embedding = nn.Embedding(num_node, self.dim)
        self.pos_embedding = nn.Embedding(200, self.dim)

        self.w_1 = nn.Parameter(torch.Tensor(2 * self.dim, self.dim))
        self.w_2 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.glu1 = nn.Linear(self.dim, self.dim)
        self.glu2 = nn.Linear(self.dim, self.dim, bias=False)
        self.linear_transform = nn.Linear(self.dim, self.dim, bias=False)

        self.leakyrelu = nn.LeakyReLU(opt.alpha)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.gru_piece = nn.GRUCell(self.dim, self.dim).cuda()
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def sample(self, target, n_sample):

        return self.adj_all[target.view(-1)], self.num[target.view(-1)]

    def compute_scores(self, hidden, mask):
        mask = mask.float().unsqueeze(-1)

        batch_size = hidden.shape[0]
        len = hidden.shape[1]
        pos_emb = self.pos_embedding.weight[:len]
        pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)

        hs = torch.sum(hidden * mask, -2) / torch.sum(mask, 1)
        hs = hs.unsqueeze(-2).repeat(1, len, 1)
        nh = torch.matmul(torch.cat([pos_emb, hidden], -1), self.w_1)
        nh = torch.tanh(nh)
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
        beta = torch.matmul(nh, self.w_2)
        beta = beta * mask
        select = torch.sum(beta * hidden, 1)

        b = self.embedding.weight[1:]  # n_nodes x latent_size
        scores = torch.matmul(select, b.transpose(1, 0))
        return scores

    def gene_sess(self, hidden, mask):
        mask = mask.float().unsqueeze(-1)

        batch_size = hidden.shape[0]
        len = hidden.shape[1]
        pos_emb = self.pos_embedding.weight[:len]
        pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)

        hs = torch.sum(hidden * mask, -2) / torch.sum(mask, 1)
        hs = hs.unsqueeze(-2).repeat(1, len, 1)
        nh = torch.matmul(torch.cat([pos_emb, hidden], -1), self.w_1)
        nh = torch.tanh(nh)
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
        beta = torch.matmul(nh, self.w_2)
        beta = beta * mask
        select = torch.sum(beta * hidden, 1)

        return select

    def forward(self, inputs, adj, mask_item, item, data, hg_adj):
        batch_size = inputs.shape[0]
        seqs_len = inputs.shape[1]
        h = self.embedding(inputs)

        # local
        h_local_all = []
        for i in range(self.hop):
            local_agg = self.local_agg[i]
            if i == 0:
                h_local = local_agg(h, adj, mask_item)
                h_local_all.append(h_local)
            else:
                h_local_next = local_agg(h_local_all[i-1], adj, mask_item)
                h_local_all.append(h_local_next)
        # global
        item_neighbors = [inputs]
        support_size = seqs_len

        for i in range(1, self.hop + 1):
            item_sample_i, weight_sample_i = self.sample(item_neighbors[-1], self.sample_num)
            support_size *= self.sample_num #max_len * sample_num
            item_neighbors.append(item_sample_i.view(batch_size, support_size))

        entity_vectors = [self.embedding(i) for i in item_neighbors]

        session_info = []
        item_emb = self.embedding(item) * mask_item.float().unsqueeze(-1)

        # mean 均值化会话内向量？
        sum_item_emb = torch.sum(item_emb, 1) / torch.sum(mask_item.float(), -1).unsqueeze(-1)


        sum_item_emb = sum_item_emb.unsqueeze(-2)
        for i in range(self.hop):
            session_info.append(sum_item_emb.repeat(1, entity_vectors[i].shape[1], 1))
        h_global_all = []
        for n_hop in range(self.hop):
            agg_global = self.global_agg[n_hop]
            if n_hop == 0:
                h_global = agg_global(h, hg_adj, mask_item, session_info[n_hop])
                h_global_all.append(h_global)
            else:
                h_global_next = agg_global(h_global_all[n_hop-1], hg_adj, mask_item, session_info[0])
                h_global_all.append(h_global_next)

        #GRU
        h_local_res = torch.ones_like(h_local).cuda()
        h_global_res = torch.ones_like(h_global).cuda()
        for j in range(self.hop):
            h_local_gru = h_local_all[j]
            h_global_gru = h_global_all[j]
            for i in range(batch_size):
                if j == 0:
                    with torch.no_grad():
                        h_i_global = self.gru_piece(h_global[i], h_local_gru[i])
                    h_global_res[i] = h_i_global.cuda()
                else:
                    with torch.no_grad():
                        h_i_global = self.gru_piece(h_global_res[i], h_local_gru[i])
                    h_global_res[i] = h_i_global.cuda()
            for i in range(batch_size):
                if j == 0:
                    with torch.no_grad():
                        h_i_local = self.gru_piece(h_local[i], h_global_gru[i])
                    h_local_res[i] = h_i_local.cuda()
                else:
                    with torch.no_grad():
                        h_i_local = self.gru_piece(h_local_res[i], h_global_gru[i])
                    h_local_res[i] = h_i_local.cuda()
        # combine
        h_local_res = self.rate * h_local_res + h_local_all[0]
        h_global_res = self.rate *h_global_res + h_global_all[self.hop-1]
        h_local = F.dropout(h_local_res, self.dropout_local, training=self.training)
        h_global = F.dropout(h_global_res, self.dropout_global, training=self.training)
        if self.opt.dataset == "Nowplaying":
            output = h_local + 0.8*h_global
        else:
            output = h_local + h_global
        return output, h_local, h_global


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def SSL(sess_emb_hgnn, sess_emb_lgcn):
    def row_shuffle(embedding):
        corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
        return corrupted_embedding

    def row_column_shuffle(embedding):
        corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
        corrupted_embedding = corrupted_embedding[:, torch.randperm(corrupted_embedding.size()[1])]
        return corrupted_embedding

    def score(x1, x2):
        return torch.sum(torch.mul(x1, x2), 1)

    pos = score(sess_emb_hgnn, sess_emb_lgcn)
    neg1 = score(sess_emb_lgcn, row_column_shuffle(sess_emb_hgnn))
    one = torch.cuda.FloatTensor(neg1.shape[0]).fill_(1)
    con_loss = torch.sum(-torch.log(1e-8 + torch.sigmoid(pos)) - torch.log(1e-8 + (one - torch.sigmoid(neg1))))
    return con_loss

def forward(model, data):

    alias_inputs, adj, items, mask, targets, inputs, hg_adj = data
    alias_inputs = trans_to_cuda(alias_inputs).long()
    items = trans_to_cuda(items).long()
    adj = trans_to_cuda(adj).float()
    mask = trans_to_cuda(mask).long()
    inputs = trans_to_cuda(inputs).long()
    hg_adj = trans_to_cuda(hg_adj).long()

    hidden, loc, glo = model(items, adj, mask, inputs, data, hg_adj)
    get = lambda index: hidden[index][alias_inputs[index]]
    get1 = lambda index: loc[index][alias_inputs[index]]
    get2 = lambda index: glo[index][alias_inputs[index]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
    seq_loc = torch.stack([get1(i) for i in torch.arange(len(alias_inputs)).long()])
    seq_glo = torch.stack([get2(i) for i in torch.arange(len(alias_inputs)).long()])
    ses_local = model.gene_sess(seq_loc, mask)
    ses_global = model.gene_sess(seq_glo, mask)
    SSL_loss = SSL(ses_local, ses_global)

    return targets, model.compute_scores(seq_hidden, mask), SSL_loss


def train_test(model, train_data, test_data,sslrate):
    print('start training: ', datetime.datetime.now())

    model.train()
    total_loss = 0.0
    train_loader = torch.utils.data.DataLoader(train_data, num_workers=4, batch_size=model.batch_size,
                                               shuffle=True, pin_memory=True)

    for data in tqdm(train_loader):
        model.optimizer.zero_grad()
        targets, scores, ssl_loss = forward(model, data)
        model.Eiters += 1
        targets = trans_to_cuda(targets).long()
        loss = model.loss_function(scores, targets - 1)
        loss = loss + 0.01*ssl_loss
        # with torch.autograd.detect_anomaly():
        loss.backward()
        model.optimizer.step()
        total_loss += loss

    print('\tLoss:\t%.3f' % total_loss)

    model.scheduler.step()

    print('start predicting: ', datetime.datetime.now())

    model.eval()
    metrics = {}

    test_loader = torch.utils.data.DataLoader(test_data, num_workers=4, batch_size=model.batch_size,
                                              shuffle=False, pin_memory=True)

    result_20 = []

    hit_20, mrr_20 = [], []
    for data in test_loader:
        targets, scores, con_loss = forward(model, data)
        sub_scores_20 = scores.topk(20)[1]
        sub_scores_20 = trans_to_cpu(sub_scores_20).detach().numpy()

        targets = targets.numpy()
        for score, target, mask in zip(sub_scores_20, targets, test_data.mask):
            hit_20.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr_20.append(0)
            else:
                mrr_20.append(1 / (np.where(score == target - 1)[0][0] + 1))

    result_20.append(np.mean(hit_20) * 100)
    result_20.append(np.mean(mrr_20) * 100)

    return result_20
