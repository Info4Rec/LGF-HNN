import torch
import torch.nn as nn


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

class Aggregator(nn.Module):
    def __init__(self, batch_size, dim, dropout, act, name=None):
        super(Aggregator, self).__init__()
        self.dropout = dropout
        self.act = act
        self.batch_size = batch_size
        self.dim = dim

    def forward(self):
        pass


class LocalAggregator(nn.Module):
    def __init__(self, dim, alpha, dropout=0., name=None):
        super(LocalAggregator, self).__init__()
        self.dim = dim
        self.dropout = dropout

        self.a_0 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_1 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_2 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_3 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.bias = nn.Parameter(torch.Tensor(self.dim))

        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, hidden, adj, mask_item=None):
        h = hidden
        batch_size = h.shape[0]
        N = h.shape[1] #N代表max_len

        a_input = (h.repeat(1, 1, N) .view(batch_size, N * N, self.dim)
                   * h.repeat(1, N, 1)).view(batch_size, N, N, self.dim)

        e_0 = torch.matmul(a_input, self.a_0)
        e_1 = torch.matmul(a_input, self.a_1)
        e_2 = torch.matmul(a_input, self.a_2)
        e_3 = torch.matmul(a_input, self.a_3)

        e_0 = self.leakyrelu(e_0).squeeze(-1).view(batch_size, N, N)
        e_1 = self.leakyrelu(e_1).squeeze(-1).view(batch_size, N, N)
        e_2 = self.leakyrelu(e_2).squeeze(-1).view(batch_size, N, N)
        e_3 = self.leakyrelu(e_3).squeeze(-1).view(batch_size, N, N)

        mask = -9e15 * torch.ones_like(e_0)
        alpha = torch.where(adj.eq(1), e_0, mask)
        alpha = torch.where(adj.eq(2), e_1, alpha)
        alpha = torch.where(adj.eq(3), e_2, alpha)
        alpha = torch.where(adj.eq(4), e_3, alpha)
        alpha = torch.softmax(alpha, dim=-1)

        output = torch.matmul(alpha, h)
        return output


class GlobalAggregator(nn.Module):
    def __init__(self, dim, dropout, act=torch.relu,  name=None):
        super(GlobalAggregator, self).__init__()
        self.dim = dim
        self.dropout = dropout

        self.w_0 = nn.Parameter(torch.Tensor(self.dim, self.dim))
        self.w_1 = nn.Parameter(torch.Tensor(self.dim, self.dim))
        self.a_0 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.bias = nn.Parameter(torch.Tensor(self.dim))
        self.act = act

    def forward(self, h,  hg_adj, mask_item=None, session_info = None):
        batch_size = h.shape[0]
        N = h.shape[1]
        shape = (batch_size, N, N, self.dim)
        hidden = h.repeat(1, 1, N).view(shape)
        session_info_all = session_info.unsqueeze(2).repeat(1, 1, N, 1)
        tem = hidden * session_info_all
        former_1 = torch.matmul(tem , self.w_0)
        latter_1 = torch.matmul(hidden, self.w_1)
        alpha = self.act(former_1 + latter_1 + self.bias)
        alpha = torch.matmul(alpha, self.a_0)
        alpha = alpha.view(batch_size, N, N)
        mask = torch.zeros_like(alpha)
        alpha = torch.where(hg_adj.eq(1), alpha, mask)

        output = torch.matmul(alpha, h)

        return output


