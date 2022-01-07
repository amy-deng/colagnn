# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from layers import *
from utils import *
from torch.autograd import Variable
import sys
 
# [ablation] main 
class cola_gnn(nn.Module):  
    def __init__(self, args, data): 
        super().__init__()
        self.x_h = 1 
        self.f_h = data.m   
        self.m = data.m  
        self.d = data.d 
        self.w = args.window
        self.h = args.horizon
        self.adj = data.adj
        self.o_adj = data.orig_adj
        if args.cuda:
            self.adj = sparse_mx_to_torch_sparse_tensor(normalize_adj2(data.orig_adj.cpu().numpy())).to_dense().cuda()
        else:
            self.adj = sparse_mx_to_torch_sparse_tensor(normalize_adj2(data.orig_adj.cpu().numpy())).to_dense()
        self.dropout = args.dropout
        self.n_hidden = args.n_hidden
        half_hid = int(self.n_hidden/2)
        self.V = Parameter(torch.Tensor(half_hid))
        self.bv = Parameter(torch.Tensor(1))
        self.W1 = Parameter(torch.Tensor(half_hid, self.n_hidden))
        self.b1 = Parameter(torch.Tensor(half_hid))
        self.W2 = Parameter(torch.Tensor(half_hid, self.n_hidden))
        self.act = F.elu 
        self.Wb = Parameter(torch.Tensor(self.m,self.m))
        self.wb = Parameter(torch.Tensor(1))
        self.k = args.k
        self.conv = nn.Conv1d(1, self.k, self.w)
        long_kernal = self.w//2
        self.conv_long = nn.Conv1d(self.x_h, self.k, long_kernal, dilation=2)
        long_out = self.w-2*(long_kernal-1)
        self.n_spatial = 10  
        self.conv1 = GraphConvLayer((1+long_out)*self.k, self.n_hidden) # self.k
        self.conv2 = GraphConvLayer(self.n_hidden, self.n_spatial)
 
        if args.rnn_model == 'LSTM':
            self.rnn = nn.LSTM( input_size=self.x_h, hidden_size=self.n_hidden, num_layers=args.n_layer, dropout=args.dropout, batch_first=True, bidirectional=args.bi)
        elif args.rnn_model == 'GRU':
            self.rnn = nn.GRU( input_size=self.x_h, hidden_size=self.n_hidden, num_layers=args.n_layer, dropout=args.dropout, batch_first=True, bidirectional=args.bi)
        elif args.rnn_model == 'RNN':
            self.rnn = nn.RNN( input_size=self.x_h, hidden_size=self.n_hidden, num_layers=args.n_layer, dropout=args.dropout, batch_first=True, bidirectional=args.bi)
        else:
            raise LookupError (' only support LSTM, GRU and RNN')

        hidden_size = (int(args.bi) + 1) * self.n_hidden
        self.out = nn.Linear(hidden_size + self.n_spatial, 1)  

        self.residual_window = 0
        self.ratio = 1.0
        if (self.residual_window > 0):
            self.residual_window = min(self.residual_window, args.window)
            self.residual = nn.Linear(self.residual_window, 1) 
        self.init_weights()
     
    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data) # best
            else:
                stdv = 1. / math.sqrt(p.size(0))
                p.data.uniform_(-stdv, stdv)

    def forward(self, x, feat=None):
        '''
        Args:  x: (batch, time_step, m)  
            feat: [batch, window, dim, m]
        Returns: (batch, m)
        ''' 
        b, w, m = x.size()
        orig_x = x 
        x = x.permute(0, 2, 1).contiguous().view(-1, x.size(1), 1) 
        r_out, hc = self.rnn(x, None)
        last_hid = r_out[:,-1,:]
        last_hid = last_hid.view(-1,self.m, self.n_hidden)
        out_temporal = last_hid  # [b, m, 20]
        hid_rpt_m = last_hid.repeat(1,self.m,1).view(b,self.m,self.m,self.n_hidden) # b,m,m,w continuous m
        hid_rpt_w = last_hid.repeat(1,1,self.m).view(b,self.m,self.m,self.n_hidden) # b,m,m,w continuous w one window data
        a_mx = self.act( hid_rpt_m @ self.W1.t()  + hid_rpt_w @ self.W2.t() + self.b1 ) @ self.V + self.bv # row, all states influence one state 
        a_mx = F.normalize(a_mx, p=2, dim=1, eps=1e-12, out=None)
        r_l = []
        r_long_l = []
        h_mids = orig_x
        for i in range(self.m):
            h_tmp = h_mids[:,:,i:i+1].permute(0,2,1).contiguous() 
            r = self.conv(h_tmp) # [32, 10/k, 1]
            r_long = self.conv_long(h_tmp)
            r_l.append(r)
            r_long_l.append(r_long)
        r_l = torch.stack(r_l,dim=1)
        r_long_l = torch.stack(r_long_l,dim=1)
        r_l = torch.cat((r_l,r_long_l),-1)
        r_l = r_l.view(r_l.size(0),r_l.size(1),-1)
        r_l = torch.relu(r_l)
        adjs = self.adj.repeat(b,1)
        adjs = adjs.view(b,self.m, self.m)
        c = torch.sigmoid(a_mx @ self.Wb + self.wb)
        a_mx = adjs * c + a_mx * (1-c) 
        adj = a_mx 
        x = r_l  
        x = F.relu(self.conv1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        out_spatial = F.relu(self.conv2(x, adj))
        out = torch.cat((out_spatial, out_temporal),dim=-1)
        out = self.out(out)
        out = torch.squeeze(out)

        if (self.residual_window > 0):
            z = orig_x[:, -self.residual_window:, :]; #Step backward # [batch, res_window, m]
            z = z.permute(0,2,1).contiguous().view(-1, self.residual_window); #[batch*m, res_window]
            z = self.residual(z); #[batch*m, 1]
            z = z.view(-1,self.m); #[batch, m]
            out = out * self.ratio + z; #[batch, m]

        return out, None


class ARMA(nn.Module): 
    def __init__(self, args, data):
        super(ARMA, self).__init__()
        self.m = data.m
        self.w = args.window
        self.n = 2 # larger worse
        self.w = 2*self.w - self.n + 1 
        self.weight = Parameter(torch.Tensor(self.w, self.m)) # 20 * 49
        self.bias = Parameter(torch.zeros(self.m)) # 49
        nn.init.xavier_normal(self.weight)

        args.output_fun = None;
        self.output = None
        if (args.output_fun == 'sigmoid'):
            self.output = F.sigmoid;
        if (args.output_fun == 'tanh'):
            self.output = F.tanh;

    def forward(self, x):
        x_o = x
        x = x.permute(0,2,1).contiguous()
        n = self.n
        cumsum = torch.cumsum(x,dim=-1)
        cumsum[:,:,n:] = cumsum[:,:,n:] - cumsum[:,:,:-n]
        x = cumsum[:,:,n - 1:] / n
        x = x.permute(0,2,1).contiguous()
        x = torch.cat((x_o,x), dim=1)
        x = torch.sum(x * self.weight, dim=1) + self.bias
        if (self.output != None):
            x = self.output(x)
        return x, None

class AR(nn.Module):
    def __init__(self, args, data):
        super(AR, self).__init__()
        self.m = data.m
        self.w = args.window
        self.weight = Parameter(torch.Tensor(self.w, self.m)) # 20 * 49
        self.bias = Parameter(torch.zeros(self.m)) # 49
        nn.init.xavier_normal(self.weight)

        args.output_fun = None;
        self.output = None
        if (args.output_fun == 'sigmoid'):
            self.output = F.sigmoid;
        if (args.output_fun == 'tanh'):
            self.output = F.tanh;

    def forward(self, x):
        batch_size = x.size(0);
        x = torch.sum(x * self.weight, dim=1) + self.bias
        if (self.output != None):
            x = self.output(x)
        return x,None

class VAR(nn.Module):
    def __init__(self, args, data):
        super(VAR, self).__init__()
        self.m = data.m
        self.w = args.window
        self.linear = nn.Linear(self.m * self.w, self.m);
        args.output_fun = None;
        self.output = None;
        if (args.output_fun == 'sigmoid'):
            self.output = F.sigmoid;
        if (args.output_fun == 'tanh'):
            self.output = F.tanh;

    def forward(self, x):
        x = x.view(-1, self.m * self.w);
        x = self.linear(x);
        if (self.output != None):
            x = self.output(x);
        return x,None

class GAR(nn.Module):
    def __init__(self, args, data):
        super(GAR, self).__init__()
        self.m = data.m
        self.w = args.window

        self.linear = nn.Linear(self.w, 1);
        args.output_fun = None;
        self.output = None;
        if (args.output_fun == 'sigmoid'):
            self.output = F.sigmoid;
        if (args.output_fun == 'tanh'):
            self.output = F.tanh;

    def forward(self, x):
        batch_size = x.size(0);
        x = x.transpose(2,1).contiguous();
        x = x.view(batch_size * self.m, self.w);
        x = self.linear(x);
        x = x.view(batch_size, self.m);
        if (self.output != None):
            x = self.output(x);
        return x,None

class RNN(nn.Module):
    def __init__(self, args, data):
        super(RNN, self).__init__()
        n_input = 1
        self.m = data.m
        if args.rnn_model == 'LSTM':
            self.rnn = nn.LSTM( input_size=n_input, hidden_size=args.n_hidden, num_layers=args.n_layer, dropout=args.dropout,
                                batch_first=True, bidirectional=args.bi)
        elif args.rnn_model == 'GRU':
            self.rnn = nn.GRU( input_size=n_input, hidden_size=args.n_hidden, num_layers=args.n_layer, dropout=args.dropout,
                                batch_first=True, bidirectional=args.bi)
        elif args.rnn_model == 'RNN':
            self.rnn = nn.RNN( input_size=n_input, hidden_size=args.n_hidden, num_layers=args.n_layer, dropout=args.dropout,
                                batch_first=True, bidirectional=args.bi)
        else:
            raise LookupError(' only support LSTM, GRU and RNN')

        hidden_size = (int(args.bi) + 1) * args.n_hidden
        self.out = nn.Linear(hidden_size, 1) #n_output

    def forward(self, x):
        '''
        Args:
            x: (batch, time_step, m)  
        Returns:
            (batch, m)
        '''
        x = x.permute(0, 2, 1).contiguous().view(-1, x.size(1), 1)
        r_out, hc = self.rnn(x, None) # hidden state is the prediction TODO
        out = self.out(r_out[:,-1,:])
        out = out.view(-1, self.m)
        return out,None

class SelfAttnRNN(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        self.n_input = 1
        self.m = data.m
        self.w = args.window
        self.hid = args.n_hidden 
        self.rnn_cell =  nn.RNNCell(input_size=self.n_input, hidden_size=self.hid)
        self.V = Parameter(torch.Tensor(self.hid, 1))
        self.Wx = Parameter(torch.Tensor(self.hid, self.n_input))
        self.Wtlt = Parameter(torch.Tensor(self.hid, self.hid))
        self.Wh = Parameter(torch.Tensor(self.hid, self.hid))
        self.init_weights()
        self.out = nn.Linear(self.hid, 1)
    
    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data) # xavier_normal xavier_uniform_
            else:
                # nn.init.zeros_(p.data)
                stdv = 1. / math.sqrt(p.size(0))
                p.data.uniform_(-stdv, stdv)

    def forward(self, x):
        '''
        Args: x: (batch, time_step, m)  
        Returns: (batch, m)
        '''
        b, w, m = x.size()
        x = x.permute(0, 2, 1).contiguous().view(x.size(0)*x.size(2), x.size(1), self.n_input) # x, 20, 1
        Htlt = []
        H = []
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        for step in range(self.w): # forloop each history step
            x_tp1 = x[:,step,:] # [x, 1]
            if step == 0:
                hx = torch.zeros(b*m, self.hid).to(device)
                H.append(hx)
                h_tlt = torch.zeros(b*m, self.hid).to(device)
            else:
                h_tlt = Htlt[-1]
            h_his = torch.stack(H,dim=1)
            if step>0:
                x_tp1_rp = x_tp1.repeat(1,step+1).view(b*m,step+1,-1)
                h_tlt_rp = h_tlt.repeat(1,step+1).view(b*m,step+1,-1)
            else: 
                x_tp1_rp = x_tp1
                h_tlt_rp = h_tlt
            q1 = x_tp1_rp @ self.Wx.t() # [x, 20]
            q2 = h_tlt_rp @ self.Wtlt.t() # [x, 20]
            q3 = h_his @ self.Wh.t() # [x, 20]
            a = torch.tanh(q1 + q2 + q3) @ self.V # [x, 1]
            a = torch.softmax(a,dim=-1)
            h_tlt_t = h_his * a
            h_tlt_t = torch.sum(h_tlt_t,dim=1)
            Htlt.append(h_tlt_t)
            hx = self.rnn_cell(x_tp1, h_tlt_t) # [x, 20]
            H.append(hx)
        h = H[-1]
        out = self.out(h)
        out = out.view(b,m)
        return out,None

class CNNRNN_Res(nn.Module):
    def __init__(self, args, data): 
        super(CNNRNN_Res, self).__init__()
        self.ratio = 1.0   
        self.m = data.m  

        if args.rnn_model == 'LSTM':
            self.rnn = nn.LSTM( input_size=self.m, hidden_size=args.n_hidden, num_layers=args.n_layer, dropout=args.dropout, batch_first=True)
        elif args.rnn_model == 'GRU':
            self.rnn = nn.GRU( input_size=self.m, hidden_size=args.n_hidden, num_layers=args.n_layer, dropout=args.dropout, batch_first=True)
        elif args.rnn_model == 'RNN':
            self.rnn = nn.RNN( input_size=self.m, hidden_size=args.n_hidden, num_layers=args.n_layer, dropout=args.dropout, batch_first=True)
        else:
            raise LookupError(' only support LSTM, GRU and RNN')

        self.residual_window = 4

        self.mask_mat = Parameter(torch.Tensor(self.m, self.m))
        nn.init.xavier_normal(self.mask_mat)  
        self.adj = data.adj  

        self.dropout = nn.Dropout(p=args.dropout)
        self.linear1 = nn.Linear(args.n_hidden, self.m)
        if (self.residual_window > 0):
            self.residual_window = min(self.residual_window, args.window)
            self.residual = nn.Linear(self.residual_window, 1);
        self.output = None
        output_fun = None
        if (output_fun == 'sigmoid'):
            self.output = F.sigmoid
        if (output_fun == 'tanh'):
            self.output = F.tanh

    def forward(self, x):
        masked_adj = self.adj * self.mask_mat
        x = x.matmul(masked_adj)
        r_out, _ = self.rnn(x) #torch.Size([window, batch, n_hid]) torch.Size([batch, n_hid])
        r = self.dropout(r_out[:,-1,:])
        res = self.linear1(r) # ->[batch, m]
       
        if (self.residual_window > 0):
            z = x[:, -self.residual_window:, :]; #Step backward # [batch, res_window, m]
            z = z.permute(0,2,1).contiguous().view(-1, self.residual_window); #[batch*m, res_window]
            z = self.residual(z); #[batch*m, 1]
            z = z.view(-1,self.m); #[batch, m]
            res = res * self.ratio + z; #[batch, m]

        if self.output is not None:
            res = self.output(res).float()
        return res,None

class LSTNet(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        self.use_cuda = args.cuda
        self.P = args.window;
        self.m = data.m
        self.hidR = args.n_hidden;
        self.hidC = args.n_hidden;
        self.hidS = 5;
        self.Ck = 8;
        self.skip = 4;
        self.pt = (self.P - self.Ck)//self.skip
        self.hw = 4
        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size = (self.Ck, self.m));
        self.GRU1 = nn.GRU(self.hidC, self.hidR);
        self.dropout = nn.Dropout(p = args.dropout);
        if (self.skip > 0):
            self.GRUskip = nn.GRU(self.hidC, self.hidS);
            self.linear1 = nn.Linear(self.hidR + self.skip * self.hidS, self.m);
        else:
            self.linear1 = nn.Linear(self.hidR, self.m);
        if (self.hw > 0):
            self.highway = nn.Linear(self.hw, 1);
        self.output = None;
        
 
    def forward(self, x):
        batch_size = x.size(0);
        
        #CNN
        c = x.view(-1, 1, self.P, self.m);
        c = F.relu(self.conv1(c));
        c = self.dropout(c);
        c = torch.squeeze(c, 3);
        
        # RNN 
        r = c.permute(2, 0, 1).contiguous();
        _, r = self.GRU1(r);
        r = self.dropout(torch.squeeze(r,0));

        
        #skip-rnn
        if (self.skip > 0):
            s = c[:,:, int(-self.pt * self.skip):].contiguous();
            # print(s.shape,self.pt)
            s = s.view(batch_size, self.hidC, self.pt, self.skip);
            s = s.permute(2,0,3,1).contiguous();
            s = s.view(self.pt, batch_size * self.skip, self.hidC);
            _, s = self.GRUskip(s);
            s = s.view(batch_size, self.skip * self.hidS);
            s = self.dropout(s);
            r = torch.cat((r,s),1);
        
        res = self.linear1(r);
        
        #highway
        if (self.hw > 0):
            z = x[:, -self.hw:, :];
            z = z.permute(0,2,1).contiguous().view(-1, self.hw);
            z = self.highway(z);
            z = z.view(-1,self.m);
            res = res + z;
            
        if (self.output):
            res = self.output(res);
        return res,None



'''STGCN'''
class TimeBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution to each node of
    a graph in isolation.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param out_channels: Desired number of output channels at each node in
        each time step.
        :param kernel_size: Size of the 1D temporal kernel.
        """
        super(TimeBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels)
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features_out=out_channels)
        """
        # Convert into NCHW format for pytorch to perform convolutions.
        X = X.permute(0, 3, 1, 2)
        temp = self.conv1(X) + torch.sigmoid(self.conv2(X))
        out = F.relu(temp + self.conv3(X))
        # Convert back from NCHW to NHWC
        out = out.permute(0, 2, 3, 1)
        return out


class STGCNBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution on each node in
    isolation, followed by a graph convolution, followed by another temporal
    convolution on each node.
    """

    def __init__(self, in_channels, spatial_channels, out_channels,
                 num_nodes):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param spatial_channels: Number of output channels of the graph
        convolutional, spatial sub-block.
        :param out_channels: Desired number of output features at each node in
        each time step.
        :param num_nodes: Number of nodes in the graph.
        """
        super(STGCNBlock, self).__init__()
        self.temporal1 = TimeBlock(in_channels=in_channels,
                                   out_channels=out_channels)
        self.Theta1 = nn.Parameter(torch.FloatTensor(out_channels,
                                                     spatial_channels))
        self.temporal2 = TimeBlock(in_channels=spatial_channels,
                                   out_channels=out_channels)
        self.batch_norm = nn.BatchNorm2d(num_nodes)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)

    def forward(self, X, A_hat):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features=out_channels).
        """
        t = self.temporal1(X)
        lfs = torch.einsum("ij,jklm->kilm", [A_hat, t.permute(1, 0, 2, 3)])
        # t2 = F.relu(torch.einsum("ijkl,lp->ijkp", [lfs, self.Theta1]))
        t2 = F.relu(torch.matmul(lfs, self.Theta1))
        t3 = self.temporal2(t2)
        return self.batch_norm(t3)
        # return t3


class STGCN(nn.Module):
    """
    Spatio-temporal graph convolutional network as described in
    https://arxiv.org/abs/1709.04875v3 by Yu et al.
    Input should have shape (batch_size, num_nodes, num_input_time_steps,
    num_features).
    """

    def __init__(self, args, data, num_nodes, num_features, num_timesteps_input,
                 num_timesteps_output):
        """
        :param num_nodes: Number of nodes in the graph.
        :param num_features: Number of features at each node in each time step.
        :param num_timesteps_input: Number of past time steps fed into the
        network.
        :param num_timesteps_output: Desired number of future time steps
        output by the network.
        """
        super(STGCN, self).__init__()
        self.block1 = STGCNBlock(in_channels=num_features, out_channels=args.n_hidden,
                                 spatial_channels=16, num_nodes=num_nodes)
        self.block2 = STGCNBlock(in_channels=args.n_hidden, out_channels=args.n_hidden,
                                 spatial_channels=16, num_nodes=num_nodes)
        self.last_temporal = TimeBlock(in_channels=args.n_hidden, out_channels=args.n_hidden)
        self.fully = nn.Linear((num_timesteps_input - 2 * 5) * args.n_hidden,
                               num_timesteps_output)
                
        if args.cuda:
            self.adj = sparse_mx_to_torch_sparse_tensor(normalize_adj2(data.orig_adj.cpu().numpy())).to_dense().cuda()
        else:
            self.adj = sparse_mx_to_torch_sparse_tensor(normalize_adj2(data.orig_adj.cpu().numpy())).to_dense()

    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        """
        # batch_size, num_nodes, num_input_time_steps, num_features
        # [32, 20, 47]
        X = X.permute(0,2,1).contiguous()
        X = X.unsqueeze(-1)
        # print(X.shape)
        out1 = self.block1(X, self.adj)
        out2 = self.block2(out1, self.adj)
        out3 = self.last_temporal(out2)
        out4 = self.fully(out3.reshape((out3.shape[0], out3.shape[1], -1)))
        # print(out4.shape)
        return out4.squeeze(-1),None


class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

class gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):
        super(gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

