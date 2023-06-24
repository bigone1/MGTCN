import torch
import numpy as np
import os,time,random,math,csv
from sklearn.model_selection import StratifiedKFold
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import torch.optim as optim
import matplotlib.pyplot as plt
import optuna
import pickle
cuda=True
def seed_torch(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic=True
def cal_sample_weight(labels, num_class, use_sample_weight=True):
    if not use_sample_weight:
        return np.ones(len(labels)) / len(labels)
    count = np.zeros(num_class)
    for i in range(num_class):
        count[i] = np.sum(labels==i)
    sample_weight = np.zeros(labels.shape)
    for i in range(num_class):
        sample_weight[np.where(labels==i)[0]] = count[i]/np.sum(count)
    return sample_weight
def prepare_trte_data(data_folder):
    labels = np.loadtxt(os.path.join(data_folder, "read-label.csv"), delimiter=',')
    labels = labels.astype(int)
    data = []
    data.append(np.loadtxt(os.path.join(data_folder, "read-mr.csv"), delimiter=','))
    data.append(np.loadtxt(os.path.join(data_folder, "read-me.csv"), delimiter=','))
    data.append(np.loadtxt(os.path.join(data_folder, "read-mi.csv"), delimiter=','))
    data_tensor_list = []
    
    for i in range(len(data)):
        data_tensor_list.append(torch.FloatTensor(data[i]))
        if cuda:
            data_tensor_list[i] = data_tensor_list[i].cuda()
    return data_tensor_list,  labels
def cosine_distance_torch(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)
def cal_adj_mat_parameter(edge_per_node, data, metric="cosine"):
    assert metric == "cosine", "Only cosine distance implemented"
    dist = cosine_distance_torch(data, data)
    parameter = torch.sort(dist.reshape(-1,)).values[int(edge_per_node*data.shape[0])]
    return np.asscalar(parameter.data.cpu().numpy())
def graph_from_dist_tensor(dist, parameter, self_dist=True):
    if self_dist:
        assert dist.shape[0]==dist.shape[1], "Input is not pairwise dist matrix"
    g = (dist <= parameter).float()
    if self_dist:
        diag_idx = np.diag_indices(g.shape[0])
        g[diag_idx[0], diag_idx[1]] = 0
        
    return g
def gen_adj_mat_tensor(data, parameter, metric="cosine"):
    assert metric == "cosine", "Only cosine distance implemented"
    dist = cosine_distance_torch(data, data)
    g = graph_from_dist_tensor(dist, parameter, self_dist=True)
    if metric == "cosine":
        adj = 1-dist
    else:
        raise NotImplementedError
    adj = adj*g
    adj_T = adj.transpose(0,1)
    I = torch.eye(adj.shape[0])
    if cuda:
        I = I.cuda()
    adj = adj + adj_T*(adj_T > adj).float() - adj*(adj_T > adj).float()
    # adj = F.normalize(adj + I, p=1)
    # adj = to_sparse(adj)
    return adj
def gen_trte_adj_mat(data_trte_list, adj_parameter):
    adj_metric = "cosine" # cosine distance
    adj_list = []
    for i in range(len(data_trte_list)):
        adj_parameter_adaptive = cal_adj_mat_parameter(adj_parameter, data_trte_list[i], adj_metric)
        adj_list.append(gen_adj_mat_tensor(data_trte_list[i], adj_parameter_adaptive, adj_metric))
    return adj_list
def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
           m.bias.data.fill_(0.0)
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        nn.init.xavier_normal_(self.weight.data)
        if self.bias is not None:
            self.bias.data.fill_(0.0)
    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        # output = torch.sparse.mm(adj, support)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
class GCN_E(nn.Module):
    def __init__(self, in_dim, hgcn_dim, dropout):
        super().__init__()
        self.hgcn_dim=hgcn_dim
        if len(self.hgcn_dim)==1:
            self.gc1 = GraphConvolution(in_dim, hgcn_dim[0])
        elif len(self.hgcn_dim)==2:
            self.gc1 = GraphConvolution(in_dim, hgcn_dim[0])
            self.gc2 = GraphConvolution(hgcn_dim[0], hgcn_dim[1])
        else:
            self.gc1 = GraphConvolution(in_dim, hgcn_dim[0])
            self.gc2 = GraphConvolution(hgcn_dim[0], hgcn_dim[1])
            self.gc3 = GraphConvolution(hgcn_dim[1], hgcn_dim[2])
        self.dropout = dropout
    def forward(self, x, adj):
        if len(self.hgcn_dim)==1:
            x = self.gc1(x, adj)
            x = F.leaky_relu(x, 0.25)
        elif len(self.hgcn_dim)==2:
            x = self.gc1(x, adj)
            x = F.leaky_relu(x, 0.25)
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.gc2(x, adj)
            x = F.leaky_relu(x, 0.25)
        else:
            x = self.gc1(x, adj)
            x = F.leaky_relu(x, 0.25)
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.gc2(x, adj)
            x = F.leaky_relu(x, 0.25)
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.gc3(x, adj)
            x = F.leaky_relu(x, 0.25)
        return x
class VCDN(nn.Module):
    def __init__(self, num_view, num_cls, hvcdn_dim):
        super().__init__()
        self.num_cls = num_cls
        self.model = nn.Sequential(
            nn.Linear(pow(num_cls, num_view), hvcdn_dim),
            nn.LeakyReLU(0.25),
            nn.Linear(hvcdn_dim, num_cls)
        )
        self.model.apply(xavier_init)
    def forward(self, in_list):
        num_view = len(in_list)
        for i in range(num_view):
            in_list[i] = torch.sigmoid(in_list[i])
        x = torch.reshape(torch.matmul(in_list[0].unsqueeze(-1), in_list[1].unsqueeze(1)),(-1,pow(self.num_cls,2),1))
        for i in range(2,num_view):
            x = torch.reshape(torch.matmul(x, in_list[i].unsqueeze(1)),(-1,pow(self.num_cls,i+1),1))
        vcdn_feat = torch.reshape(x, (-1,pow(self.num_cls,num_view)))
        output = self.model(vcdn_feat)
        return output
class GTN(nn.Module):
    def __init__(self, num_edge, num_channels, w_in, w_out, num_class,num_layers,norm,gcn_drop,sample_weight):
        super(GTN, self).__init__()
        self.num_edge = num_edge
        self.num_channels = num_channels
        self.w_in = w_in
        self.w_out = w_out
        self.num_class = num_class
        self.num_layers = num_layers
        self.is_norm = norm
        self.gcn_drop=gcn_drop
        self.sample_weight=sample_weight
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(GTLayer(num_edge, num_channels, first=True))
            else:
                layers.append(GTLayer(num_edge, num_channels, first=False))
        self.layers = nn.ModuleList(layers)
        self.loss = nn.CrossEntropyLoss()
        self.linear1 = nn.Linear(self.w_out[-1]*self.num_channels, self.w_out[-1])
        self.linear2 = nn.Linear(self.w_out[-1], self.num_class)
        self.GCN=GCN_E(w_in,w_out,dropout=self.gcn_drop)
    def normalization(self, H):
        for i in range(self.num_channels):
            if i==0:
                H_ = self.norm(H[i,:,:]).unsqueeze(0)
            else:
                H_ = torch.cat((H_,self.norm(H[i,:,:]).unsqueeze(0)), dim=0)
        return H_
    def norm(self, H, add=False):
        H = H.t()
        if add == False:
            H = H*((torch.eye(H.shape[0])==0).type(torch.FloatTensor).cuda())
        else:
            H = H*((torch.eye(H.shape[0])==0).type(torch.FloatTensor).cuda()) + torch.eye(H.shape[0]).type(torch.FloatTensor).cuda()
        deg = torch.sum(H, dim=1)+1e-6
        deg_inv = deg.pow(-1)
        deg_inv[deg_inv == float('inf')] = 0
        deg_inv = deg_inv*torch.eye(H.shape[0]).type(torch.FloatTensor).cuda()
        H = torch.mm(deg_inv,H)
        H = H.t()
        return H
    def forward(self, A, X, target_x, target):
        A = A.unsqueeze(0).permute(0,3,1,2) 
        Ws = []
        for i in range(self.num_layers):
            if i == 0:
                H, W = self.layers[i](A)
            else:
                H = self.normalization(H)
                H, W = self.layers[i](A, H)
            Ws.append(W)
        H[0] = self.norm(H[0], add=True)
        X_=self.GCN(X,H[0].t())
        X_ = self.linear1(X_)
        X_ = F.relu(X_)
        y = self.linear2(X_[target_x])
        loss = self.loss(y, target)
        return loss, y
class GTLayer(nn.Module):
    def __init__(self, in_channels, out_channels, first=True):
        super(GTLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first = first
        if self.first == True:
            self.conv1 = GTConv(in_channels, out_channels)
            self.conv2 = GTConv(in_channels, out_channels)
        else:
            self.conv1 = GTConv(in_channels, out_channels)
    def forward(self, A, H_=None):
        if self.first == True:
            a = self.conv1(A)
            b = self.conv2(A)
            H = torch.bmm(a,b)
            W = [(F.softmax(self.conv1.weight, dim=1)).detach(),(F.softmax(self.conv2.weight, dim=1)).detach()]
        else:
            a = self.conv1(A)
            H = torch.bmm(H_,a)
            W = [(F.softmax(self.conv1.weight, dim=1)).detach()]
        return H,W
class GTConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GTConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.Tensor(out_channels,in_channels,1,1))
        self.bias = None
        self.scale = nn.Parameter(torch.Tensor([0.1]), requires_grad=False)
        self.reset_parameters()
    def reset_parameters(self):
        n = self.in_channels
        nn.init.constant_(self.weight, 0.1)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    def forward(self, A):
        A = torch.sum(A*F.softmax(self.weight, dim=1), dim=1)
        return A
def init_model_dict(num_view, num_class, dim_list, dim_he_list, dim_hc, sample_weight,gcn_dopout,num_edge=4):
    model_dict = {}
    for i in range(num_view):
        model_dict["E{:}".format(i+1)] = GTN(num_edge=num_edge,num_channels=1,w_in=dim_list[i],w_out=dim_he_list,num_class=2,num_layers=1,norm=True,gcn_drop=gcn_dopout,sample_weight=sample_weight)
    if num_view >= 2:
        model_dict["C"] = VCDN(num_view, num_class, dim_hc)
    return model_dict
def init_optim(num_view, model_dict, optimizer_name,lr_e=1e-4, lr_c=1e-4):
    optim_dict = {}
    for i in range(num_view):
        optim_dict["E{:}".format(i+1)] = getattr(optim,optimizer_name)(model_dict["E{:}".format(i+1)].parameters(), lr=lr_e)
    if num_view >= 2:
        optim_dict["C"] = getattr(optim,optimizer_name)(model_dict["C"].parameters(), lr=lr_c,weight_decay=0)
    return optim_dict
def train_epoch(data_list, adj, label, sample_weight, trind, model_dict, optim_dict):
    loss_dict = {}
    prob=None
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    for m in model_dict:
        model_dict[m].train()    
    num_view = len(data_list)
    for i in range(num_view):
        optim_dict["E{:}".format(i+1)].zero_grad()
        ci_loss = model_dict["E{:}".format(i+1)](adj,data_list[i],trind,label[trind])[0]
        ci_loss.backward()
        optim_dict["E{:}".format(i+1)].step()
        loss_dict["E{:}".format(i+1)] = ci_loss.detach().cpu().numpy().item()
    if num_view >= 2:
        optim_dict["C"].zero_grad()
        c_loss = 0
        ci_list = []
        for i in range(num_view):
            ci_list.append(model_dict["E{:}".format(i+1)](adj,data_list[i],trind,label[trind])[1])
        c = model_dict["C"](ci_list)    
        c_loss = torch.mean(torch.mul(criterion(c, label[trind]),sample_weight[trind]))
        c_loss.backward()
        optim_dict["C"].step()
        loss_dict["C"] = c_loss.detach().cpu().numpy().item()
        prob = F.softmax(c, dim=1).data.cpu().numpy()
    return loss_dict,prob
def test_epoch(data_list, adj, label, sample_weight, teind, model_dict):
    criterion = torch.nn.CrossEntropyLoss(reduction='none')#
    for m in model_dict:
        model_dict[m].eval()
    num_view = len(data_list)
    ci_list = []
    for i in range(num_view):
        ci_list.append(model_dict["E{:}".format(i+1)](adj,data_list[i],teind,label[teind])[1])
    if num_view >= 2:
        c = model_dict["C"](ci_list)    
    else:
        c = ci_list[0]
    prob = F.softmax(c, dim=1).data.cpu().numpy()
    ll=torch.mean(torch.mul(criterion(c, label[teind]),sample_weight[teind]))
    return prob,ll
def objective(trial):
    seed_torch()
    data_folder='/home/zhoulin/ucsc/read/all'
    num_class=2
    num_view = 3
    dim_hvcdn = pow(num_class,num_view)
    num_epoch=400
    p={'gcn_layer':trial.suggest_int('gcn_layer',1,3)}
    params={
            'adj_parameter':trial.suggest_int('adj_parameter',2,6),
            'dim_he_list':[trial.suggest_int(f'dim_he_list_{i}',8,136,step=4) for i in range(p["gcn_layer"])],
            'optimizer_name':trial.suggest_categorical('optimizer_name',["Adadelta","Adagrad","Adam","RMSprop","SGD"]),
            'lr_e':trial.suggest_loguniform('lr_e',1e-4,1e-2),
            'lr_c':trial.suggest_loguniform('lr_c',1e-4,1e-2),
            'weight_decay':trial.suggest_loguniform('weight_decay',1e-4,1e-3),
            'gcn_drop':trial.suggest_float('gcn_drop',0.1,0.5,step=0.1)
        }
    data_list, labels = prepare_trte_data(data_folder)
    labels_tensor = torch.LongTensor(labels)
    sample_weight = torch.FloatTensor(cal_sample_weight(labels, num_class))
    if cuda:
        labels_tensor = labels_tensor.cuda()
        sample_weight = sample_weight.cuda()
    adj_list = gen_trte_adj_mat(data_list, params['adj_parameter'])
    for i,edge in enumerate(adj_list):
        if i==0:
            A=edge.unsqueeze(-1)
        else:
            A=torch.cat([A,edge.unsqueeze(-1)],dim=-1)
    A=torch.cat([A,torch.eye(adj_list[0].shape[0]).cuda().unsqueeze(-1)],dim=-1)
    dim_list = [x.shape[1] for x in data_list]
    skf=StratifiedKFold(n_splits=5,random_state=0,shuffle=True)
    sizhetest=[]
    for i,(trind,teind) in enumerate(skf.split(data_list[0],labels)):
        model_dict = init_model_dict(num_view, num_class, dim_list, params['dim_he_list'], dim_hvcdn, sample_weight,params["gcn_drop"])
        for m in model_dict:
            if cuda:
                model_dict[m].cuda()
        optim_dict = init_optim(num_view, model_dict, params['optimizer_name'],params['lr_e'], params['lr_c'])
        te=[]
        for epoch in range(num_epoch):
            a,b=train_epoch(data_list, A, labels_tensor, sample_weight, trind,model_dict, optim_dict)
            te_prob,c = test_epoch(data_list, A, labels_tensor,sample_weight, teind,model_dict,)
            acc=accuracy_score(labels[teind], te_prob.argmax(1))
            te.append(acc)
        sizhetest.append(te[-1])
    return np.mean(sizhetest),np.std(sizhetest)
def load_model_dict(folder, model_dict):
    for module in model_dict:
        if os.path.exists(os.path.join(folder, module+".pth")):
#            print("Module {:} loaded!".format(module))
            model_dict[module].load_state_dict(torch.load(os.path.join(folder, module+".pth"), map_location="cuda:{:}".format(torch.cuda.current_device())))
        else:
            print("WARNING: Module {:} from model_dict is not loaded!".format(module))
        if cuda:
            model_dict[module].cuda()    
    return model_dict
def save_model_dict(folder, model_dict):
    if not os.path.exists(folder):
        os.makedirs(folder)
    for module in model_dict:
        torch.save(model_dict[module].state_dict(), os.path.join(folder, module+".pth"))
import copy
import pandas as pd
def cal_feat_imp(fold, model_folder):
    seed_torch()
    data_folder='/home/zhoulin/ucsc/read/all'
    num_class=2
    num_view = 3
    dim_hvcdn = pow(num_class,num_view)
    num_epoch=400
    p={'gcn_layer':1}
    params={
            'adj_parameter':6,
            'dim_he_list':[92],
            'optimizer_name':"Adam",
            'lr_e':0.003933933736199892,
            'lr_c':0.0011064028243744466,
            'weight_decay':0.00027516545023786436,
            'gcn_drop':0.5
        }
    data_list, labels = prepare_trte_data(data_folder)
    labels_tensor = torch.LongTensor(labels)
    sample_weight = torch.FloatTensor(cal_sample_weight(labels, num_class))
    if cuda:
        labels_tensor = labels_tensor.cuda()
        sample_weight = sample_weight.cuda()
    adj_list = gen_trte_adj_mat(data_list, params['adj_parameter'])
    for i,edge in enumerate(adj_list):
        if i==0:
            A=edge.unsqueeze(-1)
        else:
            A=torch.cat([A,edge.unsqueeze(-1)],dim=-1)
    A=torch.cat([A,torch.eye(adj_list[0].shape[0]).cuda().unsqueeze(-1)],dim=-1)
    dim_list = [x.shape[1] for x in data_list]
    featname_list = []
    featname_list.append(np.load('mrgene2.npy',allow_pickle=True))
    featname_list.append(np.load('megene2.npy',allow_pickle=True))
    featname_list.append(np.load('migene2.npy',allow_pickle=True))
    skf=StratifiedKFold(n_splits=5,random_state=0,shuffle=True)
    for i,(trind,teind) in enumerate(skf.split(data_list[0],labels)):
        if i==fold:
            model_dict = init_model_dict(num_view, num_class, dim_list, params['dim_he_list'], dim_hvcdn, sample_weight,params["gcn_drop"])
            for m in model_dict:
                if cuda:
                    model_dict[m].cuda()
            model_dict = load_model_dict(model_folder, model_dict)
            te_prob,c = test_epoch(data_list, A, labels_tensor,sample_weight, teind,model_dict,)
            acc=accuracy_score(labels[teind], te_prob.argmax(1))
            feat_imp_list = []
            for k in range(len(featname_list)):
                feat_imp = {"feat_name":featname_list[k]}
                feat_imp['imp'] = np.zeros(dim_list[k])
                for j in range(dim_list[k]):
                    feat_tr = data_list[k][:,j].clone()
                    data_list[k][:,j] = 0
                    adj_list = gen_trte_adj_mat(data_list, params['adj_parameter'])
                    for i,edge in enumerate(adj_list):
                        if i==0:
                            A=edge.unsqueeze(-1)
                        else:
                            A=torch.cat([A,edge.unsqueeze(-1)],dim=-1)
                    A=torch.cat([A,torch.eye(adj_list[0].shape[0]).cuda().unsqueeze(-1)],dim=-1)
                    te_prob,c = test_epoch(data_list, A, labels_tensor,sample_weight, teind,model_dict,)
                    acc_tmp=accuracy_score(labels[teind], te_prob.argmax(1))
                    feat_imp['imp'][j] = (acc-acc_tmp)*dim_list[k]
                    data_list[k][:,j] = feat_tr.clone()
                feat_imp_list.append(pd.DataFrame(data=feat_imp))    
    return feat_imp_list
def summarize_imp_feat(featimp_list_list, topn=30):
    num_rep = len(featimp_list_list)
    num_view = len(featimp_list_list[0])
    df_tmp_list = []
    for v in range(num_view):
        df_tmp = copy.deepcopy(featimp_list_list[0][v])
        df_tmp['omics'] = np.ones(df_tmp.shape[0], dtype=int)*v
        df_tmp_list.append(df_tmp.copy(deep=True))
    df_featimp = pd.concat(df_tmp_list).copy(deep=True)
    for r in range(1,num_rep):
        for v in range(num_view):
            df_tmp = copy.deepcopy(featimp_list_list[r][v])
            df_tmp['omics'] = np.ones(df_tmp.shape[0], dtype=int)*v
            df_featimp = df_featimp.append(df_tmp.copy(deep=True), ignore_index=True) 
    df_featimp_top = df_featimp.groupby(['feat_name', 'omics'])['imp'].sum()
    df_featimp_top = df_featimp_top.reset_index()
    df_featimp_top = df_featimp_top.sort_values(by='imp',ascending=False)
    df_featimp_top1 = df_featimp_top.iloc[:topn]
    print('{:}\t{:}'.format('Rank','Feature name'))
    for i in range(len(df_featimp_top1)):
        print('{:}\t{:}'.format(i+1,df_featimp_top1.iloc[i]['feat_name']))
    mr=df_featimp_top[df_featimp_top['omics']==0]
    me=df_featimp_top[df_featimp_top['omics']==1]
    mi=df_featimp_top[df_featimp_top['omics']==2]
    mr.to_csv('mrgene_rank.csv')
    me.to_csv('megene_rank.csv')
    mi.to_csv('migene_rank.csv')
def objective1():
    seed_torch()
    data_folder='/home/zhoulin/ucsc/read/all'
    num_class=2
    num_view = 3
    dim_hvcdn = pow(num_class,num_view)
    num_epoch=400
    p={'gcn_layer':1}
    params={
            'adj_parameter':7,
            'dim_he_list':[92],
            'optimizer_name':"Adam",
            'lr_e':0.003933933736199892,
            'lr_c':0.0011064028243744466,
            'weight_decay':0.00027516545023786436,
            'gcn_drop':0.5
        }
    data_list, labels = prepare_trte_data(data_folder)
    labels_tensor = torch.LongTensor(labels)
    sample_weight = torch.FloatTensor(cal_sample_weight(labels, num_class))
    if cuda:
        labels_tensor = labels_tensor.cuda()
        sample_weight = sample_weight.cuda()
    # #不同adjparameter
    # alljieguo=[]
    # for ap in range(2,11):
    #     params['adj_parameter']=ap
    # alljieguo=[]
    # for lr in [0.000001,0.00001,0.003933933736199892,0.01,0.1]:
    #     params['lr_e']=lr
    # alljieguo=[]
    # for opn in ["Adadelta","Adagrad","Adam","RMSprop","SGD"]:
    #     params['optimizer_name']=opn
    alljieguo=[]
    a=[32,64,92,108,124,140]
    for opn in a:
        params['dim_he_list']=[opn]
        seed_torch()
        adj_list = gen_trte_adj_mat(data_list, params['adj_parameter'])
        for i,edge in enumerate(adj_list):
            if i==0:
                A=edge.unsqueeze(-1)
            else:
                A=torch.cat([A,edge.unsqueeze(-1)],dim=-1)
        A=torch.cat([A,torch.eye(adj_list[0].shape[0]).cuda().unsqueeze(-1)],dim=-1)
        dim_list = [x.shape[1] for x in data_list]
        skf=StratifiedKFold(n_splits=5,random_state=0,shuffle=True)
        jieguo,sizhetest=[],[]
        for i,(trind,teind) in enumerate(skf.split(data_list[0],labels)):
            model_dict = init_model_dict(num_view, num_class, dim_list, params['dim_he_list'], dim_hvcdn, sample_weight,params["gcn_drop"])
            for m in model_dict:
                if cuda:
                    model_dict[m].cuda()
            optim_dict = init_optim(num_view, model_dict, params['optimizer_name'],params['lr_e'], params['lr_c'])
            te=[]
            for epoch in range(num_epoch):
                a,b=train_epoch(data_list, A, labels_tensor, sample_weight, trind,model_dict, optim_dict)
                te_prob,c = test_epoch(data_list, A, labels_tensor,sample_weight, teind,model_dict,)
                acc=accuracy_score(labels[teind], te_prob.argmax(1))
                te.append(acc)
                if epoch==num_epoch-1:
                    jieguo.append(te_prob)
                    jieguo.append(acc)
                    save_model_dict(f'models/{i}',model_dict)
            sizhetest.append(te[-1])
            jieguo.append(labels[teind])
        # f=open('read1a3f.pkl','wb')
        # pickle.dump(jieguo,f)
        # f.close()
        alljieguo.append(jieguo)
    # f=open('adjparam1a3f.pkl','wb')
    # pickle.dump(alljieguo,f)
    # f.close()
    # f=open('lr1a3f.pkl','wb')
    # pickle.dump(alljieguo,f)
    # f.close()
    # f=open('optim1a3f.pkl','wb')
    # pickle.dump(alljieguo,f)
    # f.close()
    f=open('units1a3f.pkl','wb')
    pickle.dump(alljieguo,f)
    f.close()
    return np.mean(sizhetest),np.std(sizhetest)
def prepare_trte_datamr(data_folder):
    labels = np.loadtxt(os.path.join(data_folder, "label2.csv"), delimiter=',')
    labels = labels.astype(int)
    data = []
    data.append(np.loadtxt(os.path.join(data_folder, "mr2.csv"), delimiter=','))
    # data.append(np.loadtxt(os.path.join(data_folder, "me2.csv"), delimiter=','))
    # data.append(np.loadtxt(os.path.join(data_folder, "mi2.csv"), delimiter=','))
    data_tensor_list = []
    
    for i in range(len(data)):
        data_tensor_list.append(torch.FloatTensor(data[i]))
        if cuda:
            data_tensor_list[i] = data_tensor_list[i].cuda()
    return data_tensor_list,  labels
def mr():
    seed_torch()
    data_folder='/home/zhoulin/ucsc/read/all'
    num_class=2
    num_view = 1
    dim_hvcdn = pow(num_class,num_view)
    num_epoch=400
    p={'gcn_layer':1}
    params={
            'adj_parameter':7,
            'dim_he_list':[92],
            'optimizer_name':"Adam",
            'lr_e':0.003933933736199892,
            'lr_c':0.0011064028243744466,
            'weight_decay':0.00027516545023786436,
            'gcn_drop':0.5
        }
    data_list, labels = prepare_trte_datamr(data_folder)
    labels_tensor = torch.LongTensor(labels)
    sample_weight = torch.FloatTensor(cal_sample_weight(labels, num_class))
    if cuda:
        labels_tensor = labels_tensor.cuda()
        sample_weight = sample_weight.cuda()
    adj_list = gen_trte_adj_mat(data_list, params['adj_parameter'])
    for i,edge in enumerate(adj_list):
        if i==0:
            A=edge.unsqueeze(-1)
        else:
            A=torch.cat([A,edge.unsqueeze(-1)],dim=-1)
    A=torch.cat([A,torch.eye(adj_list[0].shape[0]).cuda().unsqueeze(-1)],dim=-1)
    
    dim_list = [x.shape[1] for x in data_list]
    skf=StratifiedKFold(n_splits=5,random_state=0,shuffle=True)
    jieguo,sizhetest=[],[]
    for i,(trind,teind) in enumerate(skf.split(data_list[0],labels)):
        model_dict = init_model_dict(num_view, num_class, dim_list, params['dim_he_list'], dim_hvcdn, sample_weight,params["gcn_drop"],num_edge=2)
        for m in model_dict:
            if cuda:
                model_dict[m].cuda()
        optim_dict = init_optim(num_view, model_dict, params['optimizer_name'],params['lr_e'], params['lr_c'])
        te=[]
        for epoch in range(num_epoch):
            a,b=train_epoch(data_list, A, labels_tensor, sample_weight, trind,model_dict, optim_dict)
            te_prob,c = test_epoch(data_list, A, labels_tensor,sample_weight, teind,model_dict,)
            acc=accuracy_score(labels[teind], te_prob.argmax(1))
            te.append(acc)
            if epoch==num_epoch-1:
                jieguo.append(te_prob)
                jieguo.append(acc)
        sizhetest.append(te[-1])
        jieguo.append(labels[teind])
    f=open('read1a3fmr.pkl','wb')
    pickle.dump(jieguo,f)
    f.close()
    return np.mean(sizhetest),np.std(sizhetest)

def prepare_trte_datame(data_folder):
    labels = np.loadtxt(os.path.join(data_folder, "label2.csv"), delimiter=',')
    labels = labels.astype(int)
    data = []
    # data.append(np.loadtxt(os.path.join(data_folder, "mr2.csv"), delimiter=','))
    data.append(np.loadtxt(os.path.join(data_folder, "me2.csv"), delimiter=','))
    # data.append(np.loadtxt(os.path.join(data_folder, "mi2.csv"), delimiter=','))
    data_tensor_list = []
    
    for i in range(len(data)):
        data_tensor_list.append(torch.FloatTensor(data[i]))
        if cuda:
            data_tensor_list[i] = data_tensor_list[i].cuda()
    return data_tensor_list,  labels
def me():
    seed_torch()
    data_folder='/home/zhoulin/ucsc/read/all'
    num_class=2
    num_view = 1
    dim_hvcdn = pow(num_class,num_view)
    num_epoch=400
    p={'gcn_layer':1}
    params={
            'adj_parameter':7,
            'dim_he_list':[92],
            'optimizer_name':"Adam",
            'lr_e':0.003933933736199892,
            'lr_c':0.0011064028243744466,
            'weight_decay':0.00027516545023786436,
            'gcn_drop':0.5
        }
    data_list, labels = prepare_trte_datame(data_folder)
    labels_tensor = torch.LongTensor(labels)
    sample_weight = torch.FloatTensor(cal_sample_weight(labels, num_class))
    if cuda:
        labels_tensor = labels_tensor.cuda()
        sample_weight = sample_weight.cuda()
    adj_list = gen_trte_adj_mat(data_list, params['adj_parameter'])
    for i,edge in enumerate(adj_list):
        if i==0:
            A=edge.unsqueeze(-1)
        else:
            A=torch.cat([A,edge.unsqueeze(-1)],dim=-1)
    A=torch.cat([A,torch.eye(adj_list[0].shape[0]).cuda().unsqueeze(-1)],dim=-1)
    
    dim_list = [x.shape[1] for x in data_list]
    skf=StratifiedKFold(n_splits=5,random_state=0,shuffle=True)
    jieguo,sizhetest=[],[]
    for i,(trind,teind) in enumerate(skf.split(data_list[0],labels)):
        model_dict = init_model_dict(num_view, num_class, dim_list, params['dim_he_list'], dim_hvcdn, sample_weight,params["gcn_drop"],num_edge=2)
        for m in model_dict:
            if cuda:
                model_dict[m].cuda()
        optim_dict = init_optim(num_view, model_dict, params['optimizer_name'],params['lr_e'], params['lr_c'])
        te=[]
        for epoch in range(num_epoch):
            a,b=train_epoch(data_list, A, labels_tensor, sample_weight, trind,model_dict, optim_dict)
            te_prob,c = test_epoch(data_list, A, labels_tensor,sample_weight, teind,model_dict,)
            acc=accuracy_score(labels[teind], te_prob.argmax(1))
            te.append(acc)
            if epoch==num_epoch-1:
                jieguo.append(te_prob)
                jieguo.append(acc)
        sizhetest.append(te[-1])
        jieguo.append(labels[teind])
    f=open('read1a3fme.pkl','wb')
    pickle.dump(jieguo,f)
    f.close()
    return np.mean(sizhetest),np.std(sizhetest)
def prepare_trte_datami(data_folder):
    labels = np.loadtxt(os.path.join(data_folder, "label2.csv"), delimiter=',')
    labels = labels.astype(int)
    data = []
    # data.append(np.loadtxt(os.path.join(data_folder, "mr2.csv"), delimiter=','))
    # data.append(np.loadtxt(os.path.join(data_folder, "me2.csv"), delimiter=','))
    data.append(np.loadtxt(os.path.join(data_folder, "mi2.csv"), delimiter=','))
    data_tensor_list = []
    
    for i in range(len(data)):
        data_tensor_list.append(torch.FloatTensor(data[i]))
        if cuda:
            data_tensor_list[i] = data_tensor_list[i].cuda()
    return data_tensor_list,  labels
def mi():
    seed_torch()
    data_folder='/home/zhoulin/ucsc/read/all'
    num_class=2
    num_view = 1
    dim_hvcdn = pow(num_class,num_view)
    num_epoch=400
    p={'gcn_layer':1}
    params={
            'adj_parameter':7,
            'dim_he_list':[92],
            'optimizer_name':"Adam",
            'lr_e':0.003933933736199892,
            'lr_c':0.0011064028243744466,
            'weight_decay':0.00027516545023786436,
            'gcn_drop':0.5
        }
    data_list, labels = prepare_trte_datami(data_folder)
    labels_tensor = torch.LongTensor(labels)
    sample_weight = torch.FloatTensor(cal_sample_weight(labels, num_class))
    if cuda:
        labels_tensor = labels_tensor.cuda()
        sample_weight = sample_weight.cuda()
    adj_list = gen_trte_adj_mat(data_list, params['adj_parameter'])
    for i,edge in enumerate(adj_list):
        if i==0:
            A=edge.unsqueeze(-1)
        else:
            A=torch.cat([A,edge.unsqueeze(-1)],dim=-1)
    A=torch.cat([A,torch.eye(adj_list[0].shape[0]).cuda().unsqueeze(-1)],dim=-1)
    
    dim_list = [x.shape[1] for x in data_list]
    skf=StratifiedKFold(n_splits=5,random_state=0,shuffle=True)
    jieguo,sizhetest=[],[]
    for i,(trind,teind) in enumerate(skf.split(data_list[0],labels)):
        model_dict = init_model_dict(num_view, num_class, dim_list, params['dim_he_list'], dim_hvcdn, sample_weight,params["gcn_drop"],num_edge=2)
        for m in model_dict:
            if cuda:
                model_dict[m].cuda()
        optim_dict = init_optim(num_view, model_dict, params['optimizer_name'],params['lr_e'], params['lr_c'])
        te=[]
        for epoch in range(num_epoch):
            a,b=train_epoch(data_list, A, labels_tensor, sample_weight, trind,model_dict, optim_dict)
            te_prob,c = test_epoch(data_list, A, labels_tensor,sample_weight, teind,model_dict,)
            acc=accuracy_score(labels[teind], te_prob.argmax(1))
            te.append(acc)
            if epoch==num_epoch-1:
                jieguo.append(te_prob)
                jieguo.append(acc)
        sizhetest.append(te[-1])
        jieguo.append(labels[teind])
    f=open('read1a3fmi.pkl','wb')
    pickle.dump(jieguo,f)
    f.close()
    return np.mean(sizhetest),np.std(sizhetest)
def prepare_trte_datamrme(data_folder):
    labels = np.loadtxt(os.path.join(data_folder, "label2.csv"), delimiter=',')
    labels = labels.astype(int)
    data = []
    data.append(np.loadtxt(os.path.join(data_folder, "mr2.csv"), delimiter=','))
    data.append(np.loadtxt(os.path.join(data_folder, "me2.csv"), delimiter=','))
    # data.append(np.loadtxt(os.path.join(data_folder, "mi2.csv"), delimiter=','))
    data_tensor_list = []
    
    for i in range(len(data)):
        data_tensor_list.append(torch.FloatTensor(data[i]))
        if cuda:
            data_tensor_list[i] = data_tensor_list[i].cuda()
    return data_tensor_list,  labels
def mrme():
    seed_torch()
    data_folder='/home/zhoulin/ucsc/read/all'
    num_class=2
    num_view = 2
    dim_hvcdn = pow(num_class,num_view)
    num_epoch=400
    p={'gcn_layer':1}
    params={
            'adj_parameter':7,
            'dim_he_list':[92],
            'optimizer_name':"Adam",
            'lr_e':0.003933933736199892,
            'lr_c':0.0011064028243744466,
            'weight_decay':0.00027516545023786436,
            'gcn_drop':0.5
        }
    data_list, labels = prepare_trte_datamrme(data_folder)
    labels_tensor = torch.LongTensor(labels)
    sample_weight = torch.FloatTensor(cal_sample_weight(labels, num_class))
    if cuda:
        labels_tensor = labels_tensor.cuda()
        sample_weight = sample_weight.cuda()
    adj_list = gen_trte_adj_mat(data_list, params['adj_parameter'])
    for i,edge in enumerate(adj_list):
        if i==0:
            A=edge.unsqueeze(-1)
        else:
            A=torch.cat([A,edge.unsqueeze(-1)],dim=-1)
    A=torch.cat([A,torch.eye(adj_list[0].shape[0]).cuda().unsqueeze(-1)],dim=-1)
    
    dim_list = [x.shape[1] for x in data_list]
    skf=StratifiedKFold(n_splits=5,random_state=0,shuffle=True)
    jieguo,sizhetest=[],[]
    for i,(trind,teind) in enumerate(skf.split(data_list[0],labels)):
        model_dict = init_model_dict(num_view, num_class, dim_list, params['dim_he_list'], dim_hvcdn, sample_weight,params["gcn_drop"],num_edge=3)
        for m in model_dict:
            if cuda:
                model_dict[m].cuda()
        optim_dict = init_optim(num_view, model_dict, params['optimizer_name'],params['lr_e'], params['lr_c'])
        te=[]
        for epoch in range(num_epoch):
            a,b=train_epoch(data_list, A, labels_tensor, sample_weight, trind,model_dict, optim_dict)
            te_prob,c = test_epoch(data_list, A, labels_tensor,sample_weight, teind,model_dict,)
            acc=accuracy_score(labels[teind], te_prob.argmax(1))
            te.append(acc)
            if epoch==num_epoch-1:
                jieguo.append(te_prob)
                jieguo.append(acc)
        sizhetest.append(te[-1])
        jieguo.append(labels[teind])
    f=open('read1a3fmrme.pkl','wb')
    pickle.dump(jieguo,f)
    f.close()
    return np.mean(sizhetest),np.std(sizhetest)
def prepare_trte_datamrmi(data_folder):
    labels = np.loadtxt(os.path.join(data_folder, "label2.csv"), delimiter=',')
    labels = labels.astype(int)
    data = []
    data.append(np.loadtxt(os.path.join(data_folder, "mr2.csv"), delimiter=','))
    # data.append(np.loadtxt(os.path.join(data_folder, "me2.csv"), delimiter=','))
    data.append(np.loadtxt(os.path.join(data_folder, "mi2.csv"), delimiter=','))
    data_tensor_list = []
    
    for i in range(len(data)):
        data_tensor_list.append(torch.FloatTensor(data[i]))
        if cuda:
            data_tensor_list[i] = data_tensor_list[i].cuda()
    return data_tensor_list,  labels
def mrmi():
    seed_torch()
    data_folder='/home/zhoulin/ucsc/read/all'
    num_class=2
    num_view = 2
    dim_hvcdn = pow(num_class,num_view)
    num_epoch=400
    p={'gcn_layer':1}
    params={
            'adj_parameter':7,
            'dim_he_list':[92],
            'optimizer_name':"Adam",
            'lr_e':0.003933933736199892,
            'lr_c':0.0011064028243744466,
            'weight_decay':0.00027516545023786436,
            'gcn_drop':0.5
        }
    data_list, labels = prepare_trte_datamrmi(data_folder)
    labels_tensor = torch.LongTensor(labels)
    sample_weight = torch.FloatTensor(cal_sample_weight(labels, num_class))
    if cuda:
        labels_tensor = labels_tensor.cuda()
        sample_weight = sample_weight.cuda()
    adj_list = gen_trte_adj_mat(data_list, params['adj_parameter'])
    for i,edge in enumerate(adj_list):
        if i==0:
            A=edge.unsqueeze(-1)
        else:
            A=torch.cat([A,edge.unsqueeze(-1)],dim=-1)
    A=torch.cat([A,torch.eye(adj_list[0].shape[0]).cuda().unsqueeze(-1)],dim=-1)
    
    dim_list = [x.shape[1] for x in data_list]
    skf=StratifiedKFold(n_splits=5,random_state=0,shuffle=True)
    jieguo,sizhetest=[],[]
    for i,(trind,teind) in enumerate(skf.split(data_list[0],labels)):
        model_dict = init_model_dict(num_view, num_class, dim_list, params['dim_he_list'], dim_hvcdn, sample_weight,params["gcn_drop"],num_edge=3)
        for m in model_dict:
            if cuda:
                model_dict[m].cuda()
        optim_dict = init_optim(num_view, model_dict, params['optimizer_name'],params['lr_e'], params['lr_c'])
        te=[]
        for epoch in range(num_epoch):
            a,b=train_epoch(data_list, A, labels_tensor, sample_weight, trind,model_dict, optim_dict)
            te_prob,c = test_epoch(data_list, A, labels_tensor,sample_weight, teind,model_dict,)
            acc=accuracy_score(labels[teind], te_prob.argmax(1))
            te.append(acc)
            if epoch==num_epoch-1:
                jieguo.append(te_prob)
                jieguo.append(acc)
        sizhetest.append(te[-1])
        jieguo.append(labels[teind])
    f=open('read1a3fmrmi.pkl','wb')
    pickle.dump(jieguo,f)
    f.close()
    return np.mean(sizhetest),np.std(sizhetest)
def prepare_trte_datamime(data_folder):
    labels = np.loadtxt(os.path.join(data_folder, "label2.csv"), delimiter=',')
    labels = labels.astype(int)
    data = []
    # data.append(np.loadtxt(os.path.join(data_folder, "mr2.csv"), delimiter=','))
    data.append(np.loadtxt(os.path.join(data_folder, "me2.csv"), delimiter=','))
    data.append(np.loadtxt(os.path.join(data_folder, "mi2.csv"), delimiter=','))
    data_tensor_list = []
    
    for i in range(len(data)):
        data_tensor_list.append(torch.FloatTensor(data[i]))
        if cuda:
            data_tensor_list[i] = data_tensor_list[i].cuda()
    return data_tensor_list,  labels
def mime():
    seed_torch()
    data_folder='/home/zhoulin/ucsc/read/all'
    num_class=2
    num_view = 2
    dim_hvcdn = pow(num_class,num_view)
    num_epoch=400
    p={'gcn_layer':1}
    params={
            'adj_parameter':7,
            'dim_he_list':[92],
            'optimizer_name':"Adam",
            'lr_e':0.003933933736199892,
            'lr_c':0.0011064028243744466,
            'weight_decay':0.00027516545023786436,
            'gcn_drop':0.5
        }
    data_list, labels = prepare_trte_datamime(data_folder)
    labels_tensor = torch.LongTensor(labels)
    sample_weight = torch.FloatTensor(cal_sample_weight(labels, num_class))
    if cuda:
        labels_tensor = labels_tensor.cuda()
        sample_weight = sample_weight.cuda()
    adj_list = gen_trte_adj_mat(data_list, params['adj_parameter'])
    for i,edge in enumerate(adj_list):
        if i==0:
            A=edge.unsqueeze(-1)
        else:
            A=torch.cat([A,edge.unsqueeze(-1)],dim=-1)
    A=torch.cat([A,torch.eye(adj_list[0].shape[0]).cuda().unsqueeze(-1)],dim=-1)
    
    dim_list = [x.shape[1] for x in data_list]
    skf=StratifiedKFold(n_splits=5,random_state=0,shuffle=True)
    jieguo,sizhetest=[],[]
    for i,(trind,teind) in enumerate(skf.split(data_list[0],labels)):
        model_dict = init_model_dict(num_view, num_class, dim_list, params['dim_he_list'], dim_hvcdn, sample_weight,params["gcn_drop"],num_edge=3)
        for m in model_dict:
            if cuda:
                model_dict[m].cuda()
        optim_dict = init_optim(num_view, model_dict, params['optimizer_name'],params['lr_e'], params['lr_c'])
        te=[]
        for epoch in range(num_epoch):
            a,b=train_epoch(data_list, A, labels_tensor, sample_weight, trind,model_dict, optim_dict)
            te_prob,c = test_epoch(data_list, A, labels_tensor,sample_weight, teind,model_dict,)
            acc=accuracy_score(labels[teind], te_prob.argmax(1))
            te.append(acc)
            if epoch==num_epoch-1:
                jieguo.append(te_prob)
                jieguo.append(acc)
        sizhetest.append(te[-1])
        jieguo.append(labels[teind])
    f=open('read1a3fmime.pkl','wb')
    pickle.dump(jieguo,f)
    f.close()
    return np.mean(sizhetest),np.std(sizhetest)
if __name__=='__main__':
    seed_torch()
    print(objective1())
    # for a in range(200,300,4):
    #     print(a)
    #     print(objective1(a))
    #     print('-----------')
    # featimp_list_list = []
    # for rep in range(5):
    #     featimp_list = cal_feat_imp(rep, f'models/{rep}')
    #     featimp_list_list.append(copy.deepcopy(featimp_list))
    # summarize_imp_feat(featimp_list_list)
    # print(mr())
    # print(me())
    # print(mi())
    # print(mrme())
    # print(mrmi())
    # print(mime())
    # start=time.time()
    # seed_torch()
    # study=optuna.create_study(directions=['maximize','minimize'])
    # study.optimize(objective,n_trials=2000,gc_after_trial=True,n_jobs=8)
    # res=[]
    # for i in range(len(study.get_trials())):
    #     tmp=[]
    #     tmp.append(study.get_trials()[i].values[0])
    #     tmp.append(study.get_trials()[i].values[1])
    #     tmp.append(study.get_trials()[i].params)
    #     res.append(tmp)
    # out=open('1A3Fgcn.csv','w',newline="")
    # csvwriter=csv.writer(out,dialect='excel')
    # csvwriter.writerow(['mean','std','parameters'])
    # csvwriter.writerows(res)
    # out.close()

    # end=time.time()
    # print('time : ',(end-start)/60)