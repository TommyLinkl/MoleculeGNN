import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, GCNConv, GATConv
import torch_geometric
from torch_geometric.nn import Set2Set, MetaLayer
from torch_geometric.nn.models.schnet import InteractionBlock
from torch.nn import Sequential, Linear, BatchNorm1d
from torch_scatter import scatter_mean, scatter


class SchNet(nn.Module):
    def __init__(
        self,
        data,
        dim1=64,
        dim2=64,
        dim3=64,
        cutoff=8,
        pre_fc_count=1,
        gc_count=3,
        post_fc_count=1,
        pool="global_mean_pool",
        dropout_rate=0.0
    ):

        super(SchNet, self).__init__()
        self.batch_track_stats = True
        self.pool = pool
        self.dropout_rate = dropout_rate

        assert gc_count > 0, "Need at least 1 GC layer"
        assert pre_fc_count > 0, "Need at least 1 pre-FC layer"
        assert post_fc_count > 0, "Need at least 1 post-FC layer"
        gc_dim = dim1
        post_fc_dim = dim1
        if data[0].y.ndim == 0:
            output_dim = 1
        else:
            output_dim = len(data[0].y[0])

        # Set up pre-GNN dense layers
        self.pre_l = nn.ModuleList()
        for i in range(pre_fc_count):
            if i == 0:
                lin = nn.Linear(data.num_features, dim1)
            else:
                lin = nn.Linear(dim1, dim1)
            nn.init.xavier_uniform_(lin.weight)  # Xavier initialization
            self.pre_l.append(lin)

        # Set up GNN layers
        self.conv_list = nn.ModuleList()
        self.bn_list = nn.ModuleList()
        for i in range(gc_count):
            conv = InteractionBlock(gc_dim, data.num_edge_features, dim3, cutoff)
            self.conv_list.append(conv)
            bn = BatchNorm1d(gc_dim, track_running_stats=self.batch_track_stats)
            self.bn_list.append(bn)

        # Set up post-GNN dense layers
        self.post_l = nn.ModuleList()
        for i in range(post_fc_count):
            if i == 0:
                lin = nn.Linear(post_fc_dim, dim2)
            else:
                lin = nn.Linear(dim2, dim2)
            nn.init.xavier_uniform_(lin.weight)  # Xavier initialization
            self.post_l.append(lin)
        self.lin_out = nn.Linear(dim2, output_dim)
        nn.init.xavier_uniform_(self.lin_out.weight)  # Xavier initialization

    def forward(self, data):
        # Pre-GNN dense layers
        out = data.x
        for layer in self.pre_l:
            out = torch.relu(layer(out))

        # GNN layers
        for layer, bn_layer in zip(self.conv_list, self.bn_list):
            out = out + layer(out, data.edge_index, data.edge_weight, data.edge_attr)
            out = bn_layer(out)
            out = nn.functional.dropout(out, p=self.dropout_rate, training=self.training)
        
        # Post-GNN dense layers
        out = getattr(torch_geometric.nn, self.pool)(out, data.batch)
        for layer in self.post_l:
            out = torch.relu(layer(out))
        out = self.lin_out(out)
        
        return out.view(-1) if out.shape[1] == 1 else out



class Megnet_EdgeModel(torch.nn.Module):
    def __init__(self, dim, dropout_rate, fc_layers=2):
        super(Megnet_EdgeModel, self).__init__()
        self.fc_layers = fc_layers
        self.batch_track_stats = True
        self.dropout_rate = dropout_rate
        
        self.edge_mlp = torch.nn.ModuleList()
        self.bn_list = torch.nn.ModuleList()
        for i in range(self.fc_layers + 1):
            if i == 0:
                lin = torch.nn.Linear(dim * 4, dim)
            else:
                lin = torch.nn.Linear(dim, dim)
            nn.init.xavier_uniform_(lin.weight)  # Xavier initialization
            self.edge_mlp.append(lin) 
            bn = BatchNorm1d(dim, track_running_stats=self.batch_track_stats)
            self.bn_list.append(bn)

    def forward(self, src, dest, edge_attr, u, batch):
        comb = torch.cat([src, dest, edge_attr, u[batch]], dim=1)
        out = comb
        for layer, bn_layer in zip(self.edge_mlp, self.bn_list):
            out = torch.relu(layer(out))
            out = bn_layer(out)
            out = nn.functional.dropout(out, p=self.dropout_rate, training=self.training)
        return out


class Megnet_NodeModel(torch.nn.Module):
    def __init__(self, dim, dropout_rate, fc_layers=2):
        super(Megnet_NodeModel, self).__init__()
        self.fc_layers = fc_layers
        self.batch_track_stats = True
        self.dropout_rate = dropout_rate
                
        self.node_mlp = torch.nn.ModuleList()
        self.bn_list = torch.nn.ModuleList()
        for i in range(self.fc_layers + 1):
            if i == 0:
                lin = torch.nn.Linear(dim * 3, dim)
            else:      
                lin = torch.nn.Linear(dim, dim)
            nn.init.xavier_uniform_(lin.weight)  # Xavier initialization
            self.node_mlp.append(lin) 
            bn = BatchNorm1d(dim, track_running_stats=self.batch_track_stats)
            self.bn_list.append(bn)    

    def forward(self, x, edge_index, edge_attr, u, batch):
        # Compute mean of edge attributes per node
        v_e = scatter_mean(edge_attr, edge_index[0, :], dim=0)
        
        # Concatenate node features, mean edge features, and u[batch]
        comb = torch.cat([x, v_e, u[batch]], dim=1)
        
        # Apply MLP layers with batch normalization and dropout
        out = comb
        for layer, bn_layer in zip(self.node_mlp, self.bn_list):
            out = torch.relu(layer(out))
            out = bn_layer(out)
            out = nn.functional.dropout(out, p=self.dropout_rate, training=self.training)
        return out
    

class Megnet_GlobalModel(torch.nn.Module):
    def __init__(self, dim, dropout_rate, fc_layers=2):
        super(Megnet_GlobalModel, self).__init__()
        self.fc_layers = fc_layers
        self.batch_track_stats = True
        self.dropout_rate = dropout_rate
                
        self.global_mlp = torch.nn.ModuleList()
        self.bn_list = torch.nn.ModuleList()
        for i in range(self.fc_layers + 1):
            if i == 0:
                lin = torch.nn.Linear(dim * 3, dim)       
            else:      
                lin = torch.nn.Linear(dim, dim)
            nn.init.xavier_uniform_(lin.weight)  # Xavier initialization
            self.global_mlp.append(lin) 
            bn = BatchNorm1d(dim, track_running_stats=self.batch_track_stats)
            self.bn_list.append(bn)  

    def forward(self, x, edge_index, edge_attr, u, batch):
        u_e = scatter_mean(edge_attr, edge_index[0, :], dim=0)
        u_e = scatter_mean(u_e, batch, dim=0)
        u_v = scatter_mean(x, batch, dim=0)
        comb = torch.cat([u_e, u_v, u], dim=1)

        out = comb
        for layer, bn_layer in zip(self.global_mlp, self.bn_list):
            out = torch.relu(layer(out))
            out = bn_layer(out)
            out = nn.functional.dropout(out, p=self.dropout_rate, training=self.training)
        return out


class MEGNet(torch.nn.Module):
    def __init__(
        self,
        data,
        dim1=64,
        dim2=64,
        dim3=64,
        pre_fc_count=1,
        gc_count=3,
        gc_fc_count=2,
        post_fc_count=1,
        pool="global_mean_pool",
        dropout_rate=0.0,
    ):
        super(MEGNet, self).__init__()
        
        self.batch_track_stats = True
        self.pool = pool
        if pool == "global_mean_pool":
            self.pool_reduce="mean"
        elif pool== "global_max_pool":
            self.pool_reduce="max" 
        elif pool== "global_sum_pool":
            self.pool_reduce="sum"
        self.dropout_rate = dropout_rate
        
        assert gc_count > 0, "Need at least 1 GC layer"
        assert pre_fc_count > 0, "Need at least 1 pre-FC layer"
        assert post_fc_count > 0, "Need at least 1 post-FC layer"
        gc_dim = dim1
        post_fc_dim = dim3
        if data[0].y.ndim == 0:
            output_dim = 1
        else:
            output_dim = len(data[0].y[0])

        # Set up pre-GNN dense layers
        self.pre_lin_list = torch.nn.ModuleList()
        for i in range(pre_fc_count):
            if i == 0:
                lin = torch.nn.Linear(data.num_features, dim1)
            else:
                lin = torch.nn.Linear(dim1, dim1)
            nn.init.xavier_uniform_(lin.weight)  # Xavier initialization
            self.pre_lin_list.append(lin)

        # Set up GNN layers
        self.e_embed_list = torch.nn.ModuleList()
        self.x_embed_list = torch.nn.ModuleList()
        self.u_embed_list = torch.nn.ModuleList()   
        self.conv_list = torch.nn.ModuleList()
        self.bn_list = torch.nn.ModuleList()
        for i in range(gc_count):
            if i == 0:
                e_embed = Sequential(Linear(data.num_edge_features, dim3), torch.relu(), Linear(dim3, dim3), torch.relu())
                x_embed = Sequential(Linear(gc_dim, dim3), torch.relu(), Linear(dim3, dim3), torch.relu())
                u_embed = Sequential(Linear((data[0].u.shape[1]), dim3), torch.relu(), Linear(dim3, dim3), torch.relu())
                self.e_embed_list.append(e_embed)
                self.x_embed_list.append(x_embed)
                self.u_embed_list.append(u_embed)
                self.conv_list.append(
                    MetaLayer(
                        Megnet_EdgeModel(dim3, self.batch_track_stats, self.dropout_rate, gc_fc_count),
                        Megnet_NodeModel(dim3, self.batch_track_stats, self.dropout_rate, gc_fc_count),
                        Megnet_GlobalModel(dim3, self.batch_track_stats, self.dropout_rate, gc_fc_count),
                    )
                )
            elif i > 0:
                e_embed = Sequential(Linear(dim3, dim3), torch.relu(), Linear(dim3, dim3), torch.relu())
                x_embed = Sequential(Linear(dim3, dim3), torch.relu(), Linear(dim3, dim3), torch.relu())
                u_embed = Sequential(Linear(dim3, dim3), torch.relu(), Linear(dim3, dim3), torch.relu())
                self.e_embed_list.append(e_embed)
                self.x_embed_list.append(x_embed)
                self.u_embed_list.append(u_embed)
                self.conv_list.append(
                    MetaLayer(
                        Megnet_EdgeModel(dim3, self.batch_track_stats, self.dropout_rate, gc_fc_count),
                        Megnet_NodeModel(dim3, self.batch_track_stats, self.dropout_rate, gc_fc_count),
                        Megnet_GlobalModel(dim3, self.batch_track_stats, self.dropout_rate, gc_fc_count),
                    )
                )

        # Set up post-GNN dense layers
        self.post_lin_list = torch.nn.ModuleList()
        for i in range(post_fc_count):
            if i == 0:
                lin = torch.nn.Linear(post_fc_dim * 3, dim2)
            else:
                lin = torch.nn.Linear(dim2, dim2)
            nn.init.xavier_uniform_(lin.weight)  # Xavier initialization
            self.post_lin_list.append(lin)
        self.lin_out = torch.nn.Linear(dim2, output_dim)
        nn.init.xavier_uniform_(self.lin_out.weight)  # Xavier initialization

    def forward(self, data):
        # Pre-GNN dense layers
        out = data.x
        for layer in self.pre_lin_list:
            out = torch.relu(layer(out))

        # GNN layers        
        for i in range(0, len(self.conv_list)):
            if i == 0:
                if len(self.pre_lin_list) == 0:
                    e_temp = self.e_embed_list[i](data.edge_attr)
                    x_temp = self.x_embed_list[i](data.x)
                    u_temp = self.u_embed_list[i](data.u)
                    x_out, e_out, u_out = self.conv_list[i](
                        x_temp, data.edge_index, e_temp, u_temp, data.batch
                    )
                    x = torch.add(x_out, x_temp)
                    e = torch.add(e_out, e_temp)
                    u = torch.add(u_out, u_temp)
                else:
                    e_temp = self.e_embed_list[i](data.edge_attr)
                    x_temp = self.x_embed_list[i](out)
                    u_temp = self.u_embed_list[i](data.u)
                    x_out, e_out, u_out = self.conv_list[i](
                        x_temp, data.edge_index, e_temp, u_temp, data.batch
                    )
                    x = torch.add(x_out, x_temp)
                    e = torch.add(e_out, e_temp)
                    u = torch.add(u_out, u_temp)
                    
            elif i > 0:
                e_temp = self.e_embed_list[i](e)
                x_temp = self.x_embed_list[i](x)
                u_temp = self.u_embed_list[i](u)
                x_out, e_out, u_out = self.conv_list[i](
                    x_temp, data.edge_index, e_temp, u_temp, data.batch
                )
                x = torch.add(x_out, x)
                e = torch.add(e_out, e)
                u = torch.add(u_out, u)

        # Post-GNN dense layers
        x_pool = scatter(x, data.batch, dim=0, reduce=self.pool_reduce)
        e_pool = scatter(e, data.edge_index[0, :], dim=0, reduce=self.pool_reduce)
        e_pool = scatter(e_pool, data.batch, dim=0, reduce=self.pool_reduce)
        out = torch.cat([x_pool, e_pool, u], dim=1)
        for layer in self.post_lin_list:
            out = torch.relu(layer(out))
        out = self.lin_out(out)
                
        return out.view(-1) if out.shape[1] == 1 else out


# Maybe add CGCNN, (or very similar GCN), or MPNN? 