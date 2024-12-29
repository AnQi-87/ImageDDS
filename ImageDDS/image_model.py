import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv # type: ignore
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import SAGPooling
import pandas as pd
import numpy as np

from losses import SupConLoss

import torchvision



#自适应特征融合   
class AttentionFusion_auto(torch.nn.Module):
    def __init__(self, n_dim_input1, n_dim_input2, lambda_1=1, lambda_2=1):
        super(AttentionFusion_auto, self).__init__()
        self.n_dim_input1, self.n_dim_input2 = n_dim_input1, n_dim_input2
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2

        #自适应特征融合
        # self.w = nn.Parameter(torch.ones(2))

        #MLP特征融合
        self.linear = nn.Linear(2 * n_dim_input1, n_dim_input2)

    def forward(self, input1, input2):

        # 自适应融合
        # w1 = torch.exp(self.w[0]/torch.sum(torch.exp(self.w)))
        # w2 = torch.exp(self.w[1]/torch.sum(torch.exp(self.w)))
        # return w1 * input1 + w2 * input2

        #MLP特征更新新加的11月21号
        mid_emb = torch.cat((input1, input2), 1)
        return F.relu(self.linear(mid_emb))

        # 自定义参数融合
        
        #return self.lambda_1 * input1 + self.lambda_2 * input2

class ImageDDS(torch.nn.Module):
    def __init__(self, n_output = 1, n_filters=32, embed_dim=128, num_features_xd=64, num_features_xt=954, output_dim=128, dropout=0.2,
                 use_cl=False, use_image_fusion=False,img_pretrained=True, temperature=0.07, base_temperature=0.07,
                 lambda_fusion_graph=1, lambda_fusion_image=1, batch_size = 128, device = torch.device('cpu')):

        super(ImageDDS, self).__init__()

        self.activate = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)

        # SMILES1 graph branch
        self.n_output = n_output 
        self.drug_conv1 = TransformerConv(78, num_features_xd * 2,  heads = 2)
        self.drug_conv2 = TransformerConv(num_features_xd * 4, num_features_xd * 8, heads = 2)
        
        self.drug_fc_g1 = torch.nn.Linear(num_features_xd * 16, num_features_xd * 8)
        self.drug_fc_g2 = torch.nn.Linear(num_features_xd * 8, output_dim )


        # DL cell featrues
        self.reduction = nn.Sequential(
            nn.Linear(num_features_xt, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, output_dim)
        )

        self.use_cl = use_cl
        self.use_img_fusion = use_image_fusion

        if self.use_img_fusion or self.use_cl:
            print('fusion_cl')
            # image drug layer
            self.image_model = torchvision.models.resnet18(pretrained=img_pretrained)
            # self.in_features_image = self.image_model.fc.in_features
            self.image_model.fc = nn.Linear(self.image_model.fc.in_features, output_dim)

        if self.use_img_fusion:
            # fusion model to fuse image and graph
            self.fusion = AttentionFusion_auto(n_dim_input1=output_dim, n_dim_input2=output_dim, lambda_1=lambda_fusion_graph, lambda_2=lambda_fusion_image)

       
       
        #最后一层MLP处理
        self.final_mlp = nn.Sequential(
            nn.Linear(output_dim*3, 1024),  # 第一层
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 256),             # 第二层
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, n_output)         # 输出层  
        )


        self.cl_loss = SupConLoss(temperature=temperature, base_temperature=base_temperature)

    def get_col_index(self, x):
        row_size = len(x[:, 0])
        row = np.zeros(row_size)
        col_size = len(x[0, :])
        for i in range(col_size):
            row[np.argmax(x[:, i])] += 1
        return row

    def save_num(self, d, path):
        d = d.cpu().numpy()
        ind = self.get_col_index(d)
        ind = pd.DataFrame(ind)
        ind.to_csv('data/case_study/' + path + '_index.csv', header=0, index=0)
        # 下面是load操作
        # read_dictionary = np.load('my_file.npy').item()
        # d = pd.DataFrame(d)
        # d.to_csv('data/result/' + path + '.csv', header=0, index=0)
 
    #GAT
    def graph_channel(self, x, edge_index, batch):
        # deal drug1
        x = self.drug1_gcn1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.drug1_gcn2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = gmp(x, batch)         # global max pooling
        x = self.drug1_fc_g1(x)
        x = self.relu(x)
        return x

    def image_channel(self, image):
        return self.image_model(image)

    def forward(self, data1, data2):
        x1, image1, edge_index1, batch1, cell = data1.x, data1.image, data1.edge_index, data1.batch, data1.cell
        x2, image2, edge_index2, batch2 = data2.x, data2.image, data2.edge_index, data2.batch

        # deal drug1
        x1 = self.drug_conv1(x1, edge_index1)
        x1 = self.activate(x1)

        x1 = self.drug_conv2(x1, edge_index1)
        x1 = self.activate(x1)

        x1 = gmp(x1, batch1)       # global max pooling    

        # flatten
        x1 = self.drug_fc_g1(x1)
        x1 = self.activate(x1)
        x1 = self.dropout(x1)

        x1 = self.drug_fc_g2(x1)
        x1 = self.dropout(x1)

        x1_graph = F.normalize(x1, 2, 1)

        # print("x1_graph:{}".format(x1_graph))

        if self.use_img_fusion or self.use_cl:
            x1_image = self.image_channel(image1)
            x1_image = F.normalize(x1_image, 2, 1)
            # print("x1_image:.{}".format(x1_image))
        else:
            x1_image = None

        if self.use_img_fusion:
            x1 = self.fusion(x1_graph, x1_image)
        else:
            x1 = x1_graph


        # deal drug2
        x2 = self.drug_conv1(x2, edge_index2)
        x2 = self.activate(x2)

        x2 = self.drug_conv2(x2, edge_index2)
        x2 = self.activate(x2)
      
        x2 = gmp(x2, batch2)       # global max pooling    

        # flatten
        x2 = self.drug_fc_g1(x2)
        x2 = self.activate(x2)
        x2 = self.dropout(x2)
        
        x2 = self.drug_fc_g2(x2)
        x2 = self.dropout(x2)

        x2_graph = F.normalize(x2, 2, 1)

        if self.use_img_fusion or self.use_cl:
            x2_image = self.image_channel(image2)
            x2_image = F.normalize(x2_image, 2, 1)
        else:
            x2_image = None

        if self.use_img_fusion:
            x2 = self.fusion(x2_graph, x2_image)
        else:
            x2 = x2_graph

        # deal cell
        cell_vector = F.normalize(cell, 2, 1)
        cell_vector = self.reduction(cell_vector)


        #拼接特征
        xc = torch.cat([x1, x2, cell_vector], dim=1)
        xc = F.normalize(xc, 2, 1) 

        # dnn_x = self.dnn_network(xc)
        #传入mlp
        # final = self.dense_final(dnn_x)
        final = self.final_mlp(xc)
        #输出预测
        outputs = torch.sigmoid(final.squeeze(1))


        data_dict = {
            "img_idx_1": data1.img_idx,
            "x1_graph": x1_graph,
            "x1_image": x1_image,
            "img_idx_2": data2.img_idx,
            "x2_graph": x2_graph,
            "x2_image": x2_image
        }
        
        return outputs, data_dict
    
    def cal_cl_loss(self, data_dict):
        img_idx_1, img_idx_2 = data_dict["img_idx_1"], data_dict["img_idx_2"]
        x1_graph, x1_image, x2_graph, x2_image = data_dict["x1_graph"], data_dict["x1_image"], data_dict["x2_graph"], \
        data_dict["x2_image"]

        # x1_graph 和 x1_image，x2_graph 和 x2_image 要接近
        loss = (self.cl_loss(x1_graph, x1_image, labels=img_idx_1) + self.cl_loss(x2_graph, x2_image,
                                                                                  labels=img_idx_2)) / 2

        return loss



