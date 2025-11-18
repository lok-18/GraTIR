import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTokenizer, CLIPTextModel
from model.graph_att import GAttNet
from model.fc import FCNet
from model.device_utils import get_device, to_device, ensure_same_device

class sgg_graph(nn.Module):
    def __init__(self, text_encoder, tokenizer, feature_dim, gat_out_dim, heads=1):
        super(sgg_graph, self).__init__()
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        
        self.text_dim = 512
        
        self.dir_num = 2  
        self.label_num = 1  
        self.in_feat_dim = self.text_dim + 1  
        self.out_feat_dim = gat_out_dim
        self.nongt_dim = 2  
        self.num_heads = heads
        self.pos_emb_dim = -1  
        
        self.gat = GAttNet(
            dir_num=self.dir_num,
            label_num=self.label_num,
            in_feat_dim=self.in_feat_dim,
            out_feat_dim=self.out_feat_dim,
            nongt_dim=self.nongt_dim,
            dropout=0.2,
            label_bias=True,
            num_heads=self.num_heads,
            pos_emb_dim=self.pos_emb_dim
        )
        
        self.edge_transform = FCNet([self.text_dim, 64, self.label_num], 'ReLU', 0.2)
        
        self.node_transform = FCNet([self.in_feat_dim, self.in_feat_dim], 'ReLU', 0.1)
        
        self.fusion = FCNet([2 * self.out_feat_dim, self.out_feat_dim, 2 * self.out_feat_dim], 'ReLU', 0.1)

    def encode_text(self, text):
        device = next(self.text_encoder.parameters()).device
        
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        outputs = self.text_encoder(**inputs)
        return outputs.last_hidden_state.mean(dim=1)  # [batch_size, 512]

    def construct_graph(self, subj_feat, obj_feat, subj_desc, relation, obj_desc):
        """
        构建图结构
        Args:
            subj_feat: 主体图像特征 [h, w]
            obj_feat: 客体图像特征 [h, w]
            subj_desc: 主体描述
            relation: 关系描述
            obj_desc: 客体描述
        
        Returns:
            v_feat: 节点特征 [batch_size, num_rois, feat_dim]
            adj_matrix: 邻接矩阵 [batch_size, num_rois, num_rois, label_num]
            pos_emb: 位置嵌入 (None，因为我们使用的是显式关系)
        """
        device = next(self.parameters()).device
        
        subj_feat, obj_feat = ensure_same_device(subj_feat, obj_feat)
        
        h, w = subj_feat.size()
        
        subj_encoded = self.encode_text([subj_desc])  # [1, 512]
        obj_encoded = self.encode_text([obj_desc])    # [1, 512]
        rel_encoded = self.encode_text([relation])    # [1, 512]
        
        subj_feat_pooled = F.adaptive_avg_pool2d(subj_feat.unsqueeze(0).unsqueeze(0), (7, 7)).view(1, -1)  # [1, 49]
        obj_feat_pooled = F.adaptive_avg_pool2d(obj_feat.unsqueeze(0).unsqueeze(0), (7, 7)).view(1, -1)    # [1, 49]
        
        subj_img_feat = torch.mean(subj_feat_pooled, dim=1, keepdim=True)  # [1, 1]
        obj_img_feat = torch.mean(obj_feat_pooled, dim=1, keepdim=True)    # [1, 1]
        
        subj_node_feature = torch.cat([subj_encoded, subj_img_feat], dim=1)  # [1, 513]
        obj_node_feature = torch.cat([obj_encoded, obj_img_feat], dim=1)     # [1, 513]
        
        subj_node_feature = self.node_transform(subj_node_feature)  # [1, 513]
        obj_node_feature = self.node_transform(obj_node_feature)    # [1, 513]
        
        v_feat = torch.stack([subj_node_feature, obj_node_feature], dim=1)  # [1, 2, 513]
        
        adj_matrix = torch.zeros(1, 2, 2, self.label_num, device=device)
        
        adj_matrix[0, 0, 1, 0] = 1.0
        
        adj_matrix[0, 1, 0, 0] = 1.0
        
        edge_weight = self.edge_transform(rel_encoded)  # [1, label_num]
        
        adj_matrix = adj_matrix * edge_weight.view(1, 1, 1, self.label_num)
        
        return v_feat, adj_matrix, None 

    def forward(self, sub_feat, obj_feat, sub_class, rel_class, obj_class):
        sub_feat, obj_feat = ensure_same_device(sub_feat, obj_feat)
        
        v_feat, adj_matrix, _ = self.construct_graph(
            sub_feat, obj_feat, sub_class, rel_class, obj_class
        )
        
        updated_features = self.gat(v_feat, adj_matrix)  # [1, 2, out_feat_dim]
        
        updated_features = updated_features.squeeze(0)  # [2, out_feat_dim]
        
        sgg_feature = torch.cat([updated_features[0], updated_features[1]], dim=-1)  # [2*out_feat_dim]
        
        sgg_feature = self.fusion(sgg_feature)  # [2*out_feat_dim]
        
        return sgg_feature
