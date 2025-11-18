import torch
import torch.nn as nn
import torch.nn.functional as F
from model.graph import GraphReasoning
from thop import profile

# from model.sgg_409 import sgg_graph
from model.sgg import sgg_graph
from RelTR.models import build_model
import argparse

from transformers import CLIPTokenizer, CLIPTextModel
from model.device_utils import get_device, to_device, ensure_same_device
# from model.transformer_decoder import FusionTransformerDecoder
from model.decoder import FusionTransformerDecoder

device = get_device()
clip_model = to_device(CLIPTextModel.from_pretrained("./clip_model"))
clip_tokenizer = CLIPTokenizer.from_pretrained("./clip_model")

# VG classes
CLASSES = [ 'N/A', 'airplane', 'animal', 'arm', 'bag', 'banana', 'basket', 'beach', 'bear', 'bed', 'bench', 'bike',
            'bird', 'board', 'boat', 'book', 'boot', 'bottle', 'bowl', 'box', 'boy', 'branch', 'building',
            'bus', 'cabinet', 'cap', 'car', 'cat', 'chair', 'child', 'clock', 'coat', 'counter', 'cow', 'cup',
            'curtain', 'desk', 'dog', 'door', 'drawer', 'ear', 'elephant', 'engine', 'eye', 'face', 'fence',
            'finger', 'flag', 'flower', 'food', 'fork', 'fruit', 'giraffe', 'girl', 'glass', 'glove', 'guy',
            'hair', 'hand', 'handle', 'hat', 'head', 'helmet', 'hill', 'horse', 'house', 'jacket', 'jean',
            'kid', 'kite', 'lady', 'lamp', 'laptop', 'leaf', 'leg', 'letter', 'light', 'logo', 'man', 'men',
            'motorcycle', 'mountain', 'mouth', 'neck', 'nose', 'number', 'orange', 'pant', 'paper', 'paw',
            'people', 'person', 'phone', 'pillow', 'pizza', 'plane', 'plant', 'plate', 'player', 'pole', 'post',
            'pot', 'racket', 'railing', 'rock', 'roof', 'room', 'screen', 'seat', 'sheep', 'shelf', 'shirt',
            'shoe', 'short', 'sidewalk', 'sign', 'sink', 'skateboard', 'ski', 'skier', 'sneaker', 'snow',
            'sock', 'stand', 'street', 'surfboard', 'table', 'tail', 'tie', 'tile', 'tire', 'toilet', 'towel',
            'tower', 'track', 'train', 'tree', 'truck', 'trunk', 'umbrella', 'vase', 'vegetable', 'vehicle',
            'wave', 'wheel', 'window', 'windshield', 'wing', 'wire', 'woman', 'zebra']

REL_CLASSES = ['__background__', 'above', 'across', 'against', 'along', 'and', 'at', 'attached to', 'behind',
            'belonging to', 'between', 'carrying', 'covered in', 'covering', 'eating', 'flying in', 'for',
            'from', 'growing on', 'hanging from', 'has', 'holding', 'in', 'in front of', 'laying on',
            'looking at', 'lying on', 'made of', 'mounted on', 'near', 'of', 'on', 'on back of', 'over',
            'painted on', 'parked on', 'part of', 'playing', 'riding', 'says', 'sitting on', 'standing on',
            'to', 'under', 'using', 'walking in', 'walking on', 'watching', 'wearing', 'wears', 'with']

class ConvLeakyRelu2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
    def forward(self,x):
        return F.leaky_relu(self.conv(x), negative_slope=0.2)

class Encoder_ir(nn.Module):
    def __init__(self, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(Encoder_ir, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation,
                               groups=groups)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation,
                               groups=groups)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation,
                               groups=groups)
        self.conv4 = nn.Conv2d(384, 512, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation,
                               groups=groups)
        self.maxpool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.avgpool = nn.AvgPool2d(2, stride=2, ceil_mode=True)

        # channel attention
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(512, 32, bias=False)
        self.fc2 = nn.Linear(32, 512, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        x1 = self.maxpool(x1)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2)
        x2 = self.maxpool(x2)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2)
        x3_max = self.maxpool(F.leaky_relu(self.conv4(x3), negative_slope=0.2))
        x3_avg = self.avgpool(F.leaky_relu(self.conv4(x3), negative_slope=0.2))
        x3 = torch.mul(x3_max, x3_avg)

        # channel attention
        x_gap = self.gap(x3)
        x_gap = x_gap.view(x_gap.size(0), -1)
        x_gap = self.relu(self.fc1(x_gap))
        x_gap = self.fc2(x_gap)
        x_weights = self.sigmoid(x_gap)
        x_weights = x_weights.view(x3.size(0), x3.size(1), 1, 1)
        x3 = x3 * x_weights.expand_as(x3)

        return x3, x2, x1

class Encoder_vis(nn.Module):
    def __init__(self, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(Encoder_vis, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation,
                               groups=groups)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation,
                               groups=groups)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation,
                               groups=groups)
        self.conv4 = nn.Conv2d(384, 512, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation,
                               groups=groups)
        self.maxpool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.avgpool = nn.AvgPool2d(2, stride=2, ceil_mode=True)

        # channel attention
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(512, 32, bias=False)
        self.fc2 = nn.Linear(32, 512, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        x1 = self.maxpool(x1)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2)
        x2 = self.maxpool(x2)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2)
        x3_max = self.maxpool(F.leaky_relu(self.conv4(x3), negative_slope=0.2))
        x3_avg = self.avgpool(F.leaky_relu(self.conv4(x3), negative_slope=0.2))
        x3 = torch.add(x3_max, x3_avg)

        # channel attention
        x_gap = self.gap(x3)
        x_gap = x_gap.view(x_gap.size(0), -1)
        x_gap = self.relu(self.fc1(x_gap))
        x_gap = self.fc2(x_gap)
        x_weights = self.sigmoid(x_gap)
        x_weights = x_weights.view(x3.size(0), x3.size(1), 1, 1)
        x3 = x3 * x_weights.expand_as(x3)

        return x3, x2, x1

class CGR(nn.Module):
    def __init__(self, n_class=2, n_iter=2, chnn_side=(512, 256, 128), chnn_targ=(512, 128, 32, 4), rd_sc=32, dila=(4, 8, 16)):
        super().__init__()
        self.n_graph = len(chnn_side)
        n_node = len(dila)
        graph = [GraphReasoning(ii, rd_sc, dila, n_iter) for ii in chnn_side]
        self.graph = nn.ModuleList(graph)
        C_cat = [nn.Sequential(
            nn.Conv2d(ii//rd_sc*n_node, ii//rd_sc, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ii//rd_sc),
            nn.ReLU(inplace=True))
            for ii in (chnn_side+chnn_side)]
        self.C_cat = nn.ModuleList(C_cat)
        idx = [ii for ii in range(len(chnn_side))]
        C_up = [nn.Sequential(
            nn.Conv2d(chnn_targ[ii]+chnn_side[ii]//rd_sc, chnn_targ[ii+1], 3, 1, 1, bias=False),
            nn.BatchNorm2d(chnn_targ[ii+1]),
            nn.ReLU(inplace=True))
            for ii in (idx+idx)]
        self.C_up = nn.ModuleList(C_up)
        
        # 传统融合方法
        self.C_cls = nn.Conv2d(chnn_targ[-1]*2, 1, 1)  # n_class -> 1

        self.conv = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        # self.upsample = nn.Upsample(size=(480, 640), mode='bilinear', align_corners=False)

    def forward(self, ir, vis):
        # img, depth = inputs
        cas_ir, cas_vis = ir[0], vis[0]
        nd_ir, nd_vis, nd_key = None, None, False
        for ii in range(self.n_graph):
            feat_ir, feat_vis = self.graph[ii]([ir[ii], vis[ii], nd_ir, nd_vis], nd_key)
            feat_ir = torch.cat(feat_ir, 1)
            feat_ir = self.C_cat[ii](feat_ir)
            feat_vis = torch.cat(feat_vis, 1)
            feat_vis = self.C_cat[self.n_graph+ii](feat_vis)
            nd_ir, nd_vis, nd_key = feat_ir, feat_vis, True
            cas_ir = torch.cat((feat_ir, cas_ir), 1)
            cas_ir = F.interpolate(cas_ir, scale_factor=2, mode='bilinear', align_corners=True)
            cas_ir = self.C_up[ii](cas_ir)
            cas_vis = torch.cat((feat_vis, cas_vis), 1)
            cas_vis = F.interpolate(cas_vis, scale_factor=2, mode='bilinear', align_corners=True)
            cas_vis = self.C_up[self.n_graph+ii](cas_vis)

        # sgg = self.conv(sgg)
        # upsample = nn.Upsample(size=(cas_dep.size(2), cas_dep.size(3)), mode='bilinear', align_corners=False)
        # sgg = upsample(sgg)

        # gat_ir = resize_feature(cas_ir, gat_ir)
        # gat_vis = resize_feature(cas_vis, gat_vis)

        # cas_ir = torch.add(cas_ir, gat_ir)  # ir场景图加进来
        # cas_vis = torch.add(cas_vis, gat_vis)  # vis场景图加进来

        # feat = torch.cat((cas_ir, cas_vis), dim=1)
        # out = self.C_cls(feat)
        return cas_ir, cas_vis

def resize_feature(target, input):
    b, c, h, w = target.shape
    required_size = b * c * h * w
    input = input.view(-1)

    if input.numel() < required_size:
        padding = required_size - input.numel()
        input = torch.cat([input, torch.zeros(padding, device=input.device)])   
    else:
        input = input[:required_size]

    return input.view(b, c, h, w)

class TransformerFusionModule(nn.Module):
    def __init__(self, feat_dim, num_layers=3, num_heads=8, ff_dim=2048):
        super(TransformerFusionModule, self).__init__()
        self.transformer_decoder = FusionTransformerDecoder(
            feat_dim=feat_dim,           
            num_layers=num_layers,       
            num_heads=num_heads,         
            ff_dim=ff_dim,               
            dropout=0.1                  
        )
    
    def forward(self, ir_feat, vis_feat, ir_sgg_feat, vis_sgg_feat):
        return self.transformer_decoder(ir_feat, vis_feat, ir_sgg_feat, vis_sgg_feat)

class backbone(nn.Module):
    def __init__(self, use_transformer=True):
        super(backbone, self).__init__()
        # GNN backbone
        self.vis_conv = Encoder_vis()
        self.inf_conv = Encoder_ir()
        self.graph = CGR()
        self.use_transformer = use_transformer
        
        if use_transformer:
            self.fusion_module = TransformerFusionModule(
                feat_dim=4,    
                num_layers=4,   
                num_heads=4,    
                ff_dim=512     
            )

        self.fusion_conv = nn.Conv2d(8, 1, kernel_size=3, padding=1)
        
        # 初始化SGG相关参数
        self.args = self.get_args_parser()
        self.sgg_model, _, _ = build_model(self.args)
        ckpt = torch.load(self.args.resume, map_location=device)
        self.sgg_model.load_state_dict(ckpt['model'])
        self.sgg_model = to_device(self.sgg_model)
        self.sgg_model.eval()

    def get_args_parser(self):
        parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
        parser.add_argument('--lr_backbone', default=1e-5, type=float)
        parser.add_argument('--dataset', default='vg')

        parser.add_argument('--img_folder', type=str, default='./RelTR/demo/',
                            help="Path folder of the test image")
        
        parser.add_argument('--output_path', type=str, default='./RelTR/results/demo/',
                            help="Path folder of output")

        parser.add_argument('--backbone', default='resnet50', type=str,
                            help="Name of the convolutional backbone to use")
        parser.add_argument('--dilation', action='store_true',
                            help="If true, we replace stride with dilation in the last convolutional block (DC5)")
        parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                            help="Type of positional embedding to use on top of the image features")

        parser.add_argument('--enc_layers', default=6, type=int,
                            help="Number of encoding layers in the transformer")
        parser.add_argument('--dec_layers', default=6, type=int,
                            help="Number of decoding layers in the transformer")
        parser.add_argument('--dim_feedforward', default=2048, type=int,
                            help="Intermediate size of the feedforward layers in the transformer blocks")
        parser.add_argument('--hidden_dim', default=256, type=int,
                            help="Size of the embeddings (dimension of the transformer)")
        parser.add_argument('--dropout', default=0.1, type=float,
                            help="Dropout applied in the transformer")
        parser.add_argument('--nheads', default=8, type=int,
                            help="Number of attention heads inside the transformer's attentions")
        parser.add_argument('--num_entities', default=100, type=int,
                            help="Number of query slots")
        parser.add_argument('--num_triplets', default=200, type=int,
                            help="Number of query slots")
        parser.add_argument('--pre_norm', action='store_true')

        parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                            help="Disables auxiliary decoding losses (loss at each layer)")

        parser.add_argument('--device', default='cuda',
                            help='device to use for training / testing')
        parser.add_argument('--resume', default='./RelTR/ckpt/checkpoint0149.pth', help='resume from checkpoint')
        parser.add_argument('--set_cost_class', default=1, type=float,
                            help="Class coefficient in the matching cost")
        parser.add_argument('--set_cost_bbox', default=5, type=float,
                            help="L1 box coefficient in the matching cost")
        parser.add_argument('--set_cost_giou', default=2, type=float,
                            help="giou box coefficient in the matching cost")
        parser.add_argument('--set_iou_threshold', default=0.7, type=float,
                            help="giou box coefficient in the matching cost")
        parser.add_argument('--bbox_loss_coef', default=5, type=float)
        parser.add_argument('--giou_loss_coef', default=2, type=float)
        parser.add_argument('--rel_loss_coef', default=1, type=float)
        parser.add_argument('--eos_coef', default=0.1, type=float,
                            help="Relative classification weight of the no-object class")

        # distributed training parameters
        parser.add_argument('--return_interm_layers', action='store_true',
                            help="Return the fpn if there is the tag")
        return parser.parse_args()

    def sgg_for_loss(self, image):
        outputs = self.sgg_model(image)

        # 一个batch内的所有场景图描述
        probas_all = outputs['rel_logits'].softmax(-1)[:, :, :-1]  # 关系词
        probas_sub_all = outputs['sub_logits'].softmax(-1)[:, :, :-1]  # 主体词
        probas_obj_all = outputs['obj_logits'].softmax(-1)[:, :, :-1]  # 客体词

        return probas_all, probas_sub_all, probas_obj_all

    def sgg(self, image):
        outputs = self.sgg_model(image)

        # 一个batch内的所有场景图描述
        probas_all = outputs['rel_logits'].softmax(-1)[:, :, :-1]  # 关系词
        probas_sub_all = outputs['sub_logits'].softmax(-1)[:, :, :-1]  # 主体词
        probas_obj_all = outputs['obj_logits'].softmax(-1)[:, :, :-1]  # 客体词

        probas, probas_sub, probas_obj = [], [], []
        all_att_sub_features, all_att_obj_features = [], []  # 存一个batch的注意力图
        all_sub_classes, all_rel_classes, all_obj_classes = [], [], []  # 存一个batch的文字描述

        # 打印每个mini-bitch场景图描述
        for i in range(probas_all.size(0)):
            probas = probas_all[i]
            probas_sub = probas_sub_all[i]
            probas_obj = probas_obj_all[i]
            
            keep = torch.logical_and(probas.max(-1).values > 0, torch.logical_and(probas_sub.max(-1).values > 0,
                                                                                probas_obj.max(-1).values > 0))

            topk = 3

            keep_queries = torch.nonzero(keep, as_tuple=True)[0]
            if keep_queries.numel() == 0:
                continue
            indices = torch.argsort(-probas[keep_queries].max(-1)[0] * probas_sub[keep_queries].max(-1)[0] * probas_obj[keep_queries].max(-1)[0])[:topk]
            keep_queries = keep_queries[indices]

            att_sub_features, att_obj_features = [], []  # 存mini-batch注意力图
            sub_classes, rel_classes, obj_classes = [], [], []  # 存mini-batch文字描述

            for query in keep_queries:
                sub_class = CLASSES[probas_sub[query].argmax()]
                rel_class = REL_CLASSES[probas[query].argmax()]
                obj_class = CLASSES[probas_obj[query].argmax()]

                sub_classes.append(sub_class)
                rel_classes.append(rel_class)
                obj_classes.append(obj_class)

                conv_features, dec_attn_weights_sub, dec_attn_weights_obj = [], [], []
                hooks = [
                    self.sgg_model.backbone[-2].register_forward_hook(
                        lambda self, input, output: conv_features.append(output)
                        ),
                    self.sgg_model.transformer.decoder.layers[-1].cross_attn_sub.register_forward_hook(
                        lambda self, input, output: dec_attn_weights_sub.append(output[1])
                    ),
                    self.sgg_model.transformer.decoder.layers[-1].cross_attn_obj.register_forward_hook(
                        lambda self, input, output: dec_attn_weights_obj.append(output[1])
                    )
                ]

                outputs = self.sgg_model(image)

                for hook in hooks:
                    hook.remove()  # 移除钩子函数

                # 为了输出可视化的特征图   一个batch_size下的attention图
                conv_features = conv_features[0]  # 场景图
                dec_attn_weights_sub = dec_attn_weights_sub[0]  # 主体attention图
                dec_attn_weights_obj = dec_attn_weights_obj[0]  # 客体attention图

                h, w = conv_features['0'].tensors.shape[-2:]
                att_sub_feature = dec_attn_weights_sub[0, query].view(h, w)
                att_obj_feature = dec_attn_weights_obj[0, query].view(h, w)

                att_sub_features.append(att_sub_feature)
                att_obj_features.append(att_obj_feature)

            all_att_sub_features.append(att_sub_features)
            all_att_obj_features.append(att_obj_features)

            all_sub_classes.append(sub_classes)
            all_rel_classes.append(rel_classes)
            all_obj_classes.append(obj_classes)

        return all_att_sub_features, all_att_obj_features, all_sub_classes, all_rel_classes, all_obj_classes

    def forward(self, image_vis_sgg, image_ir_sgg, image_vis, image_ir):
        image_vis_sgg, image_ir_sgg, image_vis, image_ir = ensure_same_device(
            image_vis_sgg, image_ir_sgg, image_vis, image_ir
        )
        
        # 场景图
        with torch.no_grad():
            all_vis_sgg_sub, all_vis_sgg_obj, all_vis_sub_classes, all_vis_rel_classes, all_vis_obj_classes = self.sgg(image_vis_sgg)
            all_ir_sgg_sub, all_ir_sgg_obj, all_ir_sub_classes, all_ir_rel_classes, all_ir_obj_classes = self.sgg(image_ir_sgg)

        vis_sgg_features, vis_all_sgg_features, ir_sgg_features, ir_all_sgg_features = [], [], [], []
        feature_dim = 513 # text_dim:512 + image_dim:1
        feature_out_dim = 128
        heads = 2

        for i, sub_list in enumerate(all_vis_sgg_sub):
            for j, vis_sgg_sub in enumerate(sub_list):
                vis_sgg_sub = all_vis_sgg_sub[i][j]
                vis_sgg_obj = all_vis_sgg_obj[i][j]
                vis_sub_classes = all_vis_sub_classes[i][j]
                vis_rel_classes = all_vis_rel_classes[i][j]
                vis_obj_classes = all_vis_obj_classes[i][j]

                ir_sgg_sub = all_ir_sgg_sub[i][j]
                ir_sgg_obj = all_ir_sgg_obj[i][j]
                ir_sub_classes = all_ir_sub_classes[i][j]
                ir_rel_classes = all_ir_rel_classes[i][j]
                ir_obj_classes = all_ir_obj_classes[i][j]
                
                graph_model = to_device(sgg_graph(clip_model, clip_tokenizer, 
                                      feature_dim=feature_dim, 
                                      gat_out_dim=feature_out_dim, 
                                      heads=heads))
                
                vis_sgg_feature = graph_model(vis_sgg_sub, vis_sgg_obj, vis_sub_classes, vis_rel_classes, vis_obj_classes)  # [256]
                ir_sgg_feature = graph_model(ir_sgg_sub, ir_sgg_obj, ir_sub_classes, ir_rel_classes, ir_obj_classes)  # [256]
                
                vis_sgg_features.append(vis_sgg_feature)
                ir_sgg_features.append(ir_sgg_feature)

            vis_sgg_features = torch.stack(vis_sgg_features, dim=0)  # [3,256]
            ir_sgg_features = torch.stack(ir_sgg_features, dim=0)  # [3,256]

            vis_all_sgg_features.append(vis_sgg_features)
            ir_all_sgg_features.append(ir_sgg_features)

            vis_sgg_features = []
            ir_sgg_features = []

        vis_all_sgg_features = torch.stack(vis_all_sgg_features, dim=0)  # [4,3,256]
        ir_all_sgg_features = torch.stack(ir_all_sgg_features, dim=0)  # [4,3,256]

        image_vis = image_vis[:, : 1]

        image_vis_features = self.vis_conv(image_vis)
        image_ir_features = self.inf_conv(image_ir)
        
        ir_feat, vis_feat = self.graph(image_ir_features, image_vis_features)
        
        if self.use_transformer:
            vis_weight, ir_weight = self.fusion_module(ir_feat, vis_feat, ir_all_sgg_features, vis_all_sgg_features)
            feat = torch.cat([ir_feat* ir_weight, vis_feat * vis_weight], dim=1)
            fused_image = self.fusion_conv(feat)
        else:
            ir_sgg_resized = resize_feature(ir_feat, ir_all_sgg_features)
            vis_sgg_resized = resize_feature(vis_feat, vis_all_sgg_features)
            
            ir_feat_enhanced = torch.add(ir_feat, ir_sgg_resized)
            vis_feat_enhanced = torch.add(vis_feat, vis_sgg_resized)
            
            feat = torch.cat((ir_feat_enhanced, vis_feat_enhanced), dim=1)
            fused_image = self.fusion_conv(feat)

        return fused_image




