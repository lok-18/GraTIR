import torch
import torch.nn as nn
import torch.nn.functional as F
from model.device_utils import ensure_same_device
from einops import rearrange
from einops.layers.torch import Rearrange

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim必须能被num_heads整除"
        
        self.to_qkv = nn.Linear(embed_dim, embed_dim*3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, attn_mask=None):
        batch_size = x.shape[0]
        
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.num_heads), qkv)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e9)
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        
        out = rearrange(attn_output, 'b h n d -> b n (h d)')
        output = self.out_proj(out)
        
        return output

class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ff = FeedForward(embed_dim, ff_dim, dropout)
        
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, self_attn_mask=None):
        residual = x
        x = self.norm(x)
        x = self.self_attn(x, self_attn_mask)
        x = self.dropout(x)
        x = residual + x
        
        residual = x
        x = self.norm(x)   
        x = self.ff(x)
        x = self.dropout(x)
        x = residual + x
        
        return self.norm(x)

class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim, num_layers, num_heads, ff_dim, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x, self_attn_mask=None):
        for layer in self.layers:
            x = layer(x, self_attn_mask)
        x = self.norm(x)
        return x

class FusionTransformerDecoder(nn.Module):
    def __init__(self, feat_dim=4, num_layers=3, num_heads=8, ff_dim=512, dropout=0.1, patch_size=16):
        super(FusionTransformerDecoder, self).__init__()
        
        self.feat_dim = feat_dim
        self.patch_size = patch_size
        
        self.embed_dim = 256
        
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(self.feat_dim, self.embed_dim, kernel_size=patch_size, stride=patch_size),
            Rearrange('b c h w -> b (h w) c'),
            nn.LayerNorm(self.embed_dim)
        )
        
        self.sgg_proj = nn.Linear(256, self.embed_dim)
        
        self.decoder = TransformerDecoder(
            embed_dim=self.embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout
        )
        
        self.fusion_conv = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.embed_dim, self.embed_dim//2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.embed_dim//2, self.embed_dim//4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.embed_dim//4, self.embed_dim//8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.embed_dim//8, self.embed_dim//16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.embed_dim//16, 1, kernel_size=3, padding=1),
            )
    
    def _project_sgg_features(self, sgg_features):
        if len(sgg_features.shape) == 4:  # [B, N1, N2, C]
            B, N1, N2, C = sgg_features.shape
            sgg_features = sgg_features.reshape(B, N1*N2, C)
        else:  # [B, N, C]
            B, N, C = sgg_features.shape
        
        if C != self.sgg_proj.in_features:
            self.sgg_proj = nn.Linear(C, self.embed_dim).to(sgg_features.device)
        
        flat_features = sgg_features.reshape(-1, C)
        projected = self.sgg_proj(flat_features)
        
        return projected.reshape(B, -1, self.embed_dim)
    
    def _create_feature_projector(self, feat, proj_attr_name):
        in_channels = feat.size(1)
    
        proj = getattr(self, proj_attr_name)
        
        if proj.in_channels != in_channels:
            new_proj = nn.Conv2d(in_channels, self.embed_dim, kernel_size=1)
            new_proj = new_proj.to(feat.device)
            setattr(self, proj_attr_name, new_proj)
            proj = new_proj
        
        return proj(feat)
    
    def forward(self, ir_feat, vis_feat, ir_sgg_feat, vis_sgg_feat):

        ir_feat, vis_feat, ir_sgg_feat, vis_sgg_feat = ensure_same_device(
            ir_feat, vis_feat, ir_sgg_feat, vis_sgg_feat
        )
        
        batch_size, _, height, width = ir_feat.shape

        h_patch_nums = height // self.patch_size
        w_patch_nums = width // self.patch_size
        patch_nums = h_patch_nums * w_patch_nums
        
        pos_embeding = nn.Parameter(torch.randn(1, patch_nums, self.embed_dim)).to(ir_feat.device)

        ir_embed = self.to_patch_embedding(ir_feat) # [B, N, embed_dim] N = H//patch_size * W//patch_size
        vis_embed = self.to_patch_embedding(vis_feat) # [B, N, embed_dim]
        ir_embed = ir_embed + pos_embeding
        vis_embed = vis_embed + pos_embeding

        ir_sgg_embed = self._project_sgg_features(ir_sgg_feat)  # [B, N_ir, embed_dim]
        vis_sgg_embed = self._project_sgg_features(vis_sgg_feat)  # [B, N_vis, embed_dim]

        x = torch.cat([
            ir_embed,    
            vis_embed,   
            ir_sgg_embed,     
            vis_sgg_embed,    
        ], dim=1)  # [B, 2N+N_ir+N_vis, embed_dim]

        decoded = self.decoder(x)  # [B, 2N+N_ir+N_vis, embed_dim]

        ir_token = decoded[:, :patch_nums, :] # [B, N, embed_dim]
        vis_token = decoded[:, patch_nums:2*patch_nums, :] # [B, N, embed_dim]

        vis_token = Rearrange('b (h w) d -> b d h w', h=h_patch_nums, w=w_patch_nums)(vis_token)
        ir_token = Rearrange('b (h w) d -> b d h w', h=h_patch_nums, w=w_patch_nums)(ir_token)
        
        vis_weight = self.fusion_conv(vis_token) #[B, 1, H, W]
        ir_weight = self.fusion_conv(ir_token) #[B, 1, H, W]
        
        return vis_weight, ir_weight 