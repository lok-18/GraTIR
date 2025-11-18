import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim

from RelTR.models import build_model
import argparse

from transformers import CLIPProcessor, CLIPModel
from model.model_gnn_sgg import backbone

clip_model = CLIPModel.from_pretrained("./clip_model")
clip_processor = CLIPProcessor.from_pretrained("./clip_model")

class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    def forward(self, x):
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)


class fusion_loss(nn.Module):
    def __init__(self):
        super(fusion_loss, self).__init__()
        self.sobelconv = Sobelxy()

    def forward(self, image_vis, image_ir, generate_img):
        image_y = image_vis[:, :1, :, :]
        # generate_img = generate_img.squeeze(0).unsqueeze(1)
        # L_int
        x_in_max = torch.max(image_y, image_ir)
        # loss_in = F.l1_loss(x_in_max, generate_img)
        x_in_mean = (image_y + image_ir) / 2
        loss_in = F.mse_loss(x_in_mean, generate_img)


        # L_edge
        y_grad = self.sobelconv(image_y)
        ir_grad = self.sobelconv(image_ir)
        generate_img_grad = self.sobelconv(generate_img)
        x_grad_joint = torch.max(y_grad, ir_grad)
        loss_grad = F.l1_loss(x_grad_joint, generate_img_grad)

        # L_ssim
        loss_ssim_ir = 1 - ssim(generate_img, image_ir, nonnegative_ssim=True)  # generate_img[0].unsqueeze(0)
        loss_ssim_vis = 1 - ssim(generate_img, image_y, nonnegative_ssim=True)  # generate_img[1].unsqueeze(0)
        loss_ssim = loss_ssim_ir + loss_ssim_vis


        # L_total
        loss_total = loss_in + 10 * loss_grad + 0.5 * loss_ssim
        return loss_total, loss_in, loss_grad, loss_ssim


class SGGFinetuneLoss(torch.nn.Module):
    def __init__(self):
        super(SGGFinetuneLoss, self).__init__()
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=1)

    def convert_to_rgb(self, image):
        if image.shape[1] == 1:
            return image.repeat(1, 3, 1, 1)
        else:
            return image

    def forward(self, ir_image, vis_image, fused_image, backbone):
        ir_image_rgb = self.convert_to_rgb(ir_image)
        vis_image_rgb = self.convert_to_rgb(vis_image)
        fused_image_rgb = self.convert_to_rgb(fused_image)
            
        with torch.no_grad():
            ir_probas_all, ir_probas_sub_all, ir_probas_obj_all = backbone.sgg_for_loss(ir_image_rgb)
            vis_probas_all, vis_probas_sub_all, vis_probas_obj_all = backbone.sgg_for_loss(vis_image_rgb)
        fused_probas_all, fused_probas_sub_all, fused_probas_obj_all = backbone.sgg_for_loss(fused_image_rgb)

        ir_probas_similarity = self.cosine_similarity(fused_probas_all, ir_probas_all)
        vis_probas_similarity = self.cosine_similarity(fused_probas_all, vis_probas_all)
        ir_probas_sub_similarity = self.cosine_similarity(fused_probas_sub_all, ir_probas_sub_all)
        vis_probas_sub_similarity = self.cosine_similarity(fused_probas_sub_all, vis_probas_sub_all)
        ir_probas_obj_similarity = self.cosine_similarity(fused_probas_obj_all, ir_probas_obj_all)
        vis_probas_obj_similarity = self.cosine_similarity(fused_probas_obj_all, vis_probas_obj_all)

        ir_similarity = torch.mean(ir_probas_similarity) + torch.mean(ir_probas_sub_similarity) + torch.mean(ir_probas_obj_similarity)
        vis_similarity = torch.mean(vis_probas_similarity) + torch.mean(vis_probas_sub_similarity) + torch.mean(vis_probas_obj_similarity)

        if ir_similarity > vis_similarity:
            loss_rel = self.cross_entropy(ir_probas_all.view(-1, ir_probas_all.size(-1)), 
                                          fused_probas_all.view(-1, fused_probas_all.size(-1)))
            loss_sub = self.cross_entropy(ir_probas_sub_all.view(-1, ir_probas_sub_all.size(-1)), 
                                          fused_probas_sub_all.view(-1, fused_probas_sub_all.size(-1)))
            loss_obj = self.cross_entropy(ir_probas_obj_all.view(-1, ir_probas_obj_all.size(-1)), 
                                          fused_probas_obj_all.view(-1, fused_probas_obj_all.size(-1)))
        else:
            loss_rel = self.cross_entropy(vis_probas_all.view(-1, vis_probas_all.size(-1)), 
                                          fused_probas_all.view(-1, fused_probas_all.size(-1)), )
            loss_sub = self.cross_entropy(vis_probas_sub_all.view(-1, vis_probas_sub_all.size(-1)), 
                                          fused_probas_sub_all.view(-1, fused_probas_sub_all.size(-1)))
            loss_obj = self.cross_entropy(vis_probas_obj_all.view(-1, vis_probas_obj_all.size(-1)), 
                                          fused_probas_obj_all.view(-1, fused_probas_obj_all.size(-1)))

        total_loss = (loss_rel + loss_sub + loss_obj) / 3

        return total_loss

if __name__ == '__main__':
    pass

