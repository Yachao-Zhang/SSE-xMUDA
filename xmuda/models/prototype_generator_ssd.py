import numpy as np
import torch
import torch.nn.functional as F


def propotype_generator(featsmix_2d, featsmix_3d, proto_2d, proto_3d, labelmix, num_classes):
    """
        Entropy loss for probabilistic prediction vectors
        input: batch_size x classes x points
        output: batch_size x 1 x points
    """
    # ************Updata the proposal_prototype_bank*****************************#
    feats_3d = featsmix_3d[labelmix >-1,:]
    feats_2d = featsmix_2d[labelmix >-1,:]
    label = labelmix[labelmix >-1]

    label_one_hot = F.one_hot(label, num_classes)
    num_points_class = torch.sum(label_one_hot, dim=0, keepdim = True).transpose(1,0)  #

    class_propotype_2d = torch.matmul(label_one_hot.transpose(1, 0).float(), feats_2d) / (num_points_class + 1e-6)
    class_propotype_3d = torch.matmul(label_one_hot.transpose(1, 0).float(), feats_3d) / (num_points_class + 1e-6)
    class_propotype_2d = F.normalize(class_propotype_2d, dim=-1)
    class_propotype_3d = F.normalize(class_propotype_3d, dim=-1)

    bate = 0.9
    for i in range(num_classes):
        if num_points_class.squeeze()[i] > 0:
            proto_2d[i, :] = (1 - bate)  * proto_2d[i, :].view(1, -1) + bate* class_propotype_2d[i, :].view(1, -1)
            proto_3d[i, :] = (1 - bate)  * proto_3d[i, :].view(1, -1) + bate* class_propotype_3d[i, :].view(1, -1)

    return proto_2d.detach(), proto_3d.detach()

def cross_consi_loss(trg_feats_2d, trg_feats_3d,proto_2d, proto_3d):
    proto_2d = F.normalize(proto_2d, dim=-1)
    proto_3d = F.normalize(proto_3d, dim=-1)
    trg_feats_2d = F.normalize(trg_feats_2d, dim=-1)
    trg_feats_3d = F.normalize(trg_feats_3d, dim=-1)
    class_num = proto_3d.shape[0]
    num_points, _ = trg_feats_2d.shape
    idx = (torch.arange(0, class_num, 1).view(1, -1)).repeat(num_points, 1).cuda()
    # cosin distance
    sem_diss_2d = torch.matmul(trg_feats_2d, proto_2d.T)
    cons_entropy_2d = -1. * torch.mul(sem_diss_2d, torch.log2(sem_diss_2d + 1e-30)).mean(-1)
    sem2d_pos, idx_2d = sem_diss_2d.max(-1)
    neg_2d = sem_diss_2d[(idx - idx_2d.view(-1, 1)) != 0].view(-1, class_num-1).mean(-1)
    #
    # neg_2d = sem_diss_2d[(sem_diss_2d-sem2d_pos.view(-1,1))<0].view(-1, class_num-1).mean(-1)
    sem_diss_3d = torch.matmul(trg_feats_3d, proto_3d.T)
    cons_entropy_3d = -1. * torch.mul(sem_diss_3d, torch.log2(sem_diss_3d + 1e-30)).mean(-1)
    sem3d_pos, idx_3d = sem_diss_3d.max(-1)
    neg_3d = sem_diss_3d[(idx - idx_3d.view(-1, 1)) != 0].view(-1, class_num-1).mean(-1)#如果两个max相似性相同,会报错吧

    consi_2d = (((1-sem2d_pos) + neg_2d)*(1-cons_entropy_2d)).mean()
    consi_3d = (((1-sem3d_pos) + neg_3d)*(1-cons_entropy_3d)).mean()
    # print(consi_2d, consi_3d)
    return consi_2d, consi_3d

def content_loss(con_feats, label):
    con_feats = F.normalize(con_feats, dim=-1)
    # label_onehot = F.one_hot(label, 2).float()

    sem_con_feats = torch.matmul(con_feats, con_feats.T)

    con_sim_loss = (1.0-sem_con_feats).mean()
    return con_sim_loss