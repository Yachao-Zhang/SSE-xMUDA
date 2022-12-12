import torch
import torch.nn.functional as F

def knn(x, k):
    N = x.shape[-1]
    if N<18000 and N>k:#消除那个8万个点的样本对相似性计算的影响
        inner = -2 * torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x ** 2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)
        idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    else:
        idx_i = torch.arange(0,N).cuda()
        idx = idx_i.view(1,N,1).repeat(1,1,k)
    return idx

def get_graph_feature(x, k, idx=None):
    batch_size ,num_points,  num_dims = x.size()
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx_tr = idx + idx_base
    idx_tr = idx_tr.view(-1)

    feature = x.view(batch_size * num_points, -1)[idx_tr, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = x - feature

    return feature

def get_style_label(B, data_batch):
    if data_batch['istrg']:
        labels = torch.Tensor([1, 1]).long()
        curr_label = labels.view(1, 2).repeat(B, 1).view(-1)
    if not data_batch['istrg']:
        if data_batch['mix']:
            labels_mix = torch.Tensor([0, 1]).long()  # 目标域是1
            labels_src = torch.Tensor([0, 0]).long()
            if data_batch['rol'] == 0:
                curr_label = torch.cat(
                    [labels_mix.view(1, 2).repeat(B // 2, 1), labels_src.view(1, 2).repeat(B // 2, 1)],
                    0).permute(1, 0).contiguous().view(-1)
            else:
                curr_label = torch.cat(
                    [(1 - labels_mix).view(1, 2).repeat(B // 2, 1), labels_src.view(1, 2).repeat(B // 2, 1)],
                    0).permute(1, 0).contiguous().view(-1)
        else:
            labels = torch.Tensor([0, 0]).long()
            curr_label = labels.view(1, 2).repeat(B, 1).view(-1)
    return curr_label


def propotype_generator(feats , proto , in_label):
    """
        Entropy loss for probabilistic prediction vectors
        input: batch_size x classes x points
        output: batch_size x 1 x points
    """

    num_classes = proto.shape[0]
    # ************Updata the proposal_prototype_bank*****************************#
    feats = feats[in_label >-1,:]
    label = in_label[in_label >-1]

    label_one_hot = F.one_hot(label, num_classes)
    num_points_class = torch.sum(label_one_hot, dim=0, keepdim = True).transpose(1,0)  #

    class_propotype = torch.matmul(label_one_hot.transpose(1, 0).float(), feats) / (num_points_class + 1e-6)
    class_propotype = F.normalize(class_propotype, dim=-1)

    class_propotype_mask = (num_points_class.squeeze()>0).long()
    # curr_pseudo_label = torch.arange(0, num_classes, 1).cuda()[torch.where(num_points_class.squeeze()>0)[0]]

    return class_propotype, class_propotype_mask

def get_pseudo_label(trg_feats,proto):
    # class_num = proto.shape[0]
    # num_points, d = trg_feats.shape
    proto = F.normalize(proto, dim=-1)
    trg_feats = F.normalize(trg_feats, dim=-1)
    # cosin distance
    sem_diss_2d = torch.matmul(trg_feats, proto.T)

    cons_entropy = -1. * torch.mul(sem_diss_2d, torch.log2(sem_diss_2d + 1e-30)).mean(-1)
    pseudo_label = sem_diss_2d.argmax(-1)

    return pseudo_label, cons_entropy

def get_proto_num(trg_feats, proto, pseudo_label=None):
    if pseudo_label is None:
        pseudo_label, cons_entropy = get_pseudo_label(trg_feats, proto)
    proto_trg, class_propotype_mask  = propotype_generator(trg_feats, proto, pseudo_label)#, curr_pseudo_label
    # num_pseudo_class, _ = proto_trg.shape

    return proto_trg, pseudo_label, class_propotype_mask

def all_fea_label(feats, proto, data_batch):

    img_indices = data_batch['img_indices']
    label = data_batch['seg_label']
    plabel = data_batch['pseudo_label']
    bitch_size = len(img_indices)
    class_num = proto.shape[0]
    d = feats.shape[-1]
    last_number = 0
    all_proto = []
    curr_sty_alllabel = []
    pseudo_label_all = []
    for i in range(bitch_size):
        num_points = img_indices[i].shape[0]
        current_number = last_number + num_points
        feats_curr = feats[last_number:current_number, :]
        curr_label = label[last_number:current_number]
        curr_plabel = plabel[last_number:current_number]

        if data_batch['istrg']:
            trg_all_proto, pseudo_label, class_propotype_mask = get_proto_num(feats_curr, proto, curr_plabel)  # , curr_pseudo_label
            curr_sty_label = (torch.ones(class_num, 1).cuda() * (1 - class_propotype_mask.view(-1, 1)) * (-100)).long()
        else:
            trg_all_proto, pseudo_label, class_propotype_mask = get_proto_num(feats_curr, proto, curr_label)  # , curr_pseudo_label
            curr_sty_label = ((1 - class_propotype_mask) * (-100)).view(-1, 1).long()
        trg_all_proto = trg_all_proto.view(class_num,1,d)
        curr_sty_alllabel.append(curr_sty_label)
        all_proto.append(trg_all_proto)
        last_number += num_points
        pseudo_label_all.append(pseudo_label)

    curr_sty_all = torch.cat(curr_sty_alllabel, 1)
    curr_all_proto = torch.cat(all_proto, 1)
    pseudo_label_all = torch.cat(pseudo_label_all, 0)

    if data_batch['istrg']:
        another_label = torch.zeros(class_num, 1).cuda().cuda().long()
    else:
        another_label = torch.ones(class_num, 1).cuda().long()

    all_label = torch.cat([another_label, curr_sty_all], 1)

    con_all_fea = torch.cat([proto.view(class_num, 1, d), curr_all_proto.view(class_num, -1,d)], 1)#前面的是一个


    return con_all_fea.view(-1,d), all_label, pseudo_label_all

def all_fea_label_batch(feats, proto, data_batch):

    img_indices = data_batch['img_indices']
    label = data_batch['seg_label']
    plabel = data_batch['pseudo_label']
    bitch_size = len(img_indices)
    class_num = proto.shape[0]
    d = feats.shape[-1]
    last_number = 0
    if data_batch['istrg']:
        trg_all_proto, pseudo_label, class_propotype_mask = get_proto_num(feats, proto,  plabel)  # , curr_pseudo_label
        curr_sty_label = (torch.ones(class_num, 1).cuda() * (1 - class_propotype_mask.view(-1, 1)) * (-100)).long()
    else:
        trg_all_proto, pseudo_label, class_propotype_mask = get_proto_num(feats, proto,label)  # , curr_pseudo_label
        curr_sty_label = ((1 - class_propotype_mask) * (-100)).view(-1, 1).long()

    if data_batch['istrg']:
        another_label = torch.zeros(class_num, 1).cuda().cuda().long()
    else:
        another_label = torch.ones(class_num, 1).cuda().long()

    all_label = torch.cat([another_label, curr_sty_label], 1)

    con_all_fea = torch.cat([proto.view(class_num, 1, d), trg_all_proto.view(class_num, -1,d)], 1)#前面的是一个

    return con_all_fea.view(-1,d), all_label, pseudo_label