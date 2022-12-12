import torch
import torch.nn as nn
from xmuda.models.resnet34_unet import UNetResNet34
from xmuda.models.scn_unet import UNetSCN
from xmuda.models.getting_function import all_fea_label, knn, get_graph_feature, all_fea_label_batch
from pytorch_revgrad import RevGrad

class DisNet(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 ):
        super(DisNet, self).__init__()
        self.RG = RevGrad(0.01)
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)
        self.linear_D1 = nn.Sequential(nn.Linear(input_dim, input_dim), self.bn1,nn.LeakyReLU(negative_slope=0.2))
        self.linear_D2 = nn.Sequential(nn.Linear(input_dim, output_dim), self.bn2, nn.LeakyReLU(negative_slope=0.2))

    def forward(self, con_all_fea):
        con_all_fea = self.RG(con_all_fea)
        con_all_fea = self.linear_D1(con_all_fea)
        con_all_fea = self.linear_D2(con_all_fea)
        return con_all_fea

class Net2DSeg(nn.Module):
    def __init__(self,
                 num_classes,
                 dual_head,
                 backbone_2d,
                 backbone_2d_kwargs
                 ):
        super(Net2DSeg, self).__init__()
        self.num_classes = num_classes
        # 2D image network
        if backbone_2d == 'UNetResNet34':
            self.net_2d = UNetResNet34(**backbone_2d_kwargs)
            self.feat_channels = 64
        else:
            raise NotImplementedError('2D backbone {} not supported'.format(backbone_2d))

        # segmentation head
        self.linear = nn.Linear(self.feat_channels, self.num_classes)

        # 2nd segmentation head
        self.dual_head = dual_head
        if dual_head:
            self.linear2 = nn.Linear(self.feat_channels, self.num_classes)

        self.trans1 = nn.Conv2d(self.feat_channels, self.feat_channels, kernel_size=1, stride=1)
        self.trans2 = nn.Conv2d(self.feat_channels, self.feat_channels//2, kernel_size=1, stride=1)#1深度预测
        self.trans3 = nn.Conv2d(self.feat_channels//2, 2, kernel_size=1, stride=1,bias=False)#1深度预测
        disnet = nn.Linear(16, 2, bias=False)
        self.dismap = DisNet(self.feat_channels, 16)
        self.disnet = []
        for i in range(self.num_classes):
            self.disnet.append(disnet.cuda())

    def forward(self, data_batch):
        img = data_batch['img']
        img_indices = data_batch['img_indices']
        # 2D network
        x = self.net_2d(img)
        preds = { 'x': x,}

        img_feats = []
        for i in range(x.shape[0]):
            img_feats.append(x.permute(0, 2, 3, 1)[i][img_indices[i][:, 0], img_indices[i][:, 1]])
        img_feats = torch.cat(img_feats, 0)
        preds['feats'] = img_feats
        logit = self.linear(img_feats)
        preds['seg_logit'] = logit

        data_batch['pseudo_label'] = torch.argmax(logit,-1)

        if data_batch['ssp']:

            x = self.trans1(x)
            depth_pre = self.trans2(x)
            depth_pre = self.trans3(depth_pre)
            img_feats_h2 = []
            last_number = 0
            point_xyz_label = []
            depth_pre_all = []
            xyz_all = (data_batch['x'][0][:, :3]).view(1, -1, 3).cuda().float()
            for i in range(x.shape[0]):
                current_number = last_number + img_indices[i].shape[0]
                img_feats_h2.append(x.permute(0, 2, 3, 1)[i][img_indices[i][:, 0], img_indices[i][:, 1]])
                depth_pre_all.append(depth_pre.permute(0, 2, 3, 1)[i][img_indices[i][:, 0], img_indices[i][:, 1]])

                point_xyz = xyz_all[:, last_number:current_number, :]
                point_xyz = (point_xyz - point_xyz.min()) / (point_xyz.max() - point_xyz.min())
                point_xyz_label.append(point_xyz.squeeze(0))  # [2250 3]
                last_number += img_indices[i].shape[0]
                # mask_label[i, 0][img_indices[i][:, 0], img_indices[i][:, 1]] = 1  # 这里是image-level有目标的标记为1

            depth_pre_all = torch.cat(depth_pre_all, 0).squeeze()
            point_xyz_label = torch.cat(point_xyz_label, 0)
            depth_label = torch.norm(point_xyz_label[:, :2], dim=1).view(-1,1)
            height_label = point_xyz_label[:, 2].view(-1,1)
            pre_label = torch.cat([depth_label, height_label],-1).detach()
            posi_loss = torch.abs(depth_pre_all - pre_label).mean()
            img_feats_h2 = torch.cat(img_feats_h2, 0)

            preds['feats'] = img_feats_h2
            preds['posi_loss'] = posi_loss

        else: img_feats_h2 = img_feats

        if self.dual_head:
            preds['seg_logit2'] = self.linear2(img_feats_h2)

        if data_batch['dis']  and data_batch['inter']>0:
            # con_all_fea, all_label, pseudo_label = all_fea_label(img_feats, data_batch['2dproto'], data_batch)
            con_all_fea, all_label, pseudo_label = all_fea_label_batch(img_feats, data_batch['2dproto'], data_batch)
            con_all_map = self.dismap(con_all_fea).view(self.num_classes, -1, 16)
            con_all_out = []
            for j in range(self.num_classes):
                con_all_out.append(self.disnet[j](con_all_map[j,:,:]).view(-1, 1 ,2))
            con_all = torch.cat(con_all_out, 1)
            preds['con_all_fea'] = con_all.view(-1,2)
            preds['all_label'] = all_label.view(-1)
            preds['pseudo_label'] = pseudo_label

        return preds

class Net3DSeg(nn.Module):
    def __init__(self,
                 num_classes,
                 dual_head,
                 backbone_3d,
                 backbone_3d_kwargs,
                 ):
        super(Net3DSeg, self).__init__()
        self.num_classe = num_classes
        # 3D network
        if backbone_3d == 'SCN':
            self.net_3d = UNetSCN(**backbone_3d_kwargs)
        else:
            raise NotImplementedError('3D backbone {} not supported'.format(backbone_3d))

        # segmentation head
        self.linear = nn.Linear(self.net_3d.out_channels, self.num_classe )

        # 2nd segmentation head
        self.dual_head = dual_head
        if dual_head:
            self.linear2 = nn.Linear(self.net_3d.out_channels, self.num_classe )

        self.region_split = [3, 4]
        self.k = 12
        self.liner3 = nn.Linear(self.net_3d.out_channels, self.net_3d.out_channels)
        self.liner4 = nn.Linear(self.net_3d.out_channels, self.net_3d.out_channels)
        self.liner5 = nn.Linear(self.net_3d.out_channels, 3 + self.k, bias=False)

        disnet = nn.Linear(16, 2, bias=False)
        self.dismap = DisNet(self.net_3d.out_channels, 16)
        self.disnet = []
        for i in range(self.num_classe):
            self.disnet.append(disnet.cuda())

    def forward(self, data_batch):
        feats = self.net_3d(data_batch['x'])
        x = self.linear(feats)
        preds = {
            'feats': feats,
            'seg_logit': x,
        }

        img_indices = data_batch['img_indices']
        bitch_size = len(img_indices)

        if data_batch['dis'] and data_batch['inter']>0:#
            con_all_fea, all_label, pseudo_label = all_fea_label(feats, data_batch['3dproto'], data_batch)
            con_all_map = self.dismap(con_all_fea).view(self.num_classe, -1, 16)
            con_all_out = []
            for j in range(self.num_classe):
                con_all_out.append(self.disnet[j](con_all_map[j,:,:]).view(-1, 1 ,2))
            con_all = torch.cat(con_all_out, 1)
            preds['con_all_fea'] = con_all.view(-1,2)
            preds['all_label'] = all_label.view(-1)
            preds['pseudo_label'] = pseudo_label

        if data_batch['ssp']:
            xyz_all = (data_batch['x'][0][:, :3]).view(1, -1, 3).cuda().float()
            last_number = 0
            idx = []
            img = data_batch['img']
            rgb = []
            for i in range(bitch_size):
                rgb.append(img.permute(0, 2, 3, 1)[i][img_indices[i][:, 0], img_indices[i][:, 1]])

                current_number = last_number + img_indices[i].shape[0]
                xyz = xyz_all[:, last_number:current_number, :]
                idx_i = knn(xyz.transpose(2, 1), self.k) + last_number
                idx.extend(idx_i)
                last_number += img_indices[i].shape[0]
            idxcat = torch.cat(idx, 0)
            rgb = torch.cat(rgb, 0)

            feats_ssp = self.liner3(feats)
            pre_3d = self.liner4(feats_ssp)
            pre_3d = self.liner5(pre_3d)
            rgb_pre, rgb_local = pre_3d[:,:3], pre_3d[:,3:]

            gray = torch.matmul(rgb, torch.Tensor([0.299, 0.587, 0.114]).cuda().view(-1,1)).view(-1,1)#
            grad = get_graph_feature(gray.view(1, -1, 1), self.k, idxcat).squeeze(0)

            grad_rgb_pre = rgb_local.view(-1, self.k, grad.shape[-1])
            grad_disstance = torch.abs(grad - grad_rgb_pre).mean()
            disstance = torch.abs(rgb_pre-rgb).mean()
            preds['point_trans_fea'] = feats_ssp
            preds['self_loss_3d'] = disstance + 0.1 * grad_disstance

        else: feats_ssp = feats

        if self.dual_head:
            preds['seg_logit2'] = self.linear2(feats_ssp)

        return preds

def test_Net2DSeg():
    # 2D
    batch_size = 2
    img_width = 400
    img_height = 225

    # 3D
    num_coords = 2000
    num_classes = 11

    # 2D
    img = torch.rand(batch_size, 3, img_height, img_width)
    u = torch.randint(high=img_height, size=(batch_size, num_coords // batch_size, 1))
    v = torch.randint(high=img_width, size=(batch_size, num_coords // batch_size, 1))
    img_indices = torch.cat([u, v], 2)

    # to cuda
    img = img.cuda()
    img_indices = img_indices.cuda()

    net_2d = Net2DSeg(num_classes,
                      backbone_2d='UNetResNet34',
                      backbone_2d_kwargs={},
                      dual_head=True)

    net_2d.cuda()
    out_dict = net_2d({
        'img': img,
        'img_indices': img_indices,
    })
    for k, v in out_dict.items():
        print('Net2DSeg:', k, v.shape)


def test_Net3DSeg():
    in_channels = 1
    num_coords = 2000
    full_scale = 4096
    num_seg_classes = 11

    coords = torch.randint(high=full_scale, size=(num_coords, 3))
    feats = torch.rand(num_coords, in_channels)

    feats = feats.cuda()

    net_3d = Net3DSeg(num_seg_classes,
                      dual_head=True,
                      backbone_3d='SCN',
                      backbone_3d_kwargs={'in_channels': in_channels})

    net_3d.cuda()
    out_dict = net_3d({
        'x': [coords, feats],
    })
    for k, v in out_dict.items():
        print('Net3DSeg:', k, v.shape)

if __name__ == '__main__':
    test_Net2DSeg()
    test_Net3DSeg()
