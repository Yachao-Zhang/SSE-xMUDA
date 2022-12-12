import numpy as np
import torch
import glob, os, sys
import random


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
from xmuda.data.utils.helper_ply import read_ply, write_ply
import plyfile as ply
import torch
from torchvision import utils as vutils
from xmuda.data.utils.trans_rtsvilla import find_best_re, find_best_re_s2t
object_color = {
    0: [255, 158, 0],  # vehicle
    1: [0, 0, 230],  # pedestrian
    2: [255, 61, 99],  # bike
    3: [0, 178, 0],  # traffic boundary
    4: [200, 200, 200],  #background
    5: [40, 40, 40],  #no lable
    6: [12, 158, 0],  # vehicle
    7: [233, 0, 230],  # pedestrian
    8: [45, 61, 99],  # bike
    9: [143, 178, 34],  # traffic boundary
    10: [183, 78, 54],  #background
    11: [40, 160, 220],  #no lable
}
def save_ply(points_i, prefix, label_i, name, p, save_path='./show_res'):
    '''
    :param points: Nx3
    :param prefix: 前缀,如1percent
    :param labels: 标签 list 长度为N
    :param name: 文件名
    :param p: 后缀
    :param save_path: 保存的路径
    :return:
    '''
    labels = label_i.copy()
    points = points_i.cpu().numpy()
    # points
    labels[labels< 0] = 5
    vertex = [(points[k, 0], points[k, 2], points[k, 1],
               object_color[labels[k]][0], object_color[labels[k]][1], object_color[labels[k]][2])
              for k in range(labels.shape[0])]
    vertex = np.array(vertex, dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),  # f4:float32 u1:uint8
    ])
    el = ply.PlyElement.describe(vertex, 'vertex')
    data = ply.PlyData([el], text=False)
    if os.path.exists(save_path) == False:
        os.makedirs(save_path)
    if prefix is None:
        prefix = ''
    if name.endswith('.ply'):  # 如果文件名中有.ply,则去除
        name = name[:-4]
    tmp = os.path.join(save_path, '{}_{}_{}.ply'.format(prefix, name, p))
    data.write(tmp)
    print('save {} successfully'.format(tmp))


def find_best_t(curr_src, curr_trg, src_x, trg_x, cut_W, W, num_row, sl):
    step = 30
    row_step = 10
    t = torch.zeros(3).long()
    t[2] = (src_x[:, 2].min() - trg_x[:, 2].min()) # .numpy()
    if sl:
        idx_trg = np.where((curr_trg[:, 1] < cut_W+step)&(curr_trg[:, 1] > cut_W)&(curr_trg[:, 0] < num_row)&(curr_trg[:, 0] > num_row-row_step))
        idx_src = np.where((curr_src[:, 1] < cut_W+step)&(curr_src[:, 1] > cut_W)&(curr_src[:, 0] < num_row)&(curr_src[:, 0] > num_row-row_step))
    else:
        idx_trg = np.where((curr_trg[:, 1] < W-cut_W) & (curr_trg[:, 1] > W-cut_W-step) & (curr_trg[:, 0] < num_row) & (
                        curr_trg[:, 0] > num_row - row_step))
        idx_src = np.where((curr_src[:, 1] < W-cut_W) & (curr_src[:, 1] > W-cut_W-step) & (curr_src[:, 0] < num_row) & (
                    curr_src[:, 0] > num_row - row_step))
    if idx_trg[0].shape[0]>0 and idx_src[0].shape[0]>0:
        minnum = min(idx_trg[0].shape[0], idx_src[0].shape[0])
        cloud_trg = trg_x[idx_trg[0][:minnum],:3]#target domain as source object
        cloud_src = src_x[idx_src[0][:minnum],:3]
        t = ((cloud_src - cloud_trg).float()).mean(0)#.numpy()
    return t.long()


def data_mixaugment_src_left(data_batch_src, data_batch_trg, cut_W):
    B, d, H, W = data_batch_src['img'].shape
    src_img = data_batch_src['img'][:B//2,:,:,:cut_W]#取源域的前半部分
    trg_img = data_batch_trg['img'][:B//2:,:,:,cut_W:]#取目标域的
    # src_img = (trg_img.reshape(B,d,-1).mean(-1) / src_img.reshape(B,d,-1).mean(-1)).view(B,d,1,1) * src_img
    mix_img = torch.cat([src_img, trg_img],-1)

    src_indices = data_batch_src['img_indices'][:B//2]
    trg_indices = data_batch_trg['img_indices'][:B//2]
    mix_x = []
    mix_x_m = []
    mix_label = []
    x_src, x_src_m = data_batch_src['x']
    x_trg, x_trg_m = data_batch_trg['x']
    last_number_src = 0
    last_number_trg = 0
    src_num_all = 0
    for i in range(B//2):

        vutils.save_image(mix_img[i, ...].cpu(), '/home/viplab/sdb1_dir/MSMnew/vis/'+str(i)+'mix.jpg')
        vutils.save_image(data_batch_src['img'][i, ...].cpu(), '/home/viplab/sdb1_dir/MSMnew/vis/'+str(i)+'src.jpg')
        vutils.save_image(data_batch_trg['img'][i, ...].cpu(), '/home/viplab/sdb1_dir/MSMnew/vis/'+str(i)+'trg.jpg')

        src_num_all += src_indices[i].shape[0]
        current_src_indices = src_indices[i]
        current_trg_indices = trg_indices[i]
        current_number_src = last_number_src + current_src_indices.shape[0]
        current_number_trg = last_number_trg + current_trg_indices.shape[0]
        cut_idx_src = np.where(current_src_indices[:,-1]<cut_W)[0]#src
        cut_idx_trg = np.where((current_trg_indices[:,-1]>cut_W) | (current_trg_indices[:,-1]==cut_W))[0]#trg

        current_label = torch.ones(cut_idx_src.shape[0]+cut_idx_trg.shape[0]).cuda()
        current_label_src = data_batch_src['seg_label'][last_number_src:current_number_src]#取的前面的batch，对这个没有影响

        current_label[:cut_idx_src.shape[0]] = current_label_src[cut_idx_src]#前半部分的是源域本来的标签
        current_label[cut_idx_src.shape[0]:] = -100#混合目标域标签全部忽略
        mix_label.append(current_label)#所有标签

        x_src_i, x_src_m_i = x_src[last_number_src:current_number_src,:],x_src_m[last_number_src:current_number_src]#取的前面的batch，对这个没有影响
        x_trg_i, x_trg_m_i = x_trg[last_number_trg:current_number_trg,:],x_trg_m[last_number_trg:current_number_trg]#取的前面的batch，对这个没有影响
        x_src_icut = x_src_i[cut_idx_src, :]#索引到源域的点
        x_trg_icut = x_trg_i[cut_idx_trg, :]#索引到目标域的点

        data_batch_src['num_src'].append( x_src_icut.shape[0])

        # t = torch.zeros(3).long()
        # t[2] = (x_src_icut[:, 2].min() - x_trg_icut[:, 2].min())  # .numpy()
        t = find_best_t(current_src_indices, current_trg_indices, x_src_i, x_trg_i, cut_W, W, H, True)
        aug_3d = x_trg_icut[:, :3] + t.view(1, 3)
        aug_3d[aug_3d < 0] = 0
        x_trg_icut[:, :3] = aug_3d

        mix_x.append(torch.cat([x_src_icut, x_trg_icut],0))#点云拼接起来
        mix_x_m.append(torch.cat([x_src_m_i[cut_idx_src], x_trg_m_i[cut_idx_trg]],0))#对应的指示也拼接起来
        current_mix_indices = np.concatenate([current_src_indices[cut_idx_src], current_trg_indices[cut_idx_trg]],0)#map索引拼接起来
        mix_x[i][:, -1] = i

        save_ply(x_src_i[:, :3], 'src', current_label_src.cpu().numpy(), str(i), '_l', save_path='/home/viplab/sdb1_dir/MSMnew/vis/show_src')
        current_trg_label = torch.ones(x_trg_i.shape[0]) * -100  # 混合目标域标签全部忽略
        save_ply(x_trg_i[:, :3], 'mix', current_trg_label.cpu().numpy(), str(i), '_l', save_path='/home/viplab/sdb1_dir/MSMnew/vis/show_trg')
        save_ply(mix_x[i][:,:3], 'mix', mix_label[i].cpu().numpy(), str(i), '_l', save_path='/home/viplab/sdb1_dir/MSMnew/vis/show_mix')

        data_batch_src['img_indices'][i] = current_mix_indices
        last_number_src = current_number_src
        last_number_trg = current_number_trg

    mix_x = torch.cat(mix_x, 0).detach()
    mix_x_m = torch.cat(mix_x_m, 0)
    mix_label = torch.cat(mix_label, 0).long()

    data_batch_src['seg_label'] = torch.cat([mix_label, data_batch_src['seg_label'][src_num_all:]],0)
    data_batch_src['img'] = torch.cat([mix_img, data_batch_src['img'][B//2:,...]])
    data_batch_src['x'][0] = torch.cat([mix_x, data_batch_src['x'][0][src_num_all:,...]], 0)
    data_batch_src['x'][1] = torch.cat([mix_x_m,data_batch_src['x'][1][src_num_all:]], 0)

    return data_batch_src

def data_mixaugment_src_right(data_batch_src, data_batch_trg, cut_W):

    B, d, H, W = data_batch_src['img'].shape

    src_img = data_batch_src['img'][:B//2,:,:,-cut_W:]#取源域的后半部分
    trg_img = data_batch_trg['img'][:B//2:,:,:,:-cut_W]#取目标域的
    mix_img = torch.cat([trg_img, src_img],-1)

    src_indices = data_batch_src['img_indices'][:B//2]
    # trg_indices = data_batch_trg['img_indices'][:B//2]
    mix_x = []
    mix_x_m = []
    mix_label = []
    x_src, x_src_m = data_batch_src['x']
    # x_trg, x_trg_m = data_batch_trg['x']
    last_number_src = 0
    last_number_trg = 0
    src_num_all = 0

    for i in range(B//2):
        # vutils.save_image(mix_img[i, ...].cpu(), './'+str(i)+'r_mix.jpg')
        # vutils.save_image(data_batch_src['img'][i, ...].cpu(), './'+str(i)+'r_src.jpg')
        # vutils.save_image(data_batch_trg['img'][i, ...].cpu(), './'+str(i)+'r_trg.jpg')
        src_num_all += src_indices[i].shape[0]
        current_src_indices = src_indices[i]
        # current_trg_indices = trg_indices[i]
        current_number_src = last_number_src + current_src_indices.shape[0]
        # current_number_trg = last_number_trg + current_trg_indices.shape[0]

        cut_idx_src = np.where(current_src_indices[:, -1] > W-cut_W)[0]  # src
        # cut_idx_trg = np.where((current_trg_indices[:, -1] < W-cut_W) | (current_trg_indices[:, -1] == W-cut_W))[0]

        current_label = torch.ones(cut_idx_src.shape[0]).cuda()
        current_label_src = data_batch_src['seg_label'][last_number_src:current_number_src]#取的前面的batch，对这个没有影响

        current_label[-cut_idx_src.shape[0]:] = current_label_src[cut_idx_src]  # 前半部分的是源域本来的标签
        # current_label[:-cut_idx_src.shape[0]] = -100  # 混合目标域标签全部忽略
        mix_label.append(current_label)#所有标签

        x_src_i, x_src_m_i = x_src[last_number_src:current_number_src,:],x_src_m[last_number_src:current_number_src]#取的前面的batch，对这个没有影响
        # x_trg_i, x_trg_m_i = x_trg[last_number_trg:current_number_trg,:],x_trg_m[last_number_trg:current_number_trg]#取的前面的batch，对这个没有影响
        x_src_icut = x_src_i[cut_idx_src, :]#索引到源域的点
        # x_trg_icut = x_trg_i[cut_idx_trg, :]#索引到目标域的点

        data_batch_src['num_src'].append( x_src_icut.shape[0])
        # t = torch.zeros(3).long()
        # t[2] = (x_src_icut[:, 2].min() - x_trg_icut[:, 2].min())  # .numpy()
        # t = find_best_t(current_src_indices, current_trg_indices, x_src_i, x_trg_i, cut_W, W, H, False)
        # aug_3d = x_trg_icut[:, :3] + t.view(1, 3)
        # aug_3d[aug_3d < 0] = 0
        # x_trg_icut[:, :3] = aug_3d


        mix_x.append( x_src_icut )  # 点云拼接起来
        mix_x_m.append(x_src_m_i[cut_idx_src])  # 对应的指示也拼接起来
        current_mix_indices = current_src_indices[cut_idx_src]  # map索引拼接起
        mix_x[i][:, -1] = i
        data_batch_src['img_indices'][i] = current_mix_indices
        last_number_src = current_number_src
        # last_number_trg = current_number_trg
        # save_ply(mix_x[i][:,:3], 'mix', mix_label[i].cpu().numpy(), str(i), 'r_', save_path='./show_res')
    mix_x = torch.cat(mix_x, 0).detach()
    mix_x_m = torch.cat(mix_x_m, 0)
    mix_label = torch.cat(mix_label, 0).long()

    data_batch_src['seg_label'] = torch.cat([mix_label, data_batch_src['seg_label'][src_num_all:]],0)
    data_batch_src['img'] = torch.cat([mix_img, data_batch_src['img'][B//2:,...]])
    data_batch_src['x'][0] = torch.cat([mix_x, data_batch_src['x'][0][src_num_all:,...]], 0)
    data_batch_src['x'][1] = torch.cat([mix_x_m,data_batch_src['x'][1][src_num_all:]], 0)

    return data_batch_src


def data_mixaugment_src_left_ak(data_batch_src, data_batch_trg, cut_W):
    B, d, H, W = data_batch_src['img'].shape
    src_img = data_batch_src['img'][:B//2,:,:,:cut_W]#取源域的前半部分
    trg_img = data_batch_trg['img'][:B//2:,:,:,cut_W:]#取目标域的
    # src_img = (trg_img.reshape(B,d,-1).mean(-1) / src_img.reshape(B,d,-1).mean(-1)).view(B,d,1,1) * src_img
    mix_img = torch.cat([src_img, trg_img],-1)

    src_indices = data_batch_src['img_indices'][:B//2]
    # trg_indices = data_batch_trg['img_indices'][:B//2]
    mix_x = []
    mix_x_m = []
    mix_label = []
    x_src, x_src_m = data_batch_src['x']
    # x_trg, x_trg_m = data_batch_trg['x']
    last_number_src = 0
    # last_number_trg = 0
    src_num_all = 0
    for i in range(B//2):

        # vutils.save_image(mix_img[i, ...].cpu(), '/home/viplab/sdb1_dir/MSMnew/vis/'+str(i)+'mix.jpg')
        # vutils.save_image(data_batch_src['img'][i, ...].cpu(), '/home/viplab/sdb1_dir/MSMnew/vis/'+str(i)+'src.jpg')
        # vutils.save_image(data_batch_trg['img'][i, ...].cpu(), '/home/viplab/sdb1_dir/MSMnew/vis/'+str(i)+'trg.jpg')

        src_num_all += src_indices[i].shape[0]
        current_src_indices = src_indices[i]
        # current_trg_indices = trg_indices[i]
        current_number_src = last_number_src + current_src_indices.shape[0]
        # current_number_trg = last_number_trg + current_trg_indices.shape[0]
        cut_idx_src = np.where(current_src_indices[:,-1]<cut_W)[0]#src
        # cut_idx_trg = np.where((current_trg_indices[:,-1]>cut_W) | (current_trg_indices[:,-1]==cut_W))[0]#trg

        current_label = torch.ones(cut_idx_src.shape[0]).cuda()
        current_label_src = data_batch_src['seg_label'][last_number_src:current_number_src]#取的前面的batch，对这个没有影响

        current_label[:cut_idx_src.shape[0]] = current_label_src[cut_idx_src]#前半部分的是源域本来的标签
        # current_label[cut_idx_src.shape[0]:] = -100#混合目标域标签全部忽略
        mix_label.append(current_label)#所有标签

        x_src_i, x_src_m_i = x_src[last_number_src:current_number_src,:],x_src_m[last_number_src:current_number_src]#取的前面的batch，对这个没有影响
        # x_trg_i, x_trg_m_i = x_trg[last_number_trg:current_number_trg,:],x_trg_m[last_number_trg:current_number_trg]#取的前面的batch，对这个没有影响
        x_src_icut = x_src_i[cut_idx_src, :]#索引到源域的点
        # x_trg_icut = x_trg_i[cut_idx_trg, :]#索引到目标域的点

        data_batch_src['num_src'].append( x_src_icut.shape[0])

        # t = torch.zeros(3).long()
        # t[2] = (x_src_icut[:, 2].min() - x_trg_icut[:, 2].min())  # .numpy()
        # t = find_best_t(current_src_indices, current_trg_indices, x_src_i, x_trg_i, cut_W, W, H, True)
        # aug_3d = x_trg_icut[:, :3] + t.view(1, 3)
        # aug_3d[aug_3d < 0] = 0
        # x_trg_icut[:, :3] = aug_3d

        mix_x.append(x_src_icut)#点云拼接起来
        mix_x_m.append(x_src_m_i[cut_idx_src])#对应的指示也拼接起来
        current_mix_indices = current_src_indices[cut_idx_src]
        mix_x[i][:, -1] = i

        # save_ply(x_src_i[:, :3], 'src', current_label_src.cpu().numpy(), str(i), '_l', save_path='/home/viplab/sdb1_dir/MSMnew/vis/show_src')
        # current_trg_label = torch.ones(x_trg_i.shape[0]) * -100  # 混合目标域标签全部忽略
        # save_ply(x_trg_i[:, :3], 'mix', current_trg_label.cpu().numpy(), str(i), '_l', save_path='/home/viplab/sdb1_dir/MSMnew/vis/show_trg')
        # save_ply(mix_x[i][:,:3], 'mix', mix_label[i].cpu().numpy(), str(i), '_l', save_path='/home/viplab/sdb1_dir/MSMnew/vis/show_mix')

        data_batch_src['img_indices'][i] = current_mix_indices
        last_number_src = current_number_src
        # last_number_trg = current_number_trg

    mix_x = torch.cat(mix_x, 0).detach()
    mix_x_m = torch.cat(mix_x_m, 0)
    mix_label = torch.cat(mix_label, 0).long()

    data_batch_src['seg_label'] = torch.cat([mix_label, data_batch_src['seg_label'][src_num_all:]],0)
    data_batch_src['img'] = torch.cat([mix_img, data_batch_src['img'][B//2:,...]])
    data_batch_src['x'][0] = torch.cat([mix_x, data_batch_src['x'][0][src_num_all:,...]], 0)
    data_batch_src['x'][1] = torch.cat([mix_x_m,data_batch_src['x'][1][src_num_all:]], 0)

    return data_batch_src

def data_mixaugment_src_right_ak(data_batch_src, data_batch_trg, cut_W):

    B, d, H, W = data_batch_src['img'].shape

    src_img = data_batch_src['img'][:B//2,:,:,-cut_W:]#取源域的后半部分
    trg_img = data_batch_trg['img'][:B//2:,:,:,:-cut_W]#取目标域的
    mix_img = torch.cat([trg_img, src_img],-1)

    src_indices = data_batch_src['img_indices'][:B//2]
    trg_indices = data_batch_trg['img_indices'][:B//2]
    mix_x = []
    mix_x_m = []
    mix_label = []
    x_src, x_src_m = data_batch_src['x']
    x_trg, x_trg_m = data_batch_trg['x']
    last_number_src = 0
    last_number_trg = 0
    src_num_all = 0

    for i in range(B//2):
        # vutils.save_image(mix_img[i, ...].cpu(), './'+str(i)+'r_mix.jpg')
        # vutils.save_image(data_batch_src['img'][i, ...].cpu(), './'+str(i)+'r_src.jpg')
        # vutils.save_image(data_batch_trg['img'][i, ...].cpu(), './'+str(i)+'r_trg.jpg')
        src_num_all += src_indices[i].shape[0]
        current_src_indices = src_indices[i]
        current_trg_indices = trg_indices[i]
        current_number_src = last_number_src + current_src_indices.shape[0]
        current_number_trg = last_number_trg + current_trg_indices.shape[0]

        cut_idx_src = np.where(current_src_indices[:, -1] > W-cut_W)[0]  # src
        cut_idx_trg = np.where((current_trg_indices[:, -1] < W-cut_W) | (current_trg_indices[:, -1] == W-cut_W))[0]

        current_label = torch.ones(cut_idx_src.shape[0]+cut_idx_trg.shape[0]).cuda()
        current_label_src = data_batch_src['seg_label'][last_number_src:current_number_src]#取的前面的batch，对这个没有影响

        current_label[-cut_idx_src.shape[0]:] = current_label_src[cut_idx_src]  # 前半部分的是源域本来的标签
        current_label[:-cut_idx_src.shape[0]] = -100  # 混合目标域标签全部忽略
        mix_label.append(current_label)#所有标签

        x_src_i, x_src_m_i = x_src[last_number_src:current_number_src,:],x_src_m[last_number_src:current_number_src]#取的前面的batch，对这个没有影响
        x_trg_i, x_trg_m_i = x_trg[last_number_trg:current_number_trg,:],x_trg_m[last_number_trg:current_number_trg]#取的前面的batch，对这个没有影响
        x_src_icut = x_src_i[cut_idx_src, :]#索引到源域的点
        x_trg_icut = x_trg_i[cut_idx_trg, :]#索引到目标域的点

        data_batch_src['num_src'].append( x_src_icut.shape[0])
        # t = torch.zeros(3).long()
        # t[2] = (x_src_icut[:, 2].min() - x_trg_icut[:, 2].min())  # .numpy()
        t = find_best_t(current_src_indices, current_trg_indices, x_src_i, x_trg_i, cut_W, W, H, False)
        aug_3d = x_trg_icut[:, :3] + t.view(1, 3)
        aug_3d[aug_3d < 0] = 0
        x_trg_icut[:, :3] = aug_3d


        mix_x.append(torch.cat([x_trg_icut, x_src_icut], 0))  # 点云拼接起来
        mix_x_m.append(torch.cat([x_trg_m_i[cut_idx_trg], x_src_m_i[cut_idx_src]], 0))  # 对应的指示也拼接起来
        current_mix_indices = np.concatenate([current_trg_indices[cut_idx_trg], current_src_indices[cut_idx_src]],0)  # map索引拼接起
        mix_x[i][:, -1] = i
        data_batch_src['img_indices'][i] = current_mix_indices
        last_number_src = current_number_src
        last_number_trg = current_number_trg
        # save_ply(mix_x[i][:,:3], 'mix', mix_label[i].cpu().numpy(), str(i), 'r_', save_path='./show_res')
    mix_x = torch.cat(mix_x, 0).detach()
    mix_x_m = torch.cat(mix_x_m, 0)
    mix_label = torch.cat(mix_label, 0).long()

    data_batch_src['seg_label'] = torch.cat([mix_label, data_batch_src['seg_label'][src_num_all:]],0)
    data_batch_src['img'] = torch.cat([mix_img, data_batch_src['img'][B//2:,...]])
    data_batch_src['x'][0] = torch.cat([mix_x, data_batch_src['x'][0][src_num_all:,...]], 0)
    data_batch_src['x'][1] = torch.cat([mix_x_m,data_batch_src['x'][1][src_num_all:]], 0)

    return data_batch_src

def data_mixaugment(data_batch_src, data_batch_trg, cut_redio = 1/2, paced=None):
    """
    """
    if paced is not None:
        cut_redio = max(cut_redio + paced, 0)
    # cut_redio = 1/4#取了多少src 1/2#
    B, d, H, W= data_batch_src['img'].shape
    cut_W = int(W*cut_redio)
    rnd = random.randint(0, 1)

    data_batch_src['cut_W'] = cut_W
    data_batch_src['rol'] = rnd
    data_batch_src['num_src'] = []
    a2d2_kitti = True
    if a2d2_kitti:
        if rnd == 0:
            data_batch_src = data_mixaugment_src_left_ak(data_batch_src, data_batch_trg, cut_W)
        else:
            data_batch_src = data_mixaugment_src_right_ak(data_batch_src, data_batch_trg, cut_W)
    else:
        if rnd == 0:
            data_batch_src = data_mixaugment_src_left(data_batch_src, data_batch_trg, cut_W)
        else:
            data_batch_src = data_mixaugment_src_right(data_batch_src, data_batch_trg, cut_W)
    # data_batch_src = data_mixaugment_src_right(data_batch_src, data_batch_trg, cut_W)


    return data_batch_src