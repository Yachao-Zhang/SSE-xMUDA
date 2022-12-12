import numpy as np
import torch
import torch.nn.functional as F
def find_best_re(curr_src, curr_trg, src_x, trg_x, cut_W, num_row):
    idxtrg = []
    idxsrc = []
    step = 20#15
    row_step = 5
    R = torch.eye(3).float()
    t = torch.ones(3).float()
    s = 1.0
    R_xy = torch.zeros(2,2).float()
    row_posi = int(num_row / row_step)
    for i in range(row_posi-20, row_posi):
        j = (i) * row_step
        idx_trg = np.where((curr_trg[:, 1] < cut_W+step)&(curr_trg[:, 1] > cut_W)&(curr_trg[:, 0] < j+row_step)&(curr_trg[:, 0] > j))
        idx_src = np.where((curr_src[:, 1] < cut_W+step)&(curr_src[:, 1] > cut_W)&(curr_src[:, 0] < j+row_step)&(curr_src[:, 0] > j))

        if (idx_trg[0].shape[0]>2)&(idx_src[0].shape[0]>2):
            minnum = int(min(idx_trg[0].shape[0], idx_src[0].shape[0])/2)*2#保证取偶数min(idx_trg[0].shape[0], idx_src[0].shape[0])#
            idxtrg.extend(idx_trg[0][0:minnum].tolist())
            idxsrc.extend(idx_src[0][0:minnum].tolist())

    if len(idxtrg)>0:
        cloud_trg = trg_x[idxtrg,:3]#target domain as source object
        cloud_src = src_x[idxsrc,:3]
        # save_ply(trg_x, 'src', label, str(1), 'srctrg', save_path='./show_res')
        mid_cent = int(cloud_src.shape[0]/2)#根据匹配点的4#
        #计算比例点应该和计算旋转矩阵的点应该一致；
        src_super_point1, src_super_point2 = cloud_src[:mid_cent,:2].float(), cloud_src[-mid_cent:,:2].float()#.mean(0)
        trg_super_point1, trg_super_point2 = cloud_trg[:mid_cent,:2].float(), cloud_trg[-mid_cent:,:2].float()#.mean(0)

        cloud_trg[:, :2] = (cloud_trg[:, :2] * s)
        # print(s)
        if (s>0.6)&(s<1.8):
            # 旋转只发生在前两个维度
            for i in range(src_super_point1.shape[0]):
                src_super_point, trg_super_point =  torch.cat([src_super_point1[i,:].view(1,-1), src_super_point2[i,:].view(1,-1)],0), \
                                                    torch.cat([trg_super_point1[i,:].view(1,-1), trg_super_point2[i,:].view(1,-1)],0)#这里还需要修改
                trg_super_point = (trg_super_point*s)#后面的方法全是基于缩放后执行，所以最先需要缩放；
                R_xy += torch.matmul(src_super_point.T, torch.inverse(trg_super_point.T))

            R[:2,:2] = R_xy/(src_super_point1.shape[0])
            # print(R)
            t = ((cloud_src.float() - (torch.matmul(R,cloud_trg.float().T)).T).mean(0))#.numpy()
        else:
            t = ((cloud_src.float() - cloud_trg.float()).mean(0))#.numpy()
            R = torch.eye(3)
            s = 1.0
    return R, t, s

def find_best_re_s2t(curr_src, curr_trg, src_x, trg_x, cut_W, num_row):
    idxtrg = []
    idxsrc = []
    step = 10#15
    row_step = 5
    s = 1.0
    row_posi = int(num_row / row_step)
    t = torch.zeros(3).float()
    t[2] = (src_x[:, 2].min() - trg_x[:, 2].min()) # .numpy()
    R = torch.eye(3).float()
    for i in range(int(row_posi/2), row_posi):
        j = (i) * row_step
        idx_trg = np.where((curr_trg[:, 1] < cut_W+step)&(curr_trg[:, 1] > cut_W)&(curr_trg[:, 0] < j+row_step)&(curr_trg[:, 0] > j))
        idx_src = np.where((curr_src[:, 1] < cut_W+step)&(curr_src[:, 1] > cut_W)&(curr_src[:, 0] < j+row_step)&(curr_src[:, 0] > j))

        if (idx_trg[0].shape[0]>0)&(idx_src[0].shape[0]>0):
            # minnum = int(min(idx_trg[0].shape[0], idx_src[0].shape[0])/2)*2#保证取偶数min(idx_trg[0].shape[0], idx_src[0].shape[0])#
            idxtrg.append(idx_trg[0][0])
            idxsrc.append(idx_src[0][0])

    if len(idxtrg)>3:
        cloud_trg = trg_x[idxtrg,:3]#target domain as source object
        cloud_src = src_x[idxsrc,:3]
        # #计算比例点应该和计算旋转矩阵的点应该一致；
        # src_super_point1, src_super_point2 = cloud_src[:3, :2].float().mean(0), cloud_src[-1,:2].float()  # .mean(0)
        # trg_super_point1, trg_super_point2 = cloud_trg[:3, :2].float().mean(0), cloud_trg[-1,:2].float()  # .mean(0)
        # # src_super_point1, src_super_point2 = (src_super_point1 + (trg_super_point2 - src_super_point2)), trg_super_point2
        # #
        # src_super_point1 = src_super_point2 + 20*F.normalize((src_super_point1 - src_super_point2).view(1,-1).float())
        # trg_super_point1 = trg_super_point2 + 20*F.normalize((trg_super_point1 - trg_super_point2).view(1,-1).float())
        # src_super_point, trg_super_point =\
        #     torch.cat([src_super_point1.view(1, -1), src_super_point2.view(1, -1)], 0), \
        #      torch.cat([trg_super_point1.view(1, -1), trg_super_point2.view(1, -1)], 0)
        # R[:2,:2] = torch.matmul(trg_super_point.T, torch.inverse(src_super_point.T))
        t = ((cloud_trg.float() - (torch.matmul(R,cloud_src.float().T)).T).mean(0))#.numpy()
    return R, t, s
