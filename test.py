import torch
import torch.nn as nn
import numpy as np
# from GuidedFilter import *
from models import *
from utils_tools import *
from data_loading import *
from torch.autograd import Variable
import time
from torch.utils.tensorboard import SummaryWriter
import os, tarfile
from guided_filter_pytorch.guided_filter import GuidedFilter

def train(expnum, ASNL_G, syn_data, step_flag, celeba_data=None, read_first=None,
          batch_size = 10, num_epochs = 10, log_path = './results/metadata/', use_cuda=False,
          lr = 0.01, wt_decay=0.005):

    # data processing
    syn_train_csv = syn_data + '../Img1024_train.csv'
    syn_test_csv  = syn_data + '../Img1024_test.csv'
    # Load Synthetic dataset
    train_dataset, _ = get_dataset_Img1024_Sin(syn_dir=syn_data, read_from_csv=syn_train_csv, read_first=read_first, validation_split=0)
    test_dataset, _ = get_dataset_Img1024_Sin(syn_dir=syn_data, read_from_csv=syn_test_csv, read_first=read_first, validation_split=0)


    syn_train_dl  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    syn_test_dl   = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    print('celeba dataset: Train data: ', len(syn_train_dl), ' Test data: ', len(syn_test_dl))
    # os.system('mkdir -p {}'.format(model_checkpoint_dir))
    testpath = '/media/hdr/oo/result/SinDecom/Test/Img1024/'

    with torch.no_grad():
        test(syn_test_dl, ASNL_G, testpath, 0, use_cuda)

def test(syn_test_dl, ASNL_G, testpath, epoch, use_cuda):
    for bix, data in enumerate(syn_test_dl):
        face_1, normal, mask, index = data

        if use_cuda:
            mask = mask.cuda()
            face_1 = face_1.cuda()
            normal = normal.cuda()

        face_1 = face_1 * mask
        normal = normal * mask

        pA_1, pS_1, pN_1, pL_1 = ASNL_G(face_1)

        pS_1 = Image_Batch_normalization(pS_1 * mask) * mask

        pA_1 = Image_Batch_normalization(pA_1 * mask) * mask

        pN_1 = pN_1 * mask

        rec_S_1 = get_shading_DPR(pN_1, pL_1) * mask

        rec_F_1 = pA_1 * pS_1 * mask
        b,c,w,h = pS_1.shape
        wandb_log_images_Single(mask, None, index, testpath, 'mask')
        wandb_log_images_Single(face_1, None, index, testpath, 'face')
        wandb_log_images_Single(pA_1, mask, index, testpath, 'albedo')
        wandb_log_images_Single(pS_1.expand([b,3,w,h]), mask, index, testpath, 'shading')
        wandb_log_images_Single(rec_S_1.expand([b,3,w,h]), mask, index, testpath, 'shading_nl')
        wandb_log_images_Single(rec_F_1, mask, index, testpath, 'recon')
        wandb_log_images_Single(get_normal_in_range(normal_DPR2SFS(pN_1)), mask, index, testpath, 'normal')

        # break


def Smooth_kind(shading):
    [b, c, w, h] = shading.shape
    result = torch.zeros_like(shading)
    epsilon = 0.01 * torch.ones(1).cuda()
    for i in range(b):
        tp = shading[i, :, :, :]
        if tp.max() > epsilon:
            result[i, :, :, :] = tp / tp.max()
        else:
            result[i, :, :, :] = tp / epsilon
    return result


def gradient(x):
    gradient_model = Gradient_Net()
    # gradient_model = Gradient_Net_Single()

    g = gradient_model(x)
    return g


def normal_normalization(normal_decoder1):
    b, c, w, h = normal_decoder1.shape
    sqrt_sum = torch.sqrt(normal_decoder1[:, 0, :, :] * normal_decoder1[:, 0, :, :] +
                          normal_decoder1[:, 1, :, :] * normal_decoder1[:, 1, :, :] +
                          normal_decoder1[:, 2, :, :] * normal_decoder1[:, 2, :, :] + 0.0000001)

    normal_decoder1 = normal_decoder1 / sqrt_sum.reshape([b, 1, w, h]).expand(b, c, w, h)
    return normal_decoder1


def Image_Batch_normalization(albedo):
    [b, c, w, h] = albedo.size()
    result = torch.zeros_like(albedo)
    for i in range(b):
        tp = albedo[i, :, :, :]
        dev = tp.max() - tp.min()
        result[i, :, :, :] = (tp - tp.min()) / (dev + 0.000001)

    return result


def Sphere_DPR(normal, lighting):
    # ---------------- create normal for rendering half sphere ------
    img_size = 256
    x = np.linspace(-1, 1, img_size)
    z = np.linspace(1, -1, img_size)
    x, z = np.meshgrid(x, z)

    mag = np.sqrt(x ** 2 + z ** 2)
    valid = mag <= 1
    y = -np.sqrt(1 - (x * valid) ** 2 - (z * valid) ** 2)
    x = x * valid
    y = y * valid
    z = z * valid
    normal_sp = np.concatenate((x[..., None], y[..., None], z[..., None]), axis=2)
    normal_cuda = torch.from_numpy(normal_sp.astype(np.float32)).cuda().permute([2, 0, 1])  # .reshape([1,256,256,3])
    normalBatch = torch.zeros_like(normal)
    for i in range(lighting.size()[0]):
        normalBatch[i, :, :, :] = normal_cuda
    # SpVisual = get_shading_DPR_0802(normalBatch, lighting)
    return normalBatch


def normal_DPR2SFS(normal):
    tt = torch.zeros_like(normal)
    tt[:, 0, :, :] = normal[:, 1, :, :]
    tt[:, 1, :, :] = normal[:, 2, :, :]
    tt[:, 2, :, :] = -normal[:, 0, :, :]
    return tt


def normal_SFS2DPR(normal):
    tt = torch.zeros_like(normal)
    tt[:, 0, :, :] = -normal[:, 2, :, :]
    tt[:, 1, :, :] = normal[:, 0, :, :]
    tt[:, 2, :, :] = normal[:, 1, :, :]
    return tt

def get_Chrom(face):
    # modify from  A closed-Form solution to Retinex with Nonlocal Texture Constraints
    size = face.shape
    new_size = [size[0], -1, size[2], size[3]]
    tp1 = face * face
    tp2 = torch.sum(tp1, 1)
    tp3 = torch.reshape(tp2, new_size)
    tp4 = torch.max(tp3, 1)
    tp5 = tp4.values
    epsilon = (torch.ones(1) * 1e-10).cuda()
    tp6 = torch.where(tp5 > epsilon, tp5, epsilon)
    tp7 = torch.reshape(torch.sqrt(tp6), new_size)
    intensity = torch.cat((tp7, tp7, tp7), 1)
    # print('intensity...' + str(intensity))
    result = face / intensity
    return result


def get_Luma(face):
    r = face[:, 0, :, :]
    g = face[:, 1, :, :]
    b = face[:, 2, :, :]

    luminance = r * 0.2126 + g * 0.7152 + b * 0.0722
    return luminance


def neigbor_distance(image, neigbor_kernel):
    wconv_x = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1).cuda()
    fconv_x = neigbor_kernel
    wconv_x.weight = nn.Parameter(fconv_x.unsqueeze(0).unsqueeze(0))
    nbias = torch.tensor([0.0]).cuda()
    wconv_x.bias = nn.Parameter(nbias)
    wconv_x.weight.requires_grad = False

    dist = wconv_x(image)

    return dist


def lumin_distance(image, neigbor_kernel):
    wconv_x = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1).cuda()
    fconv_x = neigbor_kernel
    wconv_x.weight = nn.Parameter(fconv_x.unsqueeze(0).unsqueeze(0))
    nbias = torch.tensor([0.0]).cuda()
    wconv_x.bias = nn.Parameter(nbias)
    wconv_x.weight.requires_grad = False

    dist = wconv_x(image)

    return dist


# https://sonhua.github.io/pdf/shen-intrinsics-pami13.pdf
# local sparseness of reflectance
def local_sparseness_reflectance(albedo, Chrom):
    dist_chrom = Chrom_distance(albedo, Chrom)
    return dist_chrom


def Chrom_distance(PredAlbedo, Chrom):
    x_1 = torch.tensor([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]).cuda()
    x_2 = torch.tensor([[0.0, -1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]).cuda()
    x_3 = torch.tensor([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]).cuda()
    x_4 = torch.tensor([[0.0, 0.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]).cuda()

    x_5 = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, -1.0], [0.0, 0.0, 0.0]]).cuda()
    x_6 = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]).cuda()
    x_7 = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]).cuda()
    x_8 = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]).cuda()
    size = Chrom.shape
    # --------------------------------neigbor Chrom---------------------------------------------
    channel_r = Chrom[:, 0, :, :].reshape([size[0], 1, size[2], size[2]])
    channel_g = Chrom[:, 1, :, :].reshape([size[0], 1, size[2], size[2]])
    channel_b = Chrom[:, 2, :, :].reshape([size[0], 1, size[2], size[2]])
    dist_r_1 = neigbor_distance(channel_r, x_1)
    dist_r_2 = neigbor_distance(channel_r, x_2)
    dist_r_3 = neigbor_distance(channel_r, x_3)
    dist_r_4 = neigbor_distance(channel_r, x_4)
    dist_r_5 = neigbor_distance(channel_r, x_5)
    dist_r_6 = neigbor_distance(channel_r, x_6)
    dist_r_7 = neigbor_distance(channel_r, x_7)
    dist_r_8 = neigbor_distance(channel_r, x_8)

    dist_b_1 = neigbor_distance(channel_b, x_1)
    dist_b_2 = neigbor_distance(channel_b, x_2)
    dist_b_3 = neigbor_distance(channel_b, x_3)
    dist_b_4 = neigbor_distance(channel_b, x_4)
    dist_b_5 = neigbor_distance(channel_b, x_5)
    dist_b_6 = neigbor_distance(channel_b, x_6)
    dist_b_7 = neigbor_distance(channel_b, x_7)
    dist_b_8 = neigbor_distance(channel_b, x_8)

    dist_g_1 = neigbor_distance(channel_g, x_1)
    dist_g_2 = neigbor_distance(channel_g, x_2)
    dist_g_3 = neigbor_distance(channel_g, x_3)
    dist_g_4 = neigbor_distance(channel_g, x_4)
    dist_g_5 = neigbor_distance(channel_g, x_5)
    dist_g_6 = neigbor_distance(channel_g, x_6)
    dist_g_7 = neigbor_distance(channel_g, x_7)
    dist_g_8 = neigbor_distance(channel_g, x_8)

    neigbor_chrom_1 = torch.cat((dist_r_1, dist_g_1, dist_b_1), 1)
    neigbor_chrom_2 = torch.cat((dist_r_2, dist_g_2, dist_b_2), 1)
    neigbor_chrom_3 = torch.cat((dist_r_3, dist_g_3, dist_b_3), 1)
    neigbor_chrom_4 = torch.cat((dist_r_4, dist_g_4, dist_b_4), 1)
    neigbor_chrom_5 = torch.cat((dist_r_5, dist_g_5, dist_b_5), 1)
    neigbor_chrom_6 = torch.cat((dist_r_6, dist_g_6, dist_b_6), 1)
    neigbor_chrom_7 = torch.cat((dist_r_7, dist_g_7, dist_b_7), 1)
    neigbor_chrom_8 = torch.cat((dist_r_8, dist_g_8, dist_b_8), 1)

    norm_chrom_1 = torch.norm(neigbor_chrom_1, p=1, dim=1).reshape([size[0], 1, size[2], size[2]])
    norm_chrom_2 = torch.norm(neigbor_chrom_2, p=1, dim=1).reshape([size[0], 1, size[2], size[2]])
    norm_chrom_3 = torch.norm(neigbor_chrom_3, p=1, dim=1).reshape([size[0], 1, size[2], size[2]])
    norm_chrom_4 = torch.norm(neigbor_chrom_4, p=1, dim=1).reshape([size[0], 1, size[2], size[2]])
    norm_chrom_5 = torch.norm(neigbor_chrom_5, p=1, dim=1).reshape([size[0], 1, size[2], size[2]])
    norm_chrom_6 = torch.norm(neigbor_chrom_6, p=1, dim=1).reshape([size[0], 1, size[2], size[2]])
    norm_chrom_7 = torch.norm(neigbor_chrom_7, p=1, dim=1).reshape([size[0], 1, size[2], size[2]])
    norm_chrom_8 = torch.norm(neigbor_chrom_8, p=1, dim=1).reshape([size[0], 1, size[2], size[2]])

    average_r = (dist_r_1 + dist_r_2 + dist_r_3 + dist_r_4 + dist_r_5 + dist_r_6 + dist_r_7 + dist_r_8) / 3
    average_g = (dist_g_1 + dist_g_2 + dist_g_3 + dist_g_4 + dist_g_5 + dist_g_6 + dist_g_7 + dist_g_8) / 3
    average_b = (dist_b_1 + dist_b_2 + dist_b_3 + dist_b_4 + dist_b_5 + dist_b_6 + dist_b_7 + dist_b_8) / 3

    variance_r = (torch.pow((dist_r_1 - average_r), 2) + torch.pow((dist_r_2 - average_r), 2) + torch.pow(
        (dist_r_3 - average_r), 2) + torch.pow((dist_r_4 - average_r), 2) + torch.pow((dist_r_5 - average_r),
                                                                                      2) + torch.pow(
        (dist_r_6 - average_r), 2) + torch.pow((dist_r_7 - average_r), 2) + torch.pow((dist_r_8 - average_r), 2)) / 8

    variance_g = (torch.pow((dist_g_1 - average_g), 2) + torch.pow((dist_g_2 - average_g), 2) + torch.pow(
        (dist_g_3 - average_g), 2) + torch.pow((dist_g_4 - average_g), 2) + torch.pow((dist_g_5 - average_g),
                                                                                      2) + torch.pow(
        (dist_g_6 - average_g), 2) + torch.pow((dist_g_7 - average_g), 2) + torch.pow((dist_g_8 - average_g), 2)) / 8

    variance_b = (torch.pow((dist_b_1 - average_b), 2) + torch.pow((dist_b_2 - average_b), 2) + torch.pow(
        (dist_b_3 - average_b), 2) + torch.pow((dist_b_4 - average_b), 2) + torch.pow((dist_b_5 - average_b),
                                                                                      2) + torch.pow(
        (dist_b_6 - average_b), 2) + torch.pow((dist_b_7 - average_b), 2) + torch.pow((dist_b_8 - average_b), 2)) / 8

    average_variance = (variance_r + variance_g + variance_b) / 3

    # tc = torch.ones_like(dist_b_1)*0.02
    tc = Variable(torch.ones(1) * 0.02, requires_grad=True).cuda()
    tc_0 = Variable(torch.ones(1) * 0, requires_grad=True).cuda()

    wij_r_1 = torch.exp(-(dist_r_1 * dist_r_1) / (0.3 * variance_r * variance_r))
    wij_r_2 = torch.exp(-(dist_r_2 * dist_r_2) / (0.3 * variance_r * variance_r))
    wij_r_3 = torch.exp(-(dist_r_3 * dist_r_3) / (0.3 * variance_r * variance_r))
    wij_r_4 = torch.exp(-(dist_r_4 * dist_r_4) / (0.3 * variance_r * variance_r))
    wij_r_5 = torch.exp(-(dist_r_5 * dist_r_5) / (0.3 * variance_r * variance_r))
    wij_r_6 = torch.exp(-(dist_r_6 * dist_r_6) / (0.3 * variance_r * variance_r))
    wij_r_7 = torch.exp(-(dist_r_7 * dist_r_7) / (0.3 * variance_r * variance_r))
    wij_r_8 = torch.exp(-(dist_r_8 * dist_r_8) / (0.3 * variance_r * variance_r))

    wij_g_1 = torch.exp(-(dist_g_1 * dist_g_1) / (0.3 * variance_g * variance_g))
    wij_g_2 = torch.exp(-(dist_g_2 * dist_g_2) / (0.3 * variance_g * variance_g))
    wij_g_3 = torch.exp(-(dist_g_3 * dist_g_3) / (0.3 * variance_g * variance_g))
    wij_g_4 = torch.exp(-(dist_g_4 * dist_g_4) / (0.3 * variance_g * variance_g))
    wij_g_5 = torch.exp(-(dist_g_5 * dist_g_5) / (0.3 * variance_g * variance_g))
    wij_g_6 = torch.exp(-(dist_g_6 * dist_g_6) / (0.3 * variance_g * variance_g))
    wij_g_7 = torch.exp(-(dist_g_7 * dist_g_7) / (0.3 * variance_g * variance_g))
    wij_g_8 = torch.exp(-(dist_g_8 * dist_g_8) / (0.3 * variance_g * variance_g))

    wij_b_1 = torch.exp(-(dist_b_1 * dist_b_1) / (0.3 * variance_b * variance_b))
    wij_b_2 = torch.exp(-(dist_b_2 * dist_b_2) / (0.3 * variance_b * variance_b))
    wij_b_3 = torch.exp(-(dist_b_3 * dist_b_3) / (0.3 * variance_b * variance_b))
    wij_b_4 = torch.exp(-(dist_b_4 * dist_b_4) / (0.3 * variance_b * variance_b))
    wij_b_5 = torch.exp(-(dist_b_5 * dist_b_5) / (0.3 * variance_b * variance_b))
    wij_b_6 = torch.exp(-(dist_b_6 * dist_b_6) / (0.3 * variance_b * variance_b))
    wij_b_7 = torch.exp(-(dist_b_7 * dist_b_7) / (0.3 * variance_b * variance_b))
    wij_b_8 = torch.exp(-(dist_b_8 * dist_b_8) / (0.3 * variance_b * variance_b))

    w_ij_r_1 = torch.where(torch.abs(dist_r_1) > tc, tc_0, wij_r_1)
    w_ij_r_2 = torch.where(torch.abs(dist_r_2) > tc, tc_0, wij_r_2)
    w_ij_r_3 = torch.where(torch.abs(dist_r_3) > tc, tc_0, wij_r_3)
    w_ij_r_4 = torch.where(torch.abs(dist_r_4) > tc, tc_0, wij_r_4)
    w_ij_r_5 = torch.where(torch.abs(dist_r_5) > tc, tc_0, wij_r_5)
    w_ij_r_6 = torch.where(torch.abs(dist_r_6) > tc, tc_0, wij_r_6)
    w_ij_r_7 = torch.where(torch.abs(dist_r_7) > tc, tc_0, wij_r_7)
    w_ij_r_8 = torch.where(torch.abs(dist_r_8) > tc, tc_0, wij_r_8)

    w_ij_g_1 = torch.where(torch.abs(dist_g_1) > tc, tc_0, wij_g_1)
    w_ij_g_2 = torch.where(torch.abs(dist_g_2) > tc, tc_0, wij_g_2)
    w_ij_g_3 = torch.where(torch.abs(dist_g_3) > tc, tc_0, wij_g_3)
    w_ij_g_4 = torch.where(torch.abs(dist_g_4) > tc, tc_0, wij_g_4)
    w_ij_g_5 = torch.where(torch.abs(dist_g_5) > tc, tc_0, wij_g_5)
    w_ij_g_6 = torch.where(torch.abs(dist_g_6) > tc, tc_0, wij_g_6)
    w_ij_g_7 = torch.where(torch.abs(dist_g_7) > tc, tc_0, wij_g_7)
    w_ij_g_8 = torch.where(torch.abs(dist_g_8) > tc, tc_0, wij_g_8)

    w_ij_b_1 = torch.where(torch.abs(dist_b_1) > tc, tc_0, wij_b_1)
    w_ij_b_2 = torch.where(torch.abs(dist_b_2) > tc, tc_0, wij_b_2)
    w_ij_b_3 = torch.where(torch.abs(dist_b_3) > tc, tc_0, wij_b_3)
    w_ij_b_4 = torch.where(torch.abs(dist_b_4) > tc, tc_0, wij_b_4)
    w_ij_b_5 = torch.where(torch.abs(dist_b_5) > tc, tc_0, wij_b_5)
    w_ij_b_6 = torch.where(torch.abs(dist_b_6) > tc, tc_0, wij_b_6)
    w_ij_b_7 = torch.where(torch.abs(dist_b_7) > tc, tc_0, wij_b_7)
    w_ij_b_8 = torch.where(torch.abs(dist_b_8) > tc, tc_0, wij_b_8)

    PredAlbedo_r = PredAlbedo[:, 0, :, :].reshape([size[0], 1, size[2], size[2]])
    PredAlbedo_g = PredAlbedo[:, 1, :, :].reshape([size[0], 1, size[2], size[2]])
    PredAlbedo_b = PredAlbedo[:, 2, :, :].reshape([size[0], 1, size[2], size[2]])

    a_neigh_loss_r = torch.abs(neigbor_distance(PredAlbedo_r, w_ij_r_1 * PredAlbedo_r)) + torch.abs(
        neigbor_distance(PredAlbedo_r, w_ij_r_2 * PredAlbedo_r)) + torch.abs(
        neigbor_distance(PredAlbedo_r, w_ij_r_3 * PredAlbedo_r)) + torch.abs(
        neigbor_distance(PredAlbedo_r, w_ij_r_4 * PredAlbedo_r)) + torch.abs(
        neigbor_distance(PredAlbedo_r, w_ij_r_5 * PredAlbedo_r)) + torch.abs(
        neigbor_distance(PredAlbedo_r, w_ij_r_6 * PredAlbedo_r)) + torch.abs(
        neigbor_distance(PredAlbedo_r, w_ij_r_7 * PredAlbedo_r)) + torch.abs(
        neigbor_distance(PredAlbedo_r, w_ij_r_8 * PredAlbedo_r))
    a_neigh_loss_g = torch.abs(neigbor_distance(PredAlbedo_g, w_ij_g_1 * PredAlbedo_g)) + torch.abs(
        neigbor_distance(PredAlbedo_g, w_ij_g_2 * PredAlbedo_g)) + torch.abs(
        neigbor_distance(PredAlbedo_g, w_ij_g_3 * PredAlbedo_g)) + torch.abs(
        neigbor_distance(PredAlbedo_g, w_ij_g_4 * PredAlbedo_g)) + torch.abs(
        neigbor_distance(PredAlbedo_g, w_ij_g_5 * PredAlbedo_g)) + torch.abs(
        neigbor_distance(PredAlbedo_g, w_ij_g_6 * PredAlbedo_g)) + torch.abs(
        neigbor_distance(PredAlbedo_g, w_ij_g_7 * PredAlbedo_g)) + torch.abs(
        neigbor_distance(PredAlbedo_g, w_ij_g_8 * PredAlbedo_g))
    a_neigh_loss_b = torch.abs(neigbor_distance(PredAlbedo_b, w_ij_b_1 * PredAlbedo_b)) + torch.abs(
        neigbor_distance(PredAlbedo_b, w_ij_b_2 * PredAlbedo_b)) + torch.abs(
        neigbor_distance(PredAlbedo_b, w_ij_b_3 * PredAlbedo_b)) + torch.abs(
        neigbor_distance(PredAlbedo_b, w_ij_b_4 * PredAlbedo_b)) + torch.abs(
        neigbor_distance(PredAlbedo_b, w_ij_b_5 * PredAlbedo_b)) + torch.abs(
        neigbor_distance(PredAlbedo_b, w_ij_b_6 * PredAlbedo_b)) + torch.abs(
        neigbor_distance(PredAlbedo_b, w_ij_b_7 * PredAlbedo_b)) + torch.abs(
        neigbor_distance(PredAlbedo_b, w_ij_b_8 * PredAlbedo_b))

    sum_J = a_neigh_loss_r + a_neigh_loss_g + a_neigh_loss_b
    return sum_J


def neigbor_loss(Chrom, Lumin, pred_albedo, mask):
    size = Chrom.shape

    x_1 = torch.tensor([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]).cuda()
    x_2 = torch.tensor([[0.0, -1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]).cuda()
    x_3 = torch.tensor([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]).cuda()
    x_4 = torch.tensor([[0.0, 0.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]).cuda()

    x_5 = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, -1.0], [0.0, 0.0, 0.0]]).cuda()
    x_6 = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]).cuda()
    x_7 = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]).cuda()
    x_8 = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]).cuda()

    l_1 = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]).cuda()
    l_2 = torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]).cuda()
    l_3 = torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]).cuda()
    l_4 = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]).cuda()

    l_5 = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 0.0]]).cuda()
    l_6 = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]).cuda()
    l_7 = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]).cuda()
    l_8 = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]).cuda()

    # --------------------------------neigbor albedo---------------------------------------------
    pred_albedo_r = pred_albedo[:, 0, :, :].reshape([size[0], 1, size[2], size[2]])
    pred_albedo_g = pred_albedo[:, 1, :, :].reshape([size[0], 1, size[2], size[2]])
    pred_albedo_b = pred_albedo[:, 2, :, :].reshape([size[0], 1, size[2], size[2]])

    dist_r_1 = neigbor_distance(pred_albedo_r, x_1)
    dist_r_2 = neigbor_distance(pred_albedo_r, x_2)
    dist_r_4 = neigbor_distance(pred_albedo_r, x_4)
    dist_r_5 = neigbor_distance(pred_albedo_r, x_5)
    dist_r_6 = neigbor_distance(pred_albedo_r, x_6)
    dist_r_7 = neigbor_distance(pred_albedo_r, x_7)
    dist_r_8 = neigbor_distance(pred_albedo_r, x_8)

    dist_b_1 = neigbor_distance(pred_albedo_b, x_1)
    dist_b_2 = neigbor_distance(pred_albedo_b, x_2)
    dist_b_3 = neigbor_distance(pred_albedo_b, x_3)
    dist_b_4 = neigbor_distance(pred_albedo_b, x_4)
    size = Chrom.shape

    x_1 = torch.tensor([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]).cuda()
    x_2 = torch.tensor([[0.0, -1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]).cuda()
    x_3 = torch.tensor([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]).cuda()
    x_4 = torch.tensor([[0.0, 0.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]).cuda()

    x_5 = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, -1.0], [0.0, 0.0, 0.0]]).cuda()
    x_6 = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]).cuda()
    x_7 = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]).cuda()
    x_8 = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]).cuda()

    l_1 = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]).cuda()
    l_2 = torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]).cuda()
    l_3 = torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]).cuda()
    l_4 = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]).cuda()

    l_5 = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 0.0]]).cuda()
    l_6 = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]).cuda()
    l_7 = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]).cuda()
    l_8 = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]).cuda()

    # --------------------------------neigbor albedo---------------------------------------------
    pred_albedo_r = pred_albedo[:, 0, :, :].reshape([size[0], 1, size[2], size[2]])
    pred_albedo_g = pred_albedo[:, 1, :, :].reshape([size[0], 1, size[2], size[2]])
    pred_albedo_b = pred_albedo[:, 2, :, :].reshape([size[0], 1, size[2], size[2]])

    dist_r_1 = neigbor_distance(pred_albedo_r, x_1)
    dist_r_2 = neigbor_distance(pred_albedo_r, x_2)
    dist_r_3 = neigbor_distance(pred_albedo_r, x_3)
    dist_r_4 = neigbor_distance(pred_albedo_r, x_4)
    dist_r_5 = neigbor_distance(pred_albedo_r, x_5)
    dist_r_6 = neigbor_distance(pred_albedo_r, x_6)
    dist_r_7 = neigbor_distance(pred_albedo_r, x_7)
    dist_r_8 = neigbor_distance(pred_albedo_r, x_8)

    dist_b_1 = neigbor_distance(pred_albedo_b, x_1)
    dist_b_2 = neigbor_distance(pred_albedo_b, x_2)
    dist_b_3 = neigbor_distance(pred_albedo_b, x_3)
    dist_b_4 = neigbor_distance(pred_albedo_b, x_4)
    dist_b_5 = neigbor_distance(pred_albedo_b, x_5)
    dist_b_6 = neigbor_distance(pred_albedo_b, x_6)
    dist_b_7 = neigbor_distance(pred_albedo_b, x_7)
    dist_b_8 = neigbor_distance(pred_albedo_b, x_8)

    dist_g_1 = neigbor_distance(pred_albedo_g, x_1)
    dist_g_2 = neigbor_distance(pred_albedo_g, x_2)
    dist_g_3 = neigbor_distance(pred_albedo_g, x_3)
    dist_g_4 = neigbor_distance(pred_albedo_g, x_4)
    dist_g_5 = neigbor_distance(pred_albedo_g, x_5)
    dist_g_6 = neigbor_distance(pred_albedo_g, x_6)
    dist_g_7 = neigbor_distance(pred_albedo_g, x_7)
    dist_g_8 = neigbor_distance(pred_albedo_g, x_8)

    neigbor_albedo_1 = torch.norm(torch.cat((dist_r_1, dist_g_1, dist_b_1), 1), p=2, dim=1).reshape(
        [size[0], 1, size[2], size[2]])
    neigbor_albedo_2 = torch.norm(torch.cat((dist_r_2, dist_g_2, dist_b_2), 1), p=2, dim=1).reshape(
        [size[0], 1, size[2], size[2]])
    neigbor_albedo_3 = torch.norm(torch.cat((dist_r_3, dist_g_3, dist_b_3), 1), p=2, dim=1).reshape(
        [size[0], 1, size[2], size[2]])
    neigbor_albedo_4 = torch.norm(torch.cat((dist_r_4, dist_g_4, dist_b_4), 1), p=2, dim=1).reshape(
        [size[0], 1, size[2], size[2]])
    neigbor_albedo_5 = torch.norm(torch.cat((dist_r_5, dist_g_5, dist_b_5), 1), p=2, dim=1).reshape(
        [size[0], 1, size[2], size[2]])
    neigbor_albedo_6 = torch.norm(torch.cat((dist_r_6, dist_g_6, dist_b_6), 1), p=2, dim=1).reshape(
        [size[0], 1, size[2], size[2]])
    neigbor_albedo_7 = torch.norm(torch.cat((dist_r_7, dist_g_7, dist_b_7), 1), p=2, dim=1).reshape(
        [size[0], 1, size[2], size[2]])
    neigbor_albedo_8 = torch.norm(torch.cat((dist_r_8, dist_g_8, dist_b_8), 1), p=2, dim=1).reshape(
        [size[0], 1, size[2], size[2]])

    neigbor_albedo = torch.cat((
        neigbor_albedo_1, neigbor_albedo_2, neigbor_albedo_3, neigbor_albedo_4, neigbor_albedo_5, neigbor_albedo_6,
        neigbor_albedo_7, neigbor_albedo_8),
        1)

    # --------------------------------neigbor Lumin---------------------------------------------
    Lumin_multip = torch.log2(Lumin.reshape([size[0], 1, size[2], size[2]]) + torch.ones(1).cuda() * 1e-10)

    # dist_r_1 = neigbor_distance(Lumin_multip, l_1)
    # dist_r_2 = neigbor_distance(Lumin_multip, l_2)
    # dist_r_3 = neigbor_distance(Lumin_multip, l_3)
    # dist_r_4 = neigbor_distance(Lumin_multip, l_4)
    # dist_r_5 = neigbor_distance(Lumin_multip, l_5)
    # dist_r_6 = neigbor_distance(Lumin_multip, l_6)
    # dist_r_7 = neigbor_distance(Lumin_multip, l_7)
    # dist_r_8 = neigbor_distance(Lumin_multip, l_8)
    dist_r_1 = torch.sqrt(torch.pow(2, neigbor_distance(Lumin_multip, l_1)))
    dist_r_2 = torch.sqrt(torch.pow(2, neigbor_distance(Lumin_multip, l_2)))
    dist_r_3 = torch.sqrt(torch.pow(2, neigbor_distance(Lumin_multip, l_3)))
    dist_r_4 = torch.sqrt(torch.pow(2, neigbor_distance(Lumin_multip, l_4)))
    dist_r_5 = torch.sqrt(torch.pow(2, neigbor_distance(Lumin_multip, l_5)))
    dist_r_6 = torch.sqrt(torch.pow(2, neigbor_distance(Lumin_multip, l_6)))
    dist_r_7 = torch.sqrt(torch.pow(2, neigbor_distance(Lumin_multip, l_7)))
    dist_r_8 = torch.sqrt(torch.pow(2, neigbor_distance(Lumin_multip, l_8)))

    neigbor_lumin = torch.cat((
        dist_r_1, dist_r_2, dist_r_3, dist_r_4, dist_r_5, dist_r_6, dist_r_7, dist_r_8),
        1)

    # --------------------------------neigbor Chrom---------------------------------------------
    pred_albedo_r = Chrom[:, 0, :, :].reshape([size[0], 1, size[2], size[2]])
    pred_albedo_g = Chrom[:, 1, :, :].reshape([size[0], 1, size[2], size[2]])
    pred_albedo_b = Chrom[:, 2, :, :].reshape([size[0], 1, size[2], size[2]])
    dist_r_1 = neigbor_distance(pred_albedo_r, x_1)
    dist_r_2 = neigbor_distance(pred_albedo_r, x_2)
    dist_r_3 = neigbor_distance(pred_albedo_r, x_3)
    dist_r_4 = neigbor_distance(pred_albedo_r, x_4)
    dist_r_5 = neigbor_distance(pred_albedo_r, x_5)
    dist_r_6 = neigbor_distance(pred_albedo_r, x_6)
    dist_r_7 = neigbor_distance(pred_albedo_r, x_7)
    dist_r_8 = neigbor_distance(pred_albedo_r, x_8)

    dist_b_1 = neigbor_distance(pred_albedo_b, x_1)
    dist_b_2 = neigbor_distance(pred_albedo_b, x_2)
    dist_b_3 = neigbor_distance(pred_albedo_b, x_3)
    dist_b_4 = neigbor_distance(pred_albedo_b, x_4)
    dist_b_5 = neigbor_distance(pred_albedo_b, x_5)
    dist_b_6 = neigbor_distance(pred_albedo_b, x_6)
    dist_b_7 = neigbor_distance(pred_albedo_b, x_7)
    dist_b_8 = neigbor_distance(pred_albedo_b, x_8)

    dist_g_1 = neigbor_distance(pred_albedo_g, x_1)
    dist_g_2 = neigbor_distance(pred_albedo_g, x_2)
    dist_g_3 = neigbor_distance(pred_albedo_g, x_3)
    dist_g_4 = neigbor_distance(pred_albedo_g, x_4)
    dist_g_5 = neigbor_distance(pred_albedo_g, x_5)
    dist_g_6 = neigbor_distance(pred_albedo_g, x_6)
    dist_g_7 = neigbor_distance(pred_albedo_g, x_7)
    dist_g_8 = neigbor_distance(pred_albedo_g, x_8)

    neigbor_chrom_1 = torch.cat((dist_r_1, dist_g_1, dist_b_1), 1)
    neigbor_chrom_2 = torch.cat((dist_r_2, dist_g_2, dist_b_2), 1)
    neigbor_chrom_3 = torch.cat((dist_r_3, dist_g_3, dist_b_3), 1)
    neigbor_chrom_4 = torch.cat((dist_r_4, dist_g_4, dist_b_4), 1)
    neigbor_chrom_5 = torch.cat((dist_r_5, dist_g_5, dist_b_5), 1)
    neigbor_chrom_6 = torch.cat((dist_r_6, dist_g_6, dist_b_6), 1)
    neigbor_chrom_7 = torch.cat((dist_r_7, dist_g_7, dist_b_7), 1)
    neigbor_chrom_8 = torch.cat((dist_r_8, dist_g_8, dist_b_8), 1)

    norm_chrom_1 = torch.norm(neigbor_chrom_1, p=1, dim=1).reshape([size[0], 1, size[2], size[2]])
    norm_chrom_2 = torch.norm(neigbor_chrom_2, p=1, dim=1).reshape([size[0], 1, size[2], size[2]])
    norm_chrom_3 = torch.norm(neigbor_chrom_3, p=1, dim=1).reshape([size[0], 1, size[2], size[2]])
    norm_chrom_4 = torch.norm(neigbor_chrom_4, p=1, dim=1).reshape([size[0], 1, size[2], size[2]])
    norm_chrom_5 = torch.norm(neigbor_chrom_5, p=1, dim=1).reshape([size[0], 1, size[2], size[2]])
    norm_chrom_6 = torch.norm(neigbor_chrom_6, p=1, dim=1).reshape([size[0], 1, size[2], size[2]])
    norm_chrom_7 = torch.norm(neigbor_chrom_7, p=1, dim=1).reshape([size[0], 1, size[2], size[2]])
    norm_chrom_8 = torch.norm(neigbor_chrom_8, p=1, dim=1).reshape([size[0], 1, size[2], size[2]])

    norm_chrom = torch.cat((
        norm_chrom_1, norm_chrom_2, norm_chrom_3, norm_chrom_4, norm_chrom_5, norm_chrom_6, norm_chrom_7, norm_chrom_8),
        1)
    max_neigbor_chrom = torch.max(norm_chrom, 1)[0]
    max_neigbor_chrom = torch.where(torch.eq(max_neigbor_chrom, 0), torch.zeros_like(max_neigbor_chrom).cuda() + 1e-10,
                                    max_neigbor_chrom).reshape([size[0], 1, size[2], size[2]]).expand(
        [size[0], 8, size[2], size[2]]
    )

    tp_neigbor_chrom = torch.ones_like(max_neigbor_chrom) - norm_chrom / max_neigbor_chrom

    # tp_neigbor_chrom_1 = tp_neigbor_chrom[:, 0, :, :].reshape([size[0], 1, size[2], size[2]]).expand([size[0], 3, size[2], size[2]])
    # tp_neigbor_chrom_2 = tp_neigbor_chrom[:, 1, :, :].reshape([size[0], 1, size[2], size[2]]).expand([size[0], 3, size[2], size[2]])
    # tp_neigbor_chrom_3 = tp_neigbor_chrom[:, 2, :, :].reshape([size[0], 1, size[2], size[2]]).expand([size[0], 3, size[2], size[2]])
    # tp_neigbor_chrom_4 = tp_neigbor_chrom[:, 3, :, :].reshape([size[0], 1, size[2], size[2]]).expand([size[0], 3, size[2], size[2]])
    # tp_neigbor_chrom_5 = tp_neigbor_chrom[:, 4, :, :].reshape([size[0], 1, size[2], size[2]]).expand([size[0], 3, size[2], size[2]])
    # tp_neigbor_chrom_6 = tp_neigbor_chrom[:, 5, :, :].reshape([size[0], 1, size[2], size[2]]).expand([size[0], 3, size[2], size[2]])
    # tp_neigbor_chrom_7 = tp_neigbor_chrom[:, 6, :, :].reshape([size[0], 1, size[2], size[2]]).expand([size[0], 3, size[2], size[2]])
    # tp_neigbor_chrom_8 = tp_neigbor_chrom[:, 7, :, :].reshape([size[0], 1, size[2], size[2]]).expand([size[0], 3, size[2], size[2]])
    #

    albedo_neigbor_loss = neigbor_albedo * neigbor_lumin * tp_neigbor_chrom * mask.expand(
        [size[0], 8, size[2], size[2]])

    return albedo_neigbor_loss
    dist_b_5 = neigbor_distance(pred_albedo_b, x_5)
    dist_b_6 = neigbor_distance(pred_albedo_b, x_6)
    dist_b_7 = neigbor_distance(pred_albedo_b, x_7)
    dist_b_8 = neigbor_distance(pred_albedo_b, x_8)

    dist_g_1 = neigbor_distance(pred_albedo_g, x_1)
    dist_g_2 = neigbor_distance(pred_albedo_g, x_2)
    dist_g_3 = neigbor_distance(pred_albedo_g, x_3)
    dist_g_4 = neigbor_distance(pred_albedo_g, x_4)
    dist_g_5 = neigbor_distance(pred_albedo_g, x_5)
    dist_g_6 = neigbor_distance(pred_albedo_g, x_6)
    dist_g_7 = neigbor_distance(pred_albedo_g, x_7)
    dist_g_8 = neigbor_distance(pred_albedo_g, x_8)

    neigbor_albedo_1 = torch.norm(torch.cat((dist_r_1, dist_g_1, dist_b_1), 1), p=2, dim=1).reshape(
        [size[0], 1, size[2], size[2]])
    neigbor_albedo_2 = torch.norm(torch.cat((dist_r_2, dist_g_2, dist_b_2), 1), p=2, dim=1).reshape(
        [size[0], 1, size[2], size[2]])
    neigbor_albedo_3 = torch.norm(torch.cat((dist_r_3, dist_g_3, dist_b_3), 1), p=2, dim=1).reshape(
        [size[0], 1, size[2], size[2]])
    neigbor_albedo_4 = torch.norm(torch.cat((dist_r_4, dist_g_4, dist_b_4), 1), p=2, dim=1).reshape(
        [size[0], 1, size[2], size[2]])
    neigbor_albedo_5 = torch.norm(torch.cat((dist_r_5, dist_g_5, dist_b_5), 1), p=2, dim=1).reshape(
        [size[0], 1, size[2], size[2]])
    neigbor_albedo_6 = torch.norm(torch.cat((dist_r_6, dist_g_6, dist_b_6), 1), p=2, dim=1).reshape(
        [size[0], 1, size[2], size[2]])
    neigbor_albedo_7 = torch.norm(torch.cat((dist_r_7, dist_g_7, dist_b_7), 1), p=2, dim=1).reshape(
        [size[0], 1, size[2], size[2]])
    neigbor_albedo_8 = torch.norm(torch.cat((dist_r_8, dist_g_8, dist_b_8), 1), p=2, dim=1).reshape(
        [size[0], 1, size[2], size[2]])

    neigbor_albedo = torch.cat((
        neigbor_albedo_1, neigbor_albedo_2, neigbor_albedo_3, neigbor_albedo_4, neigbor_albedo_5, neigbor_albedo_6,
        neigbor_albedo_7, neigbor_albedo_8),
        1)

    # --------------------------------neigbor Lumin---------------------------------------------
    Lumin_multip = torch.log2(Lumin.reshape([size[0], 1, size[2], size[2]]) + torch.ones(1).cuda() * 1e-10)

    # dist_r_1 = neigbor_distance(Lumin_multip, l_1)
    # dist_r_2 = neigbor_distance(Lumin_multip, l_2)
    # dist_r_3 = neigbor_distance(Lumin_multip, l_3)
    # dist_r_4 = neigbor_distance(Lumin_multip, l_4)
    # dist_r_5 = neigbor_distance(Lumin_multip, l_5)
    # dist_r_6 = neigbor_distance(Lumin_multip, l_6)
    # dist_r_7 = neigbor_distance(Lumin_multip, l_7)
    # dist_r_8 = neigbor_distance(Lumin_multip, l_8)
    dist_r_1 = torch.sqrt(torch.pow(2, neigbor_distance(Lumin_multip, l_1)))
    dist_r_2 = torch.sqrt(torch.pow(2, neigbor_distance(Lumin_multip, l_2)))
    dist_r_3 = torch.sqrt(torch.pow(2, neigbor_distance(Lumin_multip, l_3)))
    dist_r_4 = torch.sqrt(torch.pow(2, neigbor_distance(Lumin_multip, l_4)))
    dist_r_5 = torch.sqrt(torch.pow(2, neigbor_distance(Lumin_multip, l_5)))
    dist_r_6 = torch.sqrt(torch.pow(2, neigbor_distance(Lumin_multip, l_6)))
    dist_r_7 = torch.sqrt(torch.pow(2, neigbor_distance(Lumin_multip, l_7)))
    dist_r_8 = torch.sqrt(torch.pow(2, neigbor_distance(Lumin_multip, l_8)))

    neigbor_lumin = torch.cat((
        dist_r_1, dist_r_2, dist_r_3, dist_r_4, dist_r_5, dist_r_6, dist_r_7, dist_r_8),
        1)

    # --------------------------------neigbor Chrom---------------------------------------------
    pred_albedo_r = Chrom[:, 0, :, :].reshape([size[0], 1, size[2], size[2]])
    pred_albedo_g = Chrom[:, 1, :, :].reshape([size[0], 1, size[2], size[2]])
    pred_albedo_b = Chrom[:, 2, :, :].reshape([size[0], 1, size[2], size[2]])
    dist_r_1 = neigbor_distance(pred_albedo_r, x_1)
    dist_r_2 = neigbor_distance(pred_albedo_r, x_2)
    dist_r_3 = neigbor_distance(pred_albedo_r, x_3)
    dist_r_4 = neigbor_distance(pred_albedo_r, x_4)
    dist_r_5 = neigbor_distance(pred_albedo_r, x_5)
    dist_r_6 = neigbor_distance(pred_albedo_r, x_6)
    dist_r_7 = neigbor_distance(pred_albedo_r, x_7)
    dist_r_8 = neigbor_distance(pred_albedo_r, x_8)

    dist_b_1 = neigbor_distance(pred_albedo_b, x_1)
    dist_b_2 = neigbor_distance(pred_albedo_b, x_2)
    dist_b_3 = neigbor_distance(pred_albedo_b, x_3)
    dist_b_4 = neigbor_distance(pred_albedo_b, x_4)
    dist_b_5 = neigbor_distance(pred_albedo_b, x_5)
    dist_b_6 = neigbor_distance(pred_albedo_b, x_6)
    dist_b_7 = neigbor_distance(pred_albedo_b, x_7)
    dist_b_8 = neigbor_distance(pred_albedo_b, x_8)

    dist_g_1 = neigbor_distance(pred_albedo_g, x_1)
    dist_g_2 = neigbor_distance(pred_albedo_g, x_2)
    dist_g_3 = neigbor_distance(pred_albedo_g, x_3)
    dist_g_4 = neigbor_distance(pred_albedo_g, x_4)
    dist_g_5 = neigbor_distance(pred_albedo_g, x_5)
    dist_g_6 = neigbor_distance(pred_albedo_g, x_6)
    dist_g_7 = neigbor_distance(pred_albedo_g, x_7)
    dist_g_8 = neigbor_distance(pred_albedo_g, x_8)

    neigbor_chrom_1 = torch.cat((dist_r_1, dist_g_1, dist_b_1), 1)
    neigbor_chrom_2 = torch.cat((dist_r_2, dist_g_2, dist_b_2), 1)
    neigbor_chrom_3 = torch.cat((dist_r_3, dist_g_3, dist_b_3), 1)
    neigbor_chrom_4 = torch.cat((dist_r_4, dist_g_4, dist_b_4), 1)
    neigbor_chrom_5 = torch.cat((dist_r_5, dist_g_5, dist_b_5), 1)
    neigbor_chrom_6 = torch.cat((dist_r_6, dist_g_6, dist_b_6), 1)
    neigbor_chrom_7 = torch.cat((dist_r_7, dist_g_7, dist_b_7), 1)
    neigbor_chrom_8 = torch.cat((dist_r_8, dist_g_8, dist_b_8), 1)

    norm_chrom_1 = torch.norm(neigbor_chrom_1, p=1, dim=1).reshape([size[0], 1, size[2], size[2]])
    norm_chrom_2 = torch.norm(neigbor_chrom_2, p=1, dim=1).reshape([size[0], 1, size[2], size[2]])
    norm_chrom_3 = torch.norm(neigbor_chrom_3, p=1, dim=1).reshape([size[0], 1, size[2], size[2]])
    norm_chrom_4 = torch.norm(neigbor_chrom_4, p=1, dim=1).reshape([size[0], 1, size[2], size[2]])
    norm_chrom_5 = torch.norm(neigbor_chrom_5, p=1, dim=1).reshape([size[0], 1, size[2], size[2]])
    norm_chrom_6 = torch.norm(neigbor_chrom_6, p=1, dim=1).reshape([size[0], 1, size[2], size[2]])
    norm_chrom_7 = torch.norm(neigbor_chrom_7, p=1, dim=1).reshape([size[0], 1, size[2], size[2]])
    norm_chrom_8 = torch.norm(neigbor_chrom_8, p=1, dim=1).reshape([size[0], 1, size[2], size[2]])

    norm_chrom = torch.cat((
        norm_chrom_1, norm_chrom_2, norm_chrom_3, norm_chrom_4, norm_chrom_5, norm_chrom_6, norm_chrom_7, norm_chrom_8),
        1)
    max_neigbor_chrom = torch.max(norm_chrom, 1)[0]
    max_neigbor_chrom = torch.where(torch.eq(max_neigbor_chrom, 0), torch.zeros_like(max_neigbor_chrom).cuda() + 1e-10,
                                    max_neigbor_chrom).reshape([size[0], 1, size[2], size[2]]).expand(
        [size[0], 8, size[2], size[2]]
    )

    tp_neigbor_chrom = torch.ones_like(max_neigbor_chrom) - norm_chrom / max_neigbor_chrom

    # tp_neigbor_chrom_1 = tp_neigbor_chrom[:, 0, :, :].reshape([size[0], 1, size[2], size[2]]).expand([size[0], 3, size[2], size[2]])
    # tp_neigbor_chrom_2 = tp_neigbor_chrom[:, 1, :, :].reshape([size[0], 1, size[2], size[2]]).expand([size[0], 3, size[2], size[2]])
    # tp_neigbor_chrom_3 = tp_neigbor_chrom[:, 2, :, :].reshape([size[0], 1, size[2], size[2]]).expand([size[0], 3, size[2], size[2]])
    # tp_neigbor_chrom_4 = tp_neigbor_chrom[:, 3, :, :].reshape([size[0], 1, size[2], size[2]]).expand([size[0], 3, size[2], size[2]])
    # tp_neigbor_chrom_5 = tp_neigbor_chrom[:, 4, :, :].reshape([size[0], 1, size[2], size[2]]).expand([size[0], 3, size[2], size[2]])
    # tp_neigbor_chrom_6 = tp_neigbor_chrom[:, 5, :, :].reshape([size[0], 1, size[2], size[2]]).expand([size[0], 3, size[2], size[2]])
    # tp_neigbor_chrom_7 = tp_neigbor_chrom[:, 6, :, :].reshape([size[0], 1, size[2], size[2]]).expand([size[0], 3, size[2], size[2]])
    # tp_neigbor_chrom_8 = tp_neigbor_chrom[:, 7, :, :].reshape([size[0], 1, size[2], size[2]]).expand([size[0], 3, size[2], size[2]])
    #

    albedo_neigbor_loss = neigbor_albedo * neigbor_lumin * tp_neigbor_chrom * mask.expand(
        [size[0], 8, size[2], size[2]])

    return albedo_neigbor_loss