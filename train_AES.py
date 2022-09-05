import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from models_AES import *
from utils_tools import *
from data_loading import *
from torch.autograd import Variable
import time
from torch.utils.tensorboard import SummaryWriter
import os, tarfile
from guided_filter_pytorch.guided_filter import GuidedFilter
from torchvision.transforms import GaussianBlur
GaussianBlur_img = GaussianBlur(21,9).cuda()
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from torchvision.utils import save_image

def train(expnum, ASNL_G, syn_data, step_flag, celeba_data=None, read_first=None,
          batch_size = 10, num_epochs = 10, log_path = './results/metadata/', use_cuda=False,
          lr = 0.01, wt_decay=0.005):

    # data processing
    syn_train_csv = '/media/hdr/oo/Datasets/DPR_Celeba_Img1024_train.csv'
    syn_test_csv  = '/media/hdr/oo/Datasets/Img1024_val.csv'
    # Load Synthetic dataset
    train_dataset, _ = get_dataset_Img1024_Sin(syn_dir=syn_data, read_from_csv=syn_train_csv, read_first=read_first, validation_split=0)
    test_dataset, _ = get_dataset_Img1024_Sin(syn_dir=syn_data, read_from_csv=syn_test_csv, read_first=read_first, validation_split=0)


    syn_train_dl  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    syn_test_dl   = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    print('celeba dataset: Train data: ', len(syn_train_dl), ' Test data: ', len(syn_test_dl))

    timefloder = str(expnum) + '_'+ time.strftime("%m_%d", time.localtime())

    out_syn_images_dir = log_path + '/' + step_flag + '/'
    model_checkpoint_dir = out_syn_images_dir + timefloder + '/' + 'checkpoints/'

    os.system('mkdir -p {}'.format(model_checkpoint_dir))
    os.system('mkdir -p {}'.format(out_syn_images_dir + timefloder + '/'+ 'train/'))
    os.system('mkdir -p {}'.format(out_syn_images_dir + timefloder + '/'+ 'test/'))
    os.system('mkdir -p {}'.format(out_syn_images_dir + timefloder + '/'+ 'val/'))
    testpath = out_syn_images_dir + timefloder + '/'+ 'test/'

    tar_path = out_syn_images_dir + timefloder + '/' + timefloder + '_code.tar.gz'
    make_targz(tar_path, '/home/hdr/autohdr/RefinedCodes/Gradient_ComData/')
    writer_path = out_syn_images_dir + timefloder + '/'
    writer = SummaryWriter(writer_path)

    W_H = 256
    input_shape = (3, W_H, W_H)
    D_F = Discriminator(input_shape)
    optimizer_D_F = torch.optim.Adam(D_F.parameters(), lr=lr, weight_decay=wt_decay)
    cuda = torch.cuda.is_available()
    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
    D_F_model_dir = '/home/hdr/autohdr/RefinedCodes/Gradient_ComData/4__D_F.pkl'
    D_F.load_state_dict(torch.load(D_F_model_dir))
    # Collect model parameters
        # Collect model parameters
    model_parameters = ASNL_G.parameters()
    optimizer = torch.optim.Adam(model_parameters, lr=lr) #, weight_decay=wt_decay)

    smooth_loss = nn.L1Loss()
    recon_loss = nn.L1Loss()#nn.MSELoss()  #nn.L1Loss()
    s_nl_loss = nn.L1Loss()
    normal_loss = nn.MSELoss() #nn.L1Loss()
    criterion_GAN = torch.nn.BCEWithLogitsLoss()
    albedo_face_loss = nn.L1Loss()
    light_loss = nn.L1Loss()#nn.L1Loss()
    albedo_loss = nn.L1Loss()
    shading_loss = nn.L1Loss()
    CosSim = nn.CosineSimilarity(dim=1, eps=1e-6)
    CosEmb = nn.HingeEmbeddingLoss()#nn.CosineEmbeddingLoss()
    absolute_difference = nn.L1Loss()

    if use_cuda:
        recon_loss  = recon_loss.cuda()
        normal_loss = normal_loss.cuda()
        criterion_GAN = criterion_GAN.cuda()
        albedo_face_loss = albedo_face_loss.cuda()
        light_loss = light_loss.cuda()
        albedo_loss = albedo_loss.cuda()
        smooth_loss = smooth_loss.cuda()
        s_nl_loss = s_nl_loss.cuda()
        shading_loss = shading_loss.cuda()
        CosSim = CosSim.cuda()
        CosEmb = CosEmb.cuda()
        absolute_difference = absolute_difference.cuda()

    lambda_recon_1 = 0.5
    lambda_recon_2 = 0.5
    lambda_kind = 0.01
    lambda_s_nl = 0.01
    lambda_albedo = 0.001
    lambda_n = 0.5
    lambda_l = 0.1 
    lambda_shad_lum = 1
    lambda_grad =0.01
    lambda_GAN_F=0.001

    flag = 1

    lighting_vectors = torch.transpose(torch.load('/home/hdr/autohdr/RefinedCodes/Gradient_ComData/PCA/V.pt'),0,1).cuda()
    lighting_means = torch.load('/home/hdr/autohdr/RefinedCodes/Gradient_ComData/PCA/Mean.pt').cuda()
    lightings_var = torch.ones([6]).cuda()

    for epoch in range(1, num_epochs+1):
        for bix, data in enumerate(syn_train_dl):
            face_1, normal, mask, index = data

            if use_cuda:
                mask   = mask.cuda()
                face_1   = face_1.cuda()
                normal = normal.cuda()
                D_F= D_F.cuda()
                criterion_GAN.cuda()

            valid = Variable(Tensor(np.ones((normal.mul(mask).size(0), *D_F.output_shape))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((normal.mul(mask).size(0), *D_F.output_shape))), requires_grad=False)
            optimizer.zero_grad()

            face_1 = face_1 * mask
            normal = normal * mask

            pA_1, pS_1, pN_1, pL_1 = ASNL_G(face_1)
            b,c,w,h = pA_1.shape

            normal = normal*mask

            pN_1 = F.normalize(pN_1*mask)* mask
            # pN_1 = normal_normalization(pN_1*mask)* mask

            pS_1 = Image_Batch_normalization(pS_1)*mask

            pA_1 = Image_Batch_normalization(pA_1).clamp(0,0.85)*mask


            lightings_pca = torch.matmul((pL_1 - lighting_means.expand([b,9])), torch.pinverse(lighting_vectors))
            lightings = torch.matmul(lightings_pca, lighting_vectors) + lighting_means
            var = torch.mean(lightings_pca ** 2, axis=0)
            illu_prior_loss = absolute_difference(var, lightings_var)

            E_L = lambda_l * torch.log(illu_prior_loss + 1.)
            l_1 = GetLDPR20211113(pS_1, normal).detach() # use pre_normal 
            E_L_1 = lambda_l *(light_loss(lightings, l_1))


            rec_S_1 = GetShading20211113(pN_1, lightings) * mask
            rec_F_1 = pA_1 * pS_1 * mask  # p1 pS_1, p2 rec_S_1
            rec_F_2 = pA_1 * rec_S_1 * mask  # p1 pS_1, p2 rec_S_1

            Chrom1 = get_Chrom(face_1)
            Lumin1 = get_Luma(face_1)
            # neig_1 = Albedo_Regularizer(Chrom1, Lumin1, pA_1, mask)
            E_S_Lum = lambda_shad_lum * shading_loss(pS_1,Lumin1)

            
            # E_A = lambda_albedo * local_sparseness_reflectance(pA_1, Chrom1)

            # E_A = lambda_albedo * (albedo_loss(neig_1,torch.ones_like(neig_1)))

            E_R_1 = lambda_recon_1 * recon_loss(rec_F_1, face_1) #重建损失
            E_R_2 = lambda_recon_2 * recon_loss(rec_F_2, face_1)  #重建损失

            TP = torch.zeros_like(gradient(pS_1.expand([b, 3, w, h])))
            E_kind = lambda_kind*(smooth_loss(Smooth_kind(gradient(pS_1.expand([b, 3, w, h]))), TP))

            E_S_NL = lambda_s_nl * (s_nl_loss(pS_1, rec_S_1))

            E_gradient =  lambda_grad*smooth_loss(Smooth_kind(gradient(pN_1/2+0.5)), Smooth_kind(gradient(pA_1).detach()))
            # E_gradient =  lambda_grad*(1 - ms_ssim(pN_1, face_1, data_range=1, size_average=False).mean())

            
            E_N = lambda_n *( normal_loss(pN_1, normal) )


            optimizer_D_F.zero_grad()
            loss_real_1 = criterion_GAN(D_F(face_1.mul(mask).detach()), valid)
            loss_fake_3 = criterion_GAN(D_F(rec_F_2.mul(mask).detach()), fake)
            D_loss = loss_real_1 + loss_fake_3
            D_loss.backward()
            optimizer_D_F.step()

            TrainLoss = E_S_Lum + E_R_1 + E_R_2 + E_S_NL + E_N + E_L_1 + E_L + E_kind + E_gradient# p1
            # TrainLoss = E_R_2 + E_N + E_L + E_gradient# 

            D_vail_loss_F = lambda_GAN_F * (criterion_GAN(D_F(rec_F_2.mul(mask)), valid))
            total_loss = TrainLoss + D_vail_loss_F
            total_loss.backward()
            optimizer.step()

            # # print('Epoch: {} - bix: {} - E_R_1: {:.5f}, E_A: {:.5}, E_l:{:.5} , E_gradient: {:.5f}, E_S_NL: {:.5f}, EN: {:.5f},ES_lum:{:.5}, D_vail_loss_F:{:.5}'.
            #       format(epoch, bix, E_R_1.item(), E_A.item(), E_L.item(),  E_gradient.item(), E_S_NL.item(),  E_N.item(),E_S_Lum.item(), D_vail_loss_F.item()))
            print('E:{}-b:{}-E_R1: {:.5f}, E_R2: {:.5f}, E_N: {:.5}, E_S_Lum:{:.5}, E_S_NL:{:.5}, E_l:{:.5}, E_l_1:{:.5},  E_g: {:.5f}, E_k: {:.5f}, D_F:{:.5}'.
                  format(epoch, bix, E_R_1.item(), E_R_2.item(), E_N.item(), E_S_Lum.item(), E_S_NL.item(), E_L.item(), E_L_1.item(), E_gradient.item(), E_kind.item(), D_vail_loss_F.item()))
            writer.add_scalar('E_R_1', E_R_1.item(), flag)
            writer.add_scalar('E_R_2', E_R_2.item(), flag)
            writer.add_scalar('E_kind', E_kind.item(), flag)
            writer.add_scalar('E_S_NL', E_S_NL.item(), flag)
            writer.add_scalar('E_N', E_N.item(), flag)
            writer.add_scalar('E_L_1', E_L_1.item(), flag)
            writer.add_scalar('E_L', E_L.item(), flag)
            writer.add_scalar('E_SLum', E_S_Lum.item(), flag)
            # writer.add_scalar('D_N Loss', D_vail_loss_N.item(), flag)
            writer.add_scalar('D_f Loss', D_vail_loss_F.item(), flag)

            # test(syn_test_dl, ASNL_G, testpath, epoch, use_cuda)

            flag = flag + 1
                # Log images in wandb
            if bix%1234==0:# and bix > 1:
                saveImgs = torch.cat([face_1*mask, pS_1.expand(b,3,w,h)*mask, rec_F_1*mask, rec_F_2*mask, rec_S_1.expand(b,3,w,h)*mask, pA_1*mask, GetShadingSH(Sphere_DPR(pN_1, pL_1), lightings).expand(b,3,w,h), get_normal_in_range(normal_DPR2SFS(pN_1))*mask],dim=0)
                
                file_name = out_syn_images_dir + timefloder + '/train/' + str(epoch) + '_' + str(flag) + '.png'
                save_image(saveImgs, file_name, nrow=b, normalize=False)
                

        # test(syn_test_dl, ASNL_G, testpath, epoch, use_cuda)
        torch.save(ASNL_G.state_dict(), model_checkpoint_dir + str(epoch) + '_' + '_ASNL.pkl')
        torch.save(D_F.state_dict(), model_checkpoint_dir + str(epoch) + '_' + '_D_F.pkl')


def Smooth_kind(shading):
    [b,c,w,h] = shading.shape
    result = torch.zeros_like(shading)
    epsilon = 0.01*torch.ones(1).cuda()
    for i in range(b):
        tp = shading[i,:,:,:]
        if tp.max() > epsilon:
            result[i,:,:,:] = tp / tp.max()
        else:
            result[i,:,:,:] = tp / epsilon
    return result

def gradient(x):
    gradient_model = Gradient_Net()
    # gradient_model = Gradient_Net_Single()

    g = gradient_model(x)
    return g

def normal_normalization(normal_decoder1):
    b, c, w, h = normal_decoder1.shape
    sqrt_sum = torch.sqrt(normal_decoder1[:, 0, :, :] * normal_decoder1[:, 0, :, :] +
                          normal_decoder1[:, 1, :, :] * normal_decoder1[:,1, :, :] +
                          normal_decoder1[:, 2, :, :] * normal_decoder1[:, 2, :, :] + 0.0000001)

    normal_decoder1 = normal_decoder1 / sqrt_sum.reshape([b,1,w,h]).expand(b,c,w,h)
    return normal_decoder1

def Image_Batch_normalization(albedo):
    [b,c,w,h] = albedo.size()
    result = torch.zeros_like(albedo)
    for i in range(b):
        tp = albedo[i,:,:,:]
        dev = tp.max()- tp.min()
        result[i,:,:,:] = (tp - tp.min())/(dev+0.000001)

    return result

def Sphere_DPR(normal,lighting):
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
    normal_cuda = torch.from_numpy(normal_sp.astype(np.float32)).cuda().permute([2,0,1])#.reshape([1,256,256,3])
    normalBatch = torch.zeros_like(normal)
    for i in range(lighting.size()[0]):
        normalBatch[i,:,:,:] = normal_cuda
    # SpVisual = GetShading20211113_0802(normalBatch, lighting)
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

def test(syn_test_dl, ASNL_G, testpath, epoch, use_cuda):
    l0 = '/home/hdr/autohdr/code/Single_Decom_Img1024_AE/example_light/rotate_light_00.txt'
    l1 = '/home/hdr/autohdr/code/Single_Decom_Img1024_AE/example_light/rotate_light_01.txt'
    l2 = '/home/hdr/autohdr/code/Single_Decom_Img1024_AE/example_light/rotate_light_02.txt'
    l3 = '/home/hdr/autohdr/code/Single_Decom_Img1024_AE/example_light/rotate_light_03.txt'
    l4 = '/home/hdr/autohdr/code/Single_Decom_Img1024_AE/example_light/rotate_light_04.txt'
    l5 = '/home/hdr/autohdr/code/Single_Decom_Img1024_AE/example_light/rotate_light_05.txt'
    l6 = '/home/hdr/autohdr/code/Single_Decom_Img1024_AE/example_light/rotate_light_06.txt'
    pd_sh0 = pd.read_csv(l0, sep='\t', header=None, encoding=u'gbk')
    sh0 = torch.tensor(pd_sh0.values).type(torch.float).reshape([1, 9]).cuda().expand(3,9)
    pd_sh1 = pd.read_csv(l1, sep='\t', header=None, encoding=u'gbk')
    sh1 = torch.tensor(pd_sh1.values).type(torch.float).reshape([1, 9]).cuda().expand(3,9)
    pd_sh2 = pd.read_csv(l2, sep='\t', header=None, encoding=u'gbk')
    sh2 = torch.tensor(pd_sh2.values).type(torch.float).reshape([1, 9]).cuda().expand(3,9)
    pd_sh3 = pd.read_csv(l3, sep='\t', header=None, encoding=u'gbk')
    sh3 = torch.tensor(pd_sh3.values).type(torch.float).reshape([1, 9]).cuda().expand(3,9)
    pd_sh4 = pd.read_csv(l4, sep='\t', header=None, encoding=u'gbk')
    sh4 = torch.tensor(pd_sh4.values).type(torch.float).reshape([1, 9]).cuda().expand(3,9)
    pd_sh5 = pd.read_csv(l5, sep='\t', header=None, encoding=u'gbk')
    sh5 = torch.tensor(pd_sh5.values).type(torch.float).reshape([1, 9]).cuda().expand(3,9)
    pd_sh6 = pd.read_csv(l6, sep='\t', header=None, encoding=u'gbk')
    sh6 = torch.tensor(pd_sh6.values).type(torch.float).reshape([1, 9]).cuda().expand(3,9)
    for bix, data in enumerate(syn_test_dl):
        face_1, normal, mask, index = data

        if use_cuda:
            mask   = mask.cuda()
            face_1   = face_1.cuda()

            normal = normal.cuda()

        face_1 = face_1 * mask
        normal = normal * mask

        pA_1, pS_1, pN_1, pL_1 = ASNL_G(face_1)
        
        pN_1 = normal_normalization(pN_1*mask)

        pS_1 = Image_Batch_normalization(pS_1)*mask

        pA_1 = Image_Batch_normalization(pA_1)*mask

        pN_1 = pN_1 * mask

        rec_S_1 = GetShading20211113(pN_1, pL_1) * mask
        rec_sh_0 = pA_1*GetShading20211113(pN_1, sh0) * mask
        rec_sh_1 = pA_1*GetShading20211113(pN_1, sh1) * mask
        rec_sh_2 = pA_1*GetShading20211113(pN_1, sh2) * mask
        rec_sh_3 = pA_1*GetShading20211113(pN_1, sh3) * mask
        rec_sh_4 = pA_1*GetShading20211113(pN_1, sh4) * mask
        rec_sh_5 = pA_1*GetShading20211113(pN_1, sh5) * mask
        rec_sh_6 = pA_1*GetShading20211113(pN_1, sh6) * mask

        # rec_F_1 = pA_1 * pS_1 * mask

        file_name = testpath + str(epoch) + '_'
        wandb_log_images(face_1, mask, pathName=file_name + '_face.png')
        # wandb_log_images(rec_F_1, mask, pathName=file_name + '_ASface.png')
        wandb_log_images(rec_S_1*pA_1, mask, pathName=file_name + '_ANLface.png')
        wandb_log_images(pA_1, mask, pathName=file_name + '_albedo.png')
        wandb_log_images(pS_1, mask, pathName=file_name + '_shading.png')
        wandb_log_images(rec_S_1, mask, pathName=file_name + '_NLshading.png')
        wandb_log_images(get_normal_in_range(normal_DPR2SFS(pN_1)), mask, pathName=file_name + '_pNormal.png')
        wandb_log_images(get_normal_in_range(normal_DPR2SFS(normal)), mask, pathName=file_name + '_normal.png')
        Spnormal = Sphere_DPR(pN_1, pL_1)
        wandb_log_images(GetShading20211113(Spnormal, pL_1), None, pathName=file_name + '_light.png')

        wandb_log_images(rec_sh_0, mask, pathName=(file_name + '_0.png').replace('test','val'))
        wandb_log_images(rec_sh_1, mask, pathName=(file_name + '_1.png').replace('test','val'))
        wandb_log_images(rec_sh_2, mask, pathName=(file_name + '_2.png').replace('test','val'))
        wandb_log_images(rec_sh_3, mask, pathName=(file_name + '_3.png').replace('test','val'))
        wandb_log_images(rec_sh_4, mask, pathName=(file_name + '_4.png').replace('test','val'))
        wandb_log_images(rec_sh_5, mask, pathName=(file_name + '_5.png').replace('test','val'))
        wandb_log_images(rec_sh_6, mask, pathName=(file_name + '_6.png').replace('test','val'))



        break
def get_Chrom(face):
    #modify from  A closed-Form solution to Retinex with Nonlocal Texture Constraints
    size = face.shape
    new_size = [size[0], -1, size[2], size[3]]
    tp1 = face*face
    tp2 = torch.sum(tp1,1)
    tp3 = torch.reshape(tp2,new_size)
    tp4 = torch.max(tp3, 1)
    tp5 = tp4.values
    epsilon = (torch.ones(1)*1e-10).cuda()
    tp6 = torch.where(tp5 > epsilon, tp5, epsilon)
    tp7 = torch.reshape(torch.sqrt(tp6),new_size)
    intensity = torch.cat((tp7,tp7,tp7),1)
    #print('intensity...' + str(intensity))
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

def Albedo_Regularizer(Chrom, Lumin, pred_albedo, mask):
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

    # --------------------------------Compute  Chrom---------------------------------------------
    Chrom_r = Chrom[:, 0, :, :].reshape([size[0], 1, size[2], size[2]])
    Chrom_g = Chrom[:, 1, :, :].reshape([size[0], 1, size[2], size[2]])
    Chrom_b = Chrom[:, 2, :, :].reshape([size[0], 1, size[2], size[2]])
    dist_r_1 = neigbor_distance(Chrom_r, x_1)
    dist_r_2 = neigbor_distance(Chrom_r, x_2)
    dist_r_3 = neigbor_distance(Chrom_r, x_3)
    dist_r_4 = neigbor_distance(Chrom_r, x_4)
    dist_r_5 = neigbor_distance(Chrom_r, x_5)
    dist_r_6 = neigbor_distance(Chrom_r, x_6)
    dist_r_7 = neigbor_distance(Chrom_r, x_7)
    dist_r_8 = neigbor_distance(Chrom_r, x_8)

    dist_b_1 = neigbor_distance(Chrom_b, x_1)
    dist_b_2 = neigbor_distance(Chrom_b, x_2)
    dist_b_3 = neigbor_distance(Chrom_b, x_3)
    dist_b_4 = neigbor_distance(Chrom_b, x_4)
    dist_b_5 = neigbor_distance(Chrom_b, x_5)
    dist_b_6 = neigbor_distance(Chrom_b, x_6)
    dist_b_7 = neigbor_distance(Chrom_b, x_7)
    dist_b_8 = neigbor_distance(Chrom_b, x_8)

    dist_g_1 = neigbor_distance(Chrom_g, x_1)
    dist_g_2 = neigbor_distance(Chrom_g, x_2)
    dist_g_3 = neigbor_distance(Chrom_g, x_3)
    dist_g_4 = neigbor_distance(Chrom_g, x_4)
    dist_g_5 = neigbor_distance(Chrom_g, x_5)
    dist_g_6 = neigbor_distance(Chrom_g, x_6)
    dist_g_7 = neigbor_distance(Chrom_g, x_7)
    dist_g_8 = neigbor_distance(Chrom_g, x_8)

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
    # 1- (||ch(p)-ch(q)||/max(||ch(p)-ch(q)||)
    tp_neigbor_chrom = torch.ones_like(max_neigbor_chrom) - norm_chrom / max_neigbor_chrom

    # --------------------------------neigbor Lumin---------------------------------------------
    Lumin_multip = torch.log(Lumin.reshape([size[0], 1, size[2], size[2]]) + torch.ones(1).cuda() * 1e-10)

    dist_r_1 = neigbor_distance(Lumin_multip, l_1)
    dist_r_2 = neigbor_distance(Lumin_multip, l_2)
    dist_r_3 = neigbor_distance(Lumin_multip, l_3)
    dist_r_4 = neigbor_distance(Lumin_multip, l_4)
    dist_r_5 = neigbor_distance(Lumin_multip, l_5)
    dist_r_6 = neigbor_distance(Lumin_multip, l_6)
    dist_r_7 = neigbor_distance(Lumin_multip, l_7)
    dist_r_8 = neigbor_distance(Lumin_multip, l_8)

    neigbor_lumin = torch.cat((
        dist_r_1, dist_r_2, dist_r_3, dist_r_4, dist_r_5, dist_r_6, dist_r_7, dist_r_8),
        1)

    alpha = torch.exp(torch.log(tp_neigbor_chrom)  + 0.5 *  neigbor_lumin)
    albedo_neigbor_loss = alpha * neigbor_albedo * mask.expand([size[0], 8, size[2], size[2]])

    return albedo_neigbor_loss