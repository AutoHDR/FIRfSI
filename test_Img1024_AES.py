import torch
import argparse
from data_loading import *
from utils_tools import *
from shading import *
from test_AES import *
from models_AES import *

GPU_ID = 1
torch.cuda.set_device('cuda:1')
print('\n\nGPU %d is working for training....\n\n'%(GPU_ID))
expNum = 'Img1024_V_DRP'
# with GAN  no guided filter
def main():

    parser = argparse.ArgumentParser(description='SfSNet - Shading Residual')
    parser.add_argument('--batch_size', type=int, default=12, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to pre_train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--wt_decay', type=float, default=0.0005, metavar='W',
                        help='SGD momentum (default: 0.0005)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--read_first', type=int, default=-1,
                        help='read first n rows (default: -1)')
    parser.add_argument('--details', type=str, default=None,
                        help='Explaination of the run')

    parser.add_argument('--syn_data', type=str, default='/media/hdr/oo/Datasets/Img_1024/imgs/',
                    help='Synthetic Dataset path')
    parser.add_argument('--log_dir', type=str, default='/media/hdr/oo/result/SinDecom/',
                    help='Log Path')


    parser.add_argument('--load_model', type=str, default=None,
                        help='load model from')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    seed_num = 1000
    torch.manual_seed(seed_num)

    # initialization
    syn_data = args.syn_data
    celeba_data = ''
    batch_size = args.batch_size
    lr         = args.lr
    wt_decay   = args.wt_decay
    log_dir    = args.log_dir
    epochs     = args.epochs
    read_first = args.read_first

    ASNL_G      = Unsuper_AENetS()
    if use_cuda:
        ASNL_G = ASNL_G.cuda()
        torch.cuda.manual_seed(seed_num)

    step_flag = 'Test' # 1  apply pre_normal for refining
    model_dir = None   # 1
    model_dir = '/media/hdr/oo/result/SinDecom/AE_S/Img1024_V_Syn_08_29/checkpoints/44__ASNL.pkl'
    if model_dir is not None:
        ASNL_G.load_state_dict(torch.load(model_dir))
        print('************************************************')
        print('************************************************')
        print('********          loading model            *****')
        print('************************************************')
        print('************************************************\n\n\n')
    else:
        print('\n 2020 02 27\n')
        ASNL_G.apply(weights_init)
        print('************************************************')
        print('************************************************')
        print('********           init model              *****')
        print('************************************************')
        print('************************************************\n\n\n')


    train(expNum, ASNL_G,  syn_data, step_flag, celeba_data=celeba_data, read_first=read_first,\
           batch_size=batch_size, num_epochs=epochs, log_path=log_dir, use_cuda=use_cuda, \
           lr=lr, wt_decay=wt_decay)

if __name__ == '__main__':
    main()
