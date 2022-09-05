import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from guided_filter_pytorch.guided_filter import GuidedFilter



class SplitShadingEncoder(nn.Module):
    """ Generating Albedo
    """

    def __init__(self):
        super(SplitShadingEncoder, self).__init__()
        self._block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.LeakyReLU(0.1),
            )
    def forward(self, input):
        s1 = self._block1(input)
        return s1  # s5.expand([n, 3, w, h])

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self._block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        # Layers: enc_conv(i), pool(i); i=2..5
        self._block2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        self._block3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        self._block4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        self._block5 = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))

    def forward(self, input):
        s1 = self._block1(input)
        s2 = self._block2(s1)
        s3 = self._block3(s2)
        s4 = self._block4(s3)
        s5 = self._block5(s4)

        return s1, s2, s3, s4, s5

class AlbedoDecoder(nn.Module):
    def __init__(self):
        super(AlbedoDecoder, self).__init__()
        self.a_block1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=1))

        self.a_block2 = nn.Sequential(
            nn.Conv2d(32*2, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1))

        self.a_block3 = nn.Sequential(
            nn.Conv2d(32*3, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1))

        self.a_block4 = nn.Sequential(
            nn.Conv2d(32*3, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1))

        self.a_block5 = nn.Sequential(
            nn.Conv2d(32*3, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1))

        self.a_block6 = nn.Sequential(
            nn.Conv2d(32*2 + 3, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1))

    def forward(self, x, input1, input2, input3, input4, input5): 
        #x:B*3*256*256    input1:B*32*128*128   input2:B*32*64*64  input3:B*32*32*32  input4:B*32*16*16  input5:B*32*8*8
        s1 = self.a_block1(input5)#32*16*16

        ac1 = torch.cat((s1, input4), dim=1)
        s2 = self.a_block2(ac1)#64*32*32

        ac2 = torch.cat((s2, input3), dim=1)
        s3 = self.a_block3(ac2)#64*64*64

        ac3 = torch.cat((s3, input2), dim=1)
        s4 = self.a_block4(ac3)#64*128*128

        ac4 = torch.cat((s4, input1), dim=1)
        s5 = self.a_block5(ac4)

        ac5 = torch.cat((s5, x), dim=1)
        s6 = self.a_block6(ac5)
        s7 = torch.sigmoid(s6)
        return s7

class NormalDecoder(nn.Module):
    def __init__(self):
        super(NormalDecoder, self).__init__()
        self.a_block1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=1))

        self.a_block2 = nn.Sequential(
            nn.Conv2d(32*2, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1))

        self.a_block3 = nn.Sequential(
            nn.Conv2d(32*3, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1))

        self.a_block4 = nn.Sequential(
            nn.Conv2d(32*3, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1))

        self.a_block5 = nn.Sequential(
            nn.Conv2d(32*3, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1))

        self.a_block6 = nn.Sequential(
            nn.Conv2d(32*2 + 3, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1))

    def forward(self, x, input1, input2, input3, input4, input5): 
        #x:B*3*256*256    input1:B*32*128*128   input2:B*32*64*64  input3:B*32*32*32  input4:B*32*16*16  input5:B*32*8*8
        s1 = self.a_block1(input5)#32*16*16

        ac1 = torch.cat((s1, input4), dim=1)
        s2 = self.a_block2(ac1)#64*32*32

        ac2 = torch.cat((s2, input3), dim=1)
        s3 = self.a_block3(ac2)#64*64*64

        ac3 = torch.cat((s3, input2), dim=1)
        s4 = self.a_block4(ac3)#64*128*128

        ac4 = torch.cat((s4, input1), dim=1)
        s5 = self.a_block5(ac4)

        ac5 = torch.cat((s5, x), dim=1)
        s6 = self.a_block6(ac5)
        s7 = torch.tanh(s6)
        return s7

class ShadingDecoder(nn.Module):
    def __init__(self):
        super(ShadingDecoder, self).__init__()
        self.a_block1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=1))

        self.a_block2 = nn.Sequential(
            nn.Conv2d(32*2, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1))

        self.a_block3 = nn.Sequential(
            nn.Conv2d(32*3, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1))

        self.a_block4 = nn.Sequential(
            nn.Conv2d(32*3, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1))

        self.a_block5 = nn.Sequential(
            nn.Conv2d(32*3, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1))

        self.a_block6 = nn.Sequential(
            nn.Conv2d(32*2 + 3, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1))

    def forward(self, x, input1, input2, input3, input4, input5): 
        #x:B*3*256*256    input1:B*32*128*128   input2:B*32*64*64  input3:B*32*32*32  input4:B*32*16*16  input5:B*32*8*8
        s1 = self.a_block1(input5)#32*16*16

        ac1 = torch.cat((s1, input4), dim=1)
        s2 = self.a_block2(ac1)#64*32*32

        ac2 = torch.cat((s2, input3), dim=1)
        s3 = self.a_block3(ac2)#64*64*64

        ac3 = torch.cat((s3, input2), dim=1)
        s4 = self.a_block4(ac3)#64*128*128

        ac4 = torch.cat((s4, input1), dim=1)
        s5 = self.a_block5(ac4)

        ac5 = torch.cat((s5, x), dim=1)
        s6 = self.a_block6(ac5)
        s7 = torch.sigmoid(s6)
        return s7

class LightEstimator(nn.Module):
    """ Estimate lighting from normal, albedo and conv features
    """

    def __init__(self):
        super(LightEstimator, self).__init__()
        self.conv1 = get_conv(32, 64, kernel_size=1, stride=1)
        self.pool1 = nn.AvgPool2d(4, stride=1, padding=0)
        self.conv2 = get_conv(64, 32, kernel_size=1, stride=2)
        self.pool2 = nn.AvgPool2d(3, stride=1, padding=0)
        self.fc = nn.Linear(32 * 1 * 1, 9)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.pool1(out1)
        # reshape to batch_size x 128
        out3 = self.conv2(out2)
        out4 = self.pool2(out3)
        out5 = out4.view(-1, 32 * 1 * 1)
        out5 = self.fc(out5)
        # out6 = torch.tanh(out5)
        return out5

class Unsuper_UNet_Split(nn.Module):
    """Custom U-Net architecture for Noise2Noise (see Appendix, Table 2)."""

    def __init__(self):
        """Initializes U-Net."""
        super(Unsuper_UNet_Split, self).__init__()

        self.FeatureEncoder =  Encoder()
        self.NormalEstimation= NormalDecoder()
        self.AlbedoEstimation = AlbedoDecoder()
        self.LightingEstimation = LightEstimator()
        self.ShadingEstimation = SplitShadingEncoder()

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initializes weights using He et al. (2015)."""
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x):
        """Through encoder, then decoder by adding U-skip connections. """
        # Encoder
        in_1, in_2, in_3, in_4, in_5 = self.FeatureEncoder(x)
        # albedo Decoder
        albedo = self.AlbedoEstimation(x, in_1, in_2, in_3, in_4, in_5)

        # normal Decoder
        normal = self.NormalEstimation(x, in_1, in_2, in_3, in_4, in_5)

        lighting = self.LightingEstimation(in_5)

        shading = self.ShadingEstimation(x)

        return albedo, shading, normal, lighting

    def fix_weights(self):
        dfs_freeze(self.FeatureEncoder)
        dfs_freeze(self.NormalEstimation)


class UnShadingGenerationNet(nn.Module):
    """ Generating Albedo
    """

    def __init__(self):
        super(UnShadingGenerationNet, self).__init__()
        # self.upsample = nn.UpsamplingBilinear2d(size=(128, 128), scale_factor=2)
        self.Sconv1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.Sconv2 = nn.Conv2d(64, 128, 3, 1, 1)
        self.Sconv3 = nn.Conv2d(128, 128, 3, 1, 1)
        self.Sconv4 = nn.Conv2d(128, 64, 3, 1, 1)
        self.Sconv5 = nn.Conv2d(64, 1, 3, 1, 1)

    def forward(self, input):
        s1 = self.Sconv1(input)
        s2 = self.Sconv2(F.relu(s1, 0.2))
        s3 = self.Sconv3(F.relu(s2, 0.2))
        s4 = self.Sconv4(F.relu(s3, 0.2))
        s5 = self.Sconv5(F.relu(s4, 0.2))

        # n, b, w, h = s5.shape
        return s5  # s5.expand([n, 3, w, h])



# Use following to fix weights of the model
# Ref - https://discuss.pytorch.org/t/how-the-pytorch-freeze-network-in-some-layers-only-the-rest-of-the-training/7088/15
def dfs_freeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False
        dfs_freeze(child)


class Gradient_Net(nn.Module):
    def __init__(self):
        super(Gradient_Net, self).__init__()
        kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
        kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0).cuda()

        kernel_y = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
        kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0).cuda()

        self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
        self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False)

    def forward(self, x):
        gray_input = (x[:, 0, :, :] + x[:, 1, :, :] + x[:, 2, :, :]).reshape([x.size(0), 1, x.size(2), x.size(2)]) / 3
        grad_x = F.conv2d(gray_input, self.weight_x).cuda()
        grad_y = F.conv2d(gray_input, self.weight_y).cuda()
        gradient = torch.abs(grad_x) + torch.abs(grad_y)
        return gradient


class Gradient_Net_Single(nn.Module):
    def __init__(self):
        super(Gradient_Net_Single, self).__init__()
        kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
        kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0).cuda()

        kernel_y = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
        kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0).cuda()

        self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
        self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False)

    def forward(self, x):
        gray_input = x
        grad_x = F.conv2d(gray_input, self.weight_x).cuda()
        grad_y = F.conv2d(gray_input, self.weight_y).cuda()
        gradient = torch.abs(grad_x) + torch.abs(grad_y)
        return gradient


##############################
#        Discriminator
##############################


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        channels, height, width = input_shape

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 3, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


# Base methods for creating convnet
def get_conv(in_channels, out_channels, kernel_size=3, padding=0, stride=1, dropout=0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                  padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout)
    )


class baseFeaturesExtractions(nn.Module):
    """ Base Feature extraction
    """

    def __init__(self):
        super(baseFeaturesExtractions, self).__init__()
        self.conv1 = get_conv(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = get_conv(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        return out3


def reconstruct_image(shading, albedo):
    return shading * albedo


def GetLDPR20211113(S, N):
    b, c, h, w = N.shape
    for i in range(b):
        N1 = N[i,:,:,:]
        N2 = torch.zeros_like(N1)
        N2[2,:,:] = N1[0,:,:]
        N2[1,:,:] = N1[1,:,:]
        N2[0,:,:] = N1[2,:,:]
        N3 = torch.zeros_like(N2)
        N3[0,:,:] = N2[0,:,:]
        N3[1,:,:] = N2[2,:,:]
        N3[2,:,:] = -1 * N2[1,:,:]
        N3=N3.permute([1,2,0]).reshape([-1,3])
        norm_X = N3[:,0]
        norm_Y = N3[:,1]
        norm_Z = N3[:,2]
        numElem = norm_X.shape[0]
        sh_basis = torch.from_numpy(np.zeros([numElem, 9])).type(torch.FloatTensor).cuda()
        att = torch.from_numpy(np.pi * np.array([1, 2.0 / 3.0, 1 / 4.0])).type(torch.FloatTensor).cuda()
        sh_basis[:, 0] = torch.from_numpy(np.array(0.5 / np.sqrt(np.pi), dtype=float)).type(torch.FloatTensor).cuda() * \
                         att[0]

        sh_basis[:, 1] = torch.from_numpy(np.array(np.sqrt(3) / 2 / np.sqrt(np.pi), dtype=float)).type(
            torch.FloatTensor).cuda() * norm_Y * att[1]
        sh_basis[:, 2] = torch.from_numpy(np.array(np.sqrt(3) / 2 / np.sqrt(np.pi), dtype=float)).type(
            torch.FloatTensor).cuda() * norm_Z * att[1]
        sh_basis[:, 3] = torch.from_numpy(np.array(np.sqrt(3) / 2 / np.sqrt(np.pi), dtype=float)).type(
            torch.FloatTensor).cuda() * norm_X * att[1]

        sh_basis[:, 4] = torch.from_numpy(np.array(np.sqrt(15) / 2 / np.sqrt(np.pi), dtype=float)).type(
            torch.FloatTensor).cuda() * norm_Y * norm_X * att[2]
        sh_basis[:, 5] = torch.from_numpy(np.array(np.sqrt(15) / 2 / np.sqrt(np.pi), dtype=float)).type(
            torch.FloatTensor).cuda() * norm_Y * norm_Z * att[2]
        sh_basis[:, 6] = torch.from_numpy(np.array(np.sqrt(5) / 4 / np.sqrt(np.pi), dtype=float)).type(
            torch.FloatTensor).cuda() * (3 * norm_Z ** 2 - 1) * att[2]
        sh_basis[:, 7] = torch.from_numpy(np.array(np.sqrt(15) / 2 / np.sqrt(np.pi), dtype=float)).type(
            torch.FloatTensor).cuda() * norm_X * norm_Z * att[2]
        sh_basis[:, 8] = torch.from_numpy(np.array(np.sqrt(15) / 4 / np.sqrt(np.pi), dtype=float)).type(
            torch.FloatTensor).cuda() * (norm_X ** 2 - norm_Y ** 2) * att[2]

        t_shading = S[i, 0, :, :].reshape([h * w, 1])
        # l_tp, _ = torch.lstsq(t_shading, sh_basis)
        # l_tp, _ = LeastSquares.lstq(t_shading, sh_basis, 0.01)

        l_sq = torch.pinverse(sh_basis.t().matmul(sh_basis)).matmul(sh_basis.t()).matmul(t_shading).reshape([-1, 9])
        # l_tp = l_tp[0:9].reshape([-1,9])
        # print('xxx')
        if i == 0:
            # result = l_tp
            result2 = l_sq
        else:
            # result = torch.cat([result, l_tp], axis=0)
            result2 = torch.cat([result2, l_sq], axis=0)

    return result2

def GetShading20211113(N, L):
    b, c, h, w = N.shape
    for i in range(b):
        N1 = N[i,:,:,:]
        N2 = torch.zeros_like(N1)
        N2[2,:,:] = N1[0,:,:]
        N2[1,:,:] = N1[1,:,:]
        N2[0,:,:] = N1[2,:,:]
        N3 = torch.zeros_like(N2)
        N3[0,:,:] = N2[0,:,:]
        N3[1,:,:] = N2[2,:,:]
        N3[2,:,:] = -1 * N2[1,:,:]
        N3=N3.permute([1,2,0]).reshape([-1,3])
        norm_X = N3[:,0]
        norm_Y = N3[:,1]
        norm_Z = N3[:,2]
        numElem = norm_X.shape[0]
        sh_basis = torch.from_numpy(np.zeros([numElem, 9])).type(torch.FloatTensor).cuda()
        att = torch.from_numpy(np.pi * np.array([1, 2.0 / 3.0, 1 / 4.0])).type(torch.FloatTensor).cuda()
        sh_basis[:, 0] = torch.from_numpy(np.array(0.5 / np.sqrt(np.pi), dtype=float)).type(torch.FloatTensor).cuda() * \
                         att[0]

        sh_basis[:, 1] = torch.from_numpy(np.array(np.sqrt(3) / 2 / np.sqrt(np.pi), dtype=float)).type(
            torch.FloatTensor).cuda() * norm_Y * att[1]
        sh_basis[:, 2] = torch.from_numpy(np.array(np.sqrt(3) / 2 / np.sqrt(np.pi), dtype=float)).type(
            torch.FloatTensor).cuda() * norm_Z * att[1]
        sh_basis[:, 3] = torch.from_numpy(np.array(np.sqrt(3) / 2 / np.sqrt(np.pi), dtype=float)).type(
            torch.FloatTensor).cuda() * norm_X * att[1]

        sh_basis[:, 4] = torch.from_numpy(np.array(np.sqrt(15) / 2 / np.sqrt(np.pi), dtype=float)).type(
            torch.FloatTensor).cuda() * norm_Y * norm_X * att[2]
        sh_basis[:, 5] = torch.from_numpy(np.array(np.sqrt(15) / 2 / np.sqrt(np.pi), dtype=float)).type(
            torch.FloatTensor).cuda() * norm_Y * norm_Z * att[2]
        sh_basis[:, 6] = torch.from_numpy(np.array(np.sqrt(5) / 4 / np.sqrt(np.pi), dtype=float)).type(
            torch.FloatTensor).cuda() * (3 * norm_Z ** 2 - 1) * att[2]
        sh_basis[:, 7] = torch.from_numpy(np.array(np.sqrt(15) / 2 / np.sqrt(np.pi), dtype=float)).type(
            torch.FloatTensor).cuda() * norm_X * norm_Z * att[2]
        sh_basis[:, 8] = torch.from_numpy(np.array(np.sqrt(15) / 4 / np.sqrt(np.pi), dtype=float)).type(
            torch.FloatTensor).cuda() * (norm_X ** 2 - norm_Y ** 2) * att[2]

        light = L[i, :]
        shading = torch.matmul(sh_basis, light)
        myshading = (shading - torch.min(shading)) / (torch.max(shading) - torch.min(shading))

        tp = myshading.reshape([-1, h, w])
        if i == 0:
            result = tp
        else:
            result = torch.cat([result, tp], axis=0)

    b, w, h = result.shape
    return result.reshape([b, 1, w, h])

def GetShadingSH(N, L):
    b, c, h, w = N.shape

    for i in range(b):
        N1 = N[i,:,:,:]
        N3=N1.permute([1,2,0]).reshape([-1,3])
        norm_X = N3[:,0]
        norm_Y = N3[:,1]
        norm_Z = N3[:,2]
        numElem = norm_X.shape[0]
        sh_basis = torch.from_numpy(np.zeros([numElem, 9])).type(torch.FloatTensor).cuda()
        att = torch.from_numpy(np.pi * np.array([1, 2.0 / 3.0, 1 / 4.0])).type(torch.FloatTensor).cuda()
        sh_basis[:, 0] = torch.from_numpy(np.array(0.5 / np.sqrt(np.pi), dtype=float)).type(torch.FloatTensor).cuda() * \
                         att[0]

        sh_basis[:, 1] = torch.from_numpy(np.array(np.sqrt(3) / 2 / np.sqrt(np.pi), dtype=float)).type(
            torch.FloatTensor).cuda() * norm_Y * att[1]
        sh_basis[:, 2] = torch.from_numpy(np.array(np.sqrt(3) / 2 / np.sqrt(np.pi), dtype=float)).type(
            torch.FloatTensor).cuda() * norm_Z * att[1]
        sh_basis[:, 3] = torch.from_numpy(np.array(np.sqrt(3) / 2 / np.sqrt(np.pi), dtype=float)).type(
            torch.FloatTensor).cuda() * norm_X * att[1]

        sh_basis[:, 4] = torch.from_numpy(np.array(np.sqrt(15) / 2 / np.sqrt(np.pi), dtype=float)).type(
            torch.FloatTensor).cuda() * norm_Y * norm_X * att[2]
        sh_basis[:, 5] = torch.from_numpy(np.array(np.sqrt(15) / 2 / np.sqrt(np.pi), dtype=float)).type(
            torch.FloatTensor).cuda() * norm_Y * norm_Z * att[2]
        sh_basis[:, 6] = torch.from_numpy(np.array(np.sqrt(5) / 4 / np.sqrt(np.pi), dtype=float)).type(
            torch.FloatTensor).cuda() * (3 * norm_Z ** 2 - 1) * att[2]
        sh_basis[:, 7] = torch.from_numpy(np.array(np.sqrt(15) / 2 / np.sqrt(np.pi), dtype=float)).type(
            torch.FloatTensor).cuda() * norm_X * norm_Z * att[2]
        sh_basis[:, 8] = torch.from_numpy(np.array(np.sqrt(15) / 4 / np.sqrt(np.pi), dtype=float)).type(
            torch.FloatTensor).cuda() * (norm_X ** 2 - norm_Y ** 2) * att[2]

        light = L[i, :]
        shading = torch.matmul(sh_basis, light)
        myshading = (shading - torch.min(shading)) / (torch.max(shading) - torch.min(shading))

        tp = myshading.reshape([-1, h, w])
        if i == 0:
            result = tp
        else:
            result = torch.cat([result, tp], axis=0)

    b, w, h = result.shape
    return result.reshape([b, 1, w, h])
