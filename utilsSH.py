def Sphere_DPR_xz_1(normal,lighting):
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
    # SpVisual = get_shading_DPR_0802(normalBatch, lighting)
    return normalBatch
def Sphere_DPR_xz_2(normal,lighting):
    # ---------------- create normal for rendering half sphere ------
    img_size = 256
    x = np.linspace(1, -1, img_size)
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
    # SpVisual = get_shading_DPR_0802(normalBatch, lighting)
    return normalBatch

def Sphere_DPR_xz_3(normal,lighting):
    # ---------------- create normal for rendering half sphere ------
    img_size = 256
    x = np.linspace(-1, 1, img_size)
    z = np.linspace(-1, 1, img_size)
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
    # SpVisual = get_shading_DPR_0802(normalBatch, lighting)
    return normalBatch


def Sphere_DPR_xy_1(normal,lighting):
    # ---------------- create normal for rendering half sphere ------
    img_size = 256
    nx = np.linspace(-1, 1, img_size)
    ny = np.linspace(1, -1, img_size)
    nx, ny = np.meshgrid(nx, ny)

    mag = np.sqrt(ny ** 2 + nx ** 2)
    valid = mag <= 1
    nz = -np.sqrt(1 - (ny * valid) ** 2 - (nx * valid) ** 2)
    nx = nx * valid
    ny = ny * valid
    nz = nz * valid
    normal_sp = np.concatenate((nx[..., None], ny[..., None], nz[..., None]), axis=2)
    normal_cuda = torch.from_numpy(normal_sp.astype(np.float32)).cuda().permute([2,0,1])#.reshape([1,256,256,3])
    normalBatch = torch.zeros_like(normal)
    for i in range(lighting.size()[0]):
        normalBatch[i,:,:,:] = normal_cuda
    # SpVisual = get_shading_DPR_0802(normalBatch, lighting)
    return normalBatch
def Sphere_DPR_xy_2(normal,lighting):
    # ---------------- create normal for rendering half sphere ------
    img_size = 256
    nx = np.linspace(1, -1, img_size)
    ny = np.linspace(1, -1, img_size)
    nx, ny = np.meshgrid(nx, ny)

    mag = np.sqrt(ny ** 2 + nx ** 2)
    valid = mag <= 1
    nz = -np.sqrt(1 - (ny * valid) ** 2 - (nx * valid) ** 2)
    nx = nx * valid
    ny = ny * valid
    nz = nz * valid
    normal_sp = np.concatenate((nx[..., None], ny[..., None], nz[..., None]), axis=2)
    normal_cuda = torch.from_numpy(normal_sp.astype(np.float32)).cuda().permute([2,0,1])#.reshape([1,256,256,3])
    normalBatch = torch.zeros_like(normal)
    for i in range(lighting.size()[0]):
        normalBatch[i,:,:,:] = normal_cuda
    # SpVisual = get_shading_DPR_0802(normalBatch, lighting)
    return normalBatch
def Sphere_DPR_xy_3(normal,lighting):
    # ---------------- create normal for rendering half sphere ------
    img_size = 256
    nx = np.linspace(-1, 1, img_size)
    ny = np.linspace(-1, 1, img_size)
    nx, ny = np.meshgrid(nx, ny)

    mag = np.sqrt(ny ** 2 + nx ** 2)
    valid = mag <= 1
    nz = -np.sqrt(1 - (ny * valid) ** 2 - (nx * valid) ** 2)
    nx = nx * valid
    ny = ny * valid
    nz = nz * valid
    normal_sp = np.concatenate((nx[..., None], ny[..., None], nz[..., None]), axis=2)
    normal_cuda = torch.from_numpy(normal_sp.astype(np.float32)).cuda().permute([2,0,1])#.reshape([1,256,256,3])
    normalBatch = torch.zeros_like(normal)
    for i in range(lighting.size()[0]):
        normalBatch[i,:,:,:] = normal_cuda
    # SpVisual = get_shading_DPR_0802(normalBatch, lighting)
    return normalBatch



def Sphere_DPR_yz_1(normal,lighting):
    # ---------------- create normal for rendering half sphere ------
    img_size = 256
    ny = np.linspace(-1, 1, img_size)
    nz = np.linspace(1, -1, img_size)
    ny, nz = np.meshgrid(ny, nz)

    mag = np.sqrt(nz ** 2 + ny ** 2)
    valid = mag <= 1
    nx = -np.sqrt(1 - (nz * valid) ** 2 - (ny * valid) ** 2)
    nx = nx * valid
    nz = nz * valid
    nz = nz * valid
    normal_sp = np.concatenate((ny[..., None], nz[..., None], nz[..., None]), axis=2)
    normal_cuda = torch.from_numpy(normal_sp.astype(np.float32)).cuda().permute([2,0,1])#.reshape([1,256,256,3])
    normalBatch = torch.zeros_like(normal)
    for i in range(lighting.size()[0]):
        normalBatch[i,:,:,:] = normal_cuda
    # SpVisual = get_shading_DPR_0802(normalBatch, lighting)
    return normalBatch


def Sphere_DPR_yz_2(normal,lighting):
    # ---------------- create normal for rendering half sphere ------
    img_size = 256
    ny = np.linspace(1, -1, img_size)
    nz = np.linspace(1, -1, img_size)
    ny, nz = np.meshgrid(ny, nz)

    mag = np.sqrt(nz ** 2 + ny ** 2)
    valid = mag <= 1
    nx = -np.sqrt(1 - (nz * valid) ** 2 - (ny * valid) ** 2)
    nx = nx * valid
    nz = nz * valid
    nz = nz * valid
    normal_sp = np.concatenate((ny[..., None], nz[..., None], nz[..., None]), axis=2)
    normal_cuda = torch.from_numpy(normal_sp.astype(np.float32)).cuda().permute([2,0,1])#.reshape([1,256,256,3])
    normalBatch = torch.zeros_like(normal)
    for i in range(lighting.size()[0]):
        normalBatch[i,:,:,:] = normal_cuda
    # SpVisual = get_shading_DPR_0802(normalBatch, lighting)
    return normalBatch

def Sphere_DPR_yz_3(normal,lighting):
    # ---------------- create normal for rendering half sphere ------
    img_size = 256
    ny = np.linspace(-1, 1, img_size)
    nz = np.linspace(-1, 1, img_size)
    ny, nz = np.meshgrid(ny, nz)

    mag = np.sqrt(nz ** 2 + ny ** 2)
    valid = mag <= 1
    nx = -np.sqrt(1 - (nz * valid) ** 2 - (ny * valid) ** 2)
    nx = nx * valid
    nz = nz * valid
    nz = nz * valid
    normal_sp = np.concatenate((ny[..., None], nz[..., None], nz[..., None]), axis=2)
    normal_cuda = torch.from_numpy(normal_sp.astype(np.float32)).cuda().permute([2,0,1])#.reshape([1,256,256,3])
    normalBatch = torch.zeros_like(normal)
    for i in range(lighting.size()[0]):
        normalBatch[i,:,:,:] = normal_cuda
    # SpVisual = get_shading_DPR_0802(normalBatch, lighting)
    return normalBatch





wandb_log_images(get_shading_DPR(Sphere_DPR_xy_1(pN_1pN_1, pL_1),sh), None, pathName=file_name + '_s0p-xy_____o_1.png')
wandb_log_images(get_shading_DPR(Sphere_DPR_xy_2(pN_1, pL_1),sh), None, pathName=file_name + '_s0p-xy_____o_2.png')
wandb_log_images(get_shading_DPR(Sphere_DPR_xy_3(pN_1, pL_1),sh), None, pathName=file_name + '_s0p-xy_____o_3.png')

wandb_log_images(get_shading_DPR(Sphere_DPR_xz_1(pN_1, pL_1),sh), None, pathName=file_name + '_s0p-xz_____o_1.png')
wandb_log_images(get_shading_DPR(Sphere_DPR_xz_2(pN_1, pL_1),sh), None, pathName=file_name + '_s0p-xz_____o_2.png')
wandb_log_images(get_shading_DPR(Sphere_DPR_xz_3(pN_1, pL_1),sh), None, pathName=file_name + '_s0p-xz_____o_3.png')

wandb_log_images(get_shading_DPR(Sphere_DPR_yz_1(pN_1, pL_1),sh), None, pathName=file_name + '_s0p-yz_____o_1.png')
wandb_log_images(get_shading_DPR(Sphere_DPR_yz_2(pN_1, pL_1),sh), None, pathName=file_name + '_s0p-yz_____o_2.png')
wandb_log_images(get_shading_DPR(Sphere_DPR_yz_3(pN_1, pL_1),sh), None, pathName=file_name + '_s0p-yz_____o_3.png')

SHCV = utilsSH.sh_cvt()
pd_sh_1 = pd.read_csv('example_light/rotate_light_00.txt', sep='\t', header=None)
sh_1 = torch.tensor(pd_sh_1.values).type(torch.float).reshape([-1,9]).expand([8, 9]).cuda()
sh_1 = torch.tensor(SHCV.sfs2shtools(sh_1)).type(torch.float).reshape([-1,9]).expand([8, 9]).cuda()
wandb_log_images(get_shading_DPR( Sphere_DPR_yz_2(pN_1, pL_1), sh_1), None, pathName=file_name + '_gt_l_1.png')

pd_sh_2 = pd.read_csv('example_light/rotate_light_01.txt', sep='\t', header=None)
sh_2 = torch.tensor(pd_sh_2.values).type(torch.float).reshape([-1,9]).expand([8, 9]).cuda()
sh_2 = torch.tensor(SHCV.sfs2shtools(sh_2)).type(torch.float).reshape([-1,9]).expand([8, 9]).cuda()
wandb_log_images(get_shading_DPR( Sphere_DPR_yz_2(pN_1, pL_1), sh_2), None, pathName=file_name + '_gt_l_2.png')


pd_sh_3 = pd.read_csv('example_light/rotate_light_02.txt', sep='\t', header=None)
sh_3 = torch.tensor(pd_sh_3.values).type(torch.float).reshape([-1,9]).expand([8, 9]).cuda()
sh_3 = torch.tensor(SHCV.sfs2shtools(sh_3)).type(torch.float).reshape([-1,9]).expand([8, 9]).cuda()
wandb_log_images(get_shading_DPR( Sphere_DPR_yz_2(pN_1, pL_1), sh_3), None, pathName=file_name + '_gt_l_3.png')


pd_sh_4 = pd.read_csv('example_light/rotate_light_03.txt', sep='\t', header=None)
sh_4 = torch.tensor(pd_sh_4.values).type(torch.float).reshape([-1,9]).expand([8, 9]).cuda()
sh_4 = torch.tensor(SHCV.sfs2shtools(sh_4)).type(torch.float).reshape([-1,9]).expand([8, 9]).cuda()
wandb_log_images(get_shading_DPR( Sphere_DPR_yz_2(pN_1, pL_1), sh_4), None, pathName=file_name + '_gt_l_4.png')


pd_sh_5 = pd.read_csv('example_light/rotate_light_04.txt', sep='\t', header=None)
sh_5 = torch.tensor(pd_sh_5.values).type(torch.float).reshape([-1,9]).expand([8, 9]).cuda()
sh_5 = torch.tensor(SHCV.sfs2shtools(sh_5)).type(torch.float).reshape([-1,9]).expand([8, 9]).cuda()
wandb_log_images(get_shading_DPR( Sphere_DPR_yz_2(pN_1, pL_1), sh_5), None, pathName=file_name + '_gt_l_5.png')


pd_sh_6 = pd.read_csv('example_light/rotate_light_05.txt', sep='\t', header=None)
sh_6 = torch.tensor(pd_sh_6.values).type(torch.float).reshape([-1,9]).expand([8, 9]).cuda()
sh_6 = torch.tensor(SHCV.sfs2shtools(sh_6)).type(torch.float).reshape([-1,9]).expand([8, 9]).cuda()
wandb_log_images(get_shading_DPR( Sphere_DPR_yz_2(pN_1, pL_1), sh_6), None, pathName=file_name + '_gt_l_6.png')

