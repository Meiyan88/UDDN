import os.path,random
import time,codecs
import numpy as np
import torch.utils.data
import collections
from options.train_options import TrainOptions
from options.test_options import TestOptions
from data import create_dataset,my_dataset
from models import create_model
from util.visualizer import Visualizer
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import cv2
from niqe import niqe
from pytorch_msssim import ms_ssim
import SimpleITK as sitk

def to_255(img):
    return ((img + 1) / 2) * 255

def to_image( img,save_path,img_type, index):

    img = img * 255
    out = sitk.GetImageFromArray(img)
    out.SetOrigin(out.GetOrigin())
    out.SetSpacing(out.GetSpacing())
    image_path = os.path.join(save_path, '{}_'.format(img_type)+'{}.nii.gz'.format(index))
    sitk.WriteImage(out, image_path)

def to_numpy(tensor):
    img_numpy = tensor.detach().cpu().numpy()
    img_numpy = img_numpy.squeeze()
    img = img_numpy.squeeze()
    img = np.clip(img,0,255.0)
    img = img/255.0


    return img


# def val(epochi,model,file_path,phase):
#
#     model.eval()
#
#
#     opt = TestOptions().parse()
#     opt.batch_size = 1
#     opt.serial_batches = True
#     opt.phase = phase
#     name = opt.name
#     dataset1 = create_dataset(opt)
#     dataset1size = len(dataset1)
#     path_val = os.path.join(r'/public/huangmeiyan/wby/cycelegan/checkpoints',name,phase)
#     if not os.path.exists(path_val):
#         os.makedirs(path_val)
#         print('i am here')
#
#     print('The number of {} images = '.format(phase),  dataset1size)
#     p1, p2 = 0, 0
#     s1, s2 = 0, 0
#     if epochi==1:
#         file = open(file_path,'w')
#     for i, data in enumerate(dataset1):
#         index = str(data['A_paths'])
#         index = index.split('/')[-1]
#         index = index.split('.')[0]
#         #print(index)
#
#         model.set_input(data)
#         model.test()
#         realA, realB,fakeB = model.return_img()
#         realA = realA.float()
#         realB = realB.float()
#         fakeB = fakeB.float()
#
#
#         #loss1 += loss(realA, rec_A)
#         realA = to_255(realA)
#         realB = to_255(realB)
#         fakeB = to_255(fakeB)
#
#         ms1 = ms_ssim(realA,realB,data_range=255,size_average=True)
#         ms2 = ms_ssim(fakeB,realB,data_range=255,size_average=True)
#
#         s1 += ms1.item()
#         s2 += ms2.item()
#
#
#
#         realA = to_numpy(realA)
#         realB = to_numpy(realB)
#         fakeB = to_numpy(fakeB)
#
#         p1 += psnr(realA, realB)
#         p2 += psnr(realB, fakeB)
#
#
#         p1_all = p1 / dataset1size
#         p2_all = p2 / dataset1size
#         s1_all = s1 / dataset1size
#         s2_all = s2 / dataset1size
#         # if epochi % 20 == 0:
#         #     path_epoch = os.path.join(path_val,str(epochi))
#         #     if not os.path.exists(path_epoch):
#         #         os.makedirs(path_epoch)
#         #
#         #     to_image(realA,path_epoch,img_type='real_A',index=index)
#         #     to_image(realB,path_epoch,img_type='real_B',index=index)
#         #     to_image(fakeB,path_epoch,img_type='fake_B',index=index)
#
#
#
#     print('PSNR_ORI:{}'.format(p1_all), 'PSNR_AFT:{}'.format(p2_all), 'SSIM_ORI:{}'.format(s1_all),
#           'SSIM_ORI:{}'.format(s2_all))
#     with codecs.open(file_path, mode='a', encoding='utf-8') as file_txt:
#         #file_txt.write('\n' + '{}_loss:'.format(epochi) + str(loss1.item()))
#         if epochi == 1:
#             file_txt.write(
#                 '\n' + '{}_psnr_before:'.format(epochi) + str(p1_all) + '{}_ssim_before:'.format(epochi) + str(s1_all))
#             file_txt.write(
#                 '\n' + '{}_psnr_after:'.format(epochi) + str(p2_all) + '{}_ssim_after:'.format(epochi) + str(s2_all))
#         else:
#             file_txt.write(
#                 '\n' + '----------------------------------------------------------------------------')
#             file_txt.write(
#                 '\n' + '{}_psnr_after:'.format(epochi) + str(p2_all) + '{}_ssim_after:'.format(epochi) + str(s2_all))
#

def test(epochi,model,file_path,phase):

    model.eval()


    opt = TestOptions().parse()
    opt.batch_size = 1
    opt.serial_batches = True
    opt.phase = phase
    name = opt.name
    dataset1 = create_dataset(opt)
    dataset1size = len(dataset1)
    path_val = os.path.join(r'/public/huangmeiyan/wby/cycelegan/checkpoints',name,phase)
    if not os.path.exists(path_val):
        os.makedirs(path_val)
        print('i am here')

    print('The number of {} images = '.format(phase),  dataset1size)
    p1, p2 = 0, 0
    s1, s2 = 0, 0
    for i, data in enumerate(dataset1):
        index = str(data['A_paths'])
        index = index.split('/')[-1]
        index = index.split('.')[0]
        #print(index)

        model.set_input(data)
        model.test()
        #loss = model.get_loss()
        realA, realB,fakeB = model.return_img()


        #loss1 += loss(realA, rec_A)

        #loss1 += loss(realA, rec_A)
        realA = to_255(realA)
        realB = to_255(realB)
        fakeB = to_255(fakeB)

        ms1 = ms_ssim(realA,realB,data_range=255,size_average=True)
        ms2 = ms_ssim(fakeB,realB,data_range=255,size_average=True)

        s1 += ms1.item()
        s2 += ms2.item()


        realA = to_numpy(realA)
        realB = to_numpy(realB)
        fakeB = to_numpy(fakeB)
        print(realA.min(),realA.max())
        print(realB.min(),realB.max())
        print(fakeB.min(),fakeB.max())

        p1 += psnr(realA, realB)
        p2 += psnr(realB, fakeB)

        p1_all = p1 / dataset1size
        p2_all = p2 / dataset1size
        s1_all = s1 / dataset1size
        s2_all = s2 / dataset1size

        path_epoch = os.path.join(path_val,str(epochi))
        if not os.path.exists(path_epoch):
            os.makedirs(path_epoch)
        if epochi == 80:
            to_image(realA,path_epoch,img_type='real_A',index=index)
            to_image(realB,path_epoch,img_type='real_B',index=index)
            to_image(fakeB,path_epoch,img_type='fake_B',index=index)



    print('PSNR_ORI:{}'.format(p1_all), 'PSNR_AFT:{}'.format(p2_all), 'SSIM_ORI:{}'.format(s1_all),
          'SSIM_AFT:{}'.format(s2_all))

    with codecs.open(file_path, mode='a', encoding='utf-8') as file_txt:
        file_txt.write(
            '\n' + '----------------------------------------------------------------------------')
        file_txt.write(
            '\n' + '{}_psnr_after:'.format(epochi) + str(p2_all) + '{}_ssim_after:'.format(epochi) + str(s2_all))


def test_nf(epochi,model,file_path,phase):
    model.eval()

    opt = TestOptions().parse()
    opt.batch_size = 1
    opt.serial_batches = True
    opt.phase = phase
    name = opt.name
    dataset1 = create_dataset(opt)
    dataset1size = len(dataset1)
    path_val = os.path.join(r'/public/huangmeiyan/wby/cycelegan/checkpoints', name, phase)
    if not os.path.exists(path_val):
        os.makedirs(path_val)
        print('i am here')

    print('The number of {} images = '.format(phase), dataset1size)
    niqe1, niqe2 = 0, 0
    s1, s2 = 0, 0
    for i, data in enumerate(dataset1):
        index = str(data['A_paths'])
        index = index.split('/')[-1]
        index = index.split('.')[0]
        # print(index)

        model.set_input(data)
        model.test()
        # loss = model.get_loss()
        realA, _, fakeB = model.return_img()

        # loss1 += loss(realA, rec_A)

        realA = to_numpy(realA)
        fakeB = to_numpy(fakeB)

        niqe1 = niqe1 + niqe(realA.astype('uint8'))
        niqe2 = niqe2 + niqe(fakeB.astype('uint8'))




        path_epoch = os.path.join(path_val, str(epochi))
        if not os.path.exists(path_epoch):
            os.makedirs(path_epoch)
        if epochi == 80:
            to_image(realA, path_epoch, img_type='real_A', index=index)
            #to_image(realB, path_epoch, img_type='real_B', index=index)
            to_image(fakeB, path_epoch, img_type='fake_B', index=index)
    niqe1_all = niqe1 / dataset1size
    niqe2_all = niqe2 / dataset1size
    print('NIQE_ORI:{}'.format(niqe1_all), 'NIQE_AFT:{}'.format(niqe2_all))

    with codecs.open(file_path, mode='a', encoding='utf-8') as file_txt:
        file_txt.write(
            '\n' + '----------------------------------------------------------------------------')
        file_txt.write(
            '\n' + 'NIQE_AFT:{}'.format(niqe2_all))




