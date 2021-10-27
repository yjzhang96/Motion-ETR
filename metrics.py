import ipdb
import numpy as np 
import math
import skimage
import skimage.io as io
import os
from skimage.metrics import structural_similarity 
import argparse
import time
import cv2
import torch
from skimage.transform import rescale, resize, downscale_local_mean
import lpips
from ipdb import set_trace as stc

parser = argparse.ArgumentParser()
parser.add_argument('--res_root',type=str,default='/home/yjz/VFI/' ,help='the dir of restore image')
parser.add_argument('--ref_root',type=str,default='/home/yjz/datasets/GOPRO/test' ,help='the dir of restore image')
args = parser.parse_args()
lpips_fn_alex = lpips.LPIPS(net='alex') # best forward scores

def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def ssim(img1,img2):
    return structural_similarity(img1,img2,multichannel=True)

def mse(img1,img2):
    return np.mean((img1-img2)**2)

def write_txt(file_name, line):
    with open(file_name,'a') as log:
        log.write(line+'\n')
    print(line)


def metrics():
    print("-----calculate metrics for result: %s ---------"%args.res_root)
    paths = sorted(os.listdir(args.res_root))
    path_0 = os.path.join(args.res_root, paths[0])
    if os.path.isdir(path_0):
        multiple_video = True
    else:
        multiple_video = False

    ave = 0.0
    cnt = 0.0 
    t_ssim = 0
    t_psnr = 0
    t_lpips = 0
    t_mse = 0
    record_file = os.path.join(args.res_root,'PSNR.txt')
    if os.path.exists(record_file):
        os.system('rm %s'%record_file)
    
    if multiple_video:
        # if results and dataset are organized in multiple directory
        for path in paths:
            ref_path = os.path.join(args.ref_root,path,'sharp')
            res_path = os.path.join(args.res_root,path)
            gt_files = sorted(os.listdir(ref_path))
            res_files = sorted(os.listdir(res_path))
            res_files = [i for i in res_files if i.endswith('fake_S.png')]

            print("total %d images in directory %s"%(len(res_files),res_path))
            gt_files = sorted(gt_files)
            res_files = sorted(res_files)
            # assert gt_files[0] == res_files[0]

            cnt_video = 0
            psnr_video = 0
            ssim_video = 0
            lpips_video = 0
            for i in range(len(res_files)):

                imname = os.path.join(ref_path,gt_files[i])
                img1 = io.imread(imname)
                img2_name = os.path.join(res_path,res_files[i])
                img2 = io.imread(img2_name)
                img1 = skimage.img_as_float32(img1)
                img2 = skimage.img_as_float32(img2)
                # import ipdb; ipdb.set_trace()
                per_ssim = ssim(img1,img2)
                ssim_video += per_ssim
                per_psnr = psnr(img1,img2)
                psnr_video += per_psnr
                
                ## calculate LPIPS
                img_tensor = torch.from_numpy(img1)
                img2_tensor = torch.from_numpy(img2)
                img_tensor = (img_tensor * 2 -1).permute(2,0,1).unsqueeze(0)
                img2_tensor = (img2_tensor * 2 -1).permute(2,0,1).unsqueeze(0)
                with torch.no_grad():
                    per_lpips = lpips_fn_alex.forward(img_tensor,img2_tensor)
                lpips_video += per_lpips

                cnt_video += 1
                print('psnr:%.2f, lpips:%.4f'%(per_psnr, per_lpips))
            line = 'result for video %s ssim:%.4f'%(path, ssim_video / cnt_video)
            write_txt(record_file, line)
            line = 'result for video %s psnr:%.2f'%(path, psnr_video / cnt_video)
            write_txt(record_file, line)
            t_ssim += ssim_video
            t_psnr += psnr_video
            t_lpips += lpips_video
            cnt += cnt_video
    else:
        # if results and dataset are in one directory
        ref_path = os.path.join(args.ref_root)
        res_path = os.path.join(args.res_root)
        gt_files = os.listdir(ref_path)
        res_files = os.listdir(res_path)
        gt_files = [i for i in gt_files if i.endswith('real_S.png')]
        res_files = [i for i in res_files if i.endswith('fake_S.png')]

        print("total %d images in directory %s"%(len(res_files),res_path))
        gt_files = sorted(gt_files)
        res_files = sorted(res_files)
        assert len(gt_files) == len(res_files)

        cnt_video = 0
        psnr_video = 0
        ssim_video = 0
        lpips_video = 0
        for i in range(len(res_files)):
            imname = os.path.join(ref_path,gt_files[i])
            img1 = io.imread(imname)
            img2_name = os.path.join(res_path,res_files[i])
            img2 = io.imread(img2_name)
            
            img1 = skimage.img_as_float32(img1)
            img2 = skimage.img_as_float32(img2)
            img1 = resize(img1, (img2.shape[0], img2.shape[1]))
            # H,W,_ = img1.shape
            
            per_ssim = ssim(img1,img2)
            ssim_video += per_ssim
            per_psnr = psnr(img1,img2)
            psnr_video += per_psnr

            ## calculate LPIPS
            img_tensor = torch.from_numpy(img1)
            img2_tensor = torch.from_numpy(img2)
            img_tensor = (img_tensor * 2 -1).permute(2,0,1).unsqueeze(0)
            img2_tensor = (img2_tensor * 2 -1).permute(2,0,1).unsqueeze(0)
            with torch.no_grad():
                per_lpips = lpips_fn_alex.forward(img_tensor,img2_tensor)
            lpips_video += per_lpips

            cnt_video += 1
            print('psnr:%.2f, lpips:%.4f'%(per_psnr, per_lpips))
        t_ssim += ssim_video
        t_psnr += psnr_video
        t_lpips += lpips_video
        cnt += cnt_video
    line = 'Total ssim:%.4f'%(t_ssim/cnt)
    write_txt(record_file, line)
    line = 'Total psnr:%.4f'%(t_psnr/cnt)
    write_txt(record_file, line)
    line = 'Total LPIPS:%.4f'%(t_lpips/cnt)
    write_txt(record_file, line)
 
if __name__ == "__main__":
    metrics()

