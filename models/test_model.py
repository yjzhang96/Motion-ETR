from torch.autograd import Variable
from collections import OrderedDict
import util.util as util
from .base_model import BaseModel
from . import networks
import torch
import numpy as np
import os

class TestModel(BaseModel):
    def name(self):
        return 'TestModel'

    def __init__(self, opt):
        assert(not opt.isTrain)
        super(TestModel, self).__init__(opt)
        # self.input_A = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)

        self.n_offset=16
        self.offset_net = networks.define_offset(input_nc=3,nf=16,n_offset=self.n_offset , norm='batch', gpu_ids=self.gpu_ids)
        self.deblur_net = networks.define_deblur_multikernel(input_nc=3,nf=32,n_offset=self.n_offset, norm='batch', gpu_ids=self.gpu_ids)
        which_epoch = opt.which_epoch
        self.load_network(self.offset_net, 'offset', "latest")
        self.load_network(self.deblur_net, 'deblur', opt.which_epoch)

        print('---------- Networks initialized -------------')
        networks.print_network(self.deblur_net)
        print('-----------------------------------------------')

    def set_input(self, input):
        # we need to use single_dataset mode
        inputA = input['A']
        # temp = self.input_A.clone()
        # temp.resize_(input_A.size()).copy_(input_A)
        # self.input_A = temp
        # _, self.H,self.W,_ = inputA.shape
        # tmp = self.input_A.clone()
        # tmp.resize_((1,3,640,640))
        self.input_A = inputA.to(self.device)
        self.image_paths = input['A_paths']

    def test(self):
        self.real_A = Variable(self.input_A)
        with torch.no_grad():
            # import ipdb;ipdb.set_trace()
            self.offset = self.offset_net(self.real_A)
            self.fake_B = self.deblur_net(self.real_A,self.offset)
            self.map = self.deblur_net.map

            # self.draw_offset(self.offset)
            # self.print_map(self.map)
            self.print_offset(self.offset)

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B)])

    def print_map(self,att_map):
        import cv2
        # import ipdb;ipdb.set_trace()

        att_map = att_map.cpu().detach().numpy()
        att_map = att_map[0]
        att_map = np.transpose(att_map,(1,2,0))

        img_path = self.get_image_paths()
        short_path = os.path.basename(img_path[0])
        name = os.path.splitext(short_path)[0]
        for i in range(3):
            map_name = name + '_map_%d.png'%i
            print('visiualize attention map %s'%map_name)
            save_path = os.path.join(self.opt.results_dir, self.opt.name, '%s_%s' % (self.opt.phase, self.opt.which_epoch),'images')
            map_dir = os.path.join(save_path,map_name)
            cv2.imwrite((map_dir),att_map[:,:,i]*255)
    
    def print_offset(self, offset_gpu):
        from PIL import Image
        import cv2
        from util.offset_remap import img_offset_remap 
        offset = offset_gpu.cpu()
        offset = offset.detach().numpy()
        offset = np.squeeze(offset)
        C,H,W = offset.shape
        offset	= offset.reshape(self.n_offset,-1,H,W)
        offset_xy = offset[:,:2,:,:]
        offset_xy = np.reshape(offset_xy, (-1,H,W))
        offset = np.transpose(offset_xy,(1,2,0))
        H,W,C = offset.shape
        offset = np.round(offset)
        offset = offset.reshape((H,W,-1,2))
        # coord = np.array([[-1,-1],[-1,0],[-1,1],[0,-1],[0,0],[0,1],[1,-1],[1,0],[1,1]])
        vec = offset 
        hsv = np.zeros((offset.shape[0],offset.shape[1],3),dtype=np.uint8)
        hsv[...,2] = 255

        # # method1: vector sum
        index = np.where(vec[...,1] < 0)
        vec[index] = -vec[index]  
        flow = np.sum(vec,axis=2)/self.n_offset
        mag,ang = cv2.cartToPolar(flow[...,1], -flow[...,0])
        hsv[...,0] = ang * 180 / np.pi / 2
        # import ipdb; ipdb.set_trace()
        mag[-1,-1] = max(10,mag.max())
        hsv[...,1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        img_path = self.get_image_paths()
        short_path = os.path.basename(img_path[0])
        name = os.path.splitext(short_path)[0]

        flow_name = name + '_flow.png'       
        # save_path = os.path.join(self.opt.results_dir, self.opt.name, '%s_%s' % (self.opt.phase, self.opt.which_epoch),'images')
        save_path = os.path.join(self.opt.results_dir, 'real_images')
        flow_dir = os.path.join(save_path,flow_name)
        print('visiualize motion flow %s'%flow_dir)

        cv2.imwrite((flow_dir),bgr)

        # _, thresh_blur = cv2.threshold(mag,4,255,cv2.THRESH_BINARY)
        # thresh_sharp = 255 - thresh_blur

        # up_name = name + '_map_1.png'       
        # save_path = os.path.join(self.opt.results_dir, self.opt.name, '%s_%s' % (self.opt.phase, self.opt.which_epoch),'images')
        # flow_dir = os.path.join(save_path,up_name)
        # cv2.imwrite((flow_dir),thresh_blur)
        # down_name = name + '_map_0.png'       
        # save_path = os.path.join(self.opt.results_dir, self.opt.name, '%s_%s' % (self.opt.phase, self.opt.which_epoch),'images')
        # flow_dir = os.path.join(save_path,down_name)
        # cv2.imwrite((flow_dir),thresh_sharp)


    def draw_offset(self,offset_gpu):
        import cv2
        offset = offset_gpu.cpu()
        offset = offset.detach().numpy()
        offset = np.squeeze(offset)
        C,H,W = offset.shape
        offset	= offset.reshape(self.n_offset,-1,H,W)
        offset_xy = offset[:,:2,:,:]

        flow_map = np.zeros((H,W),dtype=np.float64)
        inter = 21
        for i in range(0,H-inter,inter):
            for j in range(0,W-inter,inter):
                window = np.zeros((inter,inter))
                window_center = np.array([inter//2,inter//2])
                offset_ij = offset[:,:,i+inter//2, j+inter//2]
                # transfer offset to discrete index number
                offset_ij = np.round(offset_ij)
                offset_ij = offset_ij.astype(int)
                # print(offset_ij.shape,offset_ij)

                indexes = offset_ij + window_center
                indexes = np.clip(indexes,0,inter-1)
                # index can be repeat, so the map need normalization
                for idx in indexes:
                    window[idx[0],idx[1]] += 1
                window = window / np.max(window)
                flow_map[i:i+inter,j:j+inter] = window
        # import ipdb; ipdb.set_trace()

        img_path = self.get_image_paths()
        root, short_path = os.path.split(img_path[0])
        # root, dir = os.path.split(root)
        name = os.path.splitext(short_path)[0]
        flow_name = name + '_flow.png'       
        save_path = os.path.join(self.opt.results_dir, 'Gopro_draw_mf')
        flow_dir = os.path.join(save_path,flow_name)
        print('visiualize motion flow %s'%flow_dir)
        cv2.imwrite(flow_dir,flow_map*255)