import ipdb
import numpy as np
import torch
import os
import cv2
from collections import OrderedDict
from torch.autograd import Variable
import torch.nn.functional as F
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from .losses import init_loss,SSIMLoss
import time

try:
	xrange          # Python2
except NameError:
	xrange = range  # Python 3



class Bluring_model_Q(BaseModel):
    def name(self):
        return "Bluring_model_Q"

    def __init__(self,opt):
        super(Bluring_model_Q,self).__init__(opt)
        self.is_train = opt.isTrain

		# define tensor
        self.input_A = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
        self.input_B = self.Tensor(opt.batchSize, opt.output_nc, opt.fineSize, opt.fineSize)

		# define netwrks
        self.n_offset = opt.n_offset
        self.offset_net = networks.define_offset_quad(input_nc=3,nf=16,n_offset=self.n_offset, offset_mode=opt.offset_mode, norm='batch', gpu_ids=self.gpu_ids)
        self.blur_net = networks.define_blur(gpu_ids=self.gpu_ids)      # deformable
        
        if self.is_train:
            self.old_lr = opt.lr
            
            self.fake_AB_pool = ImagePool(opt.pool_size)
            # initialize optimizers
            self.optimizer = torch.optim.Adam( self.offset_net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999) )

            # define loss function
            self.MSEloss = torch.nn.MSELoss()	
            self.L1loss = torch.nn.L1Loss()
            self.SSIMloss = SSIMLoss('MSSSIM')

        if not self.isTrain or opt.continue_train:
            self.load_network(self.offset_net, 'offset', opt.which_epoch)
            if opt.continue_train:
                self.old_lr = opt.lr - opt.lr*(opt.epoch_count-opt.niter)/opt.niter_decay
                print('loading learning rate: %f' % (self.old_lr))
            params = self.offset_net.state_dict()
            print('--------------load network--------------')
            
            
        # print('---------- Networks initialized -------------')
        # if self.isTrain:
        #     networks.print_network(self.offset_net)
        # print('-----------------------------------------------')

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)
        B,C,H,W = self.real_A.shape

        self.offset = self.offset_net(self.real_A)
        offset_N = torch.chunk(self.offset, self.n_offset,dim=1)
        self.fake_A_n = torch.zeros(B,C*self.n_offset,H,W).cuda()
        for i in range(len(offset_N)):
            self.fake_A_n[:,i*3:(i+1)*3,:,:] = self.blur_net(self.real_B,offset_N[i])

        
    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        inputA = input['A' if AtoB else 'B']
        inputB = input['B' if AtoB else 'A']
        self.input_A.resize_(inputA.size()).copy_(inputA)
        self.input_B.resize_(inputB.size()).copy_(inputB)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def set_test_input(self, input):
        inputA = input['A']
        self.input_A.resize_(inputA.size()).copy_(inputA)
        self.image_paths = input['A_paths']        
        
    def backward(self):

        B,C,H,W = self.fake_A_n.shape
        self.offset	= self.offset.view(B,self.n_offset,-1,H,W)
        
        # # spatial tv_loss across one offset
        lambda_tv = self.opt.lambda_tv
        self.tv_loss = self.L1loss(self.offset[:,:,:,:,:-1],self.offset[:,:,:,:,1:]) + \
                        self.L1loss(self.offset[:,:,:,:-1,:],self.offset[:,:,:,1:,:])

        # # regulationl loss
        lambda_reg = self.opt.lambda_reg
        self.reg_loss = torch.mean(self.offset[:,:,:,:,:]**2)


        # MSE loloss
        self.fake_A_n = self.fake_A_n.view(B,self.n_offset,-1,H,W)
        self.fake_A = torch.sum(self.fake_A_n,dim=1)/self.n_offset
        self.loss_MSE = self.MSEloss(self.fake_A,self.real_A)

        # SSIM loss
        lambda_SSIM = self.opt.lambda_ssim
        self.ssim_loss = 1 - self.SSIMloss.get_loss(self.fake_A,self.real_A)

        self.loss_total = lambda_SSIM * self.ssim_loss + self.loss_MSE \
                            + lambda_tv * self.tv_loss + lambda_reg * self.reg_loss 
                    
        self.loss_total.backward()

    def optimize_parameters(self):
        with torch.autograd.detect_anomaly():
            self.forward()
            self.optimizer.zero_grad()
            self.backward()
            self.optimizer.step()

    def test(self):
        self.real_A = self.input_A
        self.real_B = self.input_B

        with torch.no_grad():
            self.offset = self.offset_net(self.real_A)
           

            B,C,H,W = self.real_B.shape
            offset_N = torch.chunk(self.offset, self.n_offset,dim=1)
            self.fake_A_n = torch.zeros(B,C*self.n_offset,H,W).cuda()
            for i in range(len(offset_N)):
                self.fake_A_n[:,i*3:(i+1)*3,:,:] = self.blur_net(self.real_B,offset_N[i])

            self.fake_A_n = self.fake_A_n.view(B,self.n_offset,-1,H,W)
            self.fake_A = torch.sum(self.fake_A_n,dim=1)/self.n_offset

    def get_current_errors(self):
        return OrderedDict([('L2_loss', self.loss_MSE.item()),
                            ('ssim_loss',self.ssim_loss.item()),
                            ('reg_loss',self.reg_loss.item()),
                            ('tv_loss',self.tv_loss.item()),
                            ('total_loss',self.loss_total.item())
                            ])

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_A = util.tensor2im(self.fake_A.data)
        real_B = util.tensor2im(self.real_B.data)
        return OrderedDict([('Blurry', real_A), ('Reblur', fake_A), ('Sharp', real_B)])

    def get_current_offset(self):
        offset = self.offset.cpu().numpy()
        return offset

    def save(self, label):
        self.save_network(self.offset_net, 'offset', label, self.gpu_ids)

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

    def get_image_paths(self):
        return self.image_paths

    def visualize_offset_flow(self):
        from PIL import Image
        offset = self.offset.cpu()
        offset = offset.detach().numpy()
        offset = np.squeeze(offset)
        C,H,W = offset.shape
        offset	= offset.reshape(self.n_offset,-1,H,W)
        offset_xy = offset[:,:2,:,:]
        offset_xy = np.reshape(offset_xy, (-1,H,W))
        offset = np.transpose(offset_xy,(1,2,0))
        H,W,C = offset.shape
        offset = np.round(offset)
        vec = offset 
        hsv = np.zeros((offset.shape[0],offset.shape[1],3),dtype=np.uint8)
        hsv[...,2] = 255

        # # vector norm
        # index = np.where(vec[...,1] < 0)
        # vec[index] = -vec[index]  
        # flow = np.sum(vec,axis=2)/self.n_offset
        # import ipdb; ipdb.set_trace()
        
        flow = vec[:,:,0:2] - vec[:,:,-2:]

        mag,ang = cv2.cartToPolar(flow[...,1], -flow[...,0])
        hsv[...,0] = ang * 180 / np.pi / 2
        print(mag.max())
        # mag[-1,-1] = max(10,mag.max())
        hsv[...,1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        return rgb
        

    def draw_bilinear_line(self):
        import cv2
        base_img = self.real_A.cpu().detach().numpy()
        base_img = np.transpose(np.squeeze(base_img),(1,2,0))
        base_img = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
        base_img = np.tile(np.expand_dims(base_img,-1),(1,1,3))
        base_img = 0.6 * 1 + 0.4 * base_img 
        base_img = np.uint8(base_img*255)

        # order = np.array([0,7,6,5,4,3,2,14,8,9,10,11,12,13,1])
        order = np.arange(self.n_offset)
        mid = self.n_offset//2
        order[1:mid] = np.arange(mid-1,0,-1)
        
        offset = self.offset.cpu()
        offset = offset.detach().numpy()
        offset = np.squeeze(offset)
        C,H,W = offset.shape
        offset	= offset.reshape(self.n_offset,-1,H,W)
        offset_12 = np.stack((offset[0],offset[-1]),axis=0) 
        # import ipdb;ipdb.set_trace()
        flow_map = np.zeros((H,W,3),dtype='uint8')
        inter = 17
        for i in range(0,H-inter,inter):
            for j in range(0,W-inter,inter):
                window = base_img[i:i+inter,j:j+inter]
                window_center = np.array([inter//2,inter//2])
                offset_ij = offset_12[:,:,i+inter//2, j+inter//2]

                zero_p = np.zeros((2,))
                offset_ij = np.stack((offset_ij[0],zero_p,offset_ij[1]),axis=0)
                indexes = offset_ij + window_center
                
                indexes = np.flip(indexes)

                cv2.polylines(window,np.int32([indexes]),False,color=(255,0,0),thickness=1)
                flow_map[i:i+inter,j:j+inter] = window
        return flow_map


    def draw_quadratic_line(self):
        import cv2
        base_img = self.real_A.cpu().detach().numpy()
        base_img = np.transpose(np.squeeze(base_img),(1,2,0))
        base_img = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
        base_img = np.tile(np.expand_dims(base_img,-1),(1,1,3))
        base_img = 0.6 * 1 + 0.4 * base_img 
        base_img = np.uint8(base_img*255)

        order = np.arange(self.n_offset)
        mid = self.n_offset//2
        order[1:mid] = np.arange(mid-1,0,-1)
        offset = self.offset.cpu()
        offset = offset.detach().numpy()
        offset = np.squeeze(offset)
        C,H,W = offset.shape
        offset	= offset.reshape(self.n_offset,-1,H,W)

        flow_map = np.zeros((H,W,3),dtype='uint8')
        inter = 17
        for i in range(0,H-inter,inter):
            for j in range(0,W-inter,inter):
                window = base_img[i:i+inter,j:j+inter]
                window_center = np.array([inter//2,inter//2])
                offset_ij = offset[:,:,i+inter//2, j+inter//2]
                
                offset_ij = np.round(offset_ij[order])

                indexes = offset_ij + window_center
                indexes = np.flip(indexes)
                cv2.polylines(window,np.int32([indexes]),False,color=(255,0,0),thickness=1)
                flow_map[i:i+inter,j:j+inter] = window
        return flow_map


    def save_everyframe(self):
        B,C,H,W = self.real_B.shape
        offset_N = torch.chunk(self.offset, self.n_offset, dim=1)
        fake_A_n = torch.zeros(B,C*self.n_offset,H,W).cuda()
        with torch.no_grad():
            for i in range(len(offset_N)):
                fake_A_n[:,i*3:(i+1)*3,:,:] = self.blur_net(self.real_B,offset_N[i])
        frames = torch.chunk(fake_A_n,self.n_offset,dim=1)

        frames_order = []
        order = np.arange(self.n_offset)
        mid = self.n_offset//2
        order[1:mid] = np.arange(mid-1,0,-1)
        print(order)
        for i in range(len(frames)):
            frame_i = frames[order[i]]
            frame_np = util.tensor2im(frame_i)
            frames_order.append(frame_np)
        return frames_order