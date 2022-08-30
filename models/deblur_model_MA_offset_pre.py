import numpy as np
import torch
import torch.nn as nn
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from .losses import init_loss

try:
	xrange          # Python2
except NameError:
	xrange = range  # Python 3


class MA_Deblur(BaseModel):
	def name(self):
		return "deblur_model"

	def __init__(self,opt):
		super(MA_Deblur,self).__init__(opt)
		self.is_train = opt.isTrain

		# define tensor
		self.input_A = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
		self.input_B = self.Tensor(opt.batchSize, opt.output_nc, opt.fineSize, opt.fineSize)

		# define netwrks
                
		self.n_offset=opt.n_offset
		self.MANet = networks.define_deblur_offset(input_nc=3,nf=16,n_offset=self.n_offset , offset_mode=opt.offset_mode, gpu_ids=self.gpu_ids)
		self.blur_net = networks.define_blur(gpu_ids=self.gpu_ids)      # deformable
		self.pretrain_offset_net = networks.define_offset_quad(input_nc=3,nf=16,n_offset=self.n_offset , norm='batch', gpu_ids=self.gpu_ids)

		offset_network_path = os.path.join( './pretrain_models/MTR_Gopro_quad/latest_net_offset.pth')
		if len(self.gpu_ids)>1:
			self.pretrain_offset_net.module.load_state_dict(torch.load(offset_network_path))
		else:
			self.pretrain_offset_net.load_state_dict(torch.load(offset_network_path))

		if self.is_train:
			self.old_lr = opt.lr
			# initialize optimizers
			self.optimizer = torch.optim.Adam( self.MANet.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999) )

			# define loss function
			_, self.contentLoss = init_loss(opt, self.Tensor)		

			# self.load_network offset
			# offset_network_path = os.path.join( opt.checkpoints_dir,'offset_%s' %(opt.offset_mode), '%s_net_%s.pth' % ('latest', 'offset'))
			if len(self.gpu_ids)>1:
				self.MANet.module.offset_net.load_state_dict(torch.load(offset_network_path))
			else:
				self.MANet.offset_net.load_state_dict(torch.load(offset_network_path))



			print('--------------load offset network--------------')
			# params = self.MANet.offset_net.state_dict()
			# for k, v in params.items():
			# 	print(k)

		if not self.isTrain or opt.continue_train:
			self.load_network(self.MANet, 'deblur', opt.which_epoch)
			self.save_network(self.MANet, 'deblur', 'new', self.gpu_ids)	
				
			if opt.continue_train:
				self.old_lr = opt.lr - opt.lr*(opt.epoch_count-opt.niter)/opt.niter_decay
				print('loading learning rate: %f' % (self.old_lr))
			# params = self.MANet.deblur_net.state_dict()
			# print('--------------load deblur network--------------')
			# for k, v in params.items():
			# 	print(k)
	
			
		print('---------- Networks initialized -------------')
		if self.isTrain:
			pass
			# networks.print_network(self.offset_net)
			# networks.print_network(self.deblur_net)
		print('-----------------------------------------------')

	def forward(self):
		self.real_A = Variable(self.input_A)
		self.real_B = Variable(self.input_B)
		B,C,H,W = self.real_A.shape

		self.offsets, self.fake_B = self.MANet(self.real_A)   # offset[B,2*n_offset,H,W]
		# offset_n = torch.chunk(self.offsets, self.n_offset,dim=1)
		# self.fake_A_n = torch.zeros(B,C*self.n_offset,H,W).cuda()
		# for i in range(len(offset_n)):
		# 	self.fake_A_n[:,i*3:(i+1)*3,:,:] = self.blur_net(self.real_B,offset_n[i])
		# self.fake_A_n = self.fake_A_n.view(B,self.n_offset,-1,H,W)
		# self.fake_A = torch.sum(self.fake_A_n,dim=1)/self.n_offset
		
	def set_input(self, input):
		AtoB = self.opt.which_direction == 'AtoB'
		inputA = input['A' if AtoB else 'B']
		inputB = input['B' if AtoB else 'A']
		self.input_A.resize_(inputA.size()).copy_(inputA)
		self.input_B.resize_(inputB.size()).copy_(inputB)
		self.image_paths = input['A_paths' if AtoB else 'B_paths']

		self.real_A = Variable(self.input_A)
		self.real_B = Variable(self.input_B)

	def set_test_input(self, input):
		inputA = input
		self.input_A.resize_(inputA.size()).copy_(inputA)
		# self.image_paths = input['A_paths']
		self.real_A = Variable(self.input_A)

	# no backprop gradients
	def test(self):
		# self.real_A = Variable(self.input_A)

		with torch.no_grad():
			_, self.fake_B = self.MANet(self.real_A) 
			self.offsets = self.pretrain_offset_net(self.real_A)		
		
	def test_reblur(self):
		B,C,H,W = self.fake_B.shape
		offset_N = torch.chunk(self.offsets, self.n_offset, dim=1)
		fake_A_n = torch.zeros(B,C,H,W).cuda()
		with torch.no_grad():
			for i in range(len(offset_N)):
				fake_A_n[:,:,:,:] += self.blur_net(self.real_A,offset_N[i])
			fake_A = fake_A_n/self.n_offset
		return fake_A
		
	def backward(self):
		# self.loss_blur = self.contentLoss.get_loss(self.fake_A,self.real_A)
		self.loss_blur = torch.tensor(0.0).cuda()

		self.loss_deblur = self.contentLoss.get_loss(self.fake_B,self.real_B)
		self.loss_MSE = self.loss_deblur
		self.loss_MSE.backward()

	def optimize_parameters(self):
		self.forward()
		self.optimizer.zero_grad()
		self.backward()
		self.optimizer.step()
		

	def get_current_errors(self):
		return OrderedDict([
			('loss_blur', self.loss_blur.item()),
			('loss_deblur', self.loss_deblur.item()),
			('total_loss', self.loss_MSE.item()),
							])

	def get_current_visuals(self):
		real_A = util.tensor2im(self.real_A.data)
		if self.opt.dataset_mode == 'aligned':
			fake_B = util.tensor2im(self.fake_B.data)
			real_B = util.tensor2im(self.real_B.data)
			return OrderedDict([('Blurry', real_A), ('Restore', fake_B), ('Sharp', real_B)])
		elif self.opt.dataset_mode == 'single':
			fake_B = util.tensor2im(self.fake_B.data)
			return OrderedDict([('Blurry', real_A), ('Restore', fake_B)])

	def save(self, label):
		self.save_network(self.MANet, 'deblur', label, self.gpu_ids)


	def update_learning_rate(self):
		lrd = self.opt.lr / self.opt.niter_decay
		lr = self.old_lr - lrd
		for param_group in self.optimizer.param_groups:
			param_group['lr'] = lr
		print('update learning rate: %f -> %f' % (self.old_lr, lr))
		self.old_lr = lr
	
	def get_image_paths(self):
		return self.image_paths



	def vis_everyframe(self):
		## Once you have deblured image and estimated motion offsets.
		## This function can be used for extracting video frame from the blurry image. 
		B,C,H,W = self.fake_B.shape
		offset_N = torch.chunk(self.offsets, self.n_offset, dim=1)
		fake_A_n = torch.zeros(B,C*self.n_offset,H,W).cuda()
		with torch.no_grad():
			for i in range(len(offset_N)):
				fake_A_n[:,i*3:(i+1)*3,:,:] = self.blur_net(self.fake_B,offset_N[i])

		frames = torch.chunk(fake_A_n,self.n_offset,dim=1)
		frames_order = []
		order = np.arange(self.n_offset)
		mid = self.n_offset//2
		order[1:mid] = np.arange(mid-1,0,-1)
		for i in range(len(frames)):
			frame_i = frames[order[i]]
			frame_np = util.tensor2im(frame_i)
			frames_order.append(frame_np)
		return frames_order
	
	def draw_quadratic_line(self):
		# This fuction can visualize the quadratic motion trajectory in a blur kernel form 
		# as we shown in paper.
		import cv2
		offset_gpu = self.offsets
		base_img = self.real_A.cpu().detach().numpy()
		base_img = np.transpose(np.squeeze(base_img),(1,2,0))
		base_img = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
		base_img = np.tile(np.expand_dims(base_img,-1),(1,1,3))
		base_img = 0.6 * 1 + 0.4 * base_img 
		base_img = np.uint8(base_img*255)

		order = np.array([0,7,6,5,4,3,2,14,8,9,10,11,12,13,1])
		offset = offset_gpu.cpu()
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

