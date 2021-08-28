import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from pdb import set_trace as st
from util import html
from util.metrics import PSNR
import util.util as util
import numpy as np 

from ssim import SSIM
from PIL import Image

def save_image(visuals, img_path):
	# import ipdb; ipdb.set_trace()
	root, file = os.path.split(img_path)
	# root, dir = os.path.split(root)
	new_path = './results/real_images/'
	if not os.path.exists(new_path):
		os.mkdir(new_path)
	name = os.path.splitext(file)[0]

	deblur_file = os.path.join(new_path,name+'_sharp.png')
	blur_file = os.path.join(new_path,name+'_blurry.png')

	for label, image_numpy in visuals.items():
		if label == 'fake_B':
			image_pil = Image.fromarray(image_numpy)
			image_pil.save(deblur_file)
		if label == 'real_A':
			image_pil = Image.fromarray(image_numpy)
			image_pil.save(blur_file)


opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
# opt.dataroot='/home/yjz/datasets/Synthetic_motion_flow/test_syn'
opt.dataset_mode='aligned'


extract_frame = False
visualize_traj = False
visualize_flow = True

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)
# test
avgPSNR = 0.0
avgSSIM = 0.0
counter = 0

for i, data in enumerate(dataset):
	if i >= opt.how_many:
		break

	counter = i+1
	model.set_input(data)
	model.test()
	reblur_res = model.fake_A
	blurry_input = model.real_A
	reblur_res_np = util.tensor2im(reblur_res)
	blurry_input_np = util.tensor2im(blurry_input)
	
	
	img_path = model.get_image_paths()
	root, file = os.path.split(img_path[0])
	name = os.path.splitext(file)[0]
	print('process image %s' % name)
	# write image
	input_dir = opt.dataroot.split('/')[-1]
	output_dir = os.path.join(opt.results_dir,opt.name, input_dir)
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	
	reblur_name = os.path.join(output_dir,name+'_reblur.png')
	blur_name = os.path.join(output_dir,name+'_blurry.png')
	
	util.save_image(reblur_res_np, reblur_name)
	util.save_image(blurry_input_np, blur_name)

	# visualization
	if visualize_traj:
		vis_dir = os.path.join(output_dir, 'traj')
		if not os.path.exists(vis_dir):
			os.mkdir(vis_dir)
		traj = model.draw_quadratic_line()
		vis_name = os.path.join(vis_dir,name+'_traj.png')
		util.save_image(traj, vis_name)
	if visualize_flow:
		vis_dir = os.path.join(output_dir, 'vis_flow')
		if not os.path.exists(vis_dir):
			os.mkdir(vis_dir)
		flow = model.visualize_offset_flow()
		vis_name = os.path.join(vis_dir,name+'_flow.png')
		util.save_image(flow, vis_name)
	if extract_frame:
		frame_dir = os.path.join(output_dir,'frame')
		if not os.path.exists(frame_dir):
			os.mkdir(frame_dir)
		frames = model.save_everyframe()
		for i in range(len(frames)):
			frame = frames[i]
			frame_name = os.path.join(frame_dir,name+'_frame%02d.png'%i)
			util.save_image(frame,frame_name)


