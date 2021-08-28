import time
from options.train_options import TrainOptions
import os
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util.metrics import PSNR, SSIM
from multiprocessing import freeze_support
from tensorboardX import SummaryWriter

def train(opt, data_loader, model, visualizer):
	dataset,val_dataset = data_loader.load_data()
	writer = SummaryWriter('./checkpoints/%s'%opt.name)
	dataset_size = len(data_loader)
	print('#training images = %d' % dataset_size)
	total_steps = 0

	for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
		epoch_start_time = time.time()
		epoch_iter = 0
		for i, data in enumerate(dataset):
			iter_start_time = time.time()
			total_steps += opt.batchSize
			epoch_iter += opt.batchSize
			model.set_input(data)
			model.optimize_parameters()
			
			if total_steps % opt.display_freq == 0:
				results = model.get_current_visuals()
				psnrMetric = PSNR(results['Reblur'], results['Blurry'])
				print('PSNR on Train = %f' % psnrMetric)
				visualizer.display_current_results(results, epoch)
			if total_steps % opt.print_freq == 0:
				errors = model.get_current_errors()
				writer.add_scalar('loss',errors['total_loss'],total_steps)
				t = (time.time() - iter_start_time) / opt.batchSize
				visualizer.print_current_errors(epoch, epoch_iter, errors, t)
				if opt.display_id > 0:
					visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)
				
			# if total_steps % opt.save_latest_freq == 0:
			# 	print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
			# 	model.save('latest')
		
		test_PSNR = val(val_dataset,model, epoch, 'test')
		writer.add_scalar('PSNR/test',test_PSNR,epoch)	
		if epoch % (10*opt.save_epoch_freq) == 0:
			train_PSNR = val(dataset,model, epoch, 'train')
			writer.add_scalar('PSNR/train',train_PSNR,epoch)	
		if epoch % opt.save_epoch_freq == 0:
			print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
			model.save('latest')
			model.save(epoch%100)

		print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

		if epoch > opt.niter:
			model.update_learning_rate()

def val(dataset,model,epoch,mode):
	t_psnr = 0.0
	psnr_last = 0.0
	counter = 0
	start_time = time.time()
	for i, data in enumerate(dataset):
		counter = i+1		
		model.set_input(data)
		model.test()
		visuals = model.get_current_visuals()
		psnr = PSNR(visuals['Reblur'],visuals['Blurry'])
		# print("testing image pair: %.2f"%(psnr))
		t_psnr += psnr
		if i>=200:
			break
	ave_psnr = t_psnr/counter
	log_name = os.path.join(opt.checkpoints_dir,opt.name,'psnr_log.txt')
	message = '[Epoch:%d] %s data PSNR: %.4f (time:%.3f)' %(epoch, mode, ave_psnr,time.time()-start_time)
	print(message)
	with open(log_name,'a') as log:
		log.write(message+'\n')
	return ave_psnr

if __name__ == '__main__':
	freeze_support()

	# python train.py --dataroot /.path_to_your_data --learn_residual --resize_or_crop crop --fineSize CROP_SIZE (we used 256)

	opt = TrainOptions().parse()
	# opt.dataroot = './datasets/train_xtran/'
	opt.learn_residual = True
	opt.resize_or_crop = "crop"
	opt.fineSize = 256
	opt.gan_type = "none"
	# opt.which_model_netG = "unet_256"

	# default = 5000
	opt.save_latest_freq = 100

	# default = 100
	opt.print_freq = 20

	data_loader = CreateDataLoader(opt)
	model = create_model(opt)
	visualizer = Visualizer(opt)
	train(opt, data_loader, model, visualizer)
