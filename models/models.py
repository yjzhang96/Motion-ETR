from .bluring_offset_quadratic import Bluring_model_Q
from .deblur_model_MA_offset_pre import MA_Deblur

import os

def create_model(opt):
	model = None
	if opt.model == 'test':
		assert (opt.dataset_mode == 'single')
		
	if opt.blur_direction == 'deblur':
		model = MA_Deblur(opt)
	elif opt.blur_direction == 'reblur':
		model = Bluring_model_Q( opt )

	# model.initialize(opt)
	print("model [%s] was created" % (model.name()))
	return model
