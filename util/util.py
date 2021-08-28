from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import inspect, re
import numpy as np
import os
import collections
from skimage.transform import resize
import cv2

# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8):
	image_numpy = image_tensor[0].cpu().float().numpy()
	# image_numpy = np.clip((np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0,0,1) * 255.0
	image_numpy = np.clip(np.transpose(image_numpy, (1, 2, 0)),0,1)  * 255.0
	
	# if input resize
	# image_numpy = resize(image_numpy,(image_numpy.shape[0]*2,image_numpy.shape[1]*2),anti_aliasing=False)
	return image_numpy.astype(imtype)

def load_image(filename, trans_list=None, size=None, scale=None):
    img = Image.open(filename).convert('RGB')
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    
    if trans_list:
        img = trans_list(img)
    img = img.unsqueeze(0)
    return img

def flow2heat(bmap_gpu):
    bmap = bmap_gpu.cpu()
    bmap = bmap.detach().numpy()
    bmap = np.squeeze(bmap)

    bmap = np.transpose(bmap,(1,2,0))
    H,W,C = bmap.shape
    
    vec = bmap 
    hsv = np.zeros((H,W,3),dtype=np.uint8)
    hsv[...,2] = 255

    # # method1: vector sum
    # index = np.where(vec[...,1] < 0)
    # vec[index] = -vec[index]  
    mag,ang = cv2.cartToPolar(vec[...,0], -vec[...,1])
    hsv[...,0] = ang * 180 / np.pi / 2
    # print("max:",mag.max(),"min",mag.min())
    # mag[-1,-1] = 0.25
    hsv[...,1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb

def diagnose_network(net, name='network'):
	mean = 0.0
	count = 0
	for param in net.parameters():
		if param.grad is not None:
			mean += torch.mean(torch.abs(param.grad.data))
			count += 1
	if count > 0:
		mean = mean / count
	print(name)
	print(mean)


def save_image(image_numpy, image_path):
	image_pil = None
	if image_numpy.shape[2] == 1:
		image_numpy = np.reshape(image_numpy, (image_numpy.shape[0],image_numpy.shape[1]))
		image_pil = Image.fromarray(image_numpy, 'L')
	else:
		image_pil = Image.fromarray(image_numpy)
	image_pil.save(image_path)

def info(object, spacing=10, collapse=1):
	"""Print methods and doc strings.
	Takes module, class, list, dictionary, or string."""
	methodList = [e for e in dir(object) if isinstance(getattr(object, e), collections.Callable)]
	processFunc = collapse and (lambda s: " ".join(s.split())) or (lambda s: s)
	print( "\n".join(["%s %s" %
					 (method.ljust(spacing),
					  processFunc(str(getattr(object, method).__doc__)))
					 for method in methodList]) )

def varname(p):
	for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
		m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
		if m:
			return m.group(1)

def print_numpy(x, val=True, shp=False):
	x = x.astype(np.float64)
	if shp:
		print('shape,', x.shape)
	if val:
		x = x.flatten()
		print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
			np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
	if isinstance(paths, list) and not isinstance(paths, str):
		for path in paths:
			mkdir(path)
	else:
		mkdir(paths)


def mkdir(path):
	if not os.path.exists(path):
		os.makedirs(path)
