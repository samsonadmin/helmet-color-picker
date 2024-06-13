import os, warnings, io, validators
from PIL import Image
import urllib.request as urllib
from urllib.error import HTTPError
import numpy as np

def get_pil_im(fp_url_nparray, verbose = False):
	'''
	return a PIL image object
	Args:
		im: np.array, filepath, or url
	'''
	im = fp_url_nparray
	pil_im = None
	if isinstance(im, Image.Image):
		pil_im = im
	elif type(im) == np.ndarray:
		pil_im = Image.fromarray(im)
	elif os.path.isfile(im):
		pil_im = Image.open(im)
	elif validators.url(im):
		try:
			r = urllib.Request(im, headers = {'User-Agent': "Miro's Magic Image Broswer"})
			con = urllib.urlopen(r)
			pil_im = Image.open(io.BytesIO(con.read()))
		except HTTPError as e:
			warnings.warn(f'get_pil_im: error getting {im}\n{e}')
			pil_im = None
	else:
		raise ValueError(f'get_im: im must be np array, filename, or url')

	if verbose:
		print(f'Find image of size {pil_im.size}')
	return pil_im

def im_gray2rgb(pil_im):
	return pil_im.copy().convert('RGB')

def im_3c(img_arr):
	'''return only 3 channel for RGB'''
	assert isinstance(img_arr, np.ndarray), f'img_arr needs to be a numpy array'
	assert len(img_arr.shape) == 3, f'img_arr expected to a 3D array but has shape of {img_arr.shape}'
	if img_arr.shape[2] < 3: #, f'expecting at least 3 channels'
		img_arr = np.array(im_gray2rgb(Image.fromarray(img_arr)))
	return img_arr[:,:,:3]