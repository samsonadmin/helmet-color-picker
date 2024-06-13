import os, sys, io
import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageFilter, ImageOps, ImageColor
# import bbox_visualizer as bbv  # for annotation

# Paths
cwdir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, os.path.join(cwdir, "../"))
from toolbox.st_utils import show_carryyai_logo, get_image

def equalize_hist_color(image):
	# Convert the image to HSV color space
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

	# Apply histogram equalization to the V channel
	hsv[..., 2] = cv2.equalizeHist(hsv[..., 2])

	# Convert the image back to BGR color space
	return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def mod_display():
	with st.sidebar:
		with st.expander('Input Image', expanded = True):
			uploaded_files = st.file_uploader("Choose image file(s)", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
		with st.expander('Input Column', expanded=True):
			n_col = st.slider('Input number of column(s)', min_value=1, max_value=5, value=1)

	# Handle exception
	if not uploaded_files:
		st.warning(':point_left: please upload your image(s)')
		return
	
	# Write image numpy arrays into a list
	l_im_np = []
	for i in range(len(uploaded_files)):
		im_bytes = uploaded_files[i].read()
		im_np = np.array(Image.open(io.BytesIO(im_bytes)))
		l_im_np.append(im_np)

	# Apply histogram equalization to the input image
	# im = equalize_hist_color(im)
	
	# Main page
	with st.expander('HSV filter panel', expanded = True):
		l_col, m_col, r_col = st.columns(3)
		hue_range = l_col.slider('Select range for Hue', min_value = 0, max_value = 179, value = (0, 179), step = 1)
		saturation_range = m_col.slider('Select range for Saturation', min_value = 0, max_value = 255, value = (0, 255), step = 1)
		value_range = r_col.slider('Select range for Value', min_value = 0, max_value = 255, value = (0, 255), step = 1)
		lower_hsv = np.array([hue_range[0], saturation_range[0], value_range[0]])
		higher_hsv = np.array([hue_range[1], saturation_range[1], value_range[1]])

	with st.expander('Original Image', expanded = False):
		cols = st.columns(n_col)
		for i in range(len(l_im_np)):
			position = cols[i % n_col]
			im  = l_im_np[i]
			with position:
				st.image(im, channels = 'RGB') 

	with st.expander('Filtered Image', expanded = True):
		st.subheader(f'`H:S:V ::: [{hue_range[0]}, {saturation_range[0]}, {value_range[0]}] [{hue_range[1]}, {saturation_range[1]}, {value_range[1]}]`')
		cols = st.columns(n_col)
		for i in range(len(l_im_np)):
			position = cols[i % n_col]
			im  = l_im_np[i]
			# Convert color to hsv because it is easy to track colors in this color model
			im_hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
			# Apply the cv2.inrange method to create a mask
			im_mask = cv2.inRange(im_hsv, lower_hsv, higher_hsv)
			# Apply the mask on the image to extract the original color
			im = cv2.bitwise_and(im, im, mask = im_mask)
			with position:
				st.image(im, channels = 'RGB')

		# # Convert color to hsv because it is easy to track colors in this color model
		# im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
		# # Apply the cv2.inrange method to create a mask
		# im_mask = cv2.inRange(im_hsv, lower_hsv, higher_hsv)
		# # Apply the mask on the image to extract the original color
		# im = cv2.bitwise_and(im, im, mask = im_mask)
		# st.image(im, channels = 'BGR')


def Main():
	# Page config.
	carryai_icon = Image.open('carryai_favicon.ico')
	st.set_page_config(layout = 'wide', page_title = 'Image HSV color picker', page_icon = carryai_icon)
	show_carryyai_logo()
	with st.sidebar:
		st.header('Jetson: Image HSV color picker')
		with st.expander('Info'):
			st.info('''
				This tool is for picking colors in an image by adjusting HSV values.
			''')
	mod_display()
	

if __name__ == '__main__':
	Main()