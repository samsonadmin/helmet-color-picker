import os, sys
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import urllib.request as urllib
from stqdm import stqdm
from itertools import cycle

#Paths
cwdir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, os.path.join(cwdir, "../"))
from toolbox.img_utils import get_pil_im, im_3c

def show_carryyai_logo(use_column_width = False, width = 200, st_asset = st.sidebar):
	logo_path = 'carryai-simple-dark.png'
	st_asset.image(logo_path, use_column_width = use_column_width, channels = 'BGR', output_format = 'PNG', width = width)

def get_image(st_asset = st.sidebar, as_np_arr = False, no_alpha = True,
    extension_list = ['jpg', 'jpeg', 'png'], accept_multiple_files = False,
    st_key = None, default_image_url = None, as_dict = False):
    '''
    return an image either as np array or PIL Image from an ST app
    '''
    image_path = None
    use_url = st_asset.checkbox('Use image URL',
                key = st_key + '_checkbox' if st_key else None,
                value = True if default_image_url else False)
    if use_url:
        image_path = st_asset.text_input("Enter Image URL",
                        key = st_key + '_textinput' if st_key else None,
                        value = default_image_url if default_image_url else '')
    else:
        image_fh = st_asset.file_uploader(label = "Upload your image",
                        type = extension_list,
                        accept_multiple_files = accept_multiple_files,
                        key = st_key + 'fileuploader' if st_key else None
                        )
        image_path = Image.open(image_fh) if image_fh else None
        image_filename = image_fh.name if image_fh else None
    im = get_pil_im(image_path) if image_path else None
    if use_url and image_path and isinstance(im, type(None)):
        st_asset.warning(f"Can't load image from: {image_path}")
    if im:
        im = Image.fromarray(im_3c(np.array(im))) if no_alpha else im
        im = np.array(im) if as_np_arr else im
    im_dict = {'im': im, 'name': image_path if use_url else image_filename}
    return im_dict if as_dict else im


def get_df_from_csv(st_asset, str_msg = "please provide your dataframe CSV",
        l_file_ext = ['csv'], str_help = None, **kwargs
    ):
    ''' return a pandas DF from a CSV provided by the user
    '''
    with st_asset:
        payload = st.file_uploader(label = str_msg, type = l_file_ext,
                    accept_multiple_files = False,
                    help = str_help)
    df = None
    if payload:
        df = pd.read_csv(payload, **kwargs)
    return df

def image_gallery(l_images, l_caption = None , l_text_md = None, im_width = 200, n_col = 3,
        tqdm_func = stqdm):

    l_caption = l_caption if l_caption else [None for im in l_images]
    l_text_md = l_text_md if l_text_md else [None for im in l_images]
    l_cols = st.columns(n_col)
    col_cycle = cycle(l_cols)

    for i in tqdm_func(range(len(l_images)), desc = "loading images"):
        with next(col_cycle):
            st.image(l_images[i], caption = l_caption[i],
                    width = im_width, use_column_width = False if im_width else True)
            if l_text_md[i]:
                st.markdown(l_text_md[i])
    return None