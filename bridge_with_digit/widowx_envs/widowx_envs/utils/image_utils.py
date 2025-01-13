import cv2
import moviepy.editor as mpy
import os
import numpy as np
from PIL import Image, ImageDraw
from PIL import Image

def resize_store(t, target_array, input_array):
    target_img_height, target_img_width = target_array.shape[2:4]

    if (target_img_height, target_img_width) == input_array.shape[1:3]:
        for i in range(input_array.shape[0]):
            target_array[t, i] = input_array[i]
    else:
        for i in range(input_array.shape[0]):
            target_array[t, i] = cv2.resize(input_array[i], (target_img_width, target_img_height),
                                            interpolation=cv2.INTER_AREA)


def npy_to_gif(im_list, filename, fps=4):
    save_dir = '/'.join(str.split(filename, '/')[:-1])

    if not os.path.exists(save_dir):
        print('creating directory: ', save_dir)
        os.makedirs(save_dir)

    clip = mpy.ImageSequenceClip(im_list, fps=fps)
    clip.write_gif(filename + '.gif')


def npy_to_mp4(im_list, filename, fps=4):
    save_dir = '/'.join(str.split(filename, '/')[:-1])

    if not os.path.exists(save_dir):
        print('creating directory: ', save_dir)
        os.mkdir(save_dir)

    clip = mpy.ImageSequenceClip(im_list, fps=fps)
    clip.write_videofile(filename + '.mp4')

def draw_text_image(text, background_color=(255,255,255), image_size=(30, 64), dtype=np.float32):

    text_image = Image.new('RGB', image_size[::-1], background_color)
    draw = ImageDraw.Draw(text_image)
    if text:
        draw.text((4, 0), text, fill=(0, 0, 0))
    if dtype == np.float32:
        return np.array(text_image).astype(np.float32)/255.
    else:
        return np.array(text_image)


def draw_text_onimage(text, image, color=(255, 0, 0)):
    if image.dtype == np.float32:
        image = (image*255.).astype(np.uint8)
    assert image.dtype == np.uint8
    text_image = Image.fromarray(image)
    draw = ImageDraw.Draw(text_image)
    draw.text((4, 0), text, fill=color)
    return np.array(text_image).astype(np.float32)/255.
