import os
import glob
import argparse
import time
from PIL import Image
import tensorrt as trt

import caffe_inference as inference_utils

def preprocess_voc(voc_root):
    """Resizes all VOC images to 300x300 and saves them into .ppm files."""
    voc_jpegs = glob.glob(os.path.join(voc_root, 'JPEGImages', '*.jpg'))
    voc_ppms = glob.glob(os.path.join(voc_root, 'PPMImages', '*.ppm'))
    voc_ppm_img_path = os.path.join(voc_root, 'PPMImages', '{}.ppm')
    # Check if preprocessing is needed by comparing
    # image names between JPEGImages and PPMImages
    voc_jpegs_basenames = \
        [os.path.splitext(os.path.basename(p))[0] for p in voc_jpegs]
    voc_ppms_basenames = \
        [os.path.splitext(os.path.basename(p))[0] for p in voc_ppms]
    # If lists are not the same, preprocessing is needed
    if sorted(voc_jpegs_basenames) != sorted(voc_ppms_basenames):
        print("Preprocessing VOC dataset. It may take few minutes.")
        # Make PPMImages directory if it doesn't exist
        voc_ppms_path = voc_ppm_img_path
        if not os.path.exists(os.path.dirname(voc_ppms_path)):
            os.makedirs(os.path.dirname(voc_ppms_path))
        # For each .jpg file, make a resized
        # .ppm copy to fit model input expectations
        for voc_jpeg_path in voc_jpegs:
            voc_jpeg_basename = os.path.basename(voc_jpeg_path)
            voc_ppm_path = voc_ppms_path.format(
                os.path.splitext(voc_jpeg_basename)[0])
            if not os.path.exists(voc_ppm_path):
                img_pil = Image.open(voc_jpeg_path)
                img_pil = img_pil.resize(
                    size=(300,300),
                    resample=Image.BILINEAR
                )
                img_pil.save(voc_ppm_path)

def get_engine_path(precision, max_batch_size):
	return os.path.join('./engines/',str(precision),'engine_bs_{}.buf'.format(max_batch_size))

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-mf', '--model_file', type=str)
	parser.add_argument('-df', '--deploy_file', type=str)
	parser.add_argument('-p', '--precision', type=str, choices=['FP32', 'FP16'], default='FP32')
	parser.add_argument('-b', '--max_batch_size', type=int, default=32)
	parser.add_argument('-d', '--voc_dir', type=str, default='./VOCdevkit/VOC2007')
	args = parser.parse_args()
	model_file = args.model_file
	deploy_file = args.deploy_file
	precision = args.precision
	max_batch_size = args.max_batch_size
	voc_dir = args.voc_dir

	# Prepare image dataset
	voc_image_set_path = os.path.join(voc_dir, 'ImageSets', 'Main', 'test.txt')
	voc_jpg_img_path = os.path.join(voc_dir, 'JPEGImages', '{}.jpg')
	voc_ppm_img_path = os.path.join(voc_dir, 'PPMImages', '{}.ppm')
	preprocess_voc(voc_dir)
	with open(voc_image_set_path, 'r') as f:
		voc_image_numbers = f.readlines()
		image_numbers = [line.strip() for line in voc_image_numbers]
	image_path = voc_ppm_img_path
	total_imgs = len(image_numbers)

	# Prepare TRT engine path
	trt_engine_path = get_engine_path(precision, max_batch_size)
	if not os.path.exists(os.path.dirname(trt_engine_path)):
		os.makedirs(os.path.dirname(trt_engine_path))

	# Start measuring time
	inference_start_time = time.time()
	# Initialize inference model
	trt_inference_wrapper = inference_utils.TRTInference(\
		model_file, deploy_file, trt_engine_path, precison_mode="FP32", batch_size=1)
	# Inference with batching
	for idx in range(0, len(image_numbers), batch_size):
		print("Infering image {}/{}".format(idx+1, total_imgs))
		imgs = image_numbers[idx:idx+batch_size]
		image_paths = [image_path.format(img) for img in imgs]
		pred = trt_inference_wrapper.infer_batch(image_paths)

	# Output total [img load + inference] time
	print("Total time taken for inference: {} ms\n".format(int(round((time.time() - inference_start_time) * 1000))))


if __name__ == '__main__':
	main()
