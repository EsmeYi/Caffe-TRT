# Utility functions for building/saving/loading TensorRT Engine
import sys
import os

import tensorrt as trt
import pycuda.driver as cuda
import numpy as np


DTYPE = trt.float32
OUTPUT_NAME = 'cls_prob_1st'


# Allocate host and device buffers, and create a stream.
def allocate_buffers(engine):
    # Determine dimensions and create page-locked memory buffers (i.e. won't be swapped to disk) to hold host inputs/outputs.
    h_input = cuda.pagelocked_empty(
    	trt.volume(engine.get_binding_shape(0)) * engine.max_batch_size, 
    	dtype=trt.nptype(DTYPE))
    h_output = cuda.pagelocked_empty(
    	trt.volume(engine.get_binding_shape(1)) * engine.max_batch_size, 
    	dtype=trt.nptype(DTYPE))
    # Allocate device memory for inputs and outputs.
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    # Create a stream in which to copy inputs/outputs and run inference.
    stream = cuda.Stream()
    return h_input, d_input, h_output, d_output, stream


# Building 
def build_engine(model_file, deploy_file, trt_logger, batch_size=1, precison_mode='FP32'):
	DTYPE = trt.float32
	if precison_mode == 'FP16':
		DTYPE = trt.float16
	with trt.Builder(trt_logger) as builder, builder.create_network() as network, trt.CaffeParser() as parser:
		# Workspace size for building an engine.
		builder.max_workspace_size = 1 << 30
		# Max batch size
		builder.max_batch_size = batch_size
		print("Building TensorRT engine. This may take few minutes.")
		model_tensors = parser.parse(deploy=deploy_file, model=model_file, network=network, dtype=DTYPE)
		# Find the name of the output tensor in model_tensors.
		network.mark_output(model_tensors.find(OUTPUT_NAME))
		return builder.build_cuda_engine(network)

# Saving 
def save_engine(engine, engine_dest_path):
	buf = engine.serialize()
	with open(engine_dest_path, 'wb') as f:
		f.write(buf)

# Loading
def load_engine(trt_runtime, engine_path):
	with open(engine_path, 'rb') as f:
		engine_data = f.read()
	engine = trt_runtime.deserialize_cuda_engine(engine_data)
	return engine
