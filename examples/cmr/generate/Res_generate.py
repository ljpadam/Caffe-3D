#!/usr/bin/env python
"""
Generate the residule learning network.
Author: Yemin Shi
Email: shiyemin@pku.edu.cn

MSRA Paper: http://arxiv.org/pdf/1512.03385v1.pdf
"""
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def parse_args():
    """Parse input arguments
    """

    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('solver_file',
                        help='Output solver.prototxt file')
    parser.add_argument('train_val_file',
                        help='Output train_val.prototxt file')
    parser.add_argument('--layer_number', nargs='*',
                        help=('Layer number for each layer stage.'),
                        default=[5, 5])
    parser.add_argument('-t', '--type', type=int,
                        help=('0 for deploy.prototxt, 1 for train_val.prototxt.'),
                        default=1)

    args = parser.parse_args()
    return args

def generate_data_layer():
    data_layer_str = '''name: "ResNet"
layer {
  name: "data"
  type: "VideoDataSeg"
  top: "data"
  top: "label"
  include {
    phase: TRAIN
  }
  video_data_param {
    root_folder: ""
    source: "./smalltrain.txt"
    batch_size: 1
    shuffle: true
    new_length: 30
  }
}
layer {
  name: "data"
  type: "VideoDataSeg"
  top: "data"
  top: "label"
  include {
    phase: TEST
  }
  video_data_param {
    root_folder: ""
    source: "./smalltrain.txt"
    #Just setup the network. No real online testing
    batch_size: 1
    shuffle: false
    new_length: 30
  }
}
'''
    return data_layer_str

def generate_conv_layer(kernel_size, kernel_num, stride, pad, layer_name, bottom, top, filler="msra"):
    conv_layer_str = '''layer {
  name: "%s"
  type: "Convolution"
  bottom: "%s"
  top: "%s"
  convolution_param {
    num_output: %s
    kernel_size:%d
    stride:%d
    pad:%d
    bias_term: false
    engine:CAFFE
    weight_filler {
      type: "%s"
    }
  }
}
'''%(layer_name, bottom, top, kernel_num, kernel_size, stride,pad, filler)
    return conv_layer_str
  
  
def generate_deconv_layer(name, bottom, top, num, filter='thrlinear'):
    conv_layer_str = '''layer {
  name: "%s"
  type: "Deconvolution"
  bottom: "%s"
  top: "%s"
  convolution_param {
    num_output: %d
    kernel_size:4
    stride:2
    pad:1
    bias_term: false
    engine:CAFFE
    weight_filler {
      type: "%s"
    }
    group:%d
  }
  param { lr_mult: 0 decay_mult: 0 }
}
'''%(name, bottom, top, num, filter, num)
    return conv_layer_str


def generate_pooling_layer(kernel_size, stride, pool_type, layer_name, bottom, top):
    pool_layer_str = '''layer {
  name: "%s"
  type: "NdPooling"
  bottom: "%s"
  top: "%s"
  pooling_param {
    pool: %s
    kernel_shape { dim: %d dim: %d dim: %d }
    stride_shape { dim: %d dim: %d dim: %d }
  }
}
'''%(layer_name, bottom, top, pool_type, kernel_size, kernel_size, kernel_size, stride, stride, stride)
    return pool_layer_str

def generate_fc_layer(num_output, layer_name, bottom, top, filler="msra"):
    fc_layer_str = '''layer {
  name: "%s"
  type: "InnerProduct"
  bottom: "%s"
  top: "%s"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  inner_product_param {
     num_output: %d
     weight_filler {
       type: "%s"
       std: 0.001
     }
     bias_filler {
       type: "constant"
       value: 0
     }
  }
}
'''%(layer_name, bottom, top, num_output, filler)
    return fc_layer_str

def generate_eltwise_layer(layer_name, bottom_1, bottom_2, top, op_type="SUM"):
    eltwise_layer_str = '''layer {
  name: "%s"
  type: "Eltwise"
  bottom: "%s"
  bottom: "%s"
  top: "%s"
  eltwise_param {
    operation: %s
  }
}
'''%(layer_name, bottom_1, bottom_2, top, op_type)
    return eltwise_layer_str

def generate_activation_layer(layer_name, bottom, top, act_type="ReLU"):
    act_layer_str = '''layer {
  name: "%s"
  type: "%s"
  bottom: "%s"
  top: "%s"
}
'''%(layer_name, act_type, bottom, top)
    return act_layer_str

def generate_softmax_loss(bottom):
    softmax_loss_str = '''layer {
  name: "loss"
  type: "SoftmaxWithLoss"
  bottom: "%s"
  bottom: "label"
  top: "loss"
}
layer {
  name: "accuracy5"
  type: "Accuracy"
  bottom: "%s"
  bottom: "label"
  top: "accuracy5"
}
'''%(bottom, bottom)
    return softmax_loss_str

def generate_bn_layer(layer_name, bottom, top):
    bn_layer_str = '''layer {
  name: "%s"
  type: "BatchNorm"
  bottom: "%s"
  top: "%s"
  batch_norm_param { use_global_stats: false }
  param {
    lr_mult: 0
  }
  param {
    lr_mult: 0
  }
  param {
    lr_mult: 0
  }
}
layer {
	bottom: "%s"
	top: "%s"
	name: "scale_%s"
	type: "Scale"
	scale_param {
		bias_term: true
	}
}
'''%(layer_name, bottom, top, top, top ,top)
    return bn_layer_str


'''conv: kernel_size, kernel_num, stride, pad, layer_name, bottom, top, filler="msra"
    bn:  layer_name, bottom, top
activation:layer_name, bottom, top, act_type="ReLU"
eltwise: layer_name, bottom_1, bottom_2, top, op_type="SUM"
deconv: kernel_num, layer_name, bottom, top, filler="thrlinear"
pool: kernel_size, stride, pool_type, layer_name, bottom, top
'''
def generate_train_val():
    args = parse_args()
    network_str = generate_data_layer()
    '''before stage'''
    last_output = 'data'
    network_str += generate_conv_layer(3, 128, 1, 1, 'conv1', last_output, 'conv1')
    network_str += generate_bn_layer('conv1_bn', 'conv1', 'conv1')
    network_str += generate_activation_layer('poo1_relu', 'conv1', 'conv1', 'ReLU')
    
    network_str += generate_conv_layer(3, 128, 1, 1, 'conv2', 'conv1', 'conv2')
    network_str += generate_bn_layer('conv2_bn', 'conv2', 'conv2')
    network_str += generate_activation_layer('conv2_relu', 'conv2', 'conv2', 'ReLU')
    '''stage 1'''
    last_output = 'conv2'
    for l in xrange(1, args.layer_number[0]+1):
	onum=128
        if l==1:
            network_str += generate_conv_layer(3, onum, 1, 1, 'conv2_%d_1'%l, last_output, 'conv2_%d_1'%l)
        else:
            network_str += generate_bn_layer('conv2_%d_0_bn'%l, last_output, 'conv2_%d_0_bn'%l)
            network_str += generate_activation_layer('conv2_%d_0_relu'%l, 'conv2_%d_0_bn'%l, 'conv2_%d_0_bn'%l, 'ReLU')
            network_str += generate_conv_layer(3, onum, 1, 1, 'conv2_%d_1'%l, 'conv2_%d_0_bn'%l, 'conv2_%d_1'%l)
        network_str += generate_bn_layer('conv2_%d_1_bn'%l, 'conv2_%d_1'%l, 'conv2_%d_1'%l)
        network_str += generate_activation_layer('conv2_%d_1_relu'%l, 'conv2_%d_1'%l, 'conv2_%d_1'%l, 'ReLU')
        network_str += generate_conv_layer(3, onum, 1, 1, 'conv2_%d_2'%l, 'conv2_%d_1'%l, 'conv2_%d_2'%l)
        network_str += generate_eltwise_layer('conv2_%d_sum'%l, last_output, 'conv2_%d_2'%l, 'conv2_%d_sum'%l, 'SUM') 
        last_output = 'conv2_%d_sum'%l 
    '''stage 2'''
    network_str += generate_bn_layer('pool2_bn', last_output, last_output)
    network_str += generate_activation_layer('pool2_relu', last_output, last_output, 'ReLU')
    network_str += generate_conv_layer(1, 128, 1, 0, 'conv3', last_output, 'conv3')
    last_output = 'conv3'
    for l in xrange(1, args.layer_number[1]+1):
	onum=128
        if l==1:
            network_str += generate_conv_layer(3, onum, 1, 1, 'conv3_%d_1'%l, last_output, 'conv3_%d_1'%l)
        else:
            network_str += generate_bn_layer('conv3_%d_0_bn'%l, last_output, 'conv3_%d_0_bn'%l)
            network_str += generate_activation_layer('conv3_%d_0_relu'%l, 'conv3_%d_0_bn'%l, 'conv3_%d_0_bn'%l, 'ReLU')
            network_str += generate_conv_layer(3, onum, 1, 1, 'conv3_%d_1'%l, 'conv3_%d_0_bn'%l, 'conv3_%d_1'%l)
        network_str += generate_bn_layer('conv3_%d_1_bn'%l, 'conv3_%d_1'%l, 'conv3_%d_1'%l)
        network_str += generate_activation_layer('conv3_%d_1_relu'%l, 'conv3_%d_1'%l, 'conv3_%d_1'%l, 'ReLU')
        network_str += generate_conv_layer(3, onum, 1, 1, 'conv3_%d_2'%l, 'conv3_%d_1'%l, 'conv3_%d_2'%l)
        network_str += generate_eltwise_layer('conv3_%d_sum'%l, last_output, 'conv3_%d_2'%l, 'conv3_%d_sum'%l, 'SUM')
        last_output = 'conv3_%d_sum'%l
    '''deconv'''
    network_str += generate_conv_layer(1, 128, 1, 0, 'conv4', last_output, 'conv4')
    network_str += generate_conv_layer(1, 128, 1, 0, 'conv5', 'conv4', 'conv5')
    network_str += generate_conv_layer(1, 3, 1, 0, 'conv6', 'conv5', 'conv6')
    network_str += generate_softmax_loss('conv6')
    return network_str

def generate_solver(train_val_name):
    solver_str = '''net: "%s"
test_iter: 1000
test_interval: 6000
test_initialization: false
display: 60
base_lr: 0.1
lr_policy: "multistep"
stepvalue: 300000
stepvalue: 500000
gamma: 0.1
max_iter: 600000
momentum: 0.9
weight_decay: 0.0001
snapshot: 6000
snapshot_prefix: "pku_resnet"
solver_mode: GPU
device_id: 1'''%(train_val_name)
    return solver_str

def main():
    args = parse_args()
    solver_str = generate_solver(args.train_val_file)
    network_str = generate_train_val()
    fp = open(args.solver_file, 'w')
    fp.write(solver_str)
    fp.close()
    fp = open(args.train_val_file, 'w')
    fp.write(network_str)
    fp.close()

if __name__ == '__main__':
    main()
