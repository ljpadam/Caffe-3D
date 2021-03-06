# -*- coding: utf-8 -*-
"""
Created on Fri May 27 21:45:04 2016

@author: ljp
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.cm as cm
import scipy.misc
import Image
import scipy.io
import os
import cv2


# Make sure that caffe is on the python path:
caffe_root = '../../'  # this file is expected to be in {caffe_root}/examples/hed/
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

with open('test.txt') as f:   
    test_lst_full=f.readlines()
test_lst=[x.split(' ')[0] for x in test_lst_full]
start_lst=[x.split(' ')[1] for x in test_lst_full]
j=0
# load net 
model_root = '/home/ljp/code/caffe-3d/examples/cmr/'
net = caffe.Net('/home/ljp/code/caffe-3d/examples/cmr/test.prototxt', '/home/ljp/code/caffe-3d/examples/cmr/net/19⁄06 best/multi/89/cmr_iter_2400.caffemodel', caffe.TEST)
caffe.set_mode_gpu()
caffe.set_device(0)
depth=200
k=0
for i in test_lst:
    print i
    im_lst=[]
    for j in range(int(start_lst[k]),int(start_lst[k])+depth):
        #print i+'/img_'+str(j)+'.png'
        im=Image.open(i+'/img_'+str(j)+'.png')
        in_=np.array(im,dtype=np.float32)
        #in_=in_.transpose()
        im_lst.append(in_)
    
    volume = np.ones([1,depth,in_.shape[0],in_.shape[1]],np.float32)
    final = np.zeros([1,depth,in_.shape[0],in_.shape[1]],np.float32)
    for j in range(0,depth):
        volume[0,j,:,:]=im_lst[j]
	# shape for input (data blob is N x C x H x W), set data
    
    volume=volume/10
    net.forward()
    out1 = net.blobs['softmax'].data[0]
    temp=out1[1]+out1[2]
    print temp.max()
    out1=out1*255.0
    temp=out1[1]+out1[2]
    print temp.max()
    if i[-2]>='0' and i[-2]<='9':
	num=i[-2]+i[-1]
    else:
	num=i[-1]
    if not os.path.isdir('./result/pro/'+num):
	os.mkdir('./result/pro/'+num)
    if not os.path.isdir('./result/pro/'+num+'/class1'):
	os.mkdir('./result/pro/'+num+'/class1/')
    if not os.path.isdir('./result/pro/'+num+'/class2'):
	os.mkdir('./result/pro/'+num+'/class2/')
    #volume=volume/65535*255
    for j in range(int(start_lst[k]),int(start_lst[k]) + depth):
        cv2.imwrite('./result/pro/'+num+'/class1/'+str(j)+'.png',out1[1,j-int(start_lst[k]),:,:])
        cv2.imwrite('./result/pro/'+num+'/class2/'+str(j)+'.png',out1[2,j-int(start_lst[k]),:,:])
    k=k+1

    