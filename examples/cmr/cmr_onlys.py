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
net = caffe.Net('/home/ljp/code/caffe-3d/examples/cmr/test.prototxt', '/home/ljp/hd1/caffemodel/caffe3d/cmr_iter_9000.caffemodel', caffe.TEST)
caffe.set_mode_gpu()
caffe.set_device(0)
depth=200
k=0
for i in test_lst:
    print i
    im_lst=[]
    for j in range(int(start_lst[k]),int(start_lst[k])+depth):
        #print i+'/img_'+str(j)+'.png'
        print i+'/img_'+str(j)+'.png'
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
    #net.blobs['data'].reshape(1,*volume.shape)
    
    #net.blobs['data'].data[...] = volume
    #volume=net.blobs['data'].data[0]  
    #print volume[0,1,:,:].max()
    net.forward()
    	#print net.blobs['accuracy5'].data
    	#out1 = net.blobs['sigmoid-fuse'].data[0][1,:,:]
    out1 = net.blobs['softmax'].data[0]
    out1=out1*255
    #out1[out1<0.5]=255
    #out1[out1>=0.5]=0
    
    #out1=out1*255; 
    volume=net.blobs['data'].data[0]
    print net.blobs['data'].data.max()
    #print volume.dtype
    #volume=volume
    #print volume[0,1,:,:]
    #print volume.shape
    if i[-2]>='0' and i[-2]<='9':
	num=i[-2]+i[-1]
    else:
	num=i[-1]
    if not os.path.isdir('./result/'+num):
	os.mkdir('./result/'+num)
    #volume=volume/65535*255
    for j in range(int(start_lst[k]),int(start_lst[k]) + depth):
	#print '/result/'+i[-1]+'/'+str(j)+'.png'
        #print i[-1]
        #print out1.shape
        print j
        cv2.imwrite('./result/'+num+'/'+str(j)+'.png',out1[1,j-int(start_lst[k]),:,:])
        #cv2.imwrite('./result/'+i[-1]+'/'+str(j)+'.png',volume[0,j-int(start_lst[k]),:,:])
        #print volume[0,j-int(start_lst[k]),:,:].max()
    #prediction[prediction>=0.5]=255;
    #prediction[prediction<0.5]=0
    #prediction=prediction*255
    k=k+1

    