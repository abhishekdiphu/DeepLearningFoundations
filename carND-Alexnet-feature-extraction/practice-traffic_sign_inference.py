"""
The traffic signs are 32x32 so you
have to resize them to be 227x227 before
passing them to AlexNet.
"""

import time 
import tensorflow as tf
import numpy as np
from matplotlib.pyplot import imread
from caffe_classes import class_names 
from alexnet import AlexNet




##creating the placeholders 
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
resized = tf.image.resize_images(x,(227,227))

probs = AlexNet(resized)
init  = tf.global_variables_initializer()
sess  = tf.Session()
sess.run(init)


## reading the images 

im1 = imread("construction.jpg").astype(np.float32)
im1 = im1 - np.mean(im1)


im2 = imread("stop.jpg").astype(np.float32)
im2 = im2 - np.mean(im2)



##lets run the infernence 
t = time.time()

output= sess.run(probs , feed_dict ={ x : [im1 , im2]})


##print output 

for  input_im_ind in range(output.shape[0]):
    inds = np.argsort(output[input_im_ind, :])
    print("images" , input_im_ind)
    
    for i in range(5):
        print("%s: %.3f" % (class_names[inds[-1 - i]], output[input_im_ind, inds[-1 - i]]))
    print()
print("Time: %.3f seconds" % (time.time() - t))










