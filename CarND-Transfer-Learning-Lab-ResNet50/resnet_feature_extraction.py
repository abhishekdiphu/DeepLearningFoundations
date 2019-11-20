import pickle
import tensorflow as tf
import numpy as np 
from keras.layers import Input , Flatten, Dense
from keras.models import Model


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('training_file', '', "Bottle features training file (.p) ")
flags.DEFINE_string('validation_file', '', "Bottle neck features validation file (.p)")

flags.DEFINE_string('epochs' , 50 , "the number of epochs")
flags.DEFINE_string('batch_size' , 256 , "the batch size")



def load_bo