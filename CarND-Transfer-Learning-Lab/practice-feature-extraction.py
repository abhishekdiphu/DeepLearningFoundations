import pickle 
import tensorflow as tf
import numpy as np
from keras.layers import Input, Flatten, Dense
from keras.models import Model



flags =tf.app.flags
FLAGS =flags.FLAGS

flags.DEFINE_string('training_file', '', "Bottleneck features training file (.p)")
flags.DEFINE_string('validation_file', '', "Bottleneck features validation file (.p)")
flags.DEFINE_integer('epochs', 50, "The number of epochs.")
flags.DEFINE_integer('batch_size', 256, "The batch size.")


def load_bottleneck_data(training_file, validation_file):
  
        
    print("Training_files" , training_file)
    print("Validation files" , validation_file)
    
    
    with open(training_file, 'rb') as   f:
        train_data = pickle.load(f)
        
    with open(validation_file, 'rb') as f:
        validation_data = pickle.load(f)
        
        
    x_train = train_data['features']
    y_train = train_data['labels']
    
    x_valid = validation_data['features']
    y_valid = validation_data['labels']
    
    
    return x_train, y_train, x_valid, y_valid


def main(_):
    
    #load the bottleneck data:
    x_train, y_train, x_valid, y_valid = load_bottleneck_data(training_file =FLAGS.training_file, \
                                                              validation_file =FLAGS.validation_file)
    
    
    print(x_train.shape, y_train.shape)
    print(x_valid.shape, y_valid.shape)
    
    
    nb_classes = len(np.unique(y_train))
    
    
    
    ##defining  model defination:
    
    input_shape = x_train.shape[1:]
    inp = Input(shape = input_shape)
    x   = Flatten()(inp)
    x   = Dense(nb_classes , activation ='softmax')(x)
    
    model = Model(inp , x) 
    model.compile(optimizer = 'adam' , loss = 'sparse_categorical_crossentropy' , metrics =['accuracy'])
    
    #train model
    
    model.fit(x_train , y_train ,FLAGS.batch_size  , FLAGS.epochs , validation_data=(x_valid , 
                                                                                     y_valid) , shuffle =True )
    
    
    
    

    
if __name__ == '__main__':
    tf.app.run()
    
    
    
    
    
    
    