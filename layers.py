#custom l1 distance layer module : we need it to load the custom model

#dependencies
import tensorflow as tf 
from tensorflow.keras.layers import Layer

class L1Dist(Layer):
    
    # Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__()
       
    # Magic happens here - similarity calculation
    def call(self, input_embedding, validation_embedding):
        # Ensure we're working with the actual tensor values, not lists
        if isinstance(input_embedding, list):
            input_embedding = input_embedding[0]
        if isinstance(validation_embedding, list):
            validation_embedding = validation_embedding[0]
            
        return tf.math.abs(input_embedding - validation_embedding)