import sys
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
import struct
import keras
from keras.preprocessing import sequence
from keras.layers import Dense, LSTM
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import SimpleRNN
from keras.layers import Dense, Dropout, Embedding, LSTM, Input, Bidirectional
from keras.initializations import normal, identity
from keras.optimizers import RMSprop
from keras.utils import np_utils
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.models import save_model, load_model

def main(argv):
    if len(argv) <= 2:
        print('usage:\n\t'+argv[0]+' [src_model] [dst_data]')
        sys.exit(0)
    np.set_printoptions(threshold='nan')
    src_model = argv[1]
    dst_data = argv[2]
    print('----------------------loading model----------------------')
    model = load_model(src_model)
    
    arch = json.loads(model.to_json())

    with open(dst_data,'w') as f:
        f.write('layers ' + str(len(model.layers)) + '\n')
        layers = []
        for ind,layer in enumerate(arch['config']):
            class_name = layer['class_name']
            f.write('layer ' + str(ind) + ' ' + class_name + '\n')
            layers += [class_name]
            if class_name == 'Activation':
                f.write(layer['config']['activation']+'\n')
            elif class_name == 'Dense':
                weights = model.layers[ind].get_weights()
                assert(len(weights) == 2)
                weight = weights[0]
                bias = weights[1]
                f.write(str(weight.shape[0]) + ' ' + str(weight.shape[1]) + '\n')
                f.write('weight:\n')
                for row in weight:
                    for num in row:
                        f.write(str(num)+' ')
                f.write('\n')
                f.write('bias:\n')
                for num in bias:
                    f.write(str(num)+' ')
                f.write('\n')
            elif class_name == 'LSTM':
                layer_config = layer['config']
                inner_activation = layer_config['inner_activation']
                output_dim = layer_config['output_dim']
                input_dim = layer_config['input_dim']
                unroll = layer_config['unroll']
                activation = layer_config['activation']
                stateful = layer_config['stateful']
                go_backwards = layer_config['go_backwards']
                return_sequences = layer_config['return_sequences']
                ###########
                f.write('inner_activation : ' + str(inner_activation) + '\n')
                f.write('output_dim : ' + str(output_dim) + '\n')
                f.write('input_dim : ' + str(input_dim) + '\n')
                f.write('unroll : ' + str(unroll) + '\n')
                f.write('activation : ' + str(activation) + '\n')
                f.write('stateful : ' + str(stateful) + '\n')
                f.write('go_backwards : ' + str(go_backwards) + '\n')
                f.write('return_sequences : ' + str(return_sequences) + '\n')
                ###########
                weights = model.layers[ind].get_weights()
                assert(len(weights) == 12)
                begin = 0
                end = 12
                f.write('weight of input to hidden :\n')
                for i in range(begin,end):
                    weight = weights[i]
                    if (i-0) % 3 == 0:
                        for row in weight:
                            for num in row:
                                f.write(str(num)+' ')
                f.write('\n')
                f.write('weight of hidden to hidden :\n')
                for i in range(begin,end):
                    weight = weights[i]
                    if (i-1) % 3 == 0:
                        for row in weight:
                            for num in row:
                                f.write(str(num)+' ')
                f.write('\n')
                f.write('bias :\n')
                for i in range(begin,end):
                    weight = weights[i]
                    if (i-2) % 3 == 0:
                        for num in weight:
                            f.write(str(num)+' ')
                f.write('\n')

            elif class_name == 'Bidirectional':
                inner_layer = layer['config']['layer']
                inner_layer_config = inner_layer['config']
                inner_class = inner_layer['class_name']
                assert(inner_class == 'LSTM')
                f.write('inner_class : ' + inner_class+'\n')
                inner_activation = layer['config']['inner_activation']
                f.write('inner_activation : ' + inner_activation+'\n')
                merge_mode = layer['config']['merge_mode']
                f.write('merge_mode : ' + merge_mode+'\n')
                input_dim = inner_layer_config['input_dim']
                f.write('input_dim : ' + str(input_dim)+'\n')
                output_dim = inner_layer_config['output_dim']
                f.write('output_dim : ' + str(output_dim)+'\n')
                unroll = inner_layer_config['unroll']
                f.write('unroll : ' + str(unroll)+'\n')
                return_sequences = inner_layer_config['return_sequences']
                f.write('return_sequences : ' + str(return_sequences)+'\n')
                weights = model.layers[ind].get_weights()
                assert(len(weights) == 24)
                for id in range(2):
                    begin = 0
                    end = 12
                    if id == 1:
                        begin=12
                        end=24
                    f.write('weight of input to hidden :\n')
                    for i in range(begin,end):
                        weight = weights[i]
                        if (i-0) % 3 == 0:
                            for row in weight:
                                for num in row:
                                    f.write(str(num)+' ')
                    f.write('\n')
                    f.write('weight of hidden to hidden :\n')
                    for i in range(begin,end):
                        weight = weights[i]
                        if (i-1) % 3 == 0:
                            for row in weight:
                                for num in row:
                                    f.write(str(num)+' ')
                    f.write('\n')
                    f.write('bias :\n')
                    for i in range(begin,end):
                        weight = weights[i]
                        if (i-2) % 3 == 0:
                            for num in weight:
                                f.write(str(num)+' ')
                    f.write('\n')


if __name__ == '__main__':
    main(sys.argv)
