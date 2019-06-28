from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import random
import pprint
import sys
import time
import numpy as np
from optparse import OptionParser
import pickle
import math
import cv2
import copy
from matplotlib import pyplot as plt
import tensorflow as tf
import pandas as pd
import os

from sklearn.metrics import average_precision_score

from keras import backend as K
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, Dropout
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, TimeDistributed
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.objectives import categorical_crossentropy
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.utils import generic_utils
from keras.engine import Layer, InputSpec
from keras import initializers, regularizers
import argparse
from utils import *
##############     some constant parameters   ###############################

lambda_rpn_regr = 1.0
lambda_rpn_class = 1.0

lambda_cls_regr = 1.0
lambda_cls_class = 1.0

epsilon = 1e-4

##########################################################################

def parse_arguments():
    parser = argparse.ArgumentParser(description='Weights path, image_path, results_path')
    
    parser.add_argument(
        "--base_path",
        type = str,
        help = "Base path. ",
        default='/data/zhangshihao/Faster_RCNN_for_Open_Images_Dataset_Keras/'
    )
    parser.add_argument(
        "--train_path",
        type = str,
        help = "Path to save detection result. annotation.txt",
        default = 'annotation.txt'
    )
    
    parser.add_argument(
        "--base_weights_path",
        type = str,
        help = "Path to load resnet50 model parameters. ",
        default = 'model/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
    )
    parser.add_argument(
        "--gpu",
        type = str,
        help = "Which gpu will you use",
        default = '4'
    
    )

    parser.add_argument(

        "--type",
        type = str,
        help = "vgg, resnet50,vgg_deform,resnet50_deform",
        default = "vgg"
    )
    parser.add_argument(
    
        "--config_output_filename",
        type = str,
        help = "Config file ",
        default = 'resnet50_model_vgg_config.pickle'
    )
    parser.add_argument(
    
        "--save_path",
        type=str,
        help="File save path. ",
        default='resnet50_model/'
        )
    parser.add_argument(
    
        "--pre_trained_model_weights",
        type=str,
        help="Pre-trained model weights, :::resnet50_model/fiveScale_frcnn_vgg_epoch_0.hdf5",
        default='resnet50_model/fiveScale_frcnn_vgg_epoch_0.hdf5'
    
    
    
    )
    parser.add_argument(
    
        "--train_size",
        type=int,
        help="How mush sample will be used. ",
        default=10000


    )
    
    return parser.parse_args()





if __name__ == '__main__':
    
    
    args = parse_arguments()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    #base_path = 'drive/My Drive/AI/Faster_RCNN'
    base_path = args.base_path
    train_path =  args.train_path # Training data (annotation file)

    num_rois = 4 # Number of RoIs to process at once.

    # Augmentation flag
    horizontal_flips = True # Augment with horizontal flips in training. 
    vertical_flips = True   # Augment with vertical flips in training. 
    rot_90 = True           # Augment with 90 degree rotations in training. 

    output_weight_path = os.path.join(base_path, args.pre_trained_model_weights)

    record_path = os.path.join(base_path, args.save_path+'record.csv') # Record data (used to save the losses, classification accuracy and mean average precision)

    base_weight_path = os.path.join(base_path, args.base_weights_path)

    config_output_filename = os.path.join(base_path, args.config_output_filename)
    
    
    # Create the config
    C = Config()

    C.use_horizontal_flips = horizontal_flips
    C.use_vertical_flips = vertical_flips
    C.rot_90 = rot_90

    C.record_path = record_path
    C.model_path = output_weight_path
    C.num_rois = num_rois

    C.base_net_weights = base_weight_path

    #--------------------------------------------------------#
    # This step will spend some time to load the data        #
    #--------------------------------------------------------#
    st = time.time()
    train_imgs, classes_count, class_mapping = get_data(train_path)
    print()
    print('Spend %0.2f mins to load the data' % ((time.time()-st)/60) )

    if 'bg' not in classes_count:
        classes_count['bg'] = 0
        class_mapping['bg'] = len(class_mapping)
    # e.g.
    #    classes_count: {'Car': 2383, 'Mobile phone': 1108, 'Person': 3745, 'bg': 0}
    #    class_mapping: {'Person': 0, 'Car': 1, 'Mobile phone': 2, 'bg': 3}
    C.class_mapping = class_mapping

    print('Training images per class:')
    pprint.pprint(classes_count)
    print('Num classes (including bg) = {}'.format(len(classes_count)))
    print(class_mapping)

    # Save the configuration
    with open(config_output_filename, 'wb') as config_f:
        pickle.dump(C,config_f)
        print('Config has been written to {}, and can be loaded when testing to ensure correct results'.format(config_output_filename))
    # Shuffle the images with seed
    random.seed(1)
    random.shuffle(train_imgs)

    print('Num train samples (images) {}'.format(len(train_imgs)))
    ################   get data generater...............#######################################
    data_gen_train = get_anchor_gt(train_imgs, C, get_img_output_length, mode='train')
    #X, Y, image_data, debug_img, debug_num_pos = next(data_gen_train)
    ####################   Build the model    ###################################################
    #input_shape_img = (None,None,3)
    #input_shape_img = (864,640,3)
    #img_input = Input(shape=input_shape_img)
    roi_input = Input(shape=(None,4))
    # define the base network (VGG here, can be Resnet50, Inception, etc)
    if args.type == 'vgg':
        input_shape_img = (None,None,3)
        img_input = Input(shape=input_shape_img)
        shared_layers = nn_base(img_input, trainable=True)
    if args.type == 'resnet':
        input_shape_img = (None,None,3)
        img_input = Input(shape=input_shape_img)
        shared_layers = res_nn(inputs=img_input)
    if args.type == 'vgg_deform':
        input_shape_img = (864,640,3)
        img_input = Input(shape=input_shape_img)
        
        shared_layers = nn_base_deform(img_input, trainable=True)
    if args.type == 'resnet101':
        input_shape_img = (None,None,3)
        img_input = Input(shape=input_shape_img)
        shared_layers = resnet101(img_input)
    if args.type == 'resnet50_deform':
        input_shape_img = (1280,800,3)
        img_input = Input(shape=input_shape_img)
        shared_layers = resnet50_deform(img_input)    
    #img_input, shared_layers = res_nn(input=input_shape_img, trainable=True)
    ###  define the RPN, build on the base layers

    # define the RPN, built on the base layers
    num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios) # 9
    rpn = rpn_layer(shared_layers, num_anchors)

    classifier = classifier_layer(shared_layers, roi_input, C.num_rois, nb_classes=len(classes_count))

    model_rpn = Model(img_input, rpn[:2])
    model_classifier = Model([img_input, roi_input], classifier)

    # this is a model that holds both the RPN and the classifier, used to load/save weights for the models
    model_all = Model([img_input, roi_input], rpn[:2] + classifier)

    # Because the google colab can only run the session several hours one time (then you need to connect again), 
    # we need to save the model and load the model to continue training
    ### .        load vgg-16 weights...................#################################
    if not os.path.isfile(C.model_path):
        #If this is the begin of the training, load the pre-traind base network such as vgg-16
        try:
            print('This is the first time of your training')
            print('loading weights from {}'.format(C.base_net_weights))
            model_rpn.load_weights(C.base_net_weights, by_name=True)
            model_classifier.load_weights(C.base_net_weights, by_name=True)
        except:
            print('Could not load pretrained model weights. Weights can be found in the keras application folder \
                https://github.com/fchollet/keras/tree/master/keras/applications')
        
        # Create the record.csv file to record losses, acc and mAP
        record_df = pd.DataFrame(columns=['mean_overlapping_bboxes', 'class_acc', 'loss_rpn_cls', 'loss_rpn_regr', 'loss_class_cls', 'loss_class_regr', 'curr_loss', 'elapsed_time', 'mAP'])
    else:
        # If this is a continued training, load the trained model from before
        print('Continue training based on previous trained model')
        print('Loading weights from {}'.format(C.model_path))
        model_rpn.load_weights(C.model_path, by_name=True)
        model_classifier.load_weights(C.model_path, by_name=True)
        
        # Load the records
        record_df = pd.read_csv(record_path)

        r_mean_overlapping_bboxes = record_df['mean_overlapping_bboxes']
        r_class_acc = record_df['class_acc']
        r_loss_rpn_cls = record_df['loss_rpn_cls']
        r_loss_rpn_regr = record_df['loss_rpn_regr']
        r_loss_class_cls = record_df['loss_class_cls']
        r_loss_class_regr = record_df['loss_class_regr']
        r_curr_loss = record_df['curr_loss']
        r_elapsed_time = record_df['elapsed_time']
        r_mAP = record_df['mAP']

        print('Already train %dw batches'% (len(record_df)))
    ##### .       define optimizer,,,,,,,,,,,,,,,,,,,,,,,,,,,,###############################
    #optimizer = Adam(lr=0.000125)
    #optimizer_classifier = Adam(lr=0.000125)
    optimizer = Adam(lr=1e-5)
    optimizer_classifier = Adam(lr=1e-5)
    model_rpn.compile(optimizer=optimizer, loss=[rpn_loss_cls(num_anchors), rpn_loss_regr(num_anchors)])
    model_classifier.compile(optimizer=optimizer_classifier, loss=[class_loss_cls, class_loss_regr(len(classes_count)-1)], metrics={'dense_class_{}'.format(len(classes_count)): 'accuracy'})
    model_all.compile(optimizer='sgd', loss='mae')

    # Training setting
    
    total_epochs = len(record_df)
    r_epochs = len(record_df)

    epoch_length = args.train_size
    num_epochs = 50
    iter_num = 0

    total_epochs += num_epochs

    losses = np.zeros((epoch_length, 5))
    rpn_accuracy_rpn_monitor = []
    rpn_accuracy_for_epoch = []

    if len(record_df)==0:
        best_loss = np.Inf
    else:
        best_loss = np.min(r_curr_loss)
    print(len(record_df))
    #############################################       Begain      training      ################################
    start_time = time.time()
    for epoch_num in range(num_epochs):

        progbar = generic_utils.Progbar(epoch_length)
        print('Epoch {}/{}'.format(r_epochs + 1, total_epochs))
        C.model_path = args.save_path + 'fiveScale_frcnn_vgg_epoch_{:03d}.hdf5'.format(epoch_num)
        r_epochs += 1

        while True:
            try:

                if len(rpn_accuracy_rpn_monitor) == epoch_length and C.verbose:
                    mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor))/len(rpn_accuracy_rpn_monitor)
                    rpn_accuracy_rpn_monitor = []
    #                 print('Average number of overlapping bounding boxes from RPN = {} for {} previous iterations'.format(mean_overlapping_bboxes, epoch_length))
                    if mean_overlapping_bboxes == 0:
                        print('RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')

                # Generate X (x_img) and label Y ([y_rpn_cls, y_rpn_regr])
                X, Y, img_data, debug_img, debug_num_pos = next(data_gen_train)

                # Train rpn model and get loss value [_, loss_rpn_cls, loss_rpn_regr]
                loss_rpn = model_rpn.train_on_batch(X, Y)

                # Get predicted rpn from rpn model [rpn_cls, rpn_regr]
                P_rpn = model_rpn.predict_on_batch(X)

                # R: bboxes (shape=(300,4))
                # Convert rpn layer to roi bboxes
                R = rpn_to_roi(P_rpn[0], P_rpn[1], C, K.image_dim_ordering(), use_regr=True, overlap_thresh=0.7, max_boxes=300)
                
                # note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
                # X2: bboxes that iou > C.classifier_min_overlap for all gt bboxes in 300 non_max_suppression bboxes
                # Y1: one hot code for bboxes from above => x_roi (X)
                # Y2: corresponding labels and corresponding gt bboxes
                X2, Y1, Y2, IouS = calc_iou(R, img_data, C, class_mapping)

                # If X2 is None means there are no matching bboxes
                if X2 is None:
                    rpn_accuracy_rpn_monitor.append(0)
                    rpn_accuracy_for_epoch.append(0)
                    continue
                
                # Find out the positive anchors and negative anchors
                neg_samples = np.where(Y1[0, :, -1] == 1)
                pos_samples = np.where(Y1[0, :, -1] == 0)

                if len(neg_samples) > 0:
                    neg_samples = neg_samples[0]
                else:
                    neg_samples = []

                if len(pos_samples) > 0:
                    pos_samples = pos_samples[0]
                else:
                    pos_samples = []

                rpn_accuracy_rpn_monitor.append(len(pos_samples))
                rpn_accuracy_for_epoch.append((len(pos_samples)))

                if C.num_rois > 1:
                    # If number of positive anchors is larger than 4//2 = 2, randomly choose 2 pos samples
                    if len(pos_samples) < C.num_rois//2:
                        selected_pos_samples = pos_samples.tolist()
                    else:
                        selected_pos_samples = np.random.choice(pos_samples, C.num_rois//2, replace=False).tolist()
                    
                    # Randomly choose (num_rois - num_pos) neg samples
                    try:
                        selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace=False).tolist()
                    except:
                        selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace=True).tolist()
                    
                    # Save all the pos and neg samples in sel_samples
                    sel_samples = selected_pos_samples + selected_neg_samples
                else:
                    # in the extreme case where num_rois = 1, we pick a random pos or neg sample
                    selected_pos_samples = pos_samples.tolist()
                    selected_neg_samples = neg_samples.tolist()
                    if np.random.randint(0, 2):
                        sel_samples = random.choice(neg_samples)
                    else:
                        sel_samples = random.choice(pos_samples)

                # training_data: [X, X2[:, sel_samples, :]]
                # labels: [Y1[:, sel_samples, :], Y2[:, sel_samples, :]]
                #  X                     => img_data resized image
                #  X2[:, sel_samples, :] => num_rois (4 in here) bboxes which contains selected neg and pos
                #  Y1[:, sel_samples, :] => one hot encode for num_rois bboxes which contains selected neg and pos
                #  Y2[:, sel_samples, :] => labels and gt bboxes for num_rois bboxes which contains selected neg and pos
                loss_class = model_classifier.train_on_batch([X, X2[:, sel_samples, :]], [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])

                losses[iter_num, 0] = loss_rpn[1]
                losses[iter_num, 1] = loss_rpn[2]

                losses[iter_num, 2] = loss_class[1]
                losses[iter_num, 3] = loss_class[2]
                losses[iter_num, 4] = loss_class[3]

                iter_num += 1

                progbar.update(iter_num, [('rpn_cls', np.mean(losses[:iter_num, 0])), ('rpn_regr', np.mean(losses[:iter_num, 1])),
                                        ('final_cls', np.mean(losses[:iter_num, 2])), ('final_regr', np.mean(losses[:iter_num, 3]))])

                if iter_num == epoch_length:
                    loss_rpn_cls = np.mean(losses[:, 0])
                    loss_rpn_regr = np.mean(losses[:, 1])
                    loss_class_cls = np.mean(losses[:, 2])
                    loss_class_regr = np.mean(losses[:, 3])
                    class_acc = np.mean(losses[:, 4])

                    mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
                    rpn_accuracy_for_epoch = []

                    if C.verbose:
                        print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(mean_overlapping_bboxes))
                        print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
                        print('Loss RPN classifier: {}'.format(loss_rpn_cls))
                        print('Loss RPN regression: {}'.format(loss_rpn_regr))
                        print('Loss Detector classifier: {}'.format(loss_class_cls))
                        print('Loss Detector regression: {}'.format(loss_class_regr))
                        print('Total loss: {}'.format(loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr))
                        print('Elapsed time: {}'.format(time.time() - start_time))
                        elapsed_time = (time.time()-start_time)/60

                    curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
                    iter_num = 0
                    start_time = time.time()

                    if curr_loss < best_loss:
                        if C.verbose:
                            print('Total loss decreased from {} to {}, saving weights'.format(best_loss,curr_loss))
                        best_loss = curr_loss
                        model_all.save_weights(C.model_path)

                    new_row = {'mean_overlapping_bboxes':round(mean_overlapping_bboxes, 3), 
                            'class_acc':round(class_acc, 3), 
                            'loss_rpn_cls':round(loss_rpn_cls, 3), 
                            'loss_rpn_regr':round(loss_rpn_regr, 3), 
                            'loss_class_cls':round(loss_class_cls, 3), 
                            'loss_class_regr':round(loss_class_regr, 3), 
                            'curr_loss':round(curr_loss, 3), 
                            'elapsed_time':round(elapsed_time, 3), 
                            'mAP': 0}

                    record_df = record_df.append(new_row, ignore_index=True)
                    record_df.to_csv(record_path, index=0)

                    break

            except Exception as e:
                print('Exception: {}'.format(e))
                continue

    print('Training complete, exiting.')
        
