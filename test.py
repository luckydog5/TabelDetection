#from frcnn_train_vgg import *
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
import argparse
from utils import *
def parse_arguments():
    parser = argparse.ArgumentParser(description='Weights path, image_path, results_path')
    
    parser.add_argument(
        "--image_path",
        type = str,
        help = "Image for detection, 1 image per time. ",
        default="image/"
    )
    parser.add_argument(
        "--results_path",
        type = str,
        help = "Path to save detection result. ",
        default = 'result/'
    )
    
    parser.add_argument(

       "--type",
       type = str,
       help = "vgg or resnet or vgg_deform, or resnet_deform",
       default = 'resnet'



    )
    parser.add_argument(
        "--weights_path",
        type = str,
        help = "Path to load resnet50 model parameters. ",
        default = 'resnet_model/weights-ctpnlstm-07.hdf5'
    )
    parser.add_argument(
        "--gpu",
        type = str,
        help = "Which gpu will you use",
        default = '4'
    )
    parser.add_argument(
    
        "--config_output_filename",
        type = str,
        help = "Config file ",
        default = 'model_vgg_config.pickle'
    )
    
    return parser.parse_args()

#base_path = '/data/zhangshihao/Faster_RCNN_for_Open_Images_Dataset_Keras/'
#test_base_path = 'image/'
#config_output_filename = os.path.join(base_path, 'model_vgg_config.pickle')


def vgg16_nn_base2(input,trainable=False):
    base_model = VGG16(weights=None,include_top=False,input_shape = input)
    #base_model.load_weights('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
    if(trainable ==False):
        for ly in base_model.layers:
            ly.trainable = False
    return base_model.get_layer('block5_conv3').output
    #return base_model.input,base_model.get_layer('block5_conv3').output

def load_records():
    # Load the records
    record_df = pd.read_csv(C.record_path)

    r_epochs = len(record_df)

    plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    plt.plot(np.arange(0, r_epochs), record_df['mean_overlapping_bboxes'], 'r')
    plt.title('mean_overlapping_bboxes')

    plt.subplot(1,2,2)
    plt.plot(np.arange(0, r_epochs), record_df['class_acc'], 'r')
    plt.title('class_acc')

    plt.show()

    plt.figure(figsize=(15,5))

    plt.subplot(1,2,1)
    plt.plot(np.arange(0, r_epochs), record_df['loss_rpn_cls'], 'r')
    plt.title('loss_rpn_cls')

    plt.subplot(1,2,2)
    plt.plot(np.arange(0, r_epochs), record_df['loss_rpn_regr'], 'r')
    plt.title('loss_rpn_regr')
    plt.show()
    plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    plt.plot(np.arange(0, r_epochs), record_df['loss_class_cls'], 'r')
    plt.title('loss_class_cls')

    plt.subplot(1,2,2)
    plt.plot(np.arange(0, r_epochs), record_df['loss_class_regr'], 'r')
    plt.title('loss_class_regr')
    plt.show()
    plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    plt.plot(np.arange(0, r_epochs), record_df['curr_loss'], 'r')
    plt.title('total_loss')

    plt.subplot(1,2,2)
    plt.plot(np.arange(0, r_epochs), record_df['elapsed_time'], 'r')
    plt.title('elapsed_time')

    plt.show()

def format_img_size(img, C,flag='vgg_deform'):
	""" formats the image size based on config """
	img_min_side = float(600)
	#img_min_side = float(C.im_size)
	(height,width,_) = img.shape
	if flag == 'vgg' or flag == 'resnet':
		
		#new_height = 864
		#new_width = 640
		if width <= height:
			ratio = img_min_side/width
			new_height = int(ratio * height)
			new_width = int(img_min_side)
		else:
			ratio = img_min_side/height
			new_width = int(ratio * width)

			new_height = int(img_min_side)
	if flag == 'vgg_deform':
		new_height = 1280
		new_width = 800
		if width <= height:
			ratio = new_width/width
		else:
			ratio = new_height/height 
	if flag == 'resnet50_deform':
		new_height = 864
		new_width = 640
		if width <= height:
			ratio = new_width/width
		else:
			ratio = new_height/height 
	img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)


	
	return img, ratio	

def format_img_channels(img, C):
	""" formats the image channels based on config """
	img = img[:, :, (2, 1, 0)]
	img = img.astype(np.float32)
	img[:, :, 0] -= C.img_channel_mean[0]
	img[:, :, 1] -= C.img_channel_mean[1]
	img[:, :, 2] -= C.img_channel_mean[2]
	img /= C.img_scaling_factor
	img = np.transpose(img, (2, 0, 1))
	img = np.expand_dims(img, axis=0)
	return img

def format_img(img, C, flag='vgg_deform'):
	""" formats an image for model prediction based on config """
	img, ratio = format_img_size(img, C,flag)
	#img = format_img_channels(img, C)
	return img, ratio

# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio, x1, y1, x2, y2):

	real_x1 = int(round(x1 // ratio))
	real_y1 = int(round(y1 // ratio))
	real_x2 = int(round(x2 // ratio))
	real_y2 = int(round(y2 // ratio))

	return (real_x1, real_y1, real_x2 ,real_y2)

if __name__ == '__main__':
    
    args = parse_arguments()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    num_features = 512
    config_output_filename =  args.config_output_filename
    test_base_path = args.image_path
    with open(config_output_filename, 'rb') as f_in:
        C = pickle.load(f_in)

    # turn off any data augmentation at test time
    C.use_horizontal_flips = False
    C.use_vertical_flips = False
    C.rot_90 = False
    #input_shape_img = (None, None, 3)
    input_shape_features = (None, None, num_features)

    #img_input = Input(shape=input_shape_img)
    roi_input = Input(shape=(C.num_rois, 4))
    feature_map_input = Input(shape=input_shape_features)

    # define the base network (VGG here, can be Resnet50, Inception, etc)
    if args.type == 'vgg':
        input_shape_img = (None,None,3)
        img_input = Input(shape=input_shape_img)
        shared_layers = nn_base(img_input, trainable=True)
    if args.type == 'resnet':
        input_shape_img = (None,None,3)
        img_input = Input(shape=input_shape_img)
        shared_layers = res_nn(inputs=img_input)
        print('&'*30,'\t',shared_layers)
    if args.type == 'vgg_deform':
        input_shape_img = (864,640,3)
        img_input = Input(shape=input_shape_img)
        shared_layers = nn_base_deform(img_input, trainable=True)
    if args.type == 'resnet50_deform':
        input_shape_img = (864,640,3)
        img_input = Input(shape=input_shape_img)
        shared_layers = resnet50_deform(img_input)
    #img_input, shared_layers = res_nn(input=input_shape_img, trainable=True)
    #shared_layers = vgg16_nn_base2(img_input, trainable=True)
    # define the RPN, built on the base layers
    num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
    rpn_layers = rpn_layer(shared_layers, num_anchors)

    classifier = classifier_layer(feature_map_input, roi_input, C.num_rois, nb_classes=len(C.class_mapping))
    #classifier = classifier_layer(feature_map_input, roi_input, C.num_rois, nb_classes=3)

    model_rpn = Model(img_input, rpn_layers)
    model_classifier_only = Model([feature_map_input, roi_input], classifier)

    model_classifier = Model([feature_map_input, roi_input], classifier)
    print('*'*100)
    print('Loading weights from {}'.format(args.weights_path))
    model_rpn.load_weights(args.weights_path, by_name=True)
    model_classifier.load_weights(args.weights_path, by_name=True)
    #model_classifier.load_weights(args.weights_path)
    model_rpn.compile(optimizer='sgd', loss='mse')
    model_classifier.compile(optimizer='sgd', loss='mse')
    print('*'*100)
    #class_mapping = {'no-table':0,'table':1,'bg':2}
    class_mapping = C.class_mapping
    class_mapping = {v: k for k, v in class_mapping.items()}
    print(class_mapping)
    class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}

    test_imgs = os.listdir(test_base_path)
    #######################   load  test images    ###############################
    imgs_path = []
    for i in range(len(test_imgs)):
        #idx = np.random.randint(len(test_imgs))
        imgs_path.append(test_imgs[i])

    all_imgs = []

    classes = {}

    ######################    Just  do   test ................................................##################

    # If the box classification value is less than this, we ignore this box
    bbox_threshold = 0.8

    for idx, img_name in enumerate(imgs_path):
        if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
            continue
        print(img_name)
        st = time.time()
        filepath = os.path.join(test_base_path, img_name)

        img = cv2.imread(filepath)
        #img = format_img_channels(img, C)
        #X, ratio = format_img(img, C,flag=args.type)
        img, ratio = format_img(img,C,flag=args.type)
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        b = cv2.distanceTransform(img, distanceType=cv2.DIST_L2, maskSize=5)
        g = cv2.distanceTransform(img, distanceType=cv2.DIST_L1, maskSize=5)
        r = cv2.distanceTransform(img, distanceType=cv2.DIST_C, maskSize=5)
        transformed_image = cv2.merge((b,g,r))
        img = transformed_image
        """
        X = format_img_channels(img,C)
        print('*'*100)
        print(X.shape)
        print('*'*100)
        X = np.transpose(X, (0, 2, 3, 1))

        # get output layer Y1, Y2 from the RPN and the feature maps F
        # Y1: y_rpn_cls
        # Y2: y_rpn_regr
        [Y1, Y2, F] = model_rpn.predict(X)

        # Get bboxes by applying NMS 
        # R.shape = (300, 4)
        R = rpn_to_roi(Y1, Y2, C, K.image_dim_ordering(), overlap_thresh=0.7)

        # convert from (x1,y1,x2,y2) to (x,y,w,h)
        R[:, 2] -= R[:, 0]
        R[:, 3] -= R[:, 1]

        # apply the spatial pyramid pooling to the proposed regions
        bboxes = {}
        probs = {}
        print('-'*50,R.shape)
        for jk in range(R.shape[0]//C.num_rois + 1):
            ROIs = np.expand_dims(R[C.num_rois*jk:C.num_rois*(jk+1), :], axis=0)
            if ROIs.shape[1] == 0:
                break

            if jk == R.shape[0]//C.num_rois:
                #pad R
                curr_shape = ROIs.shape
                target_shape = (curr_shape[0],C.num_rois,curr_shape[2])
                ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                ROIs_padded[:, :curr_shape[1], :] = ROIs
                ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                ROIs = ROIs_padded

            [P_cls, P_regr] = model_classifier_only.predict([F, ROIs])

            # Calculate bboxes coordinates on resized image
            for ii in range(P_cls.shape[1]):
                # Ignore 'bg' class
                
                if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                    #continue
                    #if np.max(P_cls[0, ii, :]) < bbox_threshold:
                    print('-'*50,np.max(P_cls[0, ii, :]))
                    print('^'*50,np.min(P_cls[0, ii, :]))
                    continue
                cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

                if cls_name not in bboxes:
                    bboxes[cls_name] = []
                    probs[cls_name] = []

                (x, y, w, h) = ROIs[0, ii, :]

                cls_num = np.argmax(P_cls[0, ii, :])
                try:
                    (tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
                    tx /= C.classifier_regr_std[0]
                    ty /= C.classifier_regr_std[1]
                    tw /= C.classifier_regr_std[2]
                    th /= C.classifier_regr_std[3]
                    x, y, w, h = apply_regr(x, y, w, h, tx, ty, tw, th)
                except:
                    pass
                bboxes[cls_name].append([C.rpn_stride*x, C.rpn_stride*y, C.rpn_stride*(x+w), C.rpn_stride*(y+h)])
                probs[cls_name].append(np.max(P_cls[0, ii, :]))

        all_dets = []
        txt_name = img_name.split('.')[0] + '.txt'
        for key in bboxes:
            bbox = np.array(bboxes[key])
            f = open(args.results_path+txt_name, 'w')
            new_boxes, new_probs = non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.2)
            print('-'*50,len(new_boxes))
            for jk in range(new_boxes.shape[0]):
                (x1, y1, x2, y2) = new_boxes[jk,:]
                if x2 > img.shape[1]:
                    x2 = img.shape[1]
                if y2 > img.shape[0]:
                    y2 = img.shape[0]
                # Calculate real coordinates on original image
                #(real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)
                (real_x1, real_y1, real_x2, real_y2) = (x1,y1,x2,y2)
                
                
                f.write((str(real_x1) + ','+ str(real_y1) + ',' + str(real_x2) + ',' + str(real_y2) + '\n'))
                

                cv2.rectangle(img,(real_x1, real_y1), (real_x2, real_y2), (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])),4)
                #cv2.rectangle(img,(real_x1, real_y1), (real_x2, real_y2), (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])),4)
                textLabel = '{}: {}'.format(key,int(100*new_probs[jk]))
                all_dets.append((key,100*new_probs[jk]))

                (retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,1,1)
                textOrg = (real_x1, real_y1-0)

                cv2.rectangle(img, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 0, 0), 1)
                cv2.rectangle(img, (textOrg[0] - 5,textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (255, 255, 255), -1)
                cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)
        
            f.close()
        cv2.imwrite(args.results_path+img_name, img)
        print('Elapsed time = {}'.format(time.time() - st))
        print(all_dets)
        #plt.figure(figsize=(10,10))
        #plt.grid()
	#plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
	#plt.show()
