
Here is an implementation about the paper "Table Detection using deepLearing".

Reference:

1、https://blog.goodaudience.com/table-detection-using-deep-learning-7182918d778

2、https://www.researchgate.net/publication/320243569_Table_Detection_Using_Deep_Learning
3、https://github.com/RockyXu66/Faster_RCNN_for_Open_Images_Dataset_Keras

Requirements:

python3, tensorflow 1.12, keras 2.24, sk-learn.


You can find everything you need in utils.py

distance_transfor.py is the main idea from the paper, that is an implementation of the image transform.

You should run it first to preprocess your trianing samples, after that, your training images may look like this image.


![image](http://github.com/luckydog5/TabelDetection/raw/master/new_result/1403.6535_129.jpg)




Train Phase:

You should provide  table images  and their corresponding coordinates,just like you train normal faster-rcnn, nothing special.

eg.   path_to_your_train_samples/image_name.jpg   table_region [x_min,y_min,x_max,y_max,'table']  (x_min,y_min): top_left  (x_max,y_max):bottom_right

All these are in a annotation.txt file, which inclues:

..............

Faster_RCNN_for_Open_Images_Dataset_Keras/TableBank_data/Detection_data/Latex/images/1401.0007_15.jpg,85,396,510,495,table

Faster_RCNN_for_Open_Images_Dataset_Keras/TableBank_data/Detection_data/Latex/images/1401.0045_3.jpg,39,327,127,450,table

Faster_RCNN_for_Open_Images_Dataset_Keras/TableBank_data/Detection_data/Latex/images/1401.0045_3.jpg,81,179,248,225,table

..................

I will recommand Table_Bank dataset here, you only need to fill out application form and you will receive an eamil with download link.

You need download resnet50 pre-trained  model weights first.

Annotations:

TableBank Dataset:

	--Detection_data   

		--Latex

			--images

			--Latex.json

			--url.csv

		--Word

			--images

			--Word.json

			--url.csv

	--Recognition_data

		--

Here i only need the Detection_data.

Use generate_annotation.py generate annotation.txt, you can use Latex or Word subset by adjusting the 

following parameters.

parameters are:

	--type     Latex or Word subset to use.

	--Latex_path     If the type is 'Latex',you need provide its path.

	--Word_path      If the type is 'Word', you need provide its path.

	--json_file 	 If the type is 'Latex', provide Latex.json path, otherwise Word.json.

python generate_annotation.py  --parameters above.

Then you will get the annotation.txt file.

Start training...

some parameters:
	
	--base_path: path to your workdir.Where the resides.
	
	--train_path: path to your grondtruth file annotation.txt.
	
	--base_weights_path: path to pre_trained model weights, like vgg16, resnet50 etc.
	
	--type:  use vgg16 backbone or resnet50 backbone
	
	--config_output_filename:  config file name.
	
	--save_path:  path to save checkpoints.
	
	--pre_trained_model_weights: If you want to continue training from last checkpoint, otherwise set it to None.
	
	--train_size:  train_steps per epoch, default is 10000.
	
	--gpu: Choose one gpu.

python train.py parameters

Test:

some parameters:
	
	--image_path: Path to test images.
	
	--result_path: Path to save results.
	
	--type:  Must align with the train type. vgg16 or resnet50
	
	--weights_path: Where your model weights resides.
	
	--config_output_filename: Be align with the train config_output_filename.
	
	--gpu: 

python test.py parameters.

Outputs will be saved in result_path: image_name.jpg  image_name.txt.

I will provide my config file and model weighs here:	https://pan.baidu.com/s/1CfWlaZYIkAQDVJ5XxsybVA

Here are some sample results.

![image](http://github.com/luckydog5/TabelDetection/raw/master/new_result/1807.02216_22.jpg)

![image](http://github.com/luckydog5/TabelDetection/raw/master/new_result/2.png)

Besides you will get 2.txt and 1807.02216_22.txt file which contains the table coordinates.  [x_min,y_min,x_max,y_max]

# TableDetection
