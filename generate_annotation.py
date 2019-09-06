import glob
import os
import argparse
from tqdm import tqdm



def parse_arguments():
	parser = argparse.ArgumentParser(description='Weights path, image_path, results_path')

	parser.add_argument(
		"--type",
		type = str,
		help = "Latex or Word ",
		default='Latex'
	)
	parser.add_argument(
		"--Latex_path",
		type = str,
		help = "Latex image path ",
		default='TableBank_data/Detection_data/Latex/images/'
	)
	parser.add_argument(
		"--Word_path",
		type = str,
		help = "Word image path ",
		default='TableBank_data/Detection_data/Word/images/'
	)
	parser.add_argument(
		"--json_file",
		type=str,
		help='Load json file...',
		default='TableBank_data/Detection_data/Latex/Latex.json'
	)
	return parser.parse_args()

def gen_annotation():
	
	###	Annotations and images are not seriously one-to-one correspondence.
	###	One image may has several annotations.
	###	Input: annotations  json file.
	###	Output: annotations txt file.
	f = open(args.json_file,'r')
	temp = json.load(f)
	print("Total have {} images.".format(len(temp)))
	print("Json file keys are {}.".format(temp.keys()))
	print('*'*30, 'images','*'*30)
	print(len(temp['images']))
	print(temp['images'][10:20])
	print('*'*30, 'annotations','*'*30)
	print(len(temp['annotations']))
	print(temp['annotations'][10:20])
	print('*'*30, 'categories','*'*30)
	print(len(temp['categories']))
	print(temp['categories'])
	print('*'*100)

	j = 0 
	f = open('annotation.txt','w+')
	for i in tqdm(len(temp['images']),desc='Generate annotation.txt'):
		img_name = temp['images'][i]['file_name']
		img_id = temp['images'][i]['id']
		while j < len(temp['annotations']):
			b_id = temp['annotations'][j]['image_id']
			if img_id != b_id:
				break

			elif img_id == b_id:
				segmentation = temp['annotations'][j]['segmentation']
				x1,y1,x2,y2,x3,y3,x4,y4 = map(int,segmentation[0])
				line = ''+img_name+','+str(x1)+','+str(y1)+','+str(x3)+','+str(y3)+','+'table'
				f.write(line)
				f.write('\n')
			j += 1

	f.close()

if __name__ == '__main__':

	args = parse_arguments()
	gen_annotation()





















