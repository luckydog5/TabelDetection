import cv2
import os
from tqdm import tqdm



if __name__ =='__main__':

    if not os.path.exists('new_train'):
        os.mkdir('new_train/')
    f = open('annotation.txt','r')
    lines = f.readlines()
    for line in tqdm(lines):

        img_path = line.split(',')[0]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        b = cv2.distanceTransform(img, distanceType=cv2.DIST_L2, maskSize=5)
        g = cv2.distanceTransform(img, distanceType=cv2.DIST_L1, maskSize=5)
        r = cv2.distanceTransform(img, distanceType=cv2.DIST_C, maskSize=5)
        img_name = img_path.split('/')[-1]
        transformed_image = cv2.merge((b,g,r))
        cv2.imwrite('new_train/'+img_name,transformed_image)