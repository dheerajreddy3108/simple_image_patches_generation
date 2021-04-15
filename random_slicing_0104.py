# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 15:24:42 2021

@author: studperadh6230
"""

import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from random import randint
import math 
from imgaug import augmenters as iaa
from io import BytesIO
import zipfile
import glob

#trail testing 
x = 128
y = 128
w = 100
h = 100
angle = 0
b = math.cos(math.radians(angle))*0.5
a = math.sin(math.radians(angle))*0.5
    
pt0 = (int(x-a*h - b*w),int(y+ b*h -a*w))
pt1 = (int(x+a*h-b*w),int(y-b*h+a*w))
pt2 = (int(2*x-pt0[0]),int(2*y-pt0[1]))
pt3 = (int(2*x-pt1[0]),int(2*y-pt1[1]))

#fucntion to get all vertices based on center_point, h, w and angle

def get_four_vertices(ct_pt_x,ct_pt_y,width,height):
    angle = 0
    a = math.sin(math.radians(angle))*0.5
    b = math.cos(math.radians(angle))*0.5
    pt0 = (int(ct_pt_x - a * height - b * width), int(ct_pt_y + b * height - a * width))
    pt1 = (int(ct_pt_x + a * height - b * width), int(ct_pt_y - b * height - a * width))
    pt2 = (int(2 * ct_pt_x - pt0[0]),int(2 * ct_pt_y- pt0[1]))
    pt3 = (int(2 * ct_pt_x - pt1[0]),int(2 * ct_pt_y- pt1[1]))
    center_pt = (ct_pt_x, ct_pt_y)

    
    rot_free_pts = [pt0,pt1,pt2,pt3]
    
    return rot_free_pts,center_pt

rot_free_pts,center_pt = get_four_vertices(128, 128, 128, 128)

    
def rotate_points(rotation_angle):
    pt0, pt1, pt2, pt3 = rot_free_pts
    x0,y0 = center_pt
    
    
    #point_0
    rotated_x = math.cos(rotation_angle) * (pt0[0] - x0) - math.sin(rotation_angle) * (pt0[1] - y0) + x0
    rotated_y = math.sin(rotation_angle) * (pt0[0] - x0) + math.cos(rotation_angle) * (pt0[1] - y0) + y0
    point_0 = (int(rotated_x), int(rotated_y))
    
    #point_1
    
    rotated_x = math.cos(rotation_angle) * (pt1[0] - x0) - math.sin(rotation_angle) * (pt1[1] - y0) + x0
    rotated_y = math.sin(rotation_angle) * (pt1[0] - x0) + math.cos(rotation_angle) * (pt1[1] - y0) + y0
    point_1 = (int(rotated_x), int(rotated_y))
    
    #point_2
    
    rotated_x = math.cos(rotation_angle) * (pt2[0] - x0) - math.sin(rotation_angle) * (pt2[1] - y0) + x0
    rotated_y = math.sin(rotation_angle) * (pt2[1] - x0) + math.cos(rotation_angle) * (pt2[1] - y0) + y0
    point_2 = (int(rotated_x), int(rotated_y))
    
    #point_3
    
    rotated_x = math.cos(rotation_angle) * (pt3[0] - x0) - math.sin(rotation_angle) * (pt3[1] - y0) + x0
    rotated_y = math.sin(rotation_angle) * (pt3[1] - x0) + math.cos(rotation_angle) * (pt3[1] - y0) + y0
    point_3 = (int(rotated_x), int(rotated_y))
    
    rotated_points = [point_0, point_1, point_2, point_3]
    
    return rotated_points

rot_pts = rotate_points(randint(0,30))

img = cv2.imread('D:/black_and_white.bmp')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img,(1024,1024),interpolation = cv2.INTER_AREA)

img = cv2.rectangle(img, rot_pts[0], rot_pts[3], color=(255,255,255))

plt.imshow(img) 

img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)   
    
cv2.imwrite('D:/trail_patch6.bmp',img)
    
img = cv2.line(img,rot_pts[2],rot_pts[3],color =(255,0,0),thickness =1) 
    
img = cv2.circle(img,center = rot_pts[0],color = (127,127,127),radius = 5)
img = cv2.circle(img,center = rot_pts[1],color = (255,255,255),radius = 5)
img = cv2.circle(img,center = rot_pts[2],color = (0,0,0),radius = 5)
img = cv2.circle(img,center = rot_pts[3],color = (0,255,0),radius = 5)

im = Image.open('D:/black_and_white.bmp',mode = 'r')
im = im.resize((1024,1024),resample = Image.LANCZOS)
box = (rot_pts[2][0],rot_pts[2][1],rot_pts[2][0] + 128,rot_pts[2][1] + 128)
patch = im.crop(box)
patch.save('D:/s.png')

sometimes = lambda aug : iaa.Sometimes(0.5,aug)
seq =  iaa.Sequential(
    sometimes(iaa.Affine(
        scale = { 'x' : (0.1,0.2), 'y': (0.1,0.2)}
        )))



for i in range(100):
    img = Image.open('D:/black_and_white.bmp')
    img = img.resize((1024,1024),resample = Image.LANCZOS)
    
            
    for num_patches in range(16):
        rot_free_pts, center_pt = get_four_vertices(randint(0,1024), randint(0,1024), 128, 128)
        rot_pts = rotate_points(randint(-30, 30))
        box = (rot_pts[1][0], rot_pts[1][1], rot_pts[1][0] + 128, rot_pts[1][1] + 128)
        patch = im.crop(box)
        patch.save('D:/patch_database/patch_' + str(i) + '_' + str(num_patches) + '.png')
        
        
        
img = cv2.imread('D:/sample.png',-1)


img = cv2.resize(img,(1024,1024), interpolation = cv2.INTER_AREA)
X = []

for _ in range(100):
    _ = img
    X.append(_)      
       
X = np.asarray(X)

seq = iaa.OneOf(
  
    [
     iaa.Fliplr(.25),

     iaa.Crop(percent=0.05),
     iaa.TranslateX(percent=0.01),
     iaa.TranslateY(percent=0.01),
     iaa.Rotate(rotate=(-30,30)),
     #iaa.ShearX(shear = (-30,30)),
     #iaa.ShearY(shear = (-30,30)),
     #iaa.AddToBrightness(add=(-30,30)),
     #iaa.Sharpen(alpha = (0.0,0.2),lightness = (0.8,2.)),
     #iaa.ContrastNormalization(alpha=(0.5,1.5)),
     #iaa.Invert(0.1,per_channel=False),
     #iaa.Add((-15,15),per_channel=False),
     iaa.GaussianBlur(sigma=(0.5,0.8)),
     #iaa.AverageBlur(k =3),
     #iaa.Grayscale(alpha = (0.0,1.0)),
     #iaa.MedianBlur(k=3),
     #iaa.LinearContrast((0.4,2.0),per_channel=False)
     
     

     ]
    )

imgs_aug = seq(images = X)   


for _ in range(len(imgs_aug)):
    i = imgs_aug[_]
    cv2.imwrite('D:/Full_image_db/img-'+str(_) + '.png',i)



background = Image.open('D:/bg.png',mode='r')

background = background.resize((1024,1024),resample = Image.LANCZOS)

ref_img = background.copy()
ref_img = np.array(ref_img)



z = zipfile.ZipFile('D:/patch_database/output.zip','w',zipfile.ZIP_DEFLATED)

    
for n in range(len(imgs_aug)):
    render_img = background.copy()
    img_size = 1024
    w = int(img_size*1)
    img2 = Image.open('D:/Full_image_db/img-'+str(n)+'.png',mode = 'r')
    img2 = img2.resize((1024,1024),resample= Image.LANCZOS)
    x = 0
    y = 0
    #img2 = imgs_aug[n]
    #x = randint(int(img2.width/2),background.width-int(img2.width/2))
    #y = randint(int(img2.height/2),background.height/2-int(img2.height/2))
    #img2 = Image.Image.rotate(img2,0,resample=Image.BICUBIC,expand=False)
    render_img.paste(img2,(x,y),img2)
    render_img.save('D:/Full_image_db/f-'+str(n)+'.png')
    
#loading full images for slicing patches

full_imgs = []
for img in glob.glob('D:/Full_image_db/*png'):
    n= cv2.imread(img,1)
    n = cv2.cvtColor(n, cv2.COLOR_BGR2RGB)
    #n = cv2.resize(n,(256,256),interpolation=cv2.INTER_AREA)
    full_imgs.append(n)


#slicing patches

for imgs in range(len(full_imgs)):
    im = Image.open('D:/Full_image_db/f-'+str(imgs)+'.png')
    #img = img.resize((1024,1024),resample = Image.LANCZOS)
    
    for num_patches in range(16):
        rot_free_pts, center_pt = get_four_vertices(randint(0,1024), randint(0,1024), 128, 128)
        rot_pts = rotate_points(randint(-30, 30))
        box = (rot_pts[3][0], rot_pts[3][1], rot_pts[3][0]-128, rot_pts[3][1]-128)
        if 0<box[0]<1024 and 0<box[1]<1024 and 0<box[2]<1024 and 0<box[3]<1024:
            patch = im.crop(box)
            patch.save('D:/patch_database/Pat/patch_' + str(imgs) + '_' + str(num_patches) + '.png')
        
        
if 0<box[0]<1024 and 0<box[1]<1024 and 0<box[2]<1024 and 0<box[3]<1024:
    print('true')
    






    