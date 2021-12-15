#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 19:09:25 2021

@author: ufukefe
"""

import os
import argparse
import yaml
import cv2
from DeepFeatureMatcher import DeepFeatureMatcher
from PIL import Image
import numpy as np
import time
import math
import os
import glob

#homography estimation
def get_calculate_hom(h_est,w,h,h_gt):

    cornersA = [[1-0.5,1-0.5,w-0.5,w-0.5],
                [1-0.5,h-0.5,1-0.5,h-0.5],
                [1,1,1,1]]
    #  Find corners with estimated homography
    
    cornersA_est = np.zeros(shape=(3,4))
    cornersA_est =np.dot(h_est, cornersA)
    
    x = np.zeros(shape=(1,4))
    y = np.zeros(shape=(1,4))
    
    for i in range(len(cornersA_est)):
        for j in range(len(cornersA_est[i])):
            x[0][i]= cornersA_est[0][i] / cornersA_est[2][i]+0.5
            y[0][i]= cornersA_est[1][i] / cornersA_est[2][i]+0.5
    x[0][3]= cornersA_est[0][3] / cornersA_est[2][3]+0.5
    y[0][3]= cornersA_est[1][3] / cornersA_est[2][3]+0.5
    cornersA_est = np.vstack((x,y))
    cornersA_est = cornersA_est.transpose()
    
    #   Find corners with groundtruth homography
    
    f = np.zeros(shape=(1,4))
    g = np.zeros(shape=(1,4))
    cornersA_gt = np.zeros(shape=(3,3))
    cornersA_gt =np.dot(h_gt, cornersA)
    for i in range(len(cornersA_gt)):
        for j in range(len(cornersA_gt[i])):
            f[0][i]= cornersA_gt[0][i] / cornersA_gt[2][i]+0.5
            g[0][i]= cornersA_gt[1][i] / cornersA_gt[2][i]+0.5
    f[0][3]= cornersA_gt[0][3] / cornersA_gt[2][3]+0.5
    g[0][3]= cornersA_gt[1][3] / cornersA_gt[2][3]+0.5
    
    cornersA_gt= np.vstack((f,g))
    cornersA_gt = cornersA_gt.transpose()
    
    cornersA_ext = np.zeros(shape=(2,4))
    cornersA_ext = cornersA_gt - cornersA_est 
    cornersA_sum =np.zeros(shape=(4,1))
    cornersA_sum_1 =np.zeros(shape=(4,1))
    cornersA_sum_2 =np.zeros(shape=(4,1))
    distances =np.zeros(shape=(4,1))
    for i in range(len(cornersA_ext)):
       for j in range(len(cornersA_ext[i])):
           cornersA_ext[i][j] = cornersA_ext[i][j] * cornersA_ext[i][j]
           if(j == 1):
               cornersA_sum_1[i] = cornersA_ext[i][j]
           elif( j==0 ):
               cornersA_sum_2[i] = cornersA_ext[i][j]
           cornersA_sum[i] = cornersA_sum_1[i]+cornersA_sum_2[i]
           distances[i] = math.sqrt(cornersA_sum[i])
      
    distance = distances.mean()
    homq = np.zeros(shape=(1,5))
    
    for th in range(5):
        if(distance <= th+1):
            homq[0][th]= 1
        else:
            homq[0][th]= 0
    return homq,distance
 
# calculate mma
def get_calculate_mma(pointsA,pointsB,h_gt):
    
    pointsA_x =np.zeros(shape=(len(pointsA),len(pointsA[1])))
    pointsB_x =np.zeros(shape=(len(pointsB),len(pointsB[1])))
    
    for i in range(len(pointsB)):
       for j in range(len(pointsB[i])):
           pointsB_x[i][j] =pointsB[i][j]
    for i in range(len(pointsA)):
       for j in range(len(pointsA[i])):
           pointsA_x[i][j] =pointsA[i][j] - 0.5
    
    
    pointsA_tra=np.vstack((pointsA_x,np.ones(shape=(1,len(pointsA[1])))))
    
    
    pointsB_gt = np.zeros(shape=(len(pointsA_tra),3))
    pointsB_gt =np.dot(h_gt, pointsA_tra)
    
    x = np.zeros(shape=(1,len(pointsB_gt[1])))
    y = np.zeros(shape=(1,len(pointsB_gt[1])))
    for i in range(len(pointsB_gt)):
        for j in range(len(pointsB_gt[i])):
            x[0][j]= pointsB_gt[0][j] / pointsB_gt[2][j]+0.5
            y[0][j]= pointsB_gt[1][j] / pointsB_gt[2][j]+0.5
            
            
    pointsB_gt = np.vstack((x,y))
    pointsB_gt = pointsB_gt.transpose()
    
    #BOYUTLARINI AYARLAMADIN
    pointsB_gt_ext = np.zeros(shape=(2,len(pointsB_gt)))
    pointsB_gt_ext = pointsB_gt - pointsB_x.transpose() #çıkartma yapıldı
    pointsB_gt_sum =np.zeros(shape=(len(pointsB_gt),1))
    pointsB_gt_sum_1 =np.zeros(shape=(len(pointsB_gt),1))
    pointsB_gt_sum_2 =np.zeros(shape=(len(pointsB_gt),1))
    distances =np.zeros(shape=(len(pointsB_gt),1))
    for i in range(len(pointsB_gt_ext)):
       for j in range(len(pointsB_gt_ext[i])):
           pointsB_gt_ext[i][j] = pointsB_gt_ext[i][j] * pointsB_gt_ext[i][j]
           if(j == 1):
               pointsB_gt_sum_1[i] = pointsB_gt_ext[i][j]
           elif( j==0 ):
               pointsB_gt_sum_2[i] = pointsB_gt_ext[i][j]
           pointsB_gt_sum[i] = pointsB_gt_sum_1[i]+pointsB_gt_sum_2[i]
           distances[i] = math.sqrt(pointsB_gt_sum[i])
    
    a=0
    mma = np.zeros(shape=(1,10))
    for th in range(10):
        a=0
        for i in range(len(distances)):
            for j in range(len(distances[i])):
                if(distances[i][j] <= th+1):
                    a=a+1
                    
                mma[0][th]= a
           
    print("Points found = ",len(distances))
    for i in range(len(mma)):
        for j in range(len(mma[i])):
            mma[i][j]= mma[i][j]/len(distances)
            print("\nThreshold =  {0}    Accu =  {1}".format(j,mma[i][j]))
    
    
    return mma
    

#main calculator
def get_hpatches_results(points_A, points_B, h_gt, w, h, H):
    pointsA_x =np.zeros(shape=(len(points_A),len(points_A[1])))
    for i in range(len(points_A)):
       for j in range(len(points_A[i])):
           pointsA_x[i][j] =points_A[i][j]
    
    if len(pointsA_x) == 0:
        mma = np.zeros(shape=(1,10))
        num_points = 0
        hqual_max = np.zeros(shape=(1,5))
        hqual_min = np.zeros(shape=(1,5))
        hqual_all = np.zeros(shape=(10,5))
        
    elif 1<= len(pointsA_x.transpose()) and len(pointsA_x.transpose())<4 :
        mma = get_calculate_mma (points_A, points_B ,h_gt)
        num_points = len(points_A.transpose())
        hqual_max = np.zeros(shape=(1,5))
        hqual_min = np.zeros(shape=(1,5))
        hqual_all = np.zeros(shape=(10,5))
    else:
        mma = get_calculate_mma (points_A, points_B ,h_gt);
        num_points = len(points_A.transpose())
        
    hqual_all = [] 
    for v in range(9):      
        hqual, distance = get_calculate_hom(H, img_A.shape[0],img_A.shape[1],h_gt)
        
        if v == 0:
            hqual_all_total = np.zeros(shape=(10,1))
            sum_h=0
            for j in range(len(hqual[v])):
                sum_h = sum_h+hqual[v][j]
                hqual_all_total[v][0] = sum_h
            
            hqual_all = np.vstack((hqual,hqual))
        else:
            hqual_all = np.vstack((hqual_all,hqual))
            
        sum_h=0
        
        for j in range(len(hqual_all[v])):
            sum_h = sum_h+hqual_all[v][j]
            hqual_all_total[v+1][0] = sum_h
        idx_max = hqual_all_total.max()
        hqual_max = idx_max
        idx_min = hqual_all_total.min()
        hqual_min = idx_min
        
    print("\ndistances = ",distance)
    hqual_all = hqual_all.transpose()
    return [mma, num_points, hqual_max, hqual_min, np.reshape(hqual_all, [1,50])]
    
#To draw_matches
def draw_matches(img_A, img_B, keypoints0, keypoints1):
    
    p1s = []
    p2s = []
    dmatches = []
    for i, (x1, y1) in enumerate(keypoints0):
         
        p1s.append(cv2.KeyPoint(x1, y1, 1))
        p2s.append(cv2.KeyPoint(keypoints1[i][0], keypoints1[i][1], 1))
        j = len(p1s) - 1
        dmatches.append(cv2.DMatch(j, j, 1))
        
    matched_images = cv2.drawMatches(cv2.cvtColor(img_A, cv2.COLOR_RGB2BGR), p1s, 
                                     cv2.cvtColor(img_B, cv2.COLOR_RGB2BGR), p2s, dmatches, None)
    
    return matched_images

#Take arguments and configurations
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Algorithm wrapper for Image Matching Evaluation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_pairs', type=str)

    args = parser.parse_args()  

with open("config.yml", "r") as configfile:
    config = yaml.safe_load(configfile)['configuration']
    
# Make result directory
os.makedirs(config['output_directory'], exist_ok=True)     
        
# Construct FM object
fm = DeepFeatureMatcher(enable_two_stage = config['enable_two_stage'], model = config['model'], 
                    ratio_th = config['ratio_th'], bidirectional = config['bidirectional'], )
    

total_time = 0
total_pairs = 0

#For all pairs in input_pairs perform DFM
with open(args.input_pairs) as f:
    for line in f:
        pairs = line.split(' ')
        pairs[1] = pairs[1].split('\n')[0]
        
        img_A = np.array(Image.open('./' + pairs[0]))
        img_B = np.array(Image.open('./' + pairs[1].split('\n')[0]))
        
        start = time.time()
        
        H, H_init, points_A, points_B = fm.match(img_A, img_B)
        
        files = filter(os.path.isfile, glob.glob("./file/*"))
        for name in files:
            with open(name) as fh:
                h_gt = fh.read()
                h_gt = np.matrix(h_gt)
                h_gt = h_gt.reshape((3,3))
                h_gt = np.squeeze(np.asarray(h_gt))
                
        results = get_hpatches_results (points_A, points_B, h_gt, img_A.shape[0],img_A.shape[1],H )
        
        
        end = time.time()
        
        total_time = total_time + (end - start)
        total_pairs = total_pairs + 1
        
        keypoints0 = points_A.T
        keypoints1 = points_B.T
        
        mtchs = np.vstack([np.arange(0,keypoints0.shape[0])]*2).T
        
        if pairs[0].count('/') > 0:
        
            p1 = pairs[0].split('/')[pairs[0].count('/')].split('.')[0]
            p2 = pairs[1].split('/')[pairs[0].count('/')].split('.')[0]
            
        elif pairs[0].count('/') == 0:
            p1 = pairs[0].split('.')[0]
            p2 = pairs[1].split('.')[0]
                    
        np.savez_compressed(config['output_directory'] + '/' + p1 + '_' + p2 + '_' + 'matches_vgg19', 
                            keypoints0=keypoints0, keypoints1=keypoints1, matches=mtchs)
        
        if config['display_results']: 
            cv2.imwrite(config['output_directory'] + '/' + p1 + '_' + p2 + '_' + 'matches_vgg19' + '.png',
                        draw_matches(img_A, img_B, keypoints0, keypoints1))
       
        
print(f'n \n \nAverage time is: {round(1000*total_time/total_pairs,0)} ms' )    
print(f'Results are ready in ./{config["output_directory"]} directory\n \n \n' )
























