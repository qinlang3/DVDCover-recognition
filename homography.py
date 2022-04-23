#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2 as cv
import random
import os

# Extract descriptors given image fname
def sift_feature_extraction(fname):
    img = cv.imread(fname)
    gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create()
    kp, des = sift.detectAndCompute(gray,None)
    return kp, des

# Return homography transformation given 4 point pairs
def homography(pt_a, pt_b):
    A = np.zeros((8,9))
    for i in range(4):
        x_i = pt_a[i][0]
        y_i = pt_a[i][1]
        x_i_prime = pt_b[i][0]
        y_i_prime = pt_b[i][1]
        A[i*2,:] = np.array([x_i, y_i, 1, 0, 0, 0, 
                             -x_i_prime*x_i, -x_i_prime*y_i, -x_i_prime])
        A[i*2+1,:] = np.array([0, 0, 0, x_i, y_i, 1, 
                               -y_i_prime*x_i, -y_i_prime*y_i, -y_i_prime])
        w, v = np.linalg.eig(A.T@A)     # Get eigenvalue and eigenvector
        # Get H by finding eigenvector with smallest eigenvalue
        H = np.reshape(v[:,np.argmin(w)], (3,3))    
    return H
        

def get_inlier_num(H, matches, query, train):
    count = 0
    for i in range(len(matches)):
        pt_a = query[matches[i][0].queryIdx].pt
        pt_b = train[matches[i][0].trainIdx].pt
        pt_a_prime = np.array([pt_a[0], pt_a[1], 1])
        pt_a_prime = H@pt_a_prime
        pt_a_prime = (pt_a_prime/pt_a_prime[2])[:2]
        d = np.linalg.norm(np.asarray(pt_b)-pt_a_prime)
        if d < 3:
            count += 1
    return count
            

def ransac(matches, query, train):
    first_iter = True
    for i in range(3000):
        samples = random.sample(matches, 4)
        pt_a = []
        pt_b = []
        for j in range(4):
            pt_a.append(query[samples[j][0].queryIdx].pt)
            pt_b.append(train[samples[j][0].trainIdx].pt)
        curr_H = homography(pt_a, pt_b)
        curr_inlier = get_inlier_num(curr_H, matches, query, train)
        if first_iter:
            max_inlier = curr_inlier
            H = curr_H
            first_iter = False
        else:
            if curr_inlier > max_inlier:
                max_inlier = curr_inlier
                H = curr_H
    return H, max_inlier


# Perform homography estimation with RANSAC on target image and reference image.   
def compute_homography_ransac(tagt_fname, ref_fname):
    kp_a, des_a = sift_feature_extraction(ref_fname) 
    kp_b, des_b = sift_feature_extraction(tagt_fname)
    
    # Use BFMatcher() from cv2 to compute keypoint matchings.
    # Use the code from https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des_a,des_b,k=2)
    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
    
    # Perform RANSAC
    H, inlier = ransac(good, kp_a, kp_b)
    
    return H, inlier
   
# tagt_fname is the test image
# ref_fname is the DVD cover image
def visualize_homography(tagt_fname, ref_fname, H):
    ref_img = io.imread(ref_fname)  
    tagt_img = io.imread(tagt_fname)
    # Compute 4 corners of the transformed dvd cover
    pt_a = H@(np.array([0, 0, 1]))
    pt_a = (pt_a/pt_a[2])[:2]
    pt_b = H@(np.array([ref_img.shape[1]-1, 0, 1]))
    pt_b = (pt_b/pt_b[2])[:2]
    pt_c = H@(np.array([ref_img.shape[1]-1, ref_img.shape[0]-1, 1]))
    pt_c = (pt_c/pt_c[2])[:2]
    pt_d = H@(np.array([0, ref_img.shape[0]-1, 1]))
    pt_d = (pt_d/pt_d[2])[:2]
    x = [pt_a[0], pt_b[0], pt_c[0], pt_d[0]]
    y = [pt_a[1], pt_b[1], pt_c[1], pt_d[1]]
    plt.figure(dpi=300)
    gs = gridspec.GridSpec(1, 2,width_ratios=[2,1])
    plt.subplot(gs[0])
    plt.title('Test Image', fontsize=10)
    plt.gca().add_patch(plt.Polygon(xy=list(zip(x,y)), color = 'r', fill=False))
    io.imshow(tagt_img)
    plt.subplot(gs[1])
    plt.title('DVD Cover', fontsize=10)
    io.imshow(ref_img)
    result_name = os.path.splitext(os.path.basename(tagt_fname))[0]+'_dvd_cover'
    plt.savefig(result_name)
    plt.show()
    print("* Saving results as {}".format(result_name+'.png'))
    

   