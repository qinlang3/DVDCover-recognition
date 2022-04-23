#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
from homography import sift_feature_extraction, compute_homography_ransac, visualize_homography
from sklearn.cluster import KMeans


directory = 'DVDcovers'


# Implement base on the paper:
# https://www-inst.eecs.berkeley.edu//~cs294-6/fa06/papers/nister_stewenius_cvpr2006.pdf

class VocabularyTree:
    ################################
    ## Offline method
    ################################
    
    def __init__(self, k, l):
        self.k = k                                   # branch factor 
        self.max_depth = l                           # depth limit
        self.des = np.zeros((0, 128))                # descriptors for all the images
        self.img_index = np.zeros((0, 1), dtype=int) # self.img_index[i] store the image index for descriptor i
        self.images = []                             # list contains all the images names
        idx = 0
        for f in os.listdir(directory):
            if f.endswith(".jpg"):
                fname = os.path.join(directory, f)
                kp, des = sift_feature_extraction(fname)
                self.des = np.concatenate((self.des, des), axis=0)
                img_idx = np.full((des.shape[0], 1), idx)
                self.img_index = np.concatenate((self.img_index, img_idx), axis=0)
                self.images.append(fname)
                idx += 1
        self.root = -1                                      # root node for the tree
        self.db_vectors = np.zeros((len(self.images), 0))   # database vectors
  
    # Build the tree, construct all the nodes and calculate database vectors
    def build(self):
        self.root = self.recursive_Kmeans(self.max_depth, self.des, self.img_index, len(self.images))
        self.db_vectors = self.db_vectors / np.reshape(np.linalg.norm(self.db_vectors, ord=1, axis=1), (self.db_vectors.shape[0], 1))
        
    # Recursively perform kmeans and create nodes
    def recursive_Kmeans(self, depth, des, img_idx, total_img_num):
        img_num = np.unique(img_idx).shape[0]
        root = Node(des.shape[0], np.log(total_img_num/img_num))
        vector = root.cal_db_vector(img_idx, total_img_num)
        self.db_vectors = np.concatenate((self.db_vectors, vector), axis=1)
        
        # Leaf node
        if depth == 0 or des.shape[0] < self.k:
            return root
        
        kmeans = KMeans(n_clusters=self.k, random_state=0).fit(des)
        labels = kmeans.predict(des)
        root.set_kmeans(kmeans)
        # Create child nodes
        for i in range(self.k):
            child = self.recursive_Kmeans(depth-1, des[labels == i], img_idx[labels == i], total_img_num)
            root.add_child(child)
        return root
 
    # Save the vocabulary tree into disk
    def save(self):
        file = open('vocab_tree.pickle', "wb")
        pickle.dump(self, file)
        file.close()
    
    ################################
    ## Query method
    ################################
    
    def query(self, fname):
        kp, des = sift_feature_extraction(fname)
        
        query_vector = []
        self.cal_query_vector(self.root, des.astype(float), query_vector)
        query_vector = np.reshape(np.array(query_vector), (1, len(query_vector)))
        query_vector = query_vector / np.linalg.norm(np.reshape(query_vector,(query_vector.shape[1],)), ord=1)
        scores = np.linalg.norm((query_vector - self.db_vectors), ord=1, axis=1)
        best_covers = self.get_top_10_imgs(scores)
        opt_inlier = float('-inf')
        for c in best_covers:
            H, inlier = compute_homography_ransac(fname, c)
            if inlier > opt_inlier:
                opt_cover = c
                opt_inlier = inlier
                opt_H = H
        print("* The best DVD cover matched is {}".format(opt_cover))
        print("* Visualizing results...")
        visualize_homography(fname, opt_cover, opt_H)
       
        
      

    def cal_query_vector(self, node, des, query_vector):
        vector = node.cal_query_vector(des)
        query_vector.append(vector)
        
        if len(node.children) != 0:
            if des.shape[0] == 0:
                for c in node.children:
                    self.cal_query_vector(c, des, query_vector)
            else:   
                labels = node.kmeans.predict(des)
                idx = 0
                for c in node.children:
                    self.cal_query_vector(c, des[labels == idx], query_vector)
                    idx += 1
                
        
    def get_top_10_imgs(self, scores):
        best_img_idx = np.argsort(scores)[:10]
        best_imgs = []
        for i in range(10):
            best_imgs.append(self.images[best_img_idx[i]])
        return best_imgs
            
        
    

class Node():
    def __init__(self, des_num, weight):
        self.des_num = des_num       # number of descriptors belong to this node
        self.weight = weight         # node weight
        self.kmeans = -1             # kmeans classfier of this node
        self.children = []           # children list
        
    def cal_db_vector(self, img_idx, img_num):
        vector = np.zeros((img_num, 1))
        for i in range(img_idx.shape[0]):
            vector[img_idx[i, 0], 0] += 1
        vector = vector*self.weight
        
        return vector
    
    def cal_query_vector(self, des):
        return des.shape[0]*self.weight
        
        
    def set_kmeans(self, kmeans):
        self.kmeans = kmeans
        
    def add_child(self, c):
        self.children.append(c)
        
        