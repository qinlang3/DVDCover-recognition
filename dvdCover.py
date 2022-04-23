#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os
import argparse
from vocabTree import VocabularyTree
import pickle

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='Take in an test image and recognize the DVD cover')
    argparser.add_argument('FILE', help="Input test image")
    argparser.add_argument('-r', '--rebuild', action='store_true', help="Rebuild vocabulary tree")
    argparser.add_argument('-k', '--branch_factor', type=int, help='Branch factor of the vocabulary tree')
    argparser.add_argument('-l', '--depth', type=int, help='Maximum depth of the vocabulary tree.')
    args = argparser.parse_args()
    
    if os.path.exists('vocab_tree.pickle') and not args.rebuild:
        print("* Vocabulary tree file exists. Trying to load it...")
        file = open('vocab_tree.pickle', "rb")
        vt = pickle.load(file)
        file.close()  
    else:
        if not (args.branch_factor and args.depth):
            print("[error]: Pleas provide branch factor and depth for the vocabulary tree.")
            sys.exit()
        else:
            print("* Initializing and building vocabulary tree...")
            vt = VocabularyTree(args.branch_factor, args.depth)
            vt.build()
            print("* Saving vocabulary tree...")
            vt.save()

  
        
    if not os.path.exists(args.FILE):
        print("[error]: Test image {} not found.".format(args.FILE))
        sys.exit()
    
    print("* Performing DVD cover recognition on test image " + args.FILE + "...")
    vt.query(args.FILE)
    print("* DVD cover recognition complete.")
    

    sys.exit()
