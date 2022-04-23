#!/bin/bash

# Create vocabulary tree with branch factor k=8 and depth limit l=5.
# It will save the database as 'vocab_tree.pickle'.
# Perform recognization on 'image_01.jpeg'
python dvdCover.py test/image_01.jpeg -k 8 -l 5

# Perform recognization on 'image_02.jpeg'
python dvdCover.py test/image_02.jpeg

# Perform recognization on 'image_03.jpeg'
python dvdCover.py test/image_03.jpeg

# Perform recognization on 'image_04.jpeg'
python dvdCover.py test/image_04.jpeg

# Perform recognization on 'image_05.jpeg'
python dvdCover.py test/image_05.jpeg

# Perform recognization on 'image_06.jpeg'
python dvdCover.py test/image_06.jpeg

# Perform recognization on 'image_07.jpeg'
python dvdCover.py test/image_07.jpeg