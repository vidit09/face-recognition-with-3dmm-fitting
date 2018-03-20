#!/usr/bin/python
# The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
#
#   This example program shows how to find frontal human faces in an image and
#   estimate their pose.  The pose takes the form of 68 landmarks.  These are
#   points on the face such as the corners of the mouth, along the eyebrows, on
#   the eyes, and so forth.
#
#   This face detector is made using the classic Histogram of Oriented
#   Gradients (HOG) feature combined with a linear classifier, an image pyramid,
#   and sliding window detection scheme.  The pose estimator was created by
#   using dlib's implementation of the paper:
#      One Millisecond Face Alignment with an Ensemble of Regression Trees by
#      Vahid Kazemi and Josephine Sullivan, CVPR 2014
#   and was trained on the iBUG 300-W face landmark dataset.
#
#   Also, note that you can train your own models using dlib's machine learning
#   tools. See train_shape_predictor.py to see an example.
#
#   You can get the shape_predictor_68_face_landmarks.dat file from:
#   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
#
# COMPILING/INSTALLING THE DLIB PYTHON INTERFACE
#   You can install dlib using the command:
#       pip install dlib
#
#   Alternatively, if you want to compile dlib yourself then go into the dlib
#   root folder and run:
#       python setup.py install
#   or
#       python setup.py install --yes USE_AVX_INSTRUCTIONS
#   if you have a CPU that supports AVX instructions, since this makes some
#   things run faster.  
#
#   Compiling dlib should work on any operating system so long as you have
#   CMake and boost-python installed.  On Ubuntu, this can be done easily by
#   running the command:
#       sudo apt-get install libboost-python-dev cmake
#
#   Also note that this example requires scikit-image which can be installed
#   via the command:
#       pip install scikit-image
#   Or downloaded from http://scikit-image.org/download.html. 

import sys
import os
import dlib
import glob
from scipy.misc import imread,imresize
from skimage.transform import resize

if len(sys.argv) != 4:
    print(
        "Give the path to the trained shape predictor model as the first "
        "argument and then the directory containing the facial images. Later total number of images per class.\n"

        "For example, if you are in the python_examples folder then "
        "execute this program by running:\n"
        " python -O face_landmark_detection.py shape_predictor_68_face_landmarks.dat ../lfw/images 10\n"
        )
    exit()

predictor_path = sys.argv[1]
faces_folder_path = sys.argv[2]
img_set_size = int(sys.argv[3])

landmarks_folder = faces_folder_path+"../landmarks"
print("Output Folder"+landmarks_folder)

if not os.path.exists(landmarks_folder):
            os.makedirs(landmarks_folder)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

if __debug__:
    win = dlib.image_window()
    
# print(glob.glob(os.path.join(faces_folder_path, "*.jpg")))
for folder in glob.glob(os.path.join(faces_folder_path, "*")):
    print("Processing folder: {}".format(folder))
    basename = os.path.basename(folder)
    class_folder = landmarks_folder + '/' + basename

    if not os.path.exists(class_folder):
            os.makedirs(class_folder)

    prev_label = ''
    labels = open(class_folder+'/labels.txt','w')
    single_face_images = open(class_folder+'/single_face_images.txt','w')

    processed_files = 0

    for f in glob.glob(os.path.join(folder, "*.jpg")):
        
        if processed_files >= img_set_size:
            break

        img = imread(f)
        
        if __debug__:
            win.clear_overlay()
            win.set_image(img)

        filename = os.path.splitext(os.path.basename(f))[0]
        tmp = filename.split('_')
        label = '_'.join(tmp[:len(tmp)-1])
        if prev_label != label:
            labels.write(label + '\n')
            prev_label =  label
        # Ask the detector to find the bounding boxes of each face. The 1 in the
        # second argument indicates that we should upsample the image 1 time. This
        # will make everything bigger and allow us to detect more faces.
        dets = detector(img, 0)
        print("Number of faces detected: {}".format(len(dets)))
        if len(dets) > 1:
            continue
        
        processed_files = processed_files+1

        for k, d in enumerate(dets):
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                k, d.left(), d.top(), d.right(), d.bottom()))
            # Get the landmarks/parts for the face in box d.
            shape = predictor(img, d)
            print(shape.num_parts)

            single_face_images.write(filename+'.jpg\n')

            landmark_file = open(class_folder + '/' + filename + "_" + str(k) + ".pts","w")
            landmark_file.write("version: 1\n")
            landmark_file.write("n_points:  68\n{\n")

            
            print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
                                                      shape.part(1)))
            

            # Draw the face landmarks on the screen.
            for landmark_point in range(0,shape.num_parts):
                landmark_file.write(str(shape.part(landmark_point).x) + " " + str(shape.part(landmark_point).y) + "\n")


            landmark_file.write("}\n")


            if __debug__:
                win.add_overlay(shape)

        if __debug__:
            win.add_overlay(dets)
            dlib.hit_enter_to_continue()
