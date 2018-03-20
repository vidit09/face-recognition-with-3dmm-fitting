## 3DMM fitting code for the semester Project
For Fitting Code:

1. Get the library from https://github.com/patrikhuber/eos and install in you local directory.
2. Once your setup is ready. Use the three files fit-model.h, helper.cpp and helper.hpp
3. fit-model.h brings the fit-model code for your implementation and model_data with output of the fitting
4. helper.cpp and helper.hpp are for the fitting on the single and multiple images. Later with pose normalisation
and inpainting. In this file paths to be given are
```
"path/to/eos/share/sfm_shape_3448.bin"
"path/to/eos/share/ibug2did.txt"
"path/to/imagedataset/"
```
5. Labeled Faces in Wild is used for the project- http://vis-www.cs.umass.edu/lfw/
5. "path/to/imagedataset/" should have dir structure as
```
.
./images
    /person1
      person1_001.jpg
      person1_002.jpg
./landmarks
    /person1
      person1_001_0.pts
      single_face_images.txt
```
6. Facial Landmarks are generated using <a href="https://pypi.python.org/pypi/dlib">DLib</a> landmark detection. 
6. output folder will be in "path/to/imagedataset/"
as 
```
frontalimages/ - front face image
models/  - for isomaps
```
7. For recognition part RSC algorithm is used. Code provide by Xavier Fontaine from his <a href="https://infoscience.epfl.ch/record/224338/files/1926.pdf">work.</a>