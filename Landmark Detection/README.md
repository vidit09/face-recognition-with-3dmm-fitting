## Landmark detection. Implementation from <a href="https://pypi.python.org/pypi/dlib">DLib</a>


1. Get the shape_prediction model <a href="http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2">here.</a> Unzip to use.
2. Run the script to extract the landmark points from directory consisting of images. Specify number of images in each folder.
```python
python -O face_landmark_detection.py shape_predictor_68_face_landmarks.dat ../lfw/images 10
```