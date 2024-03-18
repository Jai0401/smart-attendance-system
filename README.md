# Face Recognition using OpenCV
 - Create dataset of face images
 - - Detect faces using ```deploy.prototxt``` and ```res10_300x300_ssd_iter_140000.caffemodel```.
 - Extract face embeddings for each face present in the image using pretrained [OpenFace](https://cmusatyalab.github.io/openface/) model ```openface_nn4.small2.v1.t7```. 
 - Train a SVM model on the face embeddings to recognize faces 

## Overview of OpenFace for a single input image
1. Detect faces with a pre-trained models from dlib or OpenCV.
2. Transform the face for the neural network. This repository uses dlib's real-time pose estimation with OpenCV's affine transformation to try to make the eyes and bottom lip appear in the same location on each image.
3. Use a deep neural network to represent (or embed) the face on a 128-dimensional unit hypersphere. The embedding is a generic representation for anybody's face. Unlike other face representations, this embedding has the nice property that a larger distance between two face embeddings means that the faces are likely not of the same person. This property makes clustering, similarity detection, and classification tasks easier than other face recognition techniques where the Euclidean distance between features is not meaningful.
4. Apply clustering or classification techniques to the features to complete the face recognition task.

*Read more about [OpenFace](https://cmusatyalab.github.io/openface/)

## Getting Started
How to use
```    
cd smart-attendance-system
```
 - Create dataset of face images.
 - Place the face images in dataset folder.
 - Extract facial embeddings.
```python extract_embeddings.py```
 - Train the SVM model
```python train_model.py```
 - Test the model
```python recognize_video.py```

## Prerequisites

- Python 3.5
- OpenCV
```
sudo apt-get install python-opencv
```
## Screenshots
<img width="728" alt="Screenshot 2024-03-18 at 7 14 37 PM" src="https://github.com/Jai0401/smart-attendance-system/assets/112328542/3b1e5e9c-11a6-4bd9-bbb4-7128ab477192">
<img width="728" alt="Screenshot 2024-03-18 at 7 14 52 PM" src="https://github.com/Jai0401/smart-attendance-system/assets/112328542/0b15c576-97df-43a2-a85c-69d95a7f2f3b">
<img width="766" alt="Screenshot 2024-03-18 at 7 15 14 PM" src="https://github.com/Jai0401/smart-attendance-system/assets/112328542/6eb929d7-bbae-46de-b3db-aeb1aaca0787">



