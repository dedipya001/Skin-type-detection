# import cv2
# import time
# # import tensorflow as tf

# from Crop_img import getImg
# import FaceDetectionModule as ftm

# # Load ViT model for skin type detection
# from transformers import ViTFeatureExtractor, ViTForImageClassification
# feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
# model = ViTForImageClassification.from_pretrained('/model.a')

# # Function to preprocess image for ViT model
# def preprocess_img_vit(image):
#     inputs = feature_extractor(images=image, return_tensors="pt")
#     return inputs

# # Define skin type classes
# class_names = ["Dry Skin", "Oily Skin", "Normal Skin"]

# # Load face detection model
# detector = ftm.faceDetector(minDetectionCon=0.5)

# # Video capture
# path = "assets/oil-vid.mp4"
# cap = cv2.VideoCapture(path)

# # Initialize variables for FPS calculation
# pTime = 0

# while True:
#     ret, img = cap.read()
#     if not ret:
#         cap = cv2.VideoCapture(path)
#         ret, img = cap.read()

#     # Resize input image
#     img = cv2.resize(img, (800, 480), interpolation=cv2.INTER_AREA)
    
#     # Detect faces
#     img, bboxs = detector.findFaces(img)
    
#     # Perform skin type prediction for the first detected face
#     if bboxs:
#         x, y, w, h = bboxs[0][1]
#         face_img = img[y:y+h, x:x+w]
        
#         # Preprocess image for ViT model
#         preprocessed_img = preprocess_img_vit(face_img)
        
#         # Predict skin type
#         outputs = model(**preprocessed_img)
#         pred_prob = tf.nn.softmax(outputs.logits, axis=1).numpy()[0]
#         pred_class = class_names[pred_prob.argmax()]
#         title = f"Skin Type: {pred_class}, Prob: {pred_prob.max():.2f}%"
        
#         # Overlay skin type prediction on the image
#         cv2.putText(img, text=title, org=(x, y+h+22), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.6,
#                     color=(0, 255, 0), thickness=2)

#     # Calculate and display FPS
#     cTime = time.time()
#     if (cTime - pTime) != 0:
#         fps = 1 / (cTime - pTime)
#         pTime = cTime

#     cv2.putText(img, str(int(fps)), org=(15, 60), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=2,
#                 thickness=3, color=(255, 255, 255))
#     cv2.imshow("Video", img)

#     # Break loop if 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()



import os
import cv2
import time
import tensorflow as tf
from transformers import ViTForImageClassification, ViTFeatureExtractor
import FaceDetectionModule as ftm
from Crop_img import getImg

# Define path to model directory
model_directory = "model.h5"

# Load ViT model for skin type detection
feature_extractor = ViTFeatureExtractor.from_pretrained(model_directory)
model = ViTForImageClassification.from_pretrained(model_directory)

# Function to preprocess image for ViT model
def preprocess_img_vit(image):
    inputs = feature_extractor(images=image, return_tensors="pt")
    return inputs

# Define skin type classes
class_names = ["Dry Skin", "Oily Skin", "Normal Skin"]

# Load face detection model
detector = ftm.faceDetector(minDetectionCon=0.5)

# Video capture
path = "assets/test4.mp4"
cap = cv2.VideoCapture(path)

# Initialize variables for FPS calculation
pTime = 0

while True:
    ret, img = cap.read()
    if not ret:
        cap = cv2.VideoCapture(path)
        ret, img = cap.read()

    # Resize input image
    img = cv2.resize(img, (800, 480), interpolation=cv2.INTER_AREA)
    
    # Detect faces
    img, bboxs = detector.findFaces(img)
    
    # Perform skin type prediction for the first detected face
    if bboxs:
        x, y, w, h = bboxs[0][1]
        face_img = img[y:y+h, x:x+w]
        
        # Preprocess image for ViT model
        preprocessed_img = preprocess_img_vit(face_img)
        
        # Predict skin type
        outputs = model(**preprocessed_img)
        # pred_prob = tf.nn.softmax(outputs.logits, axis=1).numpy()[0]
        pred_prob = tf.nn.softmax(outputs.logits.detach(), axis=1).numpy()[0]

        pred_class = class_names[pred_prob.argmax()]
        title = f"Skin Type: {pred_class}, Prob: {pred_prob.max():.2f}%"
        
        # Overlay skin type prediction on the image
        cv2.putText(img, text=title, org=(x, y+h+22), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.6,
                    color=(0, 255, 0), thickness=2)

    # Calculate and display FPS
    cTime = time.time()
    if (cTime - pTime) != 0:
        fps = 1 / (cTime - pTime)
        pTime = cTime

    cv2.putText(img, str(int(fps)), org=(15, 60), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=2,
                thickness=3, color=(255, 255, 255))
    cv2.imshow("Video", img)

    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
