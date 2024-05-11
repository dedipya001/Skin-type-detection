import cv2
import time
import tensorflow as tf
from Crop_img import getImg
import FaceDetectionModule as ftm
import os

# Create RealTimeDetections folder if it doesn't exist
if not os.path.exists("./RealTimeDetections"):
    os.makedirs("./RealTimeDetections")

# path = "assets/test.mov"
path = "assets/test3.mp4"
# path = "assets/oil-vid.mp4"
class_names = ["Dry Skin", "Oily Skin","Normal Skin"]
cap = cv2.VideoCapture(path)
pTime = 0
detector = ftm.faceDetector(minDetectionCon=0.5)

# Loading the model
model = tf.keras.models.load_model("new_model.h5")

# Preprocess image function
IMG_SIZE = (224, 224)
def load_and_prep(filepath):
    img_path = tf.io.read_file(filepath)
    img = tf.io.decode_image(img_path)
    img = tf.image.resize(img, IMG_SIZE)
    return img

while True:
    ret, img = cap.read()
    if not ret:
        cap = cv2.VideoCapture(path)
        ret, img = cap.read()

    img = cv2.resize(img, (800, 480), interpolation=cv2.INTER_AREA)
    img, bboxs = detector.findFaces(img)

    if bboxs:  # If faces are detected
        x, y, w, h = bboxs[0][1]  # Assuming only one face is detected
        face_img = img[y:y+h, x:x+w]  # Capture the detected face
        cv2.imwrite("./RealTimeDetections/temp_face.jpg", face_img)  # Save the face image

        # Use the captured image
        img_pred = load_and_prep("./RealTimeDetections/temp_face.jpg")
        pred_prob = model.predict(tf.expand_dims(img_pred, axis=0))
        pred_class = class_names[pred_prob.argmax()]
        title = f"Skin Type: {pred_class}, Prob: {pred_prob.max():.2f}%"

        cv2.putText(img, text=title, org=(x, y+h+22), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.6,
                    color=(0, 255, 0), thickness=2)

    cTime = time.time()
    if (cTime - pTime) != 0:
        fps = 1 / (cTime - pTime)
        pTime = cTime

    cv2.putText(img, str(int(fps)), org=(15, 60), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=2,
                thickness=3, color=(255, 255, 255))
    cv2.imshow("Video", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
