from flask import Flask,jsonify,request
from yolo_detection_images import detectObject

import firebase_admin
from firebase_admin import credentials,storage
import sys
import numpy as np
import cv2

cred = credentials.Certificate("key.json")
app = firebase_admin.initialize_app(cred,
                    {'storageBucket':
                        'itemdetection-efb27.appspot.com'})

bucket = storage.bucket()




# Get the index of the image from command line arguments
image_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0

# Get all blobs in the bucket
blobs = list(bucket.list_blobs())

if image_index < 0 or image_index >= len(blobs):
    print('Invalid image index')
    sys.exit(1)

# Get the blob corresponding to the given index
blob = blobs[0]

print(blob)

#converting blob into string
arr = np.frombuffer(blob.download_as_string(),np.uint8)

#converting string into images
img = cv2.imdecode(arr,cv2.COLOR_BGR2BGR555)

app = Flask(__name__)
@app.route('/det',methods=["POST"])
def detect():
    imageId = request.form['id']
    # getting image from storage by name
    blob = bucket.get_blob(imageId+".jpg")
    # converting blob into string
    arr = np.frombuffer(blob.download_as_string(), np.uint8)
    # converting string into image
    img = cv2.imdecode(arr, cv2.COLOR_BGR2BGR555)

    print(imageId)
    result = detectObject(img)
    return jsonify(result)

cv2.imshow('image',img)
cv2.waitKey(0)

if __name__ == "__main__":
    app.run(host='0.0.0.0')