import tensorflow.keras
import numpy as np
import cv2

################################################################################
# snippet from teachablMachineWithGoogle

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tensorflow.keras.models.load_model('keras_model.h5')

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Variables
widthImg = 640
heightImg = 480
size = (224, 224)
classNames = ["5TL", "10TL", "20TL", "50TL", "100TL", "200TL"]
index = 0
kernel = np.ones((5,5))
maxValue = 0
################################################################################
# functions

def preProcessing(img):
    imgGray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    # cv2.imshow("imgGray", imgGray)
    imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)
    # cv2.imshow("imgBlur", imgBlur)
    imgCanny = cv2.Canny(imgBlur,200,200)
    # cv2.imshow("imgCanny", imgCanny)
    imgDil = cv2.dilate(imgCanny,kernel,iterations=2)
    # cv2.imshow("imgDil", imgDil)
    imgThres = cv2.erode(imgDil,kernel,iterations=1)
    # cv2.imshow("imgThres", imgThres)
    return imgThres

# getting the biggest contour in the image and returning it
def getContours(img):
    biggest = np.array([])
    maxArea = 0
    contours, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5000:
            # cv2.drawContours(imgOriginal, cnt, -1, (255,0,0),3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
    cv2.drawContours(imgOriginal, biggest, -1, (255, 0, 0), 20)
    return biggest

# reordering the points in the image to get the right order for the warping code
def reOrder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), np.int32)
    add = myPoints.sum(1)

    # get the index of the smallest and biggest value
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    # print('NewPoints', myPointsNew)
    return myPointsNew

# warping the image to get the bird eye view of it
def getWarp(img, biggest):
    biggest = reOrder(biggest)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [580, 0], [0, 270], [580, 270]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv2.warpPerspective(img, matrix, (580, 270))

    return imgOutput

################################################################################
# Running the predictions on the captured video input

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgOriginal = img.copy()
    # resize the image to a 224x224 with the same strategy as in TM2:
    # resizing the image to be at least 224x224 and then cropping from the center

    imgThres = preProcessing(img)
    cv2.imshow("imgThres", imgThres)
    biggest = getContours(imgThres)

    # if availabe use the warped image (for prediction)
    if len(biggest) == 4:
        imgWarped = getWarp(img, biggest)
        # print(imgWarped.shape)
        imgWarped = cv2.resize(imgWarped, (580, 270))
        cv2.imshow("imgWarped", imgWarped)

        # print(img.shape)
        img = imgWarped[0:270, 0:270]
        img = cv2.resize(img, size)
        # turn the image into a numpy array
        image_array = np.asarray(img)
        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        # Load the image into the array
        data[0] = normalized_image_array
        # run the inference
        prediction = model.predict(data)
        maxValue = max(prediction[0])
        # print(maxValue)
        # cv2.imshow('resized', img)
    #else if not availabe, use the original image (for prediction)
    else:
        img = imgOriginal
        img = cv2.resize(img, size)
        # turn the image into a numpy array
        image_array = np.asarray(img)
        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        # Load the image into the array
        data[0] = normalized_image_array
        # run the inference
        prediction = model.predict(data)
        maxValue = max(prediction[0])
        # print(maxValue)
        # cv2.imshow('resized', img)


    if maxValue >= 0.8:
        index = prediction[0].tolist().index(maxValue)
        print(prediction)
        cv2.putText(imgOriginal, classNames[index], (25, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(imgOriginal, str(maxValue), (25, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        maxValue = 0
    cv2.imshow('image', imgOriginal)
    cv2.waitKey(1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
