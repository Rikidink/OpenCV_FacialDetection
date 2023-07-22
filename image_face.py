import cv2
import matplotlib.pyplot as plt

image_path = 'jobs.jpg'
img = cv2.imread(image_path)
print("IMAGE SHAPE", img.shape)
# convert to grayscale for efficiency
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print("GRAY SHAPE", gray_image.shape)



face_classify = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# detectMultiScale() identifies faces of varying sizes in image
# gray_image is the input image
# scaleFactor scales down size of image so that algorithm detects larger faces more easily (in this case, scale down 10%)
# minNeighbours: this value is important, a valid detection is made when a certain number of neighbouring rectangles
# are identified. A low value would result in more false positives, a higher value can result in false negatives
face = face_classify.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40,40))

# drawing the box around face
# face is an array with four values:
# x and y axis which faces were detected
# width and height of the face
# (0, 255, 0) is green (green box), 6 is thickness
for (x, y, w, h) in face:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 5)

# displaying image

# use matplotlib to display image:
# first need to convert BGR to RGB:
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(40,20))
plt.imshow(img_rgb)
plt.axis('off')
plt.show()

# another way to show the image using cv2
# cv2.imshow('image', img)
# cv2.waitKey()
