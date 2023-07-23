import cv2

frontal_face_classify = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
profile_face_classify = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")

video_capture = cv2.VideoCapture(0)

# this function takes a frame of the webcam video as an input
# similar to the still image facial detection, it just detects faces and draws a box in each video frame
def frontal_detection(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = frontal_face_classify.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=6, minSize=(40,40))
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 5)
    return faces


def profile_detection(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = profile_face_classify.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=4, minSize=(40,40))
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 5)
    return faces



# now we have to keep looping for real-time facial detection
while True:
    result, video_frame = video_capture.read()
    # breaks if the frame wasn't read successfully
    if result is False:
        break

    # using previously made function
    frontal_detection(video_frame)
    profile_detection(video_frame)

    cv2.imshow("WEBCAM", video_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()