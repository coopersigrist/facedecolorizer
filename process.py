import cv2

from PIL import Image

def facecrop(input_image_path,
    output_image_path, x_shift, y_shift):

    X_SHIFT = x_shift
    Y_SHIFT = y_shift

    facedata = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(facedata)

    img = cv2.imread(input_image_path)

    faces = cascade.detectMultiScale(img, 1.1, 4)

    for f in faces:
        x, y, w, h = [ v for v in f ]
        x = max(x - X_SHIFT, 0)
        y = max(y - Y_SHIFT, 0)

        max_x = img.shape[0]
        max_y = img.shape[1]

        box_y = min(y+h+Y_SHIFT*2, max_y)
        box_x = min(x+w+X_SHIFT*2, max_x)

        # cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,255))

        sub_face = img[y:box_y, x:box_x]

    # show(sub_face)
    # contoured = contourize(sub_face)
    cv2.imwrite(output_image_path, sub_face)
    # show(contoured)
    # return contoured

def contourize(input_image_path,
    output_image_path):

    img = cv2.imread(input_image_path)
    face2 = cv2.bilateralFilter(img, 11, 17, 17)
    edged = cv2.Canny(face2, 30, 200)
    contoured = cv2.resize(edged, (edged.shape[1]*2,edged.shape[0]*2))
    cv2.imwrite(output_image_path, contoured)

def show(img):
    cv2.imshow('img', img)
    print("Press any key to continue")
    cv2.waitKey(0)

def black_and_white(input_image_path,
    output_image_path):
    img = cv2.imread(input_image_path)
    bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(output_image_path, bw)

def cap(output_image_path = "./capture.jpeg"):

    video_capture = cv2.VideoCapture(0)
    # Check success
    if not video_capture.isOpened():
        raise Exception("Could not open video device")
    # Read picture. ret === True on success
    ret, frame = video_capture.read()
    cv2.imwrite(output_image_path, frame)
    # Close device
    video_capture.release()

