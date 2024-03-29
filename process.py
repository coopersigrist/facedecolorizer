import cv2
import time

def facecrop(input_image_path, output_image_path, x_shift, y_shift):
    X_SHIFT = x_shift
    Y_SHIFT = y_shift

    facedata = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(facedata)

    img = cv2.imread(input_image_path)

    faces = cascade.detectMultiScale(img, 1.1, 4)

    for f in faces:
        x, y, w, h = [ v for v in f ]
        new_x = max(x - X_SHIFT, 0)
        new_y = max(y - Y_SHIFT, 0)

        max_x = img.shape[1]
        max_y = img.shape[0]

        box_y = min(y+h+Y_SHIFT, max_y)
        box_x = min(x+w+X_SHIFT, max_x)

        # cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,255))

        sub_face = img[new_y:box_y, new_x:box_x]

    # show(sub_face)

    cv2.imwrite(output_image_path, sub_face)




def contourize(input_image_path, output_image_path):
    img = cv2.imread(input_image_path)
    face2 = cv2.bilateralFilter(img, 11, 17, 17)
    edged = cv2.Canny(face2, 30, 200)
    contoured = cv2.resize(edged, (edged.shape[1]*2,edged.shape[0]*2))
    cv2.imwrite(output_image_path, contoured)

def svg(input_image_path, output_image_path):
    img = cv2.imread(input_image_path)
    # cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # ret,thresh = cv2.threshold(img,27,25,0)
    # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    edged = cv2.Canny(img, 30, 200)
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # show(edged)
    c = max(contours, key=cv2.contourArea)
    f = open(output_image_path, 'w+')

    f.write('<svg width="'+str(img.shape[1])+'" height="'+str(img.shape[0])+'" xmlns="http://www.w3.org/2000/svg">')
    f.write('<path d="M')

    for i in range(len(c)):
        #print(c[i][0])
        x, y = c[i][0]
        # print(x)
        f.write(str(x)+  ' ' + str(y)+' ')

    f.write('"/>')
    f.write('</svg>')
    f.close()


def black_and_white(input_image_path, output_image_path):
    img = cv2.imread(input_image_path)
    bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(output_image_path, bw)


def cap(output_image_path = "./capture.jpeg"):
    i = 3
    print("Taking picture in ", end="")
    while(i>0):
        print(str(i))
        i-=1
        time.sleep(1)
    video_capture = cv2.VideoCapture(0)
    # Check success
    if not video_capture.isOpened():
        raise Exception("Could not open video device")
    # Read picture. ret === True on success
    ret, frame = video_capture.read()
    cv2.imwrite(output_image_path, frame)
    # Close device
    video_capture.release()


def show(img):
    cv2.imshow('img', img)
    print("Press any key to continue")
    cv2.waitKey(0)
