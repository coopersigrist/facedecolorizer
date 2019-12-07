import cv2

from PIL import Image

def facecrop(input_image_path,
    output_image_path):
    facedata = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(facedata)

    img = cv2.imread(input_image_path)

    bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    minisize = (bw.shape[1],bw.shape[0])
    miniframe = cv2.resize(bw, minisize)

    faces = cascade.detectMultiScale(miniframe, 1.1, 4)


    for f in faces:
        x, y, w, h = [ v for v in f ]
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,255))

        sub_face = img[y:y+h, x:x+w]
        cv2.imwrite(output_image_path, sub_face)

    cv2.imshow('img', img)
    cv2.waitKey(0)

    return

 
def black_and_white(input_image_path,
    output_image_path):
   color_image = Image.open(input_image_path)
   bw = color_image.convert('L')
   bw.save(output_image_path)
