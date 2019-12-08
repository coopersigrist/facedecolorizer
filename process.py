import cv2
import argparse

from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--x_shift", type=int, default=0)
parser.add_argument("--y_shift", type=int, default=0)

args = parser.parse_args()

def facecrop(input_image_path,
    output_image_path):

    X_SHIFT = args.x_shift
    Y_SHIFT = args.y_shift

    facedata = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(facedata)

    img = cv2.imread(input_image_path)

    bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # minisize = (bw.shape[1],bw.shape[0])
    # miniframe = cv2.resize(bw, minisize)

    faces = cascade.detectMultiScale(bw, 1.1, 4)


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

    face2 = cv2.bilateralFilter(sub_face, 11, 17, 17)
    edged = cv2.Canny(face2, 30, 200)
    fin = cv2.resize(edged, (edged.shape[1]*6,edged.shape[0]*6))
    cv2.imwrite(output_image_path, fin)
    cv2.imshow('img', fin)
    cv2.waitKey(0)

    return

 
def black_and_white(input_image_path,
    output_image_path):
   color_image = Image.open(input_image_path)
   bw = color_image.convert('L')
   bw.save(output_image_path)

if __name__ == "__main__":

    facecrop("./faces/wolverine_face.jpeg", "./faces/contoured_face.jpeg")

    pass
