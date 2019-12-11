import argparse
import os
import process

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="default")
parser.add_argument("--x_shift", type=int, default=0)
parser.add_argument("--y_shift", type=int, default=0)

args = parser.parse_args()

try:
    os.mkdir(args.name)
    dir = args.name
except FileExistsError:
    print("Folder already exists. Please provide a different folder name with flag '--name'.")
    exit(0)


process.cap(dir + "/capture.jpeg")
process.black_and_white(dir + "/capture.jpeg", dir + "/bw_capture.jpeg")
process.facecrop(dir + "/bw_capture.jpeg", dir + "/face_capture.jpeg", args.x_shift, args.y_shift)
process.contourize(dir + "/face_capture.jpeg", dir + "/face_contoured_capture.jpeg")
