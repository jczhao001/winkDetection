from  PIL import Image
import os
import cv2
import matplotlib.pyplot as plt
train_dir_path = "E:/test/eyeclt/0/test/"
save_path = "E:/test/eyeclt/0/test_gray/"

def processimage(finame,desouse,name):
    im = Image.open(finame + name)
    im = im.convert('L')
    im.save(desouse+ name )
def run():
    os.chdir(train_dir_path)
    for i in os.listdir(os.getcwd()):
        processimage(train_dir_path,save_path,i)
run()