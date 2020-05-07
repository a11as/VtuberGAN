import cv2
import sys
import os.path
import random, string

CASCADE_FILE = "./dataset/cascade/lbpcascade_animeface.xml"
FACE_ASPECT = (64, 64)

def randomName():
    return ''.join([random.choice(string.ascii_letters + string.digits) for i in range(16)])

def detect(filename, SAVE_FOLDER):
    # ファイルが存在しないときエラー
    if not os.path.isfile(CASCADE_FILE):
        raise RuntimeError("%s: not found" % CASCADE_FILE)

    # カスケードシートを指定
    cascade = cv2.CascadeClassifier(CASCADE_FILE)
    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    # 顔部分をキャプチャ
    faces = cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 5, minSize = (24, 24))
    
    # キャプチャ部分すべてをイメージ化
    for (x, y, w, h) in faces:
        # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        face = image[y:y+h, x:x+w]
        face = cv2.resize(face, FACE_ASPECT)
        cv2.waitKey(0)
        cv2.imwrite("{0}{1}.png".format(SAVE_FOLDER, randomName()), face)

if "__main__" == __name__:
    if len(sys.argv) != 2:
        sys.stderr.write("usage: detect.py <filename>\n")
        sys.exit(-1)
    
    detect(sys.argv[1])
