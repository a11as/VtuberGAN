import os
import glob
from face_cap import detect

SAVE_FOLDER = "./dataset/"
TARGET_DIR = "./original/**/*"

files = glob.glob(TARGET_DIR)

for _ in files:
    if not os.path.exists(SAVE_FOLDER):
        os.mkdir(SAVE_FOLDER)
    print("target path : %s" % _)
    try:
        detect(_, SAVE_FOLDER)
    except:
        os.remove(_)