import pixellib
from pixellib.tune_bg import alter_bg
import cv2

change_bg = alter_bg()
change_bg.load_pascalvoc_model("/models/model.h5")
output = change_bg.blur_bg("/datasets/sample.jpg", moderate = True)
cv2.imwrite("img.jpg", output)