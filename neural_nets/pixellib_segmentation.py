import pixellib
from pixellib.instance import instance_segmentation
import cv2

capture = cv2.VideoCapture(0)

segment_video = instance_segmentation()
segment_video.load_model("mask_rcnn_coco.h5")
segment_video.process_camera(capture, frames_per_second= 15, output_video_name="output_video.mp4", show_frames= True,
frame_name= "frame")