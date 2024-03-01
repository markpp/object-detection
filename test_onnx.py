import math
import sys
import os
import time
import argparse
import torch
import cv2
import numpy as np
import onnxruntime as ort

def evaluate_video(onnx_session, video, output_dir="output"):


    # make sure output_dir exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # iterate over frames in video
    frame_idx = 0

    while(video.isOpened()):
        img_original = video.read()[1]
        
        # rotate image
        #img_original = cv2.rotate(img_original, cv2.ROTATE_90_CLOCKWISE)

        img_rgb = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB) # input is RGB
        #print(img_rgb.shape)

        # convert frame to input
        input = np.transpose(img_rgb, (2, 0, 1))/255.0 # convert to channel first and float

        # add batch dimension
        input = np.expand_dims(input, axis=0).astype(np.float32)
        print(input.shape)

        # run inference
        print("processsing frame {}".format(frame_idx))
        onnx_output = onnx_session.run(None, {onnx_session.get_inputs()[0].name: input})

        print(onnx_output)

        # unpack output
        #boxes, labels, scores, keypoints, keypoints_scores = onnx_output
        x, regression, classification, anchors = onnx_output # 

        print(x.shape)
        print(regression.shape)
        print(classification.shape)
        print(anchors.shape)



        # draw bounding boxes
        for box, key, score, labels in zip(boxes, keypoints, scores, labels):
            if score > 0.01:
                # convert box to int
                box = box.astype('int')
                cv2.rectangle(img_original, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

                print(labels)

                # draw keypoints
                key = key.astype('int') # convert key to int
                cv2.circle(img_original, (key[0][0], key[0][1]), 2, (0, 0, 255), 2)

        frame_idx += 1

        if 1:
            # show frame
            cv2.imshow('frame', img_original)
            key = cv2.waitKey(0)
            if key == 27:
                break
        
        #cv2.imwrite('frame.jpg', img_original)

        #break


def evaluate_frame(onnx_session, frame, output_dir="output"):


    # make sure output_dir exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    img_original = frame
    
    # rotate image
    #img_original = cv2.rotate(img_original, cv2.ROTATE_90_CLOCKWISE)

    # crop image to square
    img_original = img_original[0:480, 0:480, :]

    # resize image
    img_original = cv2.resize(img_original, (480, 480))

    img_rgb = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB) # input is RGB


    # convert frame to input
    input = np.transpose(img_rgb, (2, 0, 1))/255.0 # convert to channel first and float

    # add batch dimension
    input = np.expand_dims(input, axis=0).astype(np.float32)
    print(input.shape)

    # run inference
    onnx_output = onnx_session.run(None, {onnx_session.get_inputs()[0].name: input})

    print(onnx_output['output'])

    # unpack output
    x, regression, classification, anchors, y, z = onnx_output # 

    print(x.shape)
    print(regression.shape)
    print(classification.shape)
    print(anchors.shape)
    print(y.shape)
    print(z.shape)
        

    # draw bounding boxes
    for box, key, score, labels in zip(boxes, keypoints, scores, labels):
        if score > 0.01:
            # convert box to int
            box = box.astype('int')
            cv2.rectangle(img_rgb, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

            print(labels)

            # draw keypoints
            key = key.astype('int') # convert key to int
            cv2.circle(img_rgb, (key[0][0], key[0][1]), 2, (0, 0, 255), 2)

    cv2.imwrite('frame.jpg', img_rgb)

    


# python3 test_onnx.py --video_path '/home/aau3090/Datasets/teejet/azure/videos/21-10-20/2020-10-21-08-07-23-57_1515-9_7391-NA-TRI-NA-NA-2_0-1_3-33.mp4' --onnx_path 'TyNet.onnx'

# python3 test_onnx.py --image_path '/home/aau3090/GitHub/TyNet-object-detection/datasets/coco128/images/train2017/000000000154.jpg' --onnx_path 'TyNet.onnx'


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    parser.add_argument('--video_path', type=str, required=False, help='Path to a video')
    parser.add_argument('--image_path', type=str, required=False, help='Path to an image')
    parser.add_argument('--onnx_path', type=str, required=True, help='Path to the onnx file')

    args = parser.parse_args()

    # Load onnx model
    ort_session = ort.InferenceSession(args.onnx_path, providers=["CUDAExecutionProvider"]) # use GPU
    #ort_session = ort.InferenceSession(args.onnx_path, providers=["CPUExecutionProvider"]) # use CPU

    # evaluate
    if 0:
        # read video
        video_cap = cv2.VideoCapture(args.video_path)
        evaluate_video(ort_session, video_cap)
    else:
        # read image
        frame = cv2.imread(args.image_path)
        evaluate_frame(ort_session, frame)
