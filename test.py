import math
import sys
import os
import time
import argparse
import torch
import cv2
import numpy as np
import onnxruntime as ort

def evaluate_video(model, video, output_dir="output"):


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


def evaluate_frame(model, frame, size=480, output_dir="output"):


    # make sure output_dir exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    img_original = frame
    
    # rotate image
    #img_original = cv2.rotate(img_original, cv2.ROTATE_90_CLOCKWISE)

    # crop image to square
    img_original = img_original[0:size, 0:size, :]

    # resize image
    img_original = cv2.resize(img_original, (size, size))

    img_rgb = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB) # input is RGB


    # convert frame to input
    input = np.transpose(img_rgb, (2, 0, 1))/255.0 # convert to channel first and float

    # standardize input using mean and std
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225] 
    input[0] = (input[0] - mean[0])/std[0]
    input[1] = (input[1] - mean[1])/std[1]
    input[2] = (input[2] - mean[2])/std[2]

    # add batch dimension
    input = np.expand_dims(input, axis=0).astype(np.float32)
    print(input.shape)

    # convert to tensor
    input = torch.from_numpy(input).cuda()

    # run inference
    with torch.no_grad():
        output = model(input)

        #print(output)

        # unpack output
        x, regression, classification, anchors = output # 

        #regression = regression.cpu().numpy()
        #classification = classification.cpu().numpy()
        #anchors = anchors.cpu().numpy()

        print(len(x))
        print(regression.shape)
        print(classification.shape)
        print(anchors.shape)

        # find max score for each anchor
        max_scores, max_indices = torch.max(classification, dim=2)
        print(max_scores)
        print(max_scores.shape)
        
        print(max_indices.shape)

        # find max score
        max_score, max_idx = torch.max(max_scores, dim=1)
        print(max_score, max_idx)

        print("class: {}".format(max_indices[:, max_idx.item()]))

        print(regression[:, max_idx, :])
        print(classification[:, max_idx, :])
        print(anchors[:, max_idx, :])

        box = anchors[:, max_idx, :].cpu().numpy()[0][0]
        print(box)

        # draw bounding box 
        box = box.astype('int')
        cv2.rectangle(img_original, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.imwrite('frame.jpg', img_original)
        
        exit()

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

    


# python3 test.py --video_path '/home/aau3090/Datasets/teejet/azure/videos/21-10-20/2020-10-21-08-07-23-57_1515-9_7391-NA-TRI-NA-NA-2_0-1_3-33.mp4' --ckpt_path 'best.ckpt'

# python3 test.py --image_path '/home/aau3090/GitHub/TyNet-object-detection/datasets/coco128/images/train2017/000000000154.jpg' --ckpt_path 'best.ckpt'


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    parser.add_argument('--video_path', type=str, required=False, help='Path to a video')
    parser.add_argument('--image_path', type=str, required=False, help='Path to an image')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to the ckpt file')

    args = parser.parse_args()

    # Load model
    import yaml
    dataset_opt = 'coco.yml'
    with open(dataset_opt, 'r') as config:
        dataset_opt = yaml.safe_load(config)

    from models.model import TyNet
    tynet = TyNet(num_classes=len(dataset_opt['obj_list']),
                  ratios=eval(dataset_opt['anchors_ratios']), 
                  scales=eval(dataset_opt['anchors_scales']))

    '''
    #init_weights(model)
    from models.loss import FocalLoss
    loss_fn = FocalLoss()

    from models.utils import get_optimizer, get_scheduler
    from models.detector import Detector

    opt = 'training.yaml'
    with open(opt, 'r') as config:
        opt = yaml.safe_load(config)

    optimizer = get_optimizer(opt['training'], tynet)
    scheduler = get_scheduler(opt['training'], optimizer, 100)
    
    detector = Detector(model=tynet, scheduler=scheduler, optimizer=optimizer, loss=loss_fn)

    # load best checkpoint
    detector = detector.load_from_checkpoint(args.ckpt_path)
    model = detector.model.cuda()
    '''

    #load TyNet.pth'
    tynet.load_state_dict(torch.load("TyNet.pth"))

    model = tynet.cuda()


    # evaluate
    if 0:
        # read video
        video_cap = cv2.VideoCapture(args.video_path)
        evaluate_video(model, video_cap)
    else:
        # read image
        frame = cv2.imread(args.image_path)
        evaluate_frame(model, frame)
