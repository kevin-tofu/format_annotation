import os
import numpy as np
import cv2
from pycocotools.coco import COCO
import pylab as plt
from skimage.draw import line


def set_audio(path_src, path_dst):
    """
    https://kp-ft.com/684
    https://stackoverflow.com/questions/46864915/python-add-audio-to-video-opencv
    """

    import os, shutil
    import moviepy.editor as mp
    import time

    root_ext_pair = os.path.splitext(path_src)
    path_dst_copy = f"{root_ext_pair[0]}-copy{root_ext_pair[1]}"
    shutil.copyfile(path_dst, path_dst_copy)
    time.sleep(0.5)
    # print("path_dst_copy: ", path_dst_copy)

    # Extract audio from input video.                                                                     
    clip_input = mp.VideoFileClip(path_src)
    # clip_input.audio.write_audiofile(path_audio)
    # Add audio to output video.                                                                          
    clip = mp.VideoFileClip(path_dst_copy)
    clip.audio = clip_input.audio

    time.sleep(0.5)
    clip.write_videofile(path_dst)

    time.sleep(0.5)
    os.remove(path_dst_copy)


def minmax(value, _min, _max):
    value = min(value, _max)
    value = max(value, _min)
    return value

def get_pairs(anns, idx: int=0):

    cat_name = anns["categories"][idx]["name"]
    if cat_name != "person":
        # raise HTTPException(status_code=503, detail="The category is not person") 
        return None
    pairs = anns["categories"][idx]["skeleton"]
    keypoint_name = anns["categories"][idx]["keypoints"]

    return pairs, keypoint_name
    

def colored_pairs(keypoints_name, pairs, regex_list):
    """
    """
    import re
    parisList = list()
    for i in range(len(regex_list) + 1):
        parisList.append(list())

    # regex_list = ['.*[lL][eE][fF][tT].*']
    # print(pairs)
    # print(keypoints_name, len(keypoints_name))
    for p in pairs:
        print(p[0] - 1, p[1] - 1)
        key1 = keypoints_name[p[0] - 1]
        key2 = keypoints_name[p[1] - 1]
        
        count_NotMatched = 0
        for i_regex, regex in enumerate(regex_list):

            # print(key1, key2, bool(re.match(regex, key1)) and bool(re.match(regex, key2)))
            if bool(re.match(regex, key1)) and bool(re.match(regex, key2)):
                # print("matched")
                parisList[i_regex].append(p)
            else:
                count_NotMatched += 1
        if count_NotMatched == len(regex_list):
            parisList[-1].append(p)

    return parisList


def draw_keypoint2img(img, labels, pairs, color = [255, 0, 0], th=0.5):

    ret = np.copy(img)
    for label in labels:
        keypoints = np.array(label['keypoints'])
        keypoints = keypoints.reshape((keypoints.shape[0]//3, 3))
        scores = np.array(label['keyscore'])
        for pair in pairs:
            score1 = scores[pair[0]]    
            score2 = scores[pair[1]]
            if score1 < th or score2 < th:
                continue
            x1 = int(minmax(keypoints[pair[0] - 1][0], 0, ret.shape[1] - 1))
            y1 = int(minmax(keypoints[pair[0] - 1][1], 0, ret.shape[0] - 1))
            x2 = int(minmax(keypoints[pair[1] - 1][0], 0, ret.shape[1] - 1))
            y2 = int(minmax(keypoints[pair[1] - 1][1], 0, ret.shape[0] - 1))

            # print(y1, x1, y2, x2)
            rr, cc = line(y1, x1, y2, x2)
            _color = np.array(color).astype(np.uint8)
            ret[rr, cc, :] = _color
        
    return ret


def draw_keypoint2img_colors(img, labels, pairsList, colorList, th=0.5):

    ret = np.copy(img)

    for label in labels:
        keypoints = np.array(label['keypoints'])
        # print("keypoints:", keypoints.shape)
        keypoints = keypoints.reshape((keypoints.shape[0] // 3, 3)) # (22, 3) ?
        scores = keypoints[:, 2] 

        # print("keypoints:", keypoints.shape)
        # scores = np.array(label['keyscore'])
        # print(keypoints.shape)
        for pairs, color in zip(pairsList, colorList):
            for pair in pairs:
                # print(pair)
                # score1 = scores[pair[0]]    
                # score2 = scores[pair[1]]
                # if score1 < th or score2 < th:
                #     continue
                
                x1 = int(minmax(keypoints[pair[0] - 1][0], 0, ret.shape[1] - 1))
                y1 = int(minmax(keypoints[pair[0] - 1][1], 0, ret.shape[0] - 1))
                x2 = int(minmax(keypoints[pair[1] - 1][0], 0, ret.shape[1] - 1))
                y2 = int(minmax(keypoints[pair[1] - 1][1], 0, ret.shape[0] - 1))
                rr, cc = line(y1, x1, y2, x2)
                _color = np.array(color).astype(np.uint8)
                ret[rr, cc, :] = _color
        
    return ret


def draw_keypoint2video_colors(path_video, \
                               path_video_dst, \
                               labels, pairsList, colorList, th=0.5):

    from mediapipe_if.parse import set_audio

    cap = cv2.VideoCapture(path_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fname_ext = os.path.splitext(path_video_dst)[-1]
    if fname_ext == ".mp4" or fname_ext == ".MP4":
        fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    else:
        raise ValueError(" mp4 video only ")

    writer = cv2.VideoWriter(path_video_dst, fmt, fps, (width, height))
    img_id = -1
    
    while True:
    # for labels_images in labels["images"]:

        img_id += 1
        success, image = cap.read()
        # _id = labels_images["id"]
        ann_image = [loop for loop in labels["images"] if loop["id"] == img_id]
        # print(ann_image)

        if success:
            # image_height, image_width, _ = image.shape
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.asarray(image)

            if len(ann_image) == 1:
                l_point_ignore = [loop for loop in labels["annotations"] \
                              if loop["image_id"] == ann_image[0]["id"]]
                image_keypoint = draw_keypoint2img_colors(image, 
                                                          l_point_ignore, 
                                                          pairsList, 
                                                          colorList, 
                                                          th=th)
            else:
                image_keypoint = image    
            
            image_keypoint = cv2.cvtColor(image_keypoint, cv2.COLOR_RGB2BGR)
            writer.write(image_keypoint)

        else:
            break

    cap.release()
    writer.release()

    set_audio(path_video, path_video_dst)


# http://10.115.1.14/kohei/yolo/-/blob/master/visualize.py
def draw_bbox2img(img, labels, fmt="x1y1wh", th=0.5, color = (255, 0, 0)):

    from skimage.draw import rectangle, rectangle_perimeter


    ret = np.copy(img)

    for label in labels:
        bbox = label["bbox"]
        if fmt == "x1y1wh":
            x1 = int(bbox[0])
            y1 = int(bbox[1])
            w = int(bbox[2])
            h = int(bbox[3])
        elif fmt == "xywh":
            w = int(bbox[2])
            h = int(bbox[3])
            x1 = int(bbox[0]) - w / 2 
            y1 = int(bbox[1]) - h / 2
        elif fmt == "x1y1x2y2":
            x1 = int(bbox[0])
            y1 = int(bbox[1])
            w = int(bbox[2]) - x1
            h = int(bbox[3]) - y1

        x2 = x1 + w
        y2 = y1 + h
        x1 = np.clip(x1, 4, ret.shape[1] - 4)
        x2 = np.clip(x2, 4, ret.shape[1] - 4)
        y1 = np.clip(y1, 4, ret.shape[0] - 4)
        y2 = np.clip(y2, 4, ret.shape[0] - 4)
        # print(x1, y1, x2, y2)

        color_line = np.array(color, dtype=np.uint8)
        # color_line = np.array([255, 0, 0], dtype=np.uint8)
        rr, cc = rectangle_perimeter(start = (y1, x1), end = (y2, x2))
        ret[rr, cc] = color_line
        
        
    return ret