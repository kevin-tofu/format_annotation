'''
@author: Kohei Watanabe
@contact: koheitech001[at]gmail.com
'''

import os
import json
import datetime
import numpy as np


def find_file(base_path, dir_list, imgname):

    ret_find = False
    for d in dir_list:
        path_img = base_path + d + imgname
        if os.path.exists(path_img):
            print(path_img)
            ret_find = True
            break
    return ret_find, path_img

def find_file2(base_path, dir_list, imgname):
    ret_find = False
    for d in dir_list:
        path_img = base_path + d + imgname
        if os.path.exists(path_img):
            #print(path_img)
            ret_find = True
            break
    return ret_find, path_img, d + imgname


def openimg2coco_row(row, imshape, id_img, id_annotation, id_cat):
    '''
    #['segmentation', 'area', 'iscrowd', 'image_id', 'bbox', 'category_id', 'id']
    '''
    x1 = int(float(row['XMin']) * imshape[0])
    y1 = int(float(row['YMin']) * imshape[1])
    x2 = int(float(row['XMax']) * imshape[0])
    y2 = int(float(row['YMax']) * imshape[1])

    #print(id_img, type(id_img))

    cocobox = [x1, y1, x2 - x1, y2 - y1]#x1,y1,w,h
    ret = {'segmentation' : [], 
           'area' : int(cocobox[2] * cocobox[3]), 
           'iscrowd' : int(row['IsGroupOf']), 
           'image_id' : int(id_img), 
           'bbox' : cocobox, 
           'category_id' : int(id_cat), 
           'id' : int(id_annotation)}

    return ret


def draw_bbox0(img, bbox, bbcolor):
    
    from PIL import Image, ImageFilter, ImageDraw

    #y, x = int(bbox[1] - bbox[3] / 2), int(bbox[0] - bbox[2] / 2)
    #w, h = bbox[2], bbox[3]
    x, y = bbox[0], bbox[1]
    w, h = bbox[2], bbox[3]

    pilImg = Image.fromarray(np.uint8(img))
    draw = ImageDraw.Draw(pilImg)
    #text_w, text_h = draw.textsize(text)
    label_y = y if y <= text_h else y - text_h
    draw.rectangle((x, label_y, x+w, label_y+h), outline=bbcolor)
    #draw.rectangle((x, label_y, x+text_w, label_y+text_h), outline=bbcolor, fill=bbcolor)
    #draw.text((x, label_y), text, fill=textcolor)

    ret = np.asarray(pilImg)
    #ret.flags.writeable = True
    return ret

def draw_bbox(img, bbox, text, textcolor, bbcolor):

    from PIL import Image, ImageFilter, ImageDraw

    x, y = bbox[0], bbox[1]
    w, h = bbox[2], bbox[3]
    #y, x = int(bbox[1] - bbox[3] / 2), int(bbox[0] - bbox[2] / 2)
    
    pilImg = Image.fromarray(np.uint8(img))
    draw = ImageDraw.Draw(pilImg)
    text_w, text_h = draw.textsize(text)
    label_y = y if y <= text_h else y - text_h
    draw.rectangle((x, label_y, x+w, label_y+h), outline=bbcolor)
    draw.rectangle((x, label_y, x+text_w, label_y+text_h), outline=bbcolor, fill=bbcolor)
    draw.text((x, label_y), text, fill=textcolor)

    ret = np.asarray(pilImg)
    #ret.flags.writeable = True
    return ret

def get_license_coco():
    ret = [{'url': 'http://creativecommons.org/licenses/by-nc-sa/2.0/', 'id': 1, 'name': 'Attribution-NonCommercial-ShareAlike License'},
           {'url': 'http://creativecommons.org/licenses/by-nc/2.0/', 'id': 2, 'name': 'Attribution-NonCommercial License'}, 
           {'url': 'http://creativecommons.org/licenses/by-nc-nd/2.0/', 'id': 3, 'name': 'Attribution-NonCommercial-NoDerivs License'},
           {'url': 'http://creativecommons.org/licenses/by/2.0/', 'id': 4, 'name': 'Attribution License'},
           {'url': 'http://creativecommons.org/licenses/by-sa/2.0/', 'id': 5, 'name': 'Attribution-ShareAlike License'}, 
           {'url': 'http://creativecommons.org/licenses/by-nd/2.0/', 'id': 6, 'name': 'Attribution-NoDerivs License'},
           {'url': 'http://flickr.com/commons/usage/', 'id': 7, 'name': 'No known copyright restrictions'}, 
           {'url': 'http://www.usa.gov/copyright.shtml', 'id': 8, 'name': 'United States Government Work'}]
    return ret 
    
def make_coco_categories(path_coco2017_val):

    #path_coco2017_val = '/data/public_data/COCO2017/annotations/instances_val2017.json'
    anns_list_coco2017_val = json.load(open(path_coco2017_val, 'r'))
    return anns_list_coco2017_val['categories']


def make_coco_images(path_root, fname_list):
    '''
    '''

    from skimage import io

    imgid = 0
    #imgid_list = range(len(fname_list))
    coco_image = list()
    for fname in fname_list:

        path = path_root + fname
        try:
            img = io.imread(path)
        except OSError as e:
            print('Exception Raised', e)
            continue

        coco_image += make_coco_image(imgid, fname, img.shape[0], img.shape[1])
        imgid += 1

    return coco_image


def get_images_coco(path_root, coco_image):
    '''
    '''

    from skimage import io
    image_list = list()
    for ci in coco_image:

        path = path_root + ci['file_name']
        try:
            img = io.imread(path)
        except OSError as e:
            print('Exception Raised', e)
            continue

        image_list.append(img)

    return image_list

def make_coco_info(dataset_name, contributor, url='', version='1.0'):
    
    data_now = datetime.datetime.now()
    info = {'description': dataset_name, 'url': url, 
            'version': version, 'year': data_now.year, 'contributor': 'COCO Consortium', 
            'date_created': str(data_now.date())}
    return info

def make_coco_image(imgid, fn, height, width):
    
    temp = {
        'license': 4,
        'file_name': fn,
        'height': height,
        'width': width,
        'id': imgid,
        #'date_captured': '2013-11-14 17:02:52',
        #'flickr_url': 'http://farm7.staticflickr.com/6116/6255196340_da26cf2c9e_z.jpg',
        #'coco_url': 'http://images.cocodataset.org/val2017/000000397133.jpg'
    }
    
    return [temp]


def make_coco_annotations(annids, bbox, person, keypoints, maxvals):
    ret = list()
    person_loop = 0
    for b_p, is_person in zip(bbox, person):
        
        cat = int(b_p['category_id'])
        #if b_p['score'] < 0.55 or cat != 0:
        if b_p['score'] < 0.55:
            continue
        imgid = b_p['image_id']
        bbox = b_p['bbox']
        
        x1, y1 = bbox[0], bbox[1]
        #y1, x1 = int(bbox[1] - bbox[3] / 2), int(bbox[0] - bbox[2] / 2)
        w, h = bbox[2], bbox[3]
        bbox2 = [x1, y1, w, h]
        annids += 1
        
        if is_person == True:
            _keypoint = keypoints[person_loop].astype(np.int32).ravel().tolist()
            maxvals = np.around(maxvals, decimals=2)
            person_loop += 1
            d = dict(id=annids, image_id=imgid, bbox=bbox2, \
                     keypoints=_keypoint, category_id=cat, iscrowd=0, keyscore=maxvals[:, :, 0].tolist())
        else:
            d = dict(id=annids, image_id=imgid, bbox=bbox2, category_id=cat, iscrowd=0)
        ret.append(d)

    return ret


def make_coco_annotations_bbox(annids, imgid, bbox):

    ret = list()
    for b_p in bbox:
        
        cat = int(b_p['category_id'])
        if b_p['score'] < 0.55:
            continue
        bbox = b_p['bbox']
        x1, y1 = bbox[0], bbox[1]
        #y1, x1 = int(bbox[1] - bbox[3] / 2), int(bbox[0] - bbox[2] / 2)
        w, h = bbox[2], bbox[3]
        bbox2 = [x1, y1, w, h]
        annids += 1
    
        d = dict(id=annids, image_id=imgid, bbox=bbox2, category_id=cat, iscrowd=0)
        ret.append(d)

    return ret


def make_coco_annotations_key(bbox, person, keypoints, maxvals):
    """
    """

    keypoint_list = list()
    person_loop = 0
    maxvals = np.around(maxvals, decimals=2) #maxvals = np.zeros((nr_img, num_keypoints, 1))
    for b_p, is_person in zip(bbox, person):
        
        cat = int(b_p['category_id'])
        #print("cat", cat)
        if b_p['score'] < 0.55 or cat != 1:
            continue

        bbox_id = b_p['id']
        imgid_temp = b_p['image_id']
        bbox_temp = b_p['bbox']
        
        if is_person == True:

            
            _keypoint = keypoints[person_loop].astype(np.int32).ravel().tolist()
            _keyscore = maxvals[person_loop, :, 0].tolist()

            #print(keypoints[person_loop, :, 2])
            _num_keypoints = keypoints[person_loop][keypoints[person_loop, :, 2] > 0].shape[0]
            #print(_num_keypoints)
            person_loop += 1
            
            d = dict(id=bbox_id, image_id=imgid_temp, bbox=bbox_temp, \
                     keypoints=_keypoint, category_id=cat, iscrowd=0, \
                     keyscore=_keyscore, num_keypoints=_num_keypoints)

            keypoint_list.append(d)

    return keypoint_list