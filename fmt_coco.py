'''
@author: Kohei Watanabe
@contact: koheitech001[at]gmail.gom
'''

import os
import json
import datetime


import numpy as np

def find_files0(base_path, imgname):
    ret_find = False
    path_img = base_path + imgname
    if os.path.exists(path_img):
        ret_find = True

    return ret_find, path_img, imgname

def find_file(base_path, dir_list, imgname):

    ret_find = False
    for d in dir_list:
        path_img = base_path + d + imgname
        if os.path.exists(path_img):
            ret_find = True
            break

    return ret_find, path_img

def find_file2(base_path, dir_list, imgname):
    ret_find = False
    for d in dir_list:
        path_img = base_path + d + imgname
        if os.path.exists(path_img):
            ret_find = True
            break

    return ret_find, path_img, d + imgname

def openimage2coco_row(row, imshape, id_img, id_annotation, id_cat):
    '''
    [segmentation, area, iscrowd, image_id, bbox, category_id, id]
    '''

    x1 = int(float(row['XMin']) * imshape[0])
    y1 = int(float(row['YMin']) * imshape[1])
    x2 = int(float(row['XMax']) * imshape[0])
    y2 = int(float(row['YMax']) * imshape[1])

    cocobox = [x1, y1, x2 - x1, y2 - y1]
    ret = {'segmentation':[], 
           'area': int(cocobox[2] * cocobox[3]),
           'iscrowd': int(row['IsGroupOf']),
           'image_id': int(id_img),
           'bbox': cocobox,
           'category_id': int(id_cat),
           'id': int(id_annotation) 
           }

def draw_bbox(img, bbox, text, textcolor, bbcolor):
    
    from PIL import Image, ImageFilter, ImageDraw

    x, y = bbox[0], bbox[1]
    w, h = bbox[2], bbox[3]

    pilImg = Image.fromarray(np.uinit8(img))
    draw = ImageDraw.Draw(pilImg)

    text_w, text_h = draw.textsize(text)
    label_y = y if y <= text_h else y - text_h
    draw.rectangle((x, label_y, x+w, label_y+h), outline = bbcolor)
    draw.rectangle((x, label_y, x+text_w, label_y+text_h), outline = bbcolor, fill=bbcolor)
    draw.text((x, label_y), text, fill = textcolor)

    ret = np.asarray(pilImg)
    return ret

def get_license_coco():
    ret = [{'url': 'http://creativecommons.org/licenses.by-nc-sa/2.0/', 'id':1, 'name':'Attribution-NonCommercial-ShareAlike License'},
           {'url': 'http://creativecommons.org/licenses.by-nc/2.0/', 'id':2, 'name':'Attribution-NonCommercial License'},
           {'url': 'http://creativecommons.org/licenses.by-nc-nd/2.0/', 'id':3, 'name':'Attribution-NonCommercial-NoDerivs License'},
           {'url': 'http://creativecommons.org/licenses.by/2.0/', 'id':4, 'name':'Attribution License'},
           {'url': 'http://creativecommons.org/licenses.by-sa/2.0/', 'id':5, 'name':'Attribution-ShareAlike License'},
           {'url': 'http://creativecommons.org/licenses.by-nd/2.0/', 'id':6, 'name':'Attribution-NoDerivs License'},
           {'url': 'http://flickr.com/commons/usage/', 'id':7, 'name':'No known copyright restrictions'},
           {'url': 'http://www.usa.gov/copyright.shtml', 'id':8, 'name':'United States Government Work'}
    ]
    return ret

def make_coco_categories(path_coco2017_val):
    
    anns_list_coco2017_val = json.load(open(path_coco2017_val, 'r'))

    return anns_list_coco2017_val['categories']

def make_coco_images(path_root, fname_list):
    '''
    '''

    from skimage import io

    imgid = 0
    coco_image = list()

    for fname in fname_list:
        path = path_root + fname
        try:
            img = io.imread(path)
        except OSError as e:
            print('Exception Rised', e)

        coco_image += make_coco_image(imgid, fname, img.shape[0], img.shape[1])

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
            print('Exception Rised', e)
        
        image_list.append(img)
    
    return image_list


def make_coco_info(dataset_name, contributor, url='', version='1.0'):

    date_now = datetime.datetime.now()
    info = {'description': dataset_name, 'url': url, 
            'version': version, 'year':date_now.year, 'contributor':'COCO Consortium', 
            'data_created':str(date_now.date())}
    return info


def make_coco_image(imgid, fn, height, width):

    temp = {'license': 4,
            'file_name': fn,
            'height': height,
            'width': width,
            'id':imgid,
            #'data_captured: '2013-11-14 17:03:42',
            #'flickr_url':
            #'coco_url: 'https://images.cocodataset.org/val2017/0000003971333.jpg'
            }
    return [temp]


def make_coco_annotations(annids, bbox, person, keypoints, maxvals):
    
    ret = list()
    person_loop = 0
    maxvals = np.aroud(maxvals, decimals=2)
    for b_p, is_person in zip(bbox, person):
        cat = int(b_p['category_id'])
        if b_p['score'] < 0.55:
            continue
        
        imgid = b_p['image_id']
        bbox = b_p['bbox']

        x1, y1 = bbox[0], bbox[1]
        w, h = bbox[2], bbox[3]

        bbox2 = [x1, y1, w, h]

        annids += 1

        if is_person == True:
            _keypoint = keypoints[person_loop].astype(np.int32).ravel().tolist()
            _keyscore = maxvals[person_loop, :, 0].tolist()
            _num_keypoints = keypoints[person_loop][_keypoint[person_loop, :, 2] > 0].shape[0]

            person_loop += 1
            d = dict(id=annids, image_id=imgid, bbox=bbox2, 
                     keypoints=_keypoint, category_id=cat, iscrowd=0, 
                     keyscore=_keyscore, num_keypoints=_num_keypoints)
        else:
            d = dict(id=annids, image_id=imgid, bbox=bbox2, category_id=cat, iscrowd=0)
        ret.append(d)
    return ret



def maek_annotation_fname(sect, name):
    return 'instances_' + sect + name + '.json'

def make_coco_categories_base(supercat, catid, catname):
    d = dict(supercategory=supercat, id=catid, name=catname)
    return d


def make_coco_annotations_bbox(annids, imgid, bboxes, iscrowd=0):

    ret = list()

    for b_p in bboxes:
        cat = int(b_p['category_id'])
        bbox = b_p['bbox']
        x1, y1 = bbox[0], bbox[1]
        w, h = bbox[2], bbox[3]
        bbox2 = [x1, y1, w, h]
        annids += 1
        d = dict(id=annids, image_id=imgid, bbox=bbox2, category_id=cat, iscrowd=0)
        ret.append(d)
    return ret


def make_coco_annotations_key(bbox, person, keypoints, maxvals):
    '''
    '''

    keypoint_list = list()
    person_loop = 0
    maxvals = np.around(maxvals, decimals=2)

    for b_p, is_person in zip(bbox, person):
        cat = int(b_p['category_id'])
        if cat != 1:
            continue

        bbox_id = b_p['id']
        imgid_temp = b_p['image_id']
        bbox_temp = b_p['bbox']

        if is_person == True:

            _keypoint = keypoints[person_loop].astype(np.int32).ravel().tolist()
            _keyscore = maxvals[person_loop, :, 0].tolist()
            _num_keypoints = keypoints[person_loop][_keypoint[person_loop, :, 2] > 0].shape[0]
            
            person_loop += 1
            d = dict(id=bbox_id, image_id=imgid_temp, bbox=bbox_temp, 
                     keypoints=_keypoint, category_id=cat, iscrowd=0, 
                     keyscore=_keyscore, num_keypoints=_num_keypoints)

            keypoint_list.append(d)
    return keypoint_list


def make_coco_category(supercategory, id, name, keypoint=None, skeleton=None):

    ret = [{
            "supercategory": supercategory,
            "id": id,
            "name": name,
            "keypoints": keypoint,
            "skeleton": skeleton
    }]

    return ret
        
