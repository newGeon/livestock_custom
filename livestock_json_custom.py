import pandas as pd
import os
import shutil
import uuid
from datetime import datetime
import pymysql
import json
from shapely.geometry import Polygon, MultiPolygon , Point
import numpy as np
from pycocotools import mask as cocomask
import cv2
from tqdm import tqdm


action_category_json = {
    'pig': '돼지',
    'blackpig': '돼지',
    'milkcow': '소',
    'cow': '소'
}

action_json = {
    '섬': 'standing',
    '앉음': 'sitting',
    '누움': 'lying',
    '밥먹음': 'eating',
    '승가': 'mounting',
    '머리털기': 'head shaking',
    '꼬리세움': 'tailing'
}

project = {
    1: 'bbox',
    2: 'polygon',
    6: 'keypoints'
}

category_dict = {
    'pig': '양돈',
    'blackpig': '흑돼지',
    'milkcow': '젖소',
    'cow': '한우'
}

farm_dict = {
    'cowfarmA': '한우 농장A',
    'cowfarmB': '한우 농장B',
    'cowfarmC': '한우 농장C',
    'milkcowfarmA': '젖소 목장A',
    'milkcowfarmB': '젖소 목장B',
    'milkcowfarmC': '젖소 목장C',
    'pigfarmA': '양돈 농장A',
    'blackpigfarmA': '흑돼지 농장A'
}


def get_bbox_job():
    conn = pymysql.connect(host='127.0.0.1', port=3306, user='geon', passwd='1234', db='livestock', charset='utf8', autocommit=False)
    with conn.cursor() as curs:
        sql = """
            select j.id as job_id, dataset_id, dataset_name, image_width, image_height, project_id,
                j.file_id, file_path, org_file_name, file_name, frame_value, file_source, detected_object, job_date, inspection_date, reject_msg, reject_count
            from job_bbox j inner join job_file jf on j.file_id = jf.id
                            inner join dataset d on j.dataset_id = d.id
            where j.dataset_id in {}
            and job_status = 'AK02' and inspection_status = 'AL03'
            and exists(select 1 from job_bbox_result jbr where jbr.job_source_id = j.id)
        """.format(dataset_id_tuple_bbox)

        curs.execute(sql)
        return curs.fetchall()


def get_seg_job():
    conn = pymysql.connect(host='127.0.0.1', port=3306, user='geon', passwd='1234', db='livestock', charset='utf8', autocommit=False)
    with conn.cursor() as curs:
        sql = """
            select j.id as job_id, dataset_id, dataset_name, image_width, image_height, project_id,
                j.file_id, file_path, org_file_name, file_name, frame_value, file_source, detected_object, job_date, inspection_date, reject_msg, reject_count
            from job_segmentation j inner join job_file jf on j.file_id = jf.id
                            inner join dataset d on j.dataset_id = d.id
            where j.dataset_id in {}
            and job_status = 'AK02' and inspection_status = 'AL03'
            and exists(select 1 from job_segmentation_result jbr where jbr.job_source_id = j.id)
        """.format(dataset_id_tuple_seg)

        curs.execute(sql)
        return curs.fetchall()


def get_keypoints_job():
    conn = pymysql.connect(host='127.0.0.1', port=3306, user='geon', passwd='1234', db='livestock', charset='utf8', autocommit=False)
    with conn.cursor() as curs:
        sql = """
            select j.id as job_id, dataset_id, dataset_name, image_width, image_height, project_id,
                j.file_id, file_path, org_file_name, file_name, frame_value, file_source, detected_object, job_date, inspection_date, reject_msg, reject_count
            from job_keypoints j inner join job_file jf on j.file_id = jf.id
                            inner join dataset d on j.dataset_id = d.id
            where j.dataset_id in {}
            and job_status = 'AK02' and inspection_status = 'AL03'
            and exists(select 1 from job_keypoints_result jbr where jbr.job_source_id = j.id)
        """.format(dataset_id_tuple_key)

        curs.execute(sql)
        return curs.fetchall()


# result -> box좌표, 라벨명
def get_annotations(project_name, job_id, dataset_id):
    conn = pymysql.connect(host='127.0.0.1', port=3306, user='geon', passwd='1234', db='livestock', charset='utf8', autocommit=False)
    with conn.cursor() as curs:
        if project_name == 'bbox':
            sql = """
                select jbr.id, result, label_id, label
                  from job_bbox_result jbr inner join dataset_label dl on jbr.label_id = dl.id
                 where 
                job_source_id = %s and dataset_id = %s
            """
        elif project_name == 'polygon':
            sql = """
                select jbr.id, result, label_id, label
                  from job_segmentation_result jbr inner join dataset_label dl on jbr.label_id = dl.id
                 where 
                job_source_id = %s and dataset_id = %s
            """
        elif project_name == 'keypoints':
            sql = """
                select jbr.id, result, label_id, label
                  from job_keypoints_result jbr inner join dataset_label dl on jbr.label_id = dl.id
                 where 
                job_source_id = %s and dataset_id = %s
            """
        curs.execute(sql, (job_id, dataset_id))
        conn.close()
        return curs.fetchall()


def get_coordinate(project_name, object_json, image_width, image_height):
    if 'x' in object_json['value'] and 'width' in object_json['value']:
        if project_name == 'bbox':
            x = object_json['value']['x']
            y = object_json['value']['y']
            width = object_json['value']['width']
            height = object_json['value']['height']
            rectanglelabels = object_json['value']['rectanglelabels'][0]

            # 예외 처리
            if width is None:
                width = 0            
            if height is None:
                height = 0

            if x is None:
                return False

            if x < 0 :
                x = 0
            if y < 0 :
                y = 0

            xtl = int((x / 100) * image_width)
            ytl = int((y / 100) * image_height)
            box_width = int((width / 100) * image_width)
            box_height = int((height / 100) * image_height)
            return xtl, ytl, box_width, box_height
            
        elif project_name == 'keypoints':
            v_keypoints = []
            for v in object_json['value']['points']:
                key_x = float(v['x'] / 100 * image_width)
                key_y = float(v['y'] / 100 * image_height)
                v_keypoints.append(int(key_x))
                v_keypoints.append(int(key_y))
                v_keypoints.append(int(v['destroyed']))
            return v_keypoints


action_category_json = {
    'pig': '돼지',
    'blackpig': '돼지',
    'milkcow': '소',
    'cow': '소'
}


def get_action_name(label_name):
    return action_json[label_name]

def get_actrion_cateogry(detected_object):
    return action_category_json[detected_object]


def get_project_type(project_id):
    return project[project_id]

def get_category_type(detected_object):
    return category_dict[detected_object]


def get_farm_name(org_file_name):
    list_file_info = org_file_name.split('_')
    farm_english = list_file_info[0]
    
    return farm_dict[farm_english]


def get_photo_time(org_file_name):

    list_file_info = org_file_name.split('_')

    time_line = list_file_info[2]
    time_int = int(time_line[8:10])
    time_type = '새벽(0~9시)'

    if time_int >= 20:
        time_type = '야간(20~24시)'
    elif time_int >= 16:
        time_type = '저녁(16~20시)'
    elif time_int >= 12:
        time_type = '오후(12~16시)'
    elif time_int >= 9:
        time_type = '오전(9~12시)'
    else:
        time_type = '새벽(0~9시)'

    return time_type





if __name__ == "__main__":
    
    print('=== 1_2 소(한우, 젖소) 및 돼지 축산업 데이터 JSON 생성 (START) ====================')

    root_path = os.path.dirname(os.path.realpath(__file__))
    json_path = os.path.join(root_path, 'json_live_stock')

    # org_file_path = '/data/synology_nfs/livestock'
    # output_json_path = '/mnt/intflow/NIA75_2022'
    
    dataset_id_tuple_bbox = (321, 351)
    dataset_id_tuple_seg = (358, 365)
    dataset_id_tuple_key = (356, 363)

    dataset_id_bbox = 322
    dataset_id_seg = 357
    dataset_id_key = 355
    
    dataset_id_tuple_bbox = (321, 322, 346, 351)
    dataset_id_tuple_seg = (357, 358, 364, 365)
    dataset_id_tuple_key = (355, 356, 362, 363)
    

    # 321,[Bbox]소(한우) 발정행동 데이터
    # 322,[Bbox]소(젖소) 발정행동 데이터
    # 346,[Bbox]돼지(백돼지) 발정행동 데이터
    # 351,[Bbox]돼지(흑돼지) 발정행동 데이터
    # 357,[seg]소(젖소) 발정행동 데이터
    # 358,[seg]소(한우) 발정행동 데이터
    # 364,[seg]돼지(백돼지) 발정행동 데이터
    # 365,[seg]돼지(흑돼지) 발정행동 데이터
    # 355,[Keypoints]소(젖소) 발정행동 데이터
    # 356,[Keypoints]소(한우) 발정행동 데이터
    # 362,[Keypoints]돼지(백돼지) 발정행동 데이터
    # 363,[Keypoints]돼지(흑돼지) 발정행동 데이터

    df_bbox_job = pd.DataFrame(get_bbox_job(), columns=['job_id', 'dataset_id', 'dataset_name', 'image_width', 'image_height', 
                                                        'project_id', 'file_id', 'file_path', 'org_file_name', 'file_name', 'frame_value', 
                                                        'file_source', 'detected_object', 'job_date', 'inspection_date', 'reject_msg', 'reject_count'])

    df_seg_job = pd.DataFrame(get_seg_job(), columns=['job_id', 'dataset_id', 'dataset_name', 'image_width', 'image_height', 
                                                      'project_id', 'file_id', 'file_path', 'org_file_name', 'file_name', 'frame_value', 
                                                      'file_source', 'detected_object', 'job_date', 'inspection_date', 'reject_msg', 'reject_count'])

    df_key_job = pd.DataFrame(get_keypoints_job(), columns=['job_id', 'dataset_id', 'dataset_name', 'image_width', 'image_height', 
                                                            'project_id', 'file_id', 'file_path', 'org_file_name', 'file_name', 'frame_value', 
                                                            'file_source', 'detected_object', 'job_date', 'inspection_date', 'reject_msg', 'reject_count'])

    df_job_info = pd.concat([df_bbox_job, df_seg_job, df_key_job])

    df_job_info.reset_index(drop=True, inplace=True)

    tqdm.pandas()
    df_job_info['project_name'] = df_job_info.progress_apply(lambda x: get_project_type(x['project_id']), axis=1)
    df_job_info['category_type'] = df_job_info.progress_apply(lambda x: get_category_type(x['detected_object']), axis=1)
    df_job_info['farm_name'] = df_job_info.progress_apply(lambda x: get_farm_name(x['org_file_name']), axis=1)
    df_job_info['time_type'] = df_job_info.progress_apply(lambda x: get_photo_time(x['org_file_name']), axis=1)

    df_job_info2 = df_job_info.groupby(['project_name', 'dataset_name', 'farm_name', 'detected_object']).size().reset_index(name='count')

    action_ratio = []               # 행동분포비율
    output_image_path_list = []     # 이미지목록 
    
    for index, job in tqdm(df_job_info.iterrows(), total=df_job_info.shape[0]):
        images_json = {}
        annotations = []
    
        images_json = {
            'IMAGE_FILE_NAME': job.org_file_name,
            'CATEGORY_TYPE': job.category_type,
            'FARM_NAME': job.farm_name,
            'TIME_TYPE': job.time_type
        }
        
        job_result = get_annotations(job.project_name, job.job_id, job.dataset_id)

        for item in job_result:
            # jbr.id, result, label_id, label
            
            
            # if item[3].replace('/','') != get_category_name(job.image_middle_category):
            #    print("카테고리불일치>>>", job.dataset_id, job.job_id, item[3], job.image_middle_category, get_category_name(job.image_middle_category))
            #    continue

            annotations.append({
                'ID': item[0],
                'TYPE': job.project_name,
                'ACTION_NAME': get_actrion_cateogry(job.detected_object) + '_' + item[3]
            })
            
        
        # ===================================================//행동분류 분포비율
        output_json_data = {
            'INFO': {
                'DATASET_NAME': job.dataset_name,
                'DATASET_DETAIL' : '',
                'VERSION': '1.0',
                'LICENSE': '',
                'CREATE_DATE_TIME': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),            
                'URL': 'https://www.livestock.kr',
            },
            'IMAGE' : images_json,
            'ANNOTATION_INFO': annotations,
        }
        # 아래 경로에 각 json
        # /NIA75_2022/cow/annot_img/**/bbox
        # /NIA75_2022/cow/annot_img/**/keypoints
        # /NIA75_2022/cow/annot_img/**/polygon

        # 라벨링데이터(output_json_path) 넣을 폴더구조
        base_output_path = os.path.join(json_path, job.detected_object, job.project_name)
        # base_output_path = os.path.join(output_json_path, job.detected_object+' 100%', upload_folder, job.project_name)
        
        save_json_path = os.path.join(base_output_path, '라벨데이터')
        # image_path = os.path.join(output_json_path, base_output_path, '이미지데이터')
        
        # =================================================================================================
        # 디렉토리 생성
        if not os.path.exists(save_json_path):
            os.makedirs(save_json_path)
        
        """
        if not os.path.exists(image_path):
            os.makedirs(image_path)
        """

        # 파일명 생성
        save_json_file_path = os.path.join(save_json_path, os.path.splitext(job.org_file_name)[0]+'.json')
        # image_path = os.path.join(image_path, job.org_file_name)

        # print(json_path)
        # print(image_path)  
        # print(json.dumps(output_json_data, indent='\t', ensure_ascii=False, default=str))
        # output_image_path_list.append((job.livestock_path, image_path))
        # break    

        try:
            with open(save_json_file_path, 'w', encoding='utf-8') as file:
                file.write(json.dumps(output_json_data, indent='\t', ensure_ascii=False, default=str))

            # output_image_path_list.append((job.livestock_path, image_path))
        except Exception as e:
            print(e)
            print("에러>>>>", job.dataset_id, job.job_id)
        # =================================================================================================

    print('=== 1_2 소(한우, 젖소) 및 돼지 축산업 데이터 JSON 생성 (SUCCESS) ==================')
    print('================================================================================')

