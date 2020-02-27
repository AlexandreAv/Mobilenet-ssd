import json
from progressbar import ProgressBar
import pdb

with open(r"C:\dataset\COCO dataset\annotations\instances_train2014.json") as json_data:
    data_dict = json.load(json_data)
    data_set = []

    for x in data_dict:
        print(x)

    images_info = data_dict['images']
    images_annotation = data_dict['annotations']
    print(images_info[0])
    print(images_annotation[0])

    # pdb.set_trace()

    with ProgressBar(max_value=len(images_info)) as bar:
        i = 0
        for x in images_info:
            elements = None
            image_id = x['id']
            i += 1
            bar.update(i)

            data = []
            for y in images_annotation:
                if y['image_id'] == image_id:
                    data.append({'category_id': y['category_id'], 'bbox': y['bbox']})

            # pdb.set_trace()
            data_set.append({'id': image_id, 'file_name': x['file_name'], 'height': x['height'], 'width': x['width'],
                             'object_detection': data})
            data = None

    with open(r'labels_files/labels train.json', 'a') as file_labels:
        for data in data_set:
            file_labels.write(json.dumps(data))
