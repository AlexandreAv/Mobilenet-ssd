import json
from progressbar import ProgressBar

with open("C:\dataset\COCO dataset\\annotations\instances_val2014.json") as json_data:
    data_dict = json.load(json_data)
    data_set = []

    for x in data_dict:
        print(x)

    images_info = data_dict['images']
    images_annotation = data_dict['annotations']
    print(images_info[0]['id'])

    with ProgressBar(max_value=len(images_info)) as bar:
        i = 0
        for x in images_info:
            elements = None
            id = x['id']
            i += 1
            bar.update(i)

            for y in images_annotation:
                if y['id'] == id:
                    data_set.append({'id': id, 'file_name': x['file_name'], 'category_id': y['category_id'], 'bbox': y['bbox']})
                    break

    with open('C:\dataset\COCO dataset\\annotations\labels\labels_val.json', 'a') as file_labels:
        for data in data_set:
            file_labels.write(json.dumps(data))
