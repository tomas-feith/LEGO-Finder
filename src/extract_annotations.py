import glob
import json
import os
from posixpath import basename

photo_width = 1024
photo_height = 1024

annotation_src_prefix = 'captures_*'

ids = []

with open('annotation_definitions.json') as file:
    definitions = json.loads(file.read())
    for id in definitions['annotation_definitions'][0]['spec']:
        ids.append(id['label_name'])

with open('obj.names', 'w') as file:
    file.write('\n'.join(ids))

for file_path in glob.glob(annotation_src_prefix):
    with open(file_path) as file:
        annotations_object = json.loads(file.read())
        for photo in annotations_object['captures']:
            filename = os.path.basename(photo['filename'])
            basefilename = filename.split('.')[0]
            annotations = []
        
            for annotation in photo['annotations'][0]['values']:
                id = annotation['label_id']-1
                x = (annotation['x']+annotation['width']/2)/photo_width
                y = (annotation['y']+annotation['height']/2)/photo_height
                width = annotation['width']/photo_width
                height = annotation['height']/photo_height
                annotations.append('%i %f %f %f %f' % (id,x,y,width,height))
            
            photo_annotations_file = open('%s.txt' % basefilename, 'w')
            photo_annotations_file.write('\n'.join(annotations))
            photo_annotations_file.close()