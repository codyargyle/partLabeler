import os
import xml.etree.ElementTree as ET

def convert_to_yolo(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def convert_annotation(xml_path, output_dir, classes):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    image_id = root.find('filename').text.split('.')[0]
    output_path = os.path.join(output_dir, f"{image_id}.txt")

    with open(output_path, 'w') as out_file:
        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in classes or int(difficult) == 1:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
            bb = convert_to_yolo((w, h), b)
            out_file.write(f"{cls_id} " + " ".join([str(a) for a in bb]) + '\n')

def create_yolo_annotations(input_dir, output_dir, classes):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.endswith('.xml'):
            xml_path = os.path.join(input_dir, filename)
            convert_annotation(xml_path, output_dir, classes)

if __name__ == "__main__":
    input_dir = '/home/cody/Documents/codyCodes/partLabeler/data/test'  # Directory containing XML files
    output_dir = '/home/cody/Documents/codyCodes/partLabeler/data/yolo_labels'  # Directory to save YOLO formatted labels
    classes = ["radiator cap", "air intake filter", "throttle body", "intake plenum", "ecu box", 
               "upper radiator hose", "fuse box", "battery", "radiator", "power steering reservoir",
               "oil fill cap", "engine timing cover", "engine cover", "coolant overflow reservoir", "air intake snorkel", "throttle body motor",
               "MAF sensor", "grill", "air intake box", "brake master cylinder", "brake fluid reservoir", "clutch fluid reservoir"]

    create_yolo_annotations(input_dir, output_dir, classes)
