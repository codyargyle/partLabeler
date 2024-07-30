import os
import pandas as pd
import xml.etree.ElementTree as ET
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def extract_boxes(filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    boxes = []
    for box in root.findall('.//bndbox'):
        xmin = int(box.find('xmin').text)
        ymin = int(box.find('ymin').text)
        xmax = int(box.find('xmax').text)
        ymax = int(box.find('ymax').text)
        boxes.append([xmin, ymin, xmax, ymax])
    return boxes

def create_dataframe(xml_directory, img_directory):
    data = []
    print(f'Walking through the directory: {xml_directory}')
    for subdir, _, files in os.walk(xml_directory):
        print(f'Checking subdir: {subdir}')
        for file in files:
            if file.endswith('.xml'):
                print(f'Found XML file: {file}')
                file_path = os.path.join(subdir, file)
                image_file = file.replace('labeled.xml', '.jpg')
                image_path = os.path.join(img_directory, image_file)
                if os.path.exists(image_path):
                    boxes = extract_boxes(file_path)
                    for box in boxes:
                        data.append([image_file, *box])
                else:
                    print(f'Image file {image_file} not found for XML {file_path}')
    
    if not data:
        print(f'No data found in directory: {xml_directory}')
    
    df = pd.DataFrame(data, columns=['filename', 'xmin', 'ymin', 'xmax', 'ymax'])
    return df

def create_dataset(dataframe, img_directory, batch_size):
    if dataframe.empty:
        print('DataFrame is empty. Check the XML files and their paths.')
        return None
    
    print(f'DataFrame head:\n{dataframe.head()}')

    datagen = ImageDataGenerator(rescale=1.0/255.0)
    
    try:
        generator = datagen.flow_from_dataframe(
            dataframe,
            directory=img_directory,
            x_col='filename',
            y_col=['xmin', 'ymin', 'xmax', 'ymax'],
            target_size=(150, 150),
            batch_size=batch_size,
            class_mode='raw',
            shuffle=True
        )
    except KeyError as e:
        print(f'KeyError: {e}. Check the column names in the DataFrame.')
        return None
    except Exception as e:
        print(f'An error occurred: {e}')
        return None
    
    return generator

def convert_xml_to_yolo(xml_directory, img_directory, yolo_annot_dir):
    if not os.path.exists(yolo_annot_dir):
        os.makedirs(yolo_annot_dir)
    
    class_mapping = {
        'radiator cap': 0,
        'air intake filter': 1,
        'throttle body': 2,
        'intake plenum': 3,
        'ecu box': 4,
        'upper radiator hose': 5,
        'fuse box': 6,
        'battery': 7,
        'radiator': 8,
        'power steering reservoir': 9,
        'oil fill cap': 10,
        'engine timing cover': 11,
        'air intake snorkel': 12,
        'throttle body motor': 13,
        'MAF sensor': 14,
        'grill': 15,
        'brake master cylinder': 16,
        'brake fluid reservoir': 17,
        'clutch fluid reservoir': 18,
        'intake air box': 19,
        'engine cover': 20, 
        'coolant overflow reservoir': 21,
        'windshield wiper fluid reservoir': 22,
        'header': 23
    }
    
    for subdir, _, files in os.walk(xml_directory):
        for file in files:
            if file.endswith('.xml'):
                file_path = os.path.join(subdir, file)
                tree = ET.parse(file_path)
                root = tree.getroot()
                
                img_width = int(root.find('.//size/width').text)
                img_height = int(root.find('.//size/height').text)
                
                annot_file_name = file.replace('.xml', '.txt')
                annot_file_path = os.path.join(yolo_annot_dir, annot_file_name)
                
                with open(annot_file_path, 'w') as f:
                    for obj in root.findall('.//object'):
                        class_name = obj.find('name').text
                        if class_name in class_mapping:
                            class_id = class_mapping[class_name]
                            bndbox = obj.find('bndbox')
                            xmin = int(bndbox.find('xmin').text)
                            ymin = int(bndbox.find('ymin').text)
                            xmax = int(bndbox.find('xmax').text)
                            ymax = int(bndbox.find('ymax').text)
                            
                            x_center = (xmin + xmax) / 2.0 / img_width
                            y_center = (ymin + ymax) / 2.0 / img_height
                            width = (xmax - xmin) / img_width
                            height = (ymax - ymin) / img_height
                            
                            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

    print(f"Converted XML annotations to YOLO format in {yolo_annot_dir}")

