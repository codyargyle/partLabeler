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
