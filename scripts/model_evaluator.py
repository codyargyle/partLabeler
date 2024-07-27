import os
import tensorflow as tf
from data_loader import create_dataframe, create_dataset
import openpyxl

def evaluate_model(model_path, test_xml_dir, test_img_dir, batch_size, run_number):
    # Create DataFrame
    test_df = create_dataframe(test_xml_dir, test_img_dir)
    
    # Check if DataFrame is empty
    if test_df.empty:
        raise ValueError('Test DataFrame is empty. Please check the XML files and paths.')
    
    # Create dataset
    test_dataset = create_dataset(test_df, test_img_dir, batch_size)
    
    if test_dataset is None:
        raise ValueError('Test dataset could not be created. Please check the dataset paths and files.')
    
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        results = model.evaluate(test_dataset)
        loss, accuracy = results
        print(f"Test loss: {loss}")
        print(f"Test accuracy: {accuracy}")

        # Record results to Excel
        workbook_path = 'evaluation_results.xlsx'
        if not os.path.exists(workbook_path):
            workbook = openpyxl.Workbook()
            sheet = workbook.active
            sheet.append(['Run Number', 'Loss', 'Accuracy'])
        else:
            workbook = openpyxl.load_workbook(workbook_path)
            sheet = workbook.active

        sheet.append([run_number, loss, accuracy])
        workbook.save(workbook_path)
    else:
        print(f"No model found at {model_path}. Please train a model first.")

if __name__ == '__main__':
    model_save_path = '/home/cody/Documents/codyCodes/partLabeler/models/model.keras'
    test_images_dir = '/home/cody/Documents/codyCodes/partLabeler/data/test/images'
    test_xml_dir = '/home/cody/Documents/codyCodes/partLabeler/data/test'
    batch_size = 32
    run_number = 1  # Adjust this as needed

    evaluate_model(model_save_path, test_xml_dir, test_images_dir, batch_size, run_number)
