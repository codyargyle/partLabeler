import os
import subprocess
import matplotlib.pyplot as plt

def main():
    # Paths
    data_yaml_path = '/home/cody/Documents/codyCodes/partLabeler/scripts/data.yaml'
    model_save_dir = '/home/cody/Documents/codyCodes/partLabeler/models'
    weights = 'yolov5s.pt'
    
    # Create the command to run the training
    command = [
        'python', 'yolov5/train.py',
        '--img', '640',
        '--batch', '16',
        '--epochs', '20',
        '--data', data_yaml_path,
        '--cfg', 'yolov5/models/yolov5s.yaml',
        '--weights', weights,
        '--name', 'yolov5_model',
        '--project', model_save_dir
    ]
    
    # Execute the training command
    subprocess.run(command, check=True)
    
    # Load the training results
    results_path = os.path.join(model_save_dir, 'yolov5_model', 'results.txt')
    results = {}
    with open(resultspath, 'r') as f:
        for line in f:
            key, value = line.strip().split(':')
            results[key] = float(value)
    
    # Plot training history
    plt.plot(results['box_loss'], label='Box Loss')
    plt.plot(results['obj_loss'], label='Objectness Loss')
    plt.plot(results['cls_loss'], label='Class Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.show()

if __name__ == "__main__":
    main()
