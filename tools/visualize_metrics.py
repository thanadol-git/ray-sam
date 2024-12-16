import json
import os
import matplotlib.pyplot as plt

# File path to the result.json
file_path = '/storage/raysam_user/tmp/sam_finetuning_ray/sam_finetuning_ray/TorchTrainer_aec84_00000_0_2024-12-15_22-05-23/result.json'

# Output directory for saving plots
output_dir = './plots'
os.makedirs(output_dir, exist_ok=True)

# Function to read and parse the JSON file
def read_results(file_path):
    results = []
    with open(file_path, 'r') as f:
        for line in f:
            results.append(json.loads(line))
    return results

# Function to extract metrics
def extract_metrics(results):
    epochs = []
    train_loss = []
    val_loss = []
    train_instance_loss = []
    val_instance_loss = []
    train_model_iou = []
    val_model_iou = []
    metric_val = []

    for result in results:
        epochs.append(result['epoch'])
        train_loss.append(result['train']['loss'])
        val_loss.append(result['val']['loss'])
        train_instance_loss.append(result['train']['instance_loss'])
        val_instance_loss.append(result['val']['instance_loss'])
        train_model_iou.append(result['train']['model_iou'])
        val_model_iou.append(result['val']['model_iou'])
        metric_val.append(result['val']['metric_val'])

    return {
        'epochs': epochs,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_instance_loss': train_instance_loss,
        'val_instance_loss': val_instance_loss,
        'train_model_iou': train_model_iou,
        'val_model_iou': val_model_iou,
        'metric_val': metric_val,
    }

# Function to plot and save metrics
def plot_metrics(metrics, output_dir):
    # Plot training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['epochs'], metrics['train_loss'], label='Train Loss', marker='o')
    plt.plot(metrics['epochs'], metrics['val_loss'], label='Validation Loss', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, 'loss_plot.png'))
    plt.close()

    # Plot instance loss
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['epochs'], metrics['train_instance_loss'], label='Train Instance Loss', marker='o')
    plt.plot(metrics['epochs'], metrics['val_instance_loss'], label='Validation Instance Loss', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Instance Loss')
    plt.title('Training and Validation Instance Loss')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, 'instance_loss_plot.png'))
    plt.close()

    # Plot model IoU
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['epochs'], metrics['train_model_iou'], label='Train Model IoU', marker='o')
    plt.plot(metrics['epochs'], metrics['val_model_iou'], label='Validation Model IoU', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Model IoU')
    plt.title('Training and Validation Model IoU')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, 'model_iou_plot.png'))
    plt.close()

    # Plot metric_val
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['epochs'], metrics['metric_val'], label='Validation Metric Value', marker='o', color='purple')
    plt.xlabel('Epochs')
    plt.ylabel('Metric Value')
    plt.title('Validation Metric Value')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, 'metric_val_plot.png'))
    plt.close()

# Main script
if __name__ == '__main__':
    # Read the results from the JSON file
    results = read_results(file_path)

    # Extract metrics
    metrics = extract_metrics(results)

    # Plot and save metrics
    plot_metrics(metrics, output_dir)

    print(f"Plots saved in {output_dir}")