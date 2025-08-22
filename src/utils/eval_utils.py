import json
import csv
import os


def read_predictions_from_txt(file_path):
    """
    Reads predictions_list from a TXT file written by write_to_txt.
    
    Args:
        file_path (str): path to the TXT file

    Returns:
        list[dict]: list of prediction dictionaries
    """
    predictions_list = []
    current_pred = {}

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                # Empty line signals end of a sample block
                if current_pred:
                    predictions_list.append(current_pred)
                    current_pred = {}
            elif line.startswith("-" * 24):
                # separator, ignore
                continue
            else:
                if ": " in line:
                    key, value = line.split(": ", 1)
                    current_pred[key] = value

        # Add last prediction if file does not end with blank line
        if current_pred:
            predictions_list.append(current_pred)

    return predictions_list

def show_prediction_and_label_from_list(filename: str, dataset_split, split_filename2sid: dict, predictions_list: list):
    """
    Display audio, model prediction, and ground-truth label for a given filename,
    using predictions_list (list of dicts with keys: sid, filename, prediction, label)
    
    Args:
        filename (str): audio filename
        dataset_split: HF Dataset split (train/test/valid)
        split_filename2sid (dict): mapping filename -> sample_id
        predictions_list (list[dict]): list of predictions
    """
    # Build dict filename -> prediction dict for fast lookup
    filename2pred = {pred['filename']: pred for pred in predictions_list}

    if filename not in filename2pred:
        raise ValueError(f"Filename '{filename}' not found in predictions_list")

    pred_entry = filename2pred[filename]
    sample_id = pred_entry['sid']
    idx = int(sample_id)
    
    # Get dataset sample
    sample = dataset_split[idx]

    # Play audio
    waveform = sample['audio']['array']
    sr = sample['audio']['sampling_rate']
    display(Audio(waveform, rate=sr))

    # Show prediction and label
    print(f"Filename: {filename}")
    print(f"Prediction: {pred_entry['prediction']}")
    print(f"Ground-truth label: {pred_entry['label']}")




def write_to_txt(file_path, predictions_list):
    """Writes prediction results to a TXT file."""
    with open(file_path, "w", encoding="utf-8") as f:

        for prediction in predictions_list:
            for key, value in prediction.items():
                f.write(f"{key}: {value}\n")
                f.write("-" * 24 + "\n")
            # f.write("-" * 48 + "\n\n")
            f.write("\n\n")

def write_to_json(file_path, predictions_list):
    """Writes prediction results to a JSON file."""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(predictions_list, f, ensure_ascii=False, indent=4)

def write_to_csv(file_path, predictions_list):
    """Writes prediction results to a CSV file."""
    fieldnames = predictions_list[0].keys()
    with open(file_path, mode='w', newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(predictions_list)


def save_predictions(predictions_list, directory, filename):
    """
    Saves predictions in a format determined by the file extension.

    Args:
        predictions_list (list): List of prediction results.
        directory (str): Directory path to save files.
        filename (str): Filename with extension (e.g., 'results.txt', 'results.json', 'results.csv').
    """

    # Ensure directory exists
    os.makedirs(directory, exist_ok=True)

    # Extract file extension
    file_extension = filename.split('.')[-1].lower()
    file_path = os.path.join(directory, filename)

    # Choose appropriate write function
    if file_extension == "txt":
        write_to_txt(file_path, predictions_list)
    elif file_extension == "json":
        write_to_json(file_path, predictions_list)
    elif file_extension == "csv":
        write_to_csv(file_path, predictions_list)
    else:
        raise ValueError("Unsupported file extension. Use '.txt', '.json', or '.csv'.")


def save_metrics(metrics, directory, filename):
    """
    Saves evaluation metrics (e.g., accuracy) to a TXT file.

    Args:
        metrics (dict): Dictionary containing evaluation metrics.
        directory (str): Directory path to save the file.
        filename (str):  Filename with extension .txt
    """
    file_path = os.path.join(directory, filename)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"Experiment Name: {metrics.get('exp_name', 'N/A')}\n")
        f.write(f"Experiment Variant: {metrics.get('exp_variant', 'N/A')}\n")
        f.write("-" * 48 + "\n\n")
        
        for key, value in metrics.items():
            if key != "exp_name" and key != "exp_variant":  # Avoid duplicating experiment name
                f.write(f"{key.capitalize()}: {value}\n")
        
        f.write("-" * 48 + "\n")

