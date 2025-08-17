import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
from PIL import Image
import seaborn as sns
from inference_sdk import InferenceHTTPClient

# ---------------------------
# Client Initialization
# ---------------------------
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="JWAWXkE0gXbvwkazQBCS"  # Replace with your actual API key
)


# ---------------------------
# Utility Functions
# ---------------------------
def normalize_image(image):
    """
    Normalize the pixel intensity values of a grayscale image to the range [0, 255].
    """
    normalized = image / np.max(image)  # Scale to range [0,1]
    return (normalized * 255).astype(np.uint8)


def convert_bbox(prediction, image_shape):
    """
    Convert predicted bounding box from center format (x, y, width, height)
    to corner format (x_min, y_min, x_max, y_max).
    image_shape is used to ensure coordinates are within image bounds.
    """
    x, y, width, height = prediction["x"], prediction["y"], prediction["width"], prediction["height"]
    x_min = max(0, int(x - width / 2))
    x_max = min(image_shape[1], int(x + width / 2))
    y_min = max(0, int(y - height / 2))
    y_max = min(image_shape[0], int(y + height / 2))
    return (x_min, y_min, x_max, y_max)


def compute_iou(boxA, boxB):
    """
    Compute the intersection over union (IoU) between two bounding boxes.
    Both boxes must be in the format (x_min, y_min, x_max, y_max).
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    if boxAArea + boxBArea - interArea == 0:
        return 0
    return interArea / float(boxAArea + boxBArea - interArea)


# ---------------------------
# Inference & Visualization Functions
# ---------------------------
def perform_inference_with_data(image_data):
    """
    Send processed image data to the AI model for inference.
    The image is saved locally for upload.
    """
    image = Image.fromarray(image_data)
    image.save("processed_image.png")
    response = CLIENT.infer("processed_image.png", model_id="void-detection-model-c4rfe/3")
    return response


def apply_threshold_on_boxes(image, predictions):
    """
    Apply Otsu thresholding to regions defined by each predicted bounding box.
    Returns a list of regions (x_min, x_max, y_min, y_max, binary_region).
    """
    n_preds = len([p for p in predictions if p["class"] == "void"])
    fig, axes = plt.subplots(1, max(1, n_preds), figsize=(15, 5))
    thresholded_regions = []
    valid_idx = 0

    for prediction in predictions:
        if prediction["class"] == "void":
            x, y, width, height = prediction["x"], prediction["y"], prediction["width"], prediction["height"]
            x_min = max(0, int(x - width / 2))
            x_max = min(image.shape[1], int(x + width / 2))
            y_min = max(0, int(y - height / 2))
            y_max = min(image.shape[0], int(y + height / 2))
            box_region = image[y_min:y_max, x_min:x_max]

            threshold = threshold_otsu(box_region)
            binary_region = box_region > threshold
            thresholded_regions.append((x_min, x_max, y_min, y_max, binary_region))

            # Display the thresholded region on a subplot
            if n_preds > 1:
                axes[valid_idx].imshow(binary_region, cmap="gray")
                axes[valid_idx].set_title(f"Box {valid_idx + 1}")
                axes[valid_idx].axis("off")
            else:
                axes.imshow(binary_region, cmap="gray")
                axes.set_title(f"Box {valid_idx + 1}")
                axes.axis("off")
            valid_idx += 1

    plt.tight_layout()
    plt.show()
    return thresholded_regions


def visualize_results_with_threshold(original_image, predictions, thresholded_regions, confidence_threshold=0.6):
    """
    Visualize the original grayscale image with detection bounding boxes overlaid.
    Only boxes with a confidence above the threshold are shown.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title("Original Image with Thresholded Regions")
    ax.imshow(original_image, cmap="gray")

    for prediction, region in zip(predictions, thresholded_regions):
        if prediction["class"] == "void" and prediction["confidence"] >= confidence_threshold:
            x_min, x_max, y_min, y_max, _ = region
            rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                 linewidth=2, edgecolor="red", facecolor="none")
            ax.add_patch(rect)
            ax.text(x_min, y_min - 10, f"Conf: {prediction['confidence']:.2f}",
                    color="red", fontsize=8,
                    bbox=dict(facecolor="white", alpha=0.7))
    plt.axis("off")
    plt.show()


def filter_voids_by_size(predictions, min_size=100, max_size=10000):
    """
    Filter void detections based on area (width x height).
    Returns a list of predictions meeting the size criteria.
    """
    filtered_predictions = []
    for prediction in predictions:
        if prediction["class"] == "void":
            width, height = prediction["width"], prediction["height"]
            size = width * height
            if min_size <= size <= max_size:
                filtered_predictions.append(prediction)
    return filtered_predictions


def display_voids_by_size(predictions, min_size, max_size):
    """
    Display information of void detections that meet the size criteria.
    """
    if predictions:
        print(f"Voids in size range [{min_size}, {max_size}]:")
        for prediction in predictions:
            if prediction["class"] == "void":
                width, height = prediction["width"], prediction["height"]
                size = width * height
                print(f"Void - Size: {size}, Confidence: {prediction['confidence']:.2f}")
    else:
        print(f"No voids detected within the size range [{min_size}, {max_size}].")


def create_heat_map(original_image, predictions, confidence_threshold=0.6):
    """
    Create and display a heat map based on the confidence levels of the void detections.
    """
    heat_map = np.zeros(original_image.shape)

    for prediction in predictions:
        if prediction["class"] == "void" and prediction["confidence"] >= confidence_threshold:
            x, y, width, height = map(int,
                                      (prediction["x"], prediction["y"], prediction["width"], prediction["height"]))
            x_min = max(0, x - width // 2)
            x_max = min(original_image.shape[1], x + width // 2)
            y_min = max(0, y - height // 2)
            y_max = min(original_image.shape[0], y + height // 2)
            heat_map[y_min:y_max, x_min:x_max] += prediction["confidence"]

    if np.max(heat_map) > 0:
        heat_map = heat_map / heat_map.max()
        plt.figure(figsize=(10, 8))
        sns.heatmap(heat_map, cmap="YlGnBu", cbar=True, alpha=0.6)
        plt.title("Void Detection Heat Map")
        plt.axis("off")
        plt.show()
    else:
        print("No voids detected above the confidence threshold.")


# ---------------------------
# Ground Truth Extraction
# ---------------------------
def extract_ground_truth_box(color_image):
    """
    Automatically extract the ground truth bounding box from a color image
    by detecting the green-labelled region.

    This function assumes the green label has a high green channel and
    relatively low red and blue channels.
    Returns a tuple (x_min, y_min, x_max, y_max) if a region is found; otherwise, None.
    """
    # If image has an alpha channel, take only the first three channels.
    if color_image.shape[-1] > 3:
        color_image = color_image[:, :, :3]

    green_threshold = 150
    red_blue_threshold = 100

    mask = (
            (color_image[:, :, 1] > green_threshold) &
            (color_image[:, :, 0] < red_blue_threshold) &
            (color_image[:, :, 2] < red_blue_threshold)
    )

    indices = np.argwhere(mask)
    if indices.size == 0:
        print("No ground truth green label detected in the image.")
        return None

    # Compute bounding box based on the detected green pixels.
    y_min, x_min = indices.min(axis=0)
    y_max, x_max = indices.max(axis=0)
    return (x_min, y_min, x_max, y_max)


# ---------------------------
# Final Results Visualization
# ---------------------------
def show_final_results(original_color, ground_truth_box, predictions, confidence_threshold=0.6):
    """
    Display the original color image overlaid with:
      - The ground truth bounding box (in green dashed lines, if available).
      - The predicted bounding boxes (in red solid lines).
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(original_color)
    ax.set_title("Final Result: Ground Truth and Predictions")

    # Draw ground truth box (if available)
    if ground_truth_box:
        x_min, y_min, x_max, y_max = ground_truth_box
        rect_gt = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                edgecolor="green", facecolor="none",
                                linestyle="--", linewidth=2)
        ax.add_patch(rect_gt)
        ax.text(x_min, y_min - 10, "Ground Truth", color="green", fontsize=12,
                bbox=dict(facecolor="white", edgecolor="green", alpha=0.5))

    # Draw predicted boxes (if confidence is met)
    for prediction in predictions:
        if prediction["class"] == "void" and prediction["confidence"] >= confidence_threshold:
            box = convert_bbox(prediction, original_color.shape)
            x_min, y_min, x_max, y_max = box
            rect_pred = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                      edgecolor="red", facecolor="none", linewidth=2)
            ax.add_patch(rect_pred)
            ax.text(x_min, y_max + 10, f"Pred: {prediction['confidence']:.2f}",
                    color="red", fontsize=12, bbox=dict(facecolor="white", alpha=0.5))
    ax.axis("off")
    plt.show()


# ---------------------------
# Main Program
# ---------------------------
if __name__ == "__main__":
    image_path = input("Enter the path to the B-scan image (PNG or JPG): ").strip()

    try:
        # Load image in color for extracting ground truth and final result;
        # also create a grayscale version for inference.
        original_image_color = imread(image_path)
        # If the image has an alpha channel, use only the RGB channels.
        if len(original_image_color.shape) == 3 and original_image_color.shape[2] >= 3:
            original_image_color = original_image_color[:, :, :3]
        else:
            # If the image is already single channel or does not have 3 channels, leave it as is.
            pass

        if len(original_image_color.shape) == 3:
            original_image_gray = (rgb2gray(original_image_color) * 255).astype(np.uint8)
        else:
            original_image_gray = original_image_color

        print("Image loaded successfully.")

        # Automatically extract the ground truth bounding box from the green-labelled region.
        ground_truth_box = extract_ground_truth_box(original_image_color)
        if ground_truth_box:
            print(f"Extracted ground truth bounding box from green label: {ground_truth_box}")
        else:
            print("No ground truth bounding box was extracted.")

        # Run inference using the grayscale image.
        inference_result = perform_inference_with_data(original_image_gray)
        predictions = inference_result.get("predictions", [])

        if not predictions:
            print("No predictions detected from the model.")
        else:
            print("Inference completed.")

            # Print raw predictions for inspection.
            print("\nRaw Predictions (for class 'void'):")
            for idx, pred in enumerate(predictions):
                if pred["class"] == "void":
                    bbox = convert_bbox(pred, original_image_gray.shape)
                    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    print(
                        f"Prediction {idx + 1} - Bounding Box: {bbox}, Area: {area}, Confidence: {pred['confidence']:.2f}")

            # Apply Otsu thresholding on each predicted region and visualize them.
            thresholded_regions = apply_threshold_on_boxes(original_image_gray, predictions)
            visualize_results_with_threshold(original_image_gray, predictions, thresholded_regions,
                                             confidence_threshold=0.6)

            # Filter and display voids by size.
            min_size = 100
            max_size = 10000
            filtered_predictions = filter_voids_by_size(predictions, min_size=min_size, max_size=max_size)
            display_voids_by_size(filtered_predictions, min_size, max_size)

            # Display heat map.
            create_heat_map(original_image_gray, predictions, confidence_threshold=0.6)

            # IoU Calculation and Confusion Matrix Calculation
            if ground_truth_box:
                print("\nCalculating IoU for each predicted 'void' bounding box:")
                iou_threshold = 0.5  # Adjust IoU threshold here if desired
                tp = 0  # True Positives count
                fp = 0  # False Positives count
                for idx, prediction in enumerate(predictions):
                    if prediction["class"] == "void":
                        predicted_box = convert_bbox(prediction, original_image_gray.shape)
                        iou = compute_iou(ground_truth_box, predicted_box)
                        print(f"Prediction {idx + 1} -> Predicted Bounding Box: {predicted_box}, IoU: {iou:.2f}")
                        if iou >= iou_threshold:
                            tp += 1
                        else:
                            fp += 1

                # If no predictions exceed the threshold, count as one False Negative.
                fn = 0 if tp > 0 else 1

                print("\nConfusion Matrix (using IoU threshold of {}):".format(iou_threshold))
                print("===================================")
                print("               Predicted")
                print("             +         -")
                print("Actual +   {:>4}      {:>4}".format(tp, fn))
                print("       -   {:>4}      N/A".format(fp))
            else:
                print("Skipping IoU and confusion matrix calculation since no ground truth bounding box was extracted.")

            # Display final results (ground truth and predicted boxes) overlaid on the original color image.
            show_final_results(original_image_color, ground_truth_box, predictions, confidence_threshold=0.6)

    except Exception as e:
        print(f"An error occurred: {e}")
