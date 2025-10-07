import os
import cv2
import matplotlib.pyplot as plt
from typing import Literal, Callable
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from image_matching import match_orb, match_sift_bf, match_sift_flann


def process_dataset(
        method: Literal["orb", "sift_flann", "sift_bf"],
        dataset_path: str,
        results_folder: str,
):
    threshold = 20  # Adjust this based on tests

    os.makedirs(results_folder, exist_ok=True)
    y_true, y_pred = [], []

    matcher: Callable = {
        "orb": match_orb,
        "sift_flann": match_sift_flann,
        "sift_bf": match_sift_bf,
    }.get(method)

    if matcher is None:
        raise ValueError(f"Unknown method: {method}")

    for folder_name in sorted(os.listdir(dataset_path)):
        folder_path = os.path.join(dataset_path, folder_name)
        if not os.path.isdir(folder_path):
            continue

        image_files = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith(('.tif', '.png', '.jpg', '.jpeg'))
        ]

        if len(image_files) != 2:
            print(f"Skipping '{folder_name}': expected 2 images, found {len(image_files)}")
            continue

        img1_path = os.path.join(folder_path, image_files[0])
        img2_path = os.path.join(folder_path, image_files[1])

        match_count, match_img = matcher(img1_path, img2_path)

        actual_label = 1 if "same" in folder_name.lower() else 0
        predicted_label = 1 if match_count > threshold else 0

        y_true.append(actual_label)
        y_pred.append(predicted_label)

        result_text = "MATCHED" if predicted_label else "UNMATCHED"
        print(f"{folder_name}: {result_text} ({match_count} good matches)")

        if match_img is not None:
            result_filename = f"{folder_name}_{method}_{result_text.lower()}.png"
            result_path = os.path.join(results_folder, result_filename)
            cv2.imwrite(result_path, match_img)
            print(f"    Saved match image → {result_path}")

    # --- Confusion matrix ---
    labels = ["Different (0)", "Same (1)"]
    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix — {method.upper()}")

    cm_path = os.path.join(results_folder, f"{method}_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved confusion matrix at: {cm_path}")

    accuracy = sum(1 for y_t, y_p in zip(y_true, y_pred) if y_t == y_p) / len(y_true)
    return accuracy
