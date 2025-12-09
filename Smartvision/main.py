# pip install datasets pillow pandas tqdm

import os
import json
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
import random

# --- Configuration Constants ---

SELECTED_CLASSES = {
    'person': 1, 'bicycle': 2, 'car': 3, 'motorcycle': 4, 'airplane': 5,
    'bus': 6, 'truck': 8, 'traffic light': 10, 'stop sign': 13, 'bench': 15,
    'bird': 16, 'cat': 17, 'dog': 18, 'horse': 19, 'cow': 21, 'elephant': 22,
    'bottle': 44, 'cup': 47, 'bowl': 51, 'pizza': 59, 'cake': 61,
    'chair': 62, 'couch': 63, 'bed': 65, 'potted plant': 64
}

IMAGES_PER_CLASS = 100
BASE_DIR = "smartvision_dataset_simple"
MAX_ITERATIONS = 50000

class COCODatasetCreator:
    def __init__(self):
        self.dataset = None
        # Store collected images in a dict: {'class_name': [list_of_image_dicts]}
        self.class_images = {class_name: [] for class_name in SELECTED_CLASSES.keys()}

    def load_streaming_dataset(self):
        """STEP 1 & 2: Load the COCO dataset in streaming mode."""
        print("📥 Loading COCO dataset in STREAMING mode...")
        self.dataset = load_dataset("detection-datasets/coco", split="train", streaming=True)
        print("✅ Dataset loaded.")

    def collect_images_from_stream(self):
        """STEP 3: Iterate through the stream and collect target images."""
        print(f"\n🔍 Starting image collection. Target: {IMAGES_PER_CLASS} images per class.")
        
        counts = {c: 0 for c in SELECTED_CLASSES.keys()}
        total_collected = 0
        images_processed = 0
        target_total = len(SELECTED_CLASSES) * IMAGES_PER_CLASS

        for item in self.dataset:
            images_processed += 1
            if all(count >= IMAGES_PER_CLASS for count in counts.values()):
                print(f"🎉 Successfully collected all {target_total} images!")
                break
            if images_processed >= MAX_ITERATIONS:
                print(f"⚠️ Reached safety limit of {MAX_ITERATIONS} iterations.")
                break

            annotations = item['objects']
            categories = annotations['category']

            for cat_id in categories:
                for class_name, class_id in SELECTED_CLASSES.items():
                    if cat_id == class_id and counts[class_name] < IMAGES_PER_CLASS:
                        self.class_images[class_name].append({
                            'image': item['image'],
                            'annotations': item['objects']
                        })
                        counts[class_name] += 1
                        total_collected += 1
                        break # Move to next image once a valid class is found

        self._print_collection_summary(images_processed, total_collected, counts)

    def _print_collection_summary(self, processed, collected, counts):
        print("\n" + "="*60)
        print("📊 COLLECTION COMPLETE SUMMARY:")
        print(f"Images Processed: {processed} | Images Collected: {collected}")
        print("="*60)
        for class_name, count in sorted(counts.items()):
            status = "✅" if count >= IMAGES_PER_CLASS else "⚠️"
            print(f"{status} {class_name:20s}: {count:3d} images")
        print("="*60)

    def create_folder_structure(self):
        """STEP 4: Set up necessary directories."""
        print(f"\n📁 Creating project folder structure in {BASE_DIR}...")
        
        folders = [
            f"{BASE_DIR}/classification/train",
            f"{BASE_DIR}/classification/val",
            f"{BASE_DIR}/classification/test",
            f"{BASE_DIR}/detection/images",
            f"{BASE_DIR}/detection/labels"
        ]

        for folder in folders:
            os.makedirs(folder, exist_ok=True)
            if 'classification' in folder:
                for class_name in SELECTED_CLASSES.keys():
                    os.makedirs(f"{folder}/{class_name}", exist_ok=True)
                    
        print("✅ Folder structure created.")

    def prepare_train_val_test_splits(self):
        """STEP 5: Split collected data (70/15/15)."""
        print("\n🔀 Preparing Train/Val/Test splits (70/15/15 ratio)...")
        train_data, val_data, test_data = {}, {}, {}
        metadata = {'splits': {'train': 0, 'val': 0, 'test': 0}}

        for class_name, items in self.class_images.items():
            if not items: continue

            n = len(items)
            train_split = int(0.7 * n)
            val_split = int(0.85 * n)

            # Shuffle items before splitting for a better distribution
            random.shuffle(items) 
            
            train_data[class_name] = items[:train_split]
            val_data[class_name] = items[train_split:val_split]
            test_data[class_name] = items[val_split:]

            metadata['splits']['train'] += len(train_data[class_name])
            metadata['splits']['val'] += len(val_data[class_name])
            metadata['splits']['test'] += len(test_data[class_name])
            
            print(f"{class_name:20s}: Train={len(train_data[class_name]):3d} | Val={len(val_data[class_name]):2d} | Test={len(test_data[class_name]):2d}")

        print(f"\nTotal Images: {metadata['splits']['train'] + metadata['splits']['val'] + metadata['splits']['test']}")
        return train_data, val_data, test_data, metadata

    def save_classification_images(self, splits_data):
        """STEP 6 PART A: Save cropped images for classification task."""
        print("\n" + "="*70)
        print("💾 Saving Classification Images (cropped, 224x224)")
        print("="*70)

        for split_name, data_dict in [('train', splits_data[0]), 
                                      ('val', splits_data[1]), 
                                      ('test', splits_data[2])]:
            
            print(f"📂 Processing {split_name.upper()} split...")
            for class_name, items in tqdm(data_dict.items(), desc=f"  Saving {split_name}"):
                for img_idx, item in enumerate(items):
                    img = item['image']
                    annotations = item['annotations']
                    
                    # Find and crop the correct object from the image
                    cropped_img = self._crop_object_from_image(img, annotations, class_name)
                    
                    if cropped_img:
                        save_path = f"{BASE_DIR}/classification/{split_name}/{class_name}/img_{img_idx+1}.jpg"
                        # Resize to standard input size for many CNNs (224x224)
                        cropped_img.resize((224, 224), Image.Resampling.LANCZOS).save(save_path)

    def _crop_object_from_image(self, img, annotations, target_class_name):
        """Helper to find the bounding box for a specific class and crop the image."""
        target_class_id = SELECTED_CLASSES[target_class_name]
        
        bboxes = annotations['bbox']
        categories = annotations['category']
        
        for bbox, cat_id in zip(bboxes, categories):
            if cat_id == target_class_id:
                # bbox format is [x_min, y_min, width, height]
                x, y, w, h = map(int, bbox)
                # Crop format is [x_min, y_min, x_max, y_max]
                try:
                    cropped_img = img.crop((x, y, x + w, y + h))
                    return cropped_img
                except ValueError:
                    # Sometimes bounding boxes are invalid (e.g., extend outside image)
                    return None
        return None


# --- Execution Flow ---
if __name__ == "__main__":
    creator = COCODatasetCreator()
    
    # 1. Load data stream
    creator.load_streaming_dataset()
    
    # 2. Collect images
    creator.collect_images_from_stream()
    
    # 3. Setup disk structure
    creator.create_folder_structure()
    
    # 4. Split data into train/val/test in memory
    train_data, val_data, test_data, metadata = creator.prepare_train_val_test_splits()
    splits_tuple = (train_data, val_data, test_data)
    
    # 5. Save classification task images to disk
    creator.save_classification_images(splits_tuple)

    # Note: The original script stopped abruptly after saving classification part of step 6. 
    # You would add the rest of the original script logic here if needed 
    # (e.g., saving detection task images in YOLO format).
    print("\nDataset creation process completed up to Classification dataset saving.")
