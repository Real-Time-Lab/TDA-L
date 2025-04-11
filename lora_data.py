import os
import torch
import random
import argparse
import clip
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from utils import build_test_data_loader
from PIL import Image

class FilePathDataset(Dataset):
    def __init__(self, samples, transform=None):
        """
        Dataset that loads images from file paths on demand.
        samples: List of (image_path, label) tuples
        """
        self.samples = samples
        self.transform = transform
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image as fallback
            blank = Image.new('RGB', (224, 224), color=(128, 128, 128))
            if self.transform:
                blank = self.transform(blank)
            return blank, label

def extract_image_path(dataset, dataset_name, idx):
    """Get image path using dataset-specific optimizations"""
    
    # Fast path for common datasets
    if dataset_name == 'I':  # ImageNet
        # ImageNet typically has a standardized structure
        if hasattr(dataset, 'samples'):
            return dataset.samples[idx][0]
        if hasattr(dataset, 'imgs'):
            return dataset.imgs[idx][0]
    
    elif dataset_name in ['A', 'V', 'R', 'S']:  # ImageNet variants
        # For variant datasets, try known attributes first
        if hasattr(dataset, '_image_files'):
            return dataset._image_files[idx]
        if hasattr(dataset, 'images') and isinstance(dataset.images[0], str):
            return dataset.images[idx]
    
    # Generic fallback approaches (try not to load images)
    try:
        # Method 1: For datasets with impath attribute in data_source
        if hasattr(dataset, 'data_source') and hasattr(dataset.data_source[idx], 'impath'):
            return dataset.data_source[idx].impath
            
        # Method 2: For datasets with direct path access
        if hasattr(dataset, 'filepaths'):
            return dataset.filepaths[idx]
        if hasattr(dataset, 'image_paths'):
            return dataset.image_paths[idx]
            
        # Very last resort - this is slow but will work
        img, _ = dataset[idx]
        if hasattr(img, 'filename'):
            return img.filename
            
        # If all else fails, create a unique identifier
        return f"{dataset_name}_sample_{idx}"
    except Exception as e:
        print(f"Error extracting path for {dataset_name} at idx {idx}: {e}")
        return f"{dataset_name}_sample_{idx}"

def create_lora_dataset(datasets_str, data_root, output_dir, ratio=0.1, max_per_dataset=1000):
    """Create a dataset for LoRA fine-tuning by sampling from multiple datasets"""
    print(f"Creating combined dataset from: {datasets_str}")
    dataset_names = datasets_str.split("/")
    
    # Load CLIP model for preprocessing
    _, preprocess = clip.load("RN50")
    
    combined_paths = []  # Store paths instead of actual images
    dataset_stats = {}
    
    # Process each dataset
    for dataset_name in dataset_names:
        print(f"Processing dataset: {dataset_name}")
        
        try:
            # Use your project's existing function to load dataset
            loader, classnames, template = build_test_data_loader(dataset_name, data_root, preprocess)
            
            # Print dataset info
            print(f"  Dataset size: {len(loader.dataset)}")
            print(f"  Dataset type: {type(loader.dataset).__name__}")
            
            # Sample from the dataset
            num_samples = min(int(len(loader.dataset) * ratio), max_per_dataset)
            indices = random.sample(range(len(loader.dataset)), num_samples)
            
            # Try to determine the most efficient path extraction method
            # Test first sample to see which method works
            test_idx = indices[0]
            test_path = extract_image_path(loader.dataset, dataset_name, test_idx)
            print(f"  Sample path format: {test_path}")
            
            # Extract sampled data paths - with progress tracking per dataset
            dataset_samples = []
            print(f"  Extracting {num_samples} samples...")
            
            # Use batch processing for speed
            batch_size = 100
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:min(batch_start + batch_size, len(indices))]
                
                # Process batch
                for idx in tqdm(batch_indices, desc=f"Batch {batch_start//batch_size + 1}"):
                    path = extract_image_path(loader.dataset, dataset_name, idx)
                    _, label = loader.dataset[idx]
                    dataset_samples.append((path, label))
                
                # Print progress
                print(f"  Processed {min(batch_start + batch_size, len(indices))} / {len(indices)}")
            
            combined_paths.extend(dataset_samples)
            dataset_stats[dataset_name] = len(dataset_samples)
            
            print(f"  - Added {len(dataset_samples)} paths from {dataset_name}")
            
        except Exception as e:
            print(f"Error processing dataset {dataset_name}: {e}")
            continue
    
    # Save incrementally to avoid memory issues
    os.makedirs(output_dir, exist_ok=True)
    
    # Save in chunks if too large
    if len(combined_paths) > 10000:
        chunk_size = 5000
        for i in range(0, len(combined_paths), chunk_size):
            chunk = combined_paths[i:i+chunk_size]
            chunk_path = os.path.join(output_dir, f"lora_dataset_chunk{i//chunk_size}.pt")
            torch.save(chunk, chunk_path)
            print(f"Saved chunk {i//chunk_size} to {chunk_path}")
        
        # Save index file
        index_path = os.path.join(output_dir, "lora_combined_dataset.pt")
        torch.save({
            'chunk_count': (len(combined_paths) + chunk_size - 1) // chunk_size,
            'stats': dataset_stats
        }, index_path)
    else:
        # Save everything in one file if small enough
        save_path = os.path.join(output_dir, "lora_combined_dataset.pt")
        torch.save({
            'paths': combined_paths,
            'stats': dataset_stats
        }, save_path)
    
    print(f"\nSaved combined dataset with {len(combined_paths)} samples")
    print("Dataset statistics:")
    for name, count in dataset_stats.items():
        print(f"  - {name}: {count} samples")
    
    return combined_paths, dataset_stats

def main():
    parser = argparse.ArgumentParser(description="Create custom dataset for LoRA training")
    parser.add_argument("--datasets", type=str, default="I/A/V/R/S",
                        help="Dataset names separated by /")
    parser.add_argument("--data-root", type=str, default="/home/rhossain/exp/TDAQ/dataset",
                        help="Data root directory")
    parser.add_argument("--output-dir", type=str, default="./lora_data",
                        help="Output directory to save the custom dataset")
    parser.add_argument("--ratio", type=float, default=0.1,
                        help="Sampling ratio (default: 0.1)")
    parser.add_argument("--max-per-dataset", type=int, default=1000,
                        help="Maximum samples per dataset")
    
    args = parser.parse_args()
    
    create_lora_dataset(
        args.datasets, 
        args.data_root, 
        args.output_dir, 
        args.ratio, 
        args.max_per_dataset
    )
    
    print("\nTo use this dataset for LoRA training:")
    print(f"python train_lora_custom.py --custom-dataset {os.path.join(args.output_dir, 'lora_combined_dataset.pt')}")

if __name__ == "__main__":
    main()