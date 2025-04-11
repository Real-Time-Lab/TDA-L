import random
import argparse
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn.functional as F
import operator

import clip
from utils import *
import time


def get_arguments():
    """Get arguments of the test-time adaptation."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', required=True, help='settings of TDA on specific dataset in yaml format.')
    parser.add_argument('--datasets', dest='datasets', type=str, required=True, help="Datasets to process, separated by a slash (/). Example: I/A/V/R/S")
    parser.add_argument('--data-root', dest='data_root', type=str, default='/home/rhossain/exp/TDAQ/dataset', help='Path to the datasets directory. Default is ./dataset/')
    parser.add_argument('--backbone', dest='backbone', type=str, choices=['RN50', 'ViT-B/16'], required=True, help='CLIP model backbone to use: RN50 or ViT-B/16.')

    args = parser.parse_args()

    return args

def get_tensor_size(tensor):
    """
    Calculate the size of a PyTorch tensor in bytes.

    Args:
        tensor (torch.Tensor): The input tensor.

    Returns:
        int: The size of the tensor in bytes.
    """
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("Input must be a PyTorch tensor.")
    #print(tensor.element_size())
    return tensor.element_size() * tensor.numel()


def compute_real_cache_size(cache):
    """
    Compute the real memory size of a cache (positive or negative) in bytes.
    
    Args:
        cache (dict): The cache dictionary, where each entry is a list of (feature, loss) tuples.
        
    Returns:
        int: Total memory size of the cache in bytes.
    """
    total_size = 0
    for class_entries in cache.values():
        for item in class_entries:
            # Handle feature tensor (item[0])
            if isinstance(item[0], torch.Tensor):
                feature_size = get_tensor_size(item[0])
                total_size += feature_size
                
            # Handle loss or other metadata (item[1:])
            for i in range(1, len(item)):
                if isinstance(item[i], torch.Tensor):
                    total_size += get_tensor_size(item[i])
                elif isinstance(item[i], (float, int)):
                    # Estimate size of scalar values
                    total_size += 8  # Approx size of float/int
    
    return total_size


def apply_lora(features, lora_a, lora_b):
    """Apply LoRA transformation to features."""
    with torch.no_grad():
        # Ensure consistent data types
        features = features.half()
        lora_a = lora_a.half()
        lora_b = lora_b.half()
        
        # Apply LoRA with residual connection
        lora_delta = (features @ lora_a) @ lora_b
        adapted_features = features + lora_delta
        
        # Normalize
        adapted_features = adapted_features / adapted_features.norm(dim=1, keepdim=True)
        return adapted_features

def update_cache_with_lora(cache, pred, image_features, loss, shot_capacity, lora_a, lora_b):
    """
    Update cache with LoRA-adapted features instead of raw features.
    This pre-computes the LoRA transformation during cache update instead of during lookup.
    
    Args:
        cache: Dictionary cache mapping class indices to lists of features
        pred: Predicted class index
        image_features: Original image features
        loss: Loss value for this prediction
        shot_capacity: Maximum number of shots to store per class
        lora_a, lora_b: LoRA matrices for adaptation
    """
    with torch.no_grad():
        # Apply LoRA transformation when adding to cache
        image_features_half = image_features.half()
        lora_a_half = lora_a.half()
        lora_b_half = lora_b.half()
        
        # Apply LoRA adaptation
        lora_delta = (image_features_half @ lora_a_half) @ lora_b_half
        adapted_features = image_features_half + lora_delta
        adapted_features = adapted_features / adapted_features.norm(dim=1, keepdim=True)
        
        # Store adapted features in cache
        item = (adapted_features, loss)
        
        if pred in cache:
            if len(cache[pred]) < shot_capacity:
                cache[pred].append(item)
            else:
                # Replace oldest item
                cache[pred].pop(0)
                cache[pred].append(item)
        else:
            cache[pred] = [item]
            
            
def compute_logits_with_adapted_cache(image_features, cache, alpha, beta, clip_weights, lora_a, lora_b):
    """
    Compute logits using pre-adapted cache features.
    Since the cache already contains LoRA-adapted features, we only need to adapt the query features.
    
    Args:
        image_features: Query features to adapt
        cache: Dictionary with pre-adapted features
        alpha, beta: TDA hyperparameters
        clip_weights: CLIP model classifier weights
        lora_a, lora_b: LoRA matrices for query feature adaptation
    """
    with torch.no_grad():
        # Apply LoRA only to query features
        image_features_half = image_features.half()
        lora_a_half = lora_a.half() 
        lora_b_half = lora_b.half()
        
        # Adapt query features with LoRA
        lora_delta = (image_features_half @ lora_a_half) @ lora_b_half
        adapted_query = image_features_half + lora_delta
        adapted_query = adapted_query / adapted_query.norm(dim=1, keepdim=True)
        
        # Get cached features (already adapted, no need for LoRA reapplication)
        all_features = []
        class_indices = []
        
        for class_index in sorted(cache.keys()):
            for item in cache[class_index]:
                # Item[0] now contains pre-adapted features
                all_features.append(item[0])
                class_indices.append(class_index)
                
        if not all_features:
            return torch.zeros(image_features.size(0), clip_weights.size(1), 
                               device=image_features.device)
        
        # Stack all cache features
        adapted_cache = torch.cat(all_features, dim=0)
        
        # Create one-hot encoding for classes
        cache_values = F.one_hot(
            torch.tensor(class_indices, dtype=torch.int64),
            num_classes=clip_weights.size(1)
        ).to(device=image_features.device, dtype=adapted_query.dtype)
        
        # Compute affinity using adapted features
        affinity = adapted_query @ adapted_cache.T
        
        # Standard TDA formula
        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
        
        return alpha * cache_logits
    
def run_test_tda_with_lora(pos_cfg, neg_cfg, loader, clip_model, clip_weights, log_file="./lora_test_log.txt"):
    
    pos_cache, neg_cache, accuracies = {}, {}, []
    total_inference_time = 0.0
    
    with torch.no_grad():

        
        with open(log_file, "w") as f:
            f.write(f"LoRA-TDA Test Run - {datetime.now()}\n\n")
            
            # Load LoRA matrices from checkpoint (just once, not per sample!)
            try:
                print("Loading LoRA weights...")
                lora_checkpoint = torch.load("./lora_weights/lora_multi_dataset_rank4.pt", weights_only=True)
                lora_a = lora_checkpoint['lora_a'].cuda().half()  # Convert to half precision
                lora_b = lora_checkpoint['lora_b'].cuda().half()  # Convert to half precision
                print(f"LoRA matrices loaded: a={tuple(lora_a.shape)}, b={tuple(lora_b.shape)}")
            except Exception as e:
                print(f"Error loading LoRA weights: {e}")
                return 0
            
            accuracies = []
            start_global = time.time()
            total_images = 0
            # Test-time adaptation using LoRA + cache
            for i, (images, target) in enumerate(tqdm(loader, desc='Processed test images: ')):
                total_images +=1
                start_time = time.time()
                # Get features and baseline predictions
                image_features, clip_logits, loss, prob_map, pred = get_clip_logits(images, clip_model, clip_weights)
                target = target.cuda()
                prop_entropy = get_entropy(loss, clip_weights)               
                
                # Update caches with features
                # Update caches with PRE-ADAPTED features
                if pos_cfg['enabled']:
                    update_cache_with_lora(pos_cache, pred, image_features, loss, 
                                         pos_cfg['shot_capacity'], lora_a, lora_b)
                    
                if neg_cfg['enabled'] and neg_cfg['entropy_threshold']['lower'] < prop_entropy < neg_cfg['entropy_threshold']['upper']:
                    update_cache_with_lora(neg_cache, pred, image_features, loss, 
                                         neg_cfg['shot_capacity'], lora_a, lora_b)
                
                # Compute logits with pre-adapted cache
                final_logits = clip_logits.clone()
                
                if pos_cache:
                    final_logits += compute_logits_with_adapted_cache(
                        image_features, pos_cache, pos_cfg['alpha'], pos_cfg['beta'], 
                        clip_weights, lora_a, lora_b)
                
                if neg_cache:
                    final_logits -= compute_logits_with_adapted_cache(
                        image_features, neg_cache, neg_cfg['alpha'], neg_cfg['beta'],
                        clip_weights, lora_a, lora_b)
                
                acc = cls_acc(final_logits, target)
                accuracies.append(acc)
                inference_time = time.time() - start_time  # Measure total inference time
                total_inference_time += inference_time 
                # Log progress periodically
                if i % 1000 == 0:
                    pos_cache_size = compute_real_cache_size(pos_cache)
                    neg_cache_size = compute_real_cache_size(neg_cache)
                    avg_inference_time = total_inference_time / (i + 1)
                    avg_acc = sum(accuracies) / len(accuracies)
                    
                    f.write(f"---- Iteration {i} ----\n")
                    f.write(f"Average Inference Time: {avg_inference_time:.6f} seconds\n")
                    f.write(f"---- TDA's test accuracy: {avg_acc:.2f}. ----\n\n")
                    status = f"Step {i}: Acc={avg_acc:.4f}, Cache: +{pos_cache_size}/-{neg_cache_size}"
                    print(f"---- {status} ----")

            final_acc = sum(accuracies) / len(accuracies)
            avg_inference_time = total_inference_time / len(loader)
            total_time = time.time() - start_global
            throughput = total_images / total_time

            f.write("---- Final Results ----\n")
            f.write(f"Average Inference Time: {avg_inference_time:.6f} seconds\n")
            f.write(f"Total time: {total_time:.2f} seconds\n")
            f.write(f"Throughput: {throughput:.2f} images/sec\n")
            f.write(f"\nFinal accuracy: {final_acc:.4f}\n")
            print(f"---- Final accuracy: {final_acc:.4f} ----")
            print(f"Results logged to {log_file}")
            
            return final_acc



def main():
    args = get_arguments()
    config_path = args.config

    # Initialize CLIP model
    clip_model, preprocess = clip.load(args.backbone)
    clip_model.eval()

    # Set random seed
    random.seed(1)
    torch.manual_seed(1)
    
    # Run TDA on each dataset
    datasets = args.datasets.split('/')
    for dataset_name in datasets:
        print(f"Processing {dataset_name} dataset.")
        
        cfg = get_config_file(config_path, dataset_name)
        print("\nRunning dataset configurations:")
        print(cfg, "\n")
        
        test_loader, classnames, template = build_test_data_loader(dataset_name, args.data_root, preprocess)
        clip_weights = clip_classifier(classnames, template, clip_model)
        # Log directory for results
        os.makedirs("logs", exist_ok=True)
        log_file = f"logs/lora_tda_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        acc = run_test_tda_with_lora(cfg['positive'], cfg['negative'], test_loader, clip_model, clip_weights, log_file)



if __name__ == "__main__":
    main()