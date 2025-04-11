import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import clip
import os
import argparse
from torch.utils.data import DataLoader, Dataset
from PIL import Image

class LoRATrainer:
    def __init__(self, clip_model, rank=16, lr=1e-6, save_path="lora_weights"):
        # Get CLIP feature dimension
        self.clip_model = clip_model
        self.feature_dim = clip_model.visual.output_dim
        self.rank = rank
        self.save_path = save_path
        
        # Initialize LoRA matrices with very small values
        self.lora_a = nn.Parameter(torch.zeros(self.feature_dim, rank, device="cuda"))
        self.lora_b = nn.Parameter(torch.zeros(rank, self.feature_dim, device="cuda"))
        
        # Very conservative initialization
        with torch.no_grad():
            nn.init.xavier_normal_(self.lora_a, gain=0.0001)
            nn.init.xavier_normal_(self.lora_b, gain=0.0001)
        
        # Use Adam with much lower weight decay
        self.optimizer = optim.Adam([self.lora_a, self.lora_b], lr=lr, weight_decay=0.001)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=2, factor=0.5, mode='max'
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def train(self, loader, clip_weights, num_epochs=10):
        # Ensure clip_weights is on GPU and float32
        clip_weights = clip_weights.cuda().float()
        
        best_acc = 0
        scale_factor = 0.001  # Start with much smaller adaptation
            
        for epoch in range(num_epochs):
            total_loss = 0
            correct = 0
            total = 0
            valid_batches = 0
            
            for i, (images, targets) in enumerate(tqdm(loader, desc=f'Epoch {epoch+1}/{num_epochs}')):
                try:
                    self.optimizer.zero_grad()
                    
                    # Get image features
                    with torch.no_grad():
                        images = images.cuda()
                        targets = targets.cuda()
                        image_features = self.clip_model.encode_image(images)
                        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                        # Convert to float32
                        image_features = image_features.float()
                        
                        # Get baseline accuracy with original CLIP
                        if i == 0 and epoch == 0:
                            base_logits = image_features @ clip_weights
                            base_pred = base_logits.argmax(dim=1)
                            base_acc = (base_pred == targets).float().mean().item()
                            print(f"Baseline CLIP accuracy: {base_acc:.4f}")
                    
                    # Compute LoRA adaptation with small scaling factor
                    lora_term = (image_features @ self.lora_a) @ self.lora_b
                    lora_term = lora_term * scale_factor  # Scale down the adaptation
                    adapted_features = image_features + lora_term
                    
                    # Re-normalize features after adaptation
                    adapted_features = adapted_features / adapted_features.norm(dim=1, keepdim=True)
                    
                    # Calculate logits
                    logits = adapted_features @ clip_weights
                    
                    # Compute loss and backpropagate
                    loss = self.loss_fn(logits, targets)
                    
                    if torch.isfinite(loss):
                        loss.backward()
                        # Very aggressive gradient clipping
                        torch.nn.utils.clip_grad_norm_([self.lora_a, self.lora_b], max_norm=0.01)
                        self.optimizer.step()
                        
                        total_loss += loss.item()
                        valid_batches += 1
                        
                        # Track accuracy
                        pred = logits.argmax(dim=1)
                        batch_correct = (pred == targets).sum().item()
                        correct += batch_correct
                        total += targets.size(0)
                        
                        if i % 20 == 0:
                            print(f"  Batch {i}: Loss={loss.item():.4f}, Batch Acc={batch_correct/targets.size(0):.4f}")
                    else:
                        print(f"  Skipping batch {i} due to non-finite loss: {loss.item()}")
                        
                except Exception as e:
                    print(f"  Error in batch {i}: {str(e)}")
                    continue
            
            # Avoid division by zero
            avg_loss = total_loss / max(valid_batches, 1)
            accuracy = correct / max(total, 1)
            
            print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Acc = {accuracy:.4f}, Valid batches: {valid_batches}")
            
            # Update LR scheduler based on accuracy (not loss)
            self.scheduler.step(accuracy)
            
            # Only increase scale if accuracy is improving
            if accuracy >= best_acc and valid_batches > len(loader) * 0.9:
                old_scale = scale_factor
                scale_factor = min(scale_factor * 1.1, 0.01)  # More conservative increase, lower cap
                print(f"  Increasing adaptation scale from {old_scale:.6f} to {scale_factor:.6f}")
            
            # Save best model
            if accuracy > best_acc:
                best_acc = accuracy
                print(f"  New best accuracy: {best_acc:.4f}")
                standard_path = f"{self.save_path}/lora_multi_dataset_rank{self.rank}.pt"
                self.save(standard_path)

    def save(self, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            'lora_a': self.lora_a.data,
            'lora_b': self.lora_b.data,
            'feature_dim': self.feature_dim,
            'rank': self.rank
        }, save_path)
        print(f"LoRA matrices saved to {save_path}")

class CustomLoRADataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data = torch.load(data_path)
        self.paths = self.data['paths']
        self.transform = transform
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        img_path, label = self.paths[idx]
        try:
            # Load image from path
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--custom-dataset", type=str, required=True, 
                      help="Path to the custom dataset file")
    parser.add_argument("--backbone", type=str, default="RN50", help="CLIP backbone")
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--save-path", type=str, default="lora_weights", 
                      help="Path to save LoRA weights")
    args = parser.parse_args()

    # Load CLIP model
    print(f"Loading {args.backbone}...")
    clip_model, preprocess = clip.load(args.backbone)
    clip_model.eval()
    
    # Load the custom dataset
    print(f"Loading custom dataset from {args.custom_dataset}...")
    dataset = CustomLoRADataset(args.custom_dataset, preprocess)
    
    # Print statistics
    print("Dataset statistics:")
    for name, count in dataset.data['stats'].items():
        print(f"  - {name}: {count} samples")
    
    # Get data loader
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    # Get class weights from CLIP for ImageNet (default)
    from utils import clip_classifier
    import numpy as np
    
    # For simplicity, use ImageNet classes
    from datasets.imagenet import ImageNet
    imagenet = ImageNet("", None)  # Path doesn't matter since we only need classnames
    clip_weights = clip_classifier(imagenet.classnames, imagenet.template, clip_model)
    
    # Create and train LoRA
    trainer = LoRATrainer(clip_model, rank=args.rank, lr=args.lr, save_path=args.save_path)
    trainer.train(data_loader, clip_weights, num_epochs=args.epochs)
    
    print("Training complete!")

if __name__ == "__main__":
    main()