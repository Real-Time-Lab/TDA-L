#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python train_lora.py --custom-dataset ./lora_data/lora_combined_dataset.pt --rank 16 --lr 5e-6 --epochs 20 --save-path ./lora_weights 
                                                
                                               