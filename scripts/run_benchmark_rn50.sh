#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python tda_runner_lora.py     --config configs \
                                                --datasets I/A/V/R/S \
                                                --backbone RN50