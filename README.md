# TDA-L
## Overview
![abstract figure](docs/main.png)
> **<p align="justify"> Abstract:**  Vision-language models (VLMs) significantly enhance intelligent sensing by analyzing visual inputs—such as images from cameras—to recognize objects in real time, supporting applications like assistive technologies for visually impaired users. However, adapting VLMs to distribution shifts at test time, caused by factors like lighting or weather changes, remains challenging—especially in resource-constrained settings. Existing test-time adaptation (TTA) methods often rely on fine-tuning
and backpropagation, making them computationally expensive and unsuitable for real-time applications. To address this, the Training-Free Dynamic Adapter (TDA) was introduced as a lightweight alternative that uses a dynamic key-value (KV) cache and pseudo-label refinement to adapt without backpropagation. Building on this, we propose TDA-L, a novel framework that integrates Low-Rank Adaptation (LoRA) to enhance feature representations at test time using pre-learned low-rank matrices.
TDA-L applies LoRA transformations to both query and cached features during inference, improving robustness to distribution shifts while maintaining the training-free and lightweight nature of TDA. Experimental results on five ImageNet-based benchmarks show that TDA-L maintains almost the same accuracy, but achieves lower latency, and higher throughput, making it well-suited for real-world deployment in edge and low-resource environments

## Main Contributions
In summary, the contributions of this work are threefold: </br>

* **First**, We introduce TDA-L, a test-time adaptation framework that integrates LoRA-based
feature adaptation with a dynamic key-value (KV) cache, operating in a training-free
setting. </br>
* **Second**, We curate a custom dataset by combining 10% of multiple datasets, enabling the
learning of generalized LoRA matrices for diverse domain adaptation. </br>
* **Third**, We demonstrate that TDA-L preserves accuracy while significantly reducing both
latency and memory usage, especially with low-rank configurations (e.g., rank 8). </br>

## Requirements 
### Installation
Follow these steps to set up a conda environment and ensure all necessary packages are installed:

```bash
git clone git@github.com:Real-Time-Lab/TDA-L.git
cd TDA

conda create -n tda python=3.8.20
conda activate tda-l

# The results are produced with PyTorch 2.4.1 and CUDA 12.1
conda install pytorch==2.4.1 torchvision==0.15.1 torchaudio==2.0.1 cudatoolkit=12.1 -c pytorch

pip install -r requirements.txt
```

### Dataset
To set up all required datasets, kindly refer to the guidance in [DATASETS.md](docs/DATASETS.md), which incorporates steps for two benchmarks.

## Run TDA
### Configs
The configuration for TDA-L hyperparameters in `configs/dataset.yaml` can be tailored within the provided file to meet the needs of various datasets. This customization includes settings for both the positive and negative caches as outlined below:
* **Positive Cache Configuration:** Adjustments can be made to the `shot_capacity`, `alpha`, and `beta` values to optimize performance.

* **Negative Cache Configuration:** Similar to the positive cache, the negative cache can also be fine-tuned by modifying the `shot_capacity`, `alpha`, `beta`, as well as the `entropy_threshold` and `mask_threshold` parameters.

For ease of reference, the configurations provided aim to achieve optimal performance across datasets on two benchmarks, consistent with the results documented in our paper. However, specific tuning of these parameters for negative cache could potentially unlock further enhancements in performance. Adjusting parameters like `alpha` and `beta` within the positive cache lets you fine-tune things to match the unique needs of each dataset.

### Running
To execute the TDA-L, navigate to the `scripts` directory, where you'll find 4 bash scripts available.The scripts process the datasets sequentially, as indicated by the order divided by '/' in the script. 

Below are instructions for running TDA-L on various backbone architectures. Follow the steps suited to your specific needs:"

#### LoRa Dataset creation
```
bash ./scripts/run_create_lora_data.sh
```

#### LoRa finetuning
```
bash ./scripts/run_train_lora.sh
```
You might need to change the script if you want play with different ranks and epochs.

#### Benchmark
You might have to change the lora_weight file name based on the rank you used.
* **ResNet50**: Run TDA-L using the ResNet50 model:
```
bash ./scripts/run_benchmark_rn50.sh 
```
* **ViT/B-16**: Run TDA-L using the ViT/B-16 model.
```
bash ./scripts/run_benchmark_vit.sh 
```



## Contact
If you have any questions, feel free to create an issue in this repository or contact us via email at rhossain@binghamton.edu

## Acknowledgements
Our gratitude goes to the authors of [TDA](https://kdiaaa.github.io/tda/), [Tip-Adapter](https://github.com/gaopengcuhk/Tip-Adapter), [TPT](https://github.com/azshue/TPT), and [CoOp/CoCoOp](https://github.com/KaiyangZhou/CoOp) for sharing their work through open-source implementation and for providing detailed instructions on data preparation.
