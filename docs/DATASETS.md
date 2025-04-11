# How to install datasets
### Acknowledgement: We extend our gratitude to the authors of [CoOp/CoCoOp](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md#how-to-install-datasets) for their foundational work on data preparation instructions. In this project, we have adopted their data preparation format, implementing certain modifications to better suit our needs.

We suggest putting all datasets under the same folder (say `$DATA`) to ease management and following the instructions below to organize datasets to avoid modifying the source code. The file structure looks like

```
$DATA/
|–– imagenet/
|–– caltech-101/
|–– ImageNetV2/
```

If you have some datasets already installed somewhere else, you can create symbolic links in `$DATA/dataset_name` that point to the original data to avoid duplicate download.

Datasets list:
- [ImageNet](#imagenet)
- [Caltech101](#caltech101)
- [ImageNetV2](#imagenetv2)
- [ImageNet-Sketch](#imagenet-sketch)
- [ImageNet-A](#imagenet-a)
- [ImageNet-R](#imagenet-r)

The instructions to prepare each dataset are detailed below. To ensure reproducibility and fair comparison for future work, we provide fixed train/val/test splits for all datasets except ImageNet where the validation set is used as test set. The fixed splits are either from the original datasets (if available) or created by us.

### ImageNet
- Create a folder named `imagenet/` under `$DATA`.
- Create `images/` under `imagenet/`.
- Download the dataset from the [official website](https://image-net.org/index.php) and extract the validation set to `$DATA/imagenet/images`. The directory structure should look like
```
imagenet/
|–– images/
|   |–– val/ # contains 1,000 folders like n01440764, n01443537, etc.
```
- If you had downloaded the ImageNet dataset before, you can create symbolic links to map the validation set to `$DATA/imagenet/images`.
- Download the `classnames.txt` to `$DATA/imagenet/` from this [link](https://drive.google.com/file/d/1-61f_ol79pViBFDG_IDlUQSwoLcn2XXF/view?usp=sharing). The class names are copied from [CLIP](https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb).

### Caltech101
- Create a folder named `caltech-101/` under `$DATA`.
- Download `101_ObjectCategories.tar.gz` from http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz and extract the file under `$DATA/caltech-101`.
- Download `split_zhou_Caltech101.json` from this [link](https://drive.google.com/file/d/1hyarUivQE36mY6jSomru6Fjd-JzwcCzN/view?usp=sharing) and put it under `$DATA/caltech-101`. 

The directory structure should look like
```
caltech-101/
|–– 101_ObjectCategories/
|–– split_zhou_Caltech101.json
```

### ImageNetV2
- Create a folder named `imagenetv2/` under `$DATA`.
- Go to this github repo https://github.com/modestyachts/ImageNetV2.
- Download the matched-frequency dataset from https://s3-us-west-2.amazonaws.com/imagenetv2public/imagenetv2-matched-frequency.tar.gz and extract it to `$DATA/imagenetv2/`.
- Copy `$DATA/imagenet/classnames.txt` to `$DATA/imagenetv2/`.

The directory structure should look like
```
imagenetv2/
|–– imagenetv2-matched-frequency-format-val/
|–– classnames.txt
```

### ImageNet-Sketch
- Download the dataset from https://github.com/HaohanWang/ImageNet-Sketch.
- Extract the dataset to `$DATA/imagenet-sketch`.
- Copy `$DATA/imagenet/classnames.txt` to `$DATA/imagenet-sketch/`.

The directory structure should look like
```
imagenet-sketch/
|–– images/ # contains 1,000 folders whose names have the format of n*
|–– classnames.txt
```

### ImageNet-A
- Create a folder named `imagenet-adversarial/` under `$DATA`.
- Download the dataset from https://github.com/hendrycks/natural-adv-examples and extract it to `$DATA/imagenet-adversarial/`.
- Copy `$DATA/imagenet/classnames.txt` to `$DATA/imagenet-adversarial/`.

The directory structure should look like
```
imagenet-adversarial/
|–– imagenet-a/ # contains 200 folders whose names have the format of n*
|–– classnames.txt
```

### ImageNet-R
- Create a folder named `imagenet-rendition/` under `$DATA`.
- Download the dataset from https://github.com/hendrycks/imagenet-r and extract it to `$DATA/imagenet-rendition/`.
- Copy `$DATA/imagenet/classnames.txt` to `$DATA/imagenet-rendition/`.

The directory structure should look like
```
imagenet-rendition/
|–– imagenet-r/ # contains 200 folders whose names have the format of n*
|–– classnames.txt
