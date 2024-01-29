# PAP-REC

This repo presents implementation of PAP-REC: Personalized Automatic Prompt for Recommendation Language Model.

## Requirements

- Python 3.9.7
- PyTorch 1.10.1
- transformers 4.2.1
- tqdm
- numpy
- sentencepiece
- pyyaml

## Usage

0. Clone this repo

1. Install necessary packages:

```
pip install -r requirements.txt
```

2. Download preprocessed data from this [Google Drive link](https://drive.google.com/file/d/1qGxgmx7G_WB7JE4Cn_bEcZ_o_NAJLE3G/view?usp=sharing), then put them into the *data* folder. If you would like to preprocess your own data, please follow the jupyter notebooks in the *preprocess* folder. Raw data can be downloaded from this [Google Drive link](https://drive.google.com/file/d/1uE-_wpGmIiRLxaIy8wItMspOf5xRNF2O/view?usp=sharing), then put them into the *raw_data* folder.

3. Download pretrained checkpoints into *snap* folder. If you would like to train your own P5 models, *snap* folder will also be used to store P5 checkpoints.

4. Run main.py with arguments

Example:
```
python main.py --task sequential --dataset beauty --model_size small --dynamic_length 0
```

## Pretrained Checkpoints
See [CHECKPOINTS.md](snap/CHECKPOINTS.md).

## Citation

The citation of our paper will be updated soon.

This codebase is developed based on [P5](https://github.com/jeykigung/P5).
