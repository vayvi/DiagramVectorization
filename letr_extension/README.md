This repository is an extension of the LETR line segment detector for line segment and circle detection. The official code for LETR, [Line Segment Detection Using Transformers without Edges](https://arxiv.org/abs/2101.01909). [Yifan Xu*](https://yfxu.com/), [Weijian Xu*](https://weijianxu.com/), [David Cheung](https://github.com/sawsa307), and [Zhuowen Tu](https://pages.ucsd.edu/~ztu/) (CVPR2021), can be found [here](https://github.com/mlpc-ucsd/LETR/tree/master). 


The following repo is a re-implementation of their code with pytorch lightning that supports shapes other than lines.

The model is trained on synthetic data that builds on the dataset and code used in [docExtractor](https://github.com/monniert/docExtractor) and in [diagram-extraction](https://github.com/Segolene-Albouy/Diagram-extraction).

## Installation

You can create a conda environment 

```
conda env create -f environment.yml
```


## Synthetic dataset


This resource is part of the dataset used in [docExtractor](https://github.com/monniert/docExtractor) and in [diagram-extraction](https://github.com/Segolene-Albouy/Diagram-extraction). The code for generating the synthetic data is also heavily based on docExtractor.  

To get the synthetic resource (backgrounds) for the synthetic dataset you can launch: 

```
bash download_synthetic_resource.sh
```

<details><summary>Click to expand</summary>

To generate the synthetic dataset, you need to download the synthetic resource folder [here](https://www.dropbox.com/s/tiqqb166f5ygzx2/synthetic_resource.zip?dl=0) and unzip it in the data folder. 

</details>

You can install the local package for synthetic data by going to the synthetic folder and installing: 

```
cd synthetic 
pip install -e .
```

You can generate a training and validation set of synthetic data using: 

```
cd synthetic_module
python synthetic.py 
```

This will generate the synthetic data folder in raw format which requires a preprocessing step for the LETR model.


## Real dataset with SVG annotations

To process real data, you can run the following bash script.

```
bash process_real_data
```

<details><summary>Click to expand</summary>

To get a real dataset with image and svg annotation pairs, you need to have an input data directory with subfolders images and svgs where corresponding files should share the same name.  If you would like to evaluate or even train on this dataset, it has to be in the COCO raw format. To do this for the diagrams dataset, you can run 

```
python helper/parse_svg.py --data_path data/diagrams 
``` 

If you would like to test on a real dataset, it also has to be preprocessed from the raw format. For the diagrams dataset, you would have to run

```
python helper/preprocess_data.py data/diagrams data/diagrams_processed
```
</details>


## Training
Pretrained model checkpoints can be found [here]([link1](https://drive.google.com/file/d/1CUJDh9PwoyjkXdlqZGgLV0_WUkWWijNF/view?usp=sharing)) for stages 1 and 2.
You can train the model from scratch on synthetic data by running 
```
bash process_and_train.sh
```

<details><summary>Click to expand</summary>

You can train the LETR extension model from scratch for stages 1 and 2 by running 
```
python main.py --config_path config_primitives.yaml --epochs epochs_stage1
python main.py --config_path config_primitives_stage2.yaml --epochs epochs_stage2
```
 </details>



## Evaluation
To evaluate the stage1 and stage2 model on real data, 

```
bash evaluate_real_data.sh
```

<details><summary>Click to expand</summary>

To perform evaluation over a synthetic testing dataset or a real testing dataset, you need to generate the ground-truth annotations over a resized image as done in LETR by running 

```
python evaluation/generate_gt.py --data_path path/to/dataset/in/raw/format/
```

You can evaluate the model by comparing its predictions to the generated ground-truth by adding --test to the previous training commands and running
</details>

