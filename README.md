![Python 3.10](https://img.shields.io/badge/python-3.10-green)
![Pytorch 2.5](https://img.shields.io/badge/pytorch-2.5-orange)

>Codes for **Towards Robust Multimodal Emotion Recognition under Missing Modalities and Distribution Shifts**.

## Usage
### Clone the repository
    git clone https://github.com/gw-zhong/CIDer.git
### Download the datasets
+ IID: [CMU-MOSI & CMU-MOSEI (**BERT**) [align & unaligned]](https://github.com/thuiar/MMSA)
+ OOD: [CMU-MOSI & CMU-MOSEI (**BERT**) [align & unaligned]]()
### Preparation
Create (empty) folders for results:
 ```python
cd cider
 mkdir results
```
and set the ```data_path``` and the ```model_path``` correctly in main.py.
### Hyperparameter tuning
 ```python
python main.py --[FLAGS]
 ```
### Evaluation
```python
python main_eval.py --[FLAGS]
 ```
### Single Training
```python
python main_run.py --[FLAGS]
 ```