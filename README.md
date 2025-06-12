![Python 3.10](https://img.shields.io/badge/python-3.10-green)
![Pytorch 2.5](https://img.shields.io/badge/pytorch-2.5-orange)

>Codes for **Towards Robust Multimodal Emotion Recognition under Missing Modalities and Distribution Shifts**.

## Usage
### Clone the repository
    git clone https://github.com/gw-zhong/CIDer.git
### Download the datasets
+ IID: [CMU-MOSI & CMU-MOSEI (**BERT**) [align & unaligned]](https://github.com/thuiar/MMSA)
+ OOD: [CMU-MOSI & CMU-MOSEI (**BERT**) [align & unaligned]](https://pan.baidu.com/s/1Ob3VY5j1Vz1pIaJ_k_bq9Q) `code: 19db`
### Preparation
Create (empty) folder for results:
 ```python
cd cider
 mkdir results
```
and set the `data_path` and the `model_path` correctly in `main.py`.
### Hyperparameter tuning
 ```python
python main.py --[FLAGS]
 ```
Or, you can use the bash script for tuning：
 ```python
bash scripts/run_all.sh
 ```
Please note that `run_all.sh` contains **all the tasks** and uses **8 GPUs** for hyperparameter tuning. You should select one or several tasks for tuning according to your actual needs, instead of running all of them.
### Evaluation
```python
python main_eval.py --[FLAGS]
 ```
### Single Training
```python
python main_run.py --[FLAGS]
 ```
### reproduction
To facilitate the reproduction of the results in the paper, we have also uploaded the corresponding model weights:
- [BaiduYun Disk](https://pan.baidu.com/s/1mHIpZvG0lRYiIrv4xuN3bQ) `code: 885a`

You just need to run `main_eval.py` to reproduce the results.

Please note that when running the evaluation for the corresponding model, you should also modify the relevant task parameters in `main_eval.py`.
## Contact
If you have any question, feel free to contact me through [guoweizhong@zjut.edu.cn](guoweizhong@zjut.edu.cn) or [gwzhong@zju.edu.cn](gwzhong@zju.edu.cn).