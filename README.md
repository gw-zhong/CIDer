![Python 3.10](https://img.shields.io/badge/python-3.10-green)
![Pytorch 2.5](https://img.shields.io/badge/pytorch-2.5-orange)

>Codes for **[Towards Robust Multimodal Emotion Recognition under Missing Modalities and Distribution Shifts](https://arxiv.org/abs/2506.10452)**.

## Usage
### Clone the repository
    git clone https://github.com/gw-zhong/CIDer.git
### Download the datasets
+ IID: [CMU-MOSI & CMU-MOSEI (**BERT**) [align & unaligned]](https://github.com/thuiar/MMSA)
+ OOD: CMU-MOSI & CMU-MOSEI (**BERT**) [align & unaligned]
    + [BaiduYun Disk](https://pan.baidu.com/s/1Ob3VY5j1Vz1pIaJ_k_bq9Q) `code: 19db`
    + [Hugging Face](https://huggingface.co/datasets/GWZhong/MSA_OOD_Dataset_in_CIDer)
### Download the BERT models
- [BaiduYun Disk](https://pan.baidu.com/s/12zhRpTEx5589Bmo0OAF5cg) ```code: e7mw```
### Preparation
Create (empty) folder for results:
 ```
cd cider
 mkdir results
```
and set the `data_path` and the `model_path` correctly in `main.py`, `main_eval.py`, and `main_run.py`.
### Hyperparameter tuning
 ```
python main.py --[FLAGS]
 ```
Or, you can use the bash script for tuning：
 ```
bash scripts/run_all.sh
 ```
Please note that `run_all.sh` contains **all the tasks** and uses **8 GPUs** for hyperparameter tuning. You should select one or several tasks for tuning according to your actual needs, instead of running all of them.
### Evaluation
```
python main_eval.py --[FLAGS]
 ```
Guidance:

When conducting the evaluation, you need to correctly set the `missing_mode` in `main_eval.py`. The specific settings are as follows:

- Our proposed RMFM: `--missing_mode RMFM`

- Traditional RMFM: `--missing_mode RMFM_same`

- RMM: `--missing_mode RMM`

- TMFM: `--missing_mode TMFM`

- STMFM: `--missing_mode STMFM`

- SMM: `--missing_mode RMFM_same` **and** uncomment the sections in `main_eval.py` from **line 169 to line 175** and **line 188**.
### Single Training
```
python main_run.py --[FLAGS]
 ```
### Reproduction
To facilitate the reproduction of the results in the paper, we have also uploaded the corresponding model weights:
- [BaiduYun Disk](https://pan.baidu.com/s/1mHIpZvG0lRYiIrv4xuN3bQ) `code: 885a`
- [Hugging Face](https://huggingface.co/GWZhong/CIDer)

You just need to run `main_eval.py` to reproduce the results.

Please note that when running the evaluation for the corresponding model, you should also modify the relevant task parameters in `main_eval.py`.
## Citation
Please cite our paper if you find that useful for your research:
 ```bibtex
@article{zhong2025towards,
   title={Towards Robust Multimodal Emotion Recognition under Missing Modalities and Distribution Shifts},
   author={Zhong, Guowei and Huan, Ruohong and Wu, Mingzhen and Liang, Ronghua and Chen, Peng},
   journal={arXiv preprint arXiv:2506.10452},
   year={2025}
}

 ```
## Contact
If you have any question, feel free to contact me through [guoweizhong@zjut.edu.cn](guoweizhong@zjut.edu.cn) or [gwzhong@zju.edu.cn](gwzhong@zju.edu.cn).

## Acknowledgment
Our code is based on [MulT](https://github.com/yaohungt/Multimodal-Transformer) and [SELF-MM](https://github.com/thuiar/Self-MM). And our repartitioned MER OOD Datasets are based on [CLUE](https://github.com/Teng-Sun/CLUE_model). Thanks to their open-source spirit for saving us a lot of time.