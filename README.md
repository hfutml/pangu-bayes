# Pangu-Bayes: A Bayesian Ensemble Framework for Global Weather Forecasting

Pangu-Bayes is an open-source Bayesian ensemble forecasting framework for global weather forecasting. 
This repository provides the official implementation of Pangu-Bayes, including deterministic pretraining, epistemic uncertainty learning, joint epistemic-aleatoric uncertainty learning, ensemble forecast inference, probabilistic post-processing, and evaluation scripts.

## Citation

An earlier version of this work is available on arXiv:

**Bridging the Gap Between Bayesian Deep Learning and Ensemble Weather Forecasts**  
Xinlei Xiong, Wenbo Hu, Shuxun Zhou, Kaifeng Bi, Lingxi Xie, Ying Liu, Richang Hong, and Qi Tian  
arXiv:2511.14218, 2025

Paper: https://arxiv.org/abs/2511.14218

If you find this repository useful, please cite:

```bibtex
@article{xiong2025bridging,
  title={Bridging the Gap Between Bayesian Deep Learning and Ensemble Weather Forecasts},
  author={Xiong, Xinlei and Hu, Wenbo and Zhou, Shuxun and Bi, Kaifeng and Xie, Lingxi and Liu, Ying and Hong, Richang and Tian, Qi},
  journal={arXiv preprint arXiv:2511.14218},
  year={2025}
}


## Resources

- **Code:** https://github.com/hfutml/pangu-bayes
- **Model checkpoints:** https://huggingface.co/xionge/pangu-bayes
- **Preprint:** https://arxiv.org/abs/2511.14218

The current preprint is:

> Xinlei Xiong, Wenbo Hu, Shuxun Zhou, Kaifeng Bi, Lingxi Xie, Ying Liu, Richang Hong, and Qi Tian.  
> **Bridging the Gap Between Bayesian Deep Learning and Ensemble Weather Forecasts.**  
> arXiv:2511.14218, 2025.


## Installation

We recommend using Conda to create the environment.

```bash
conda create -n pangu-bayes python=3.10
conda activate pangu-bayes
pip install -r requirements.txt
```

## Data Preparation

Pangu-Bayes uses global atmospheric reanalysis data for training and evaluation. To help users quickly test the code, this repository provides a small example dataset under:

Expected upper-air variables include:

- Geopotential (Z)
- Temperature (T)
- Specific humidity (Q)
- U component of wind (U)
- V component of wind (V)

Expected surface variables includUe:

- 2m temperature (T2M)
- 10m u component of wind (10U)
- 10m v component of wind (10V)
- Mean sea level pressure (MSL)

The expected pressure levels are: 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000 hPa


```text
ERA5_examples/
├── constant_masks/
│   ├── land_mask.npy
│   ├── soil_type.npy
│   └── topography.npy
│
├── statistic/
│   ├── surface_mean_40.pt
│   ├── surface_std_40.pt
│   ├── upper_mean_40.pt
│   └── upper_std_40.pt
│
├── train/
│   ├── 1979_0000.pt
│   └── 1979_0001.pt
│
│
└── test/
    ├── 2022_0000.pt
    └── 2022_0001.pt
 
```

## Method Overview

The overall pipeline consists of four stages:

```text
Deterministic pretraining
        ↓
Epistemic uncertainty learning
        ↓
Joint epistemic-aleatoric uncertainty learning
        ↓
Operational fine-tuning
        ↓
Probabilistic post-processing
```

---

## Training

### Stage 1: Deterministic Pretraining

The deterministic backbone is trained to predict future global atmospheric fields from historical atmospheric states.

```bash
bash train_scripts/pretrain.sh
```


### Stage 2: Epistemic Uncertainty Learning

Epistemic uncertainty is introduced into the forecasting backbone through parameter uncertainty.

```
bash train_scripts/train_eu.sh
```


### Stage 3: Joint Epistemic-Aleatoric Uncertainty Learning

The joint EU-AU model combines epistemic uncertainty from Bayesian model parameters and aleatoric uncertainty from state-dependent perturbations.

```
bash train_scripts/train_eu_au.sh
```
### Stage 4: Operational fine-tuning

After completing the aleatoric uncertainty training stage, we obtained the
Pangu-Bayes model. To further enhance operational forecasting performance, we fine-tuned this model on the
high-resolution (HRES) analysis data, yielding the Pangu-Bayes(oper.) variant.
```
bash train_scripts/train_eu_au_hres.sh
```

### Stage 5: Probabilistic Post-processing

The post-processing module calibrates intensity. Before running the post-processing training script, users should first construct the post-processing training dataset from ensemble forecast outputs.

This step computes task-specific summary statistics from the ensemble forecast fields, and pairs them with the corresponding verification targets. The resulting dataset is then used to train the probabilistic post-processing network.

The post-processing workflow is:

```text
Ensemble forecast outputs
        ↓
Compute statistics
        ↓
Construct post-processing training dataset
        ↓
Train probabilistic post-processing model
```

## Global Weather Forecasting and Evaluation

To generate global ensemble forecasts using Pangu-Bayes, Global forecast evaluation is performed at the field level.

metrics include: RMSE, CRPS, Spread-skill ratio (SSR)

Example:

```bash
bash inference_ensemble_scripts/inference.sh
```


## Tropical Cyclone Forecasting and Evaluation

Although Pangu-Bayes is designed for global ensemble weather forecasting, this repository provides tropical cyclone ensemble forecasting as a downstream application. In this task, Pangu-Bayes first generates global ensemble forecast fields, and cyclone-level quantities are then extracted and evaluated using external cyclone analysis tools.

The tropical cyclone workflow consists of two main steps:

1. **Cyclone detection and tracking**  
   Tropical cyclone candidates and trajectories are identified from global forecast fields using the **TempestExtremes** tracker. The tracker extracts storm-level quantities from each ensemble member, including:

   - Cyclone center latitude
   - Cyclone center longitude
   - Maximum sustained wind speed (MSW)
   - Minimum sea-level pressure (MSLP)

2. **Track matching and verification**  
   The predicted cyclone tracks are matched with observed best-track records from the **IBTrACS** dataset using the **huracanpy** Python package. After matching, deterministic and probabilistic verification metrics are computed for track and intensity forecasts.



## Model Checkpoints

The Pangu-Bayes model checkpoints are publicly available on Hugging Face:

https://huggingface.co/xionge/pangu-bayes

The checkpoints correspond to the main stages of the forecasting framework and are organized as:

```text
checkpoint/
├── deterministic_forecast.pt
├── eu_ensemble_forecast.pt
├── eu_au_ensemble_forecast.pt
├── eu_au_ensemble_forecast_hres.pt
└── best_prob_ann.pt

## Acknowledgements

This project builds upon recent advances in data-driven global weather forecasting, Bayesian deep learning, ensemble prediction, and probabilistic forecast verification.

We thank the providers of ERA5 and best-track datasets for making global atmospheric and cyclone records available to the research community.
