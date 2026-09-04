# Resolving sources of uncertainty in AI weather forecasting

Pangu-Bayes is an open-source Bayesian ensemble forecasting framework for global weather forecasting.

This repository provides the official implementation of Pangu-Bayes, including deterministic pretraining, epistemic uncertainty learning, joint epistemic--aleatoric uncertainty learning, operational-analysis fine-tuning, ensemble forecast inference, probabilistic post-processing, and evaluation scripts.

## Resources

- **Code:** https://github.com/hfutml/pangu-bayes
- **Model checkpoints:** https://huggingface.co/xionge/pangu-bayes
- **Preprint:** https://arxiv.org/abs/2511.14218

An earlier version of this work is available as:

> Wenbo Hu¹*, Xinlei Xiong¹*, Shuxun Zhou¹, Kaifeng Bi², Lingxi Xie², Jun Zhu³, Richang Hong¹, and Qi Tian²†.  
> **Resolving sources of uncertainty in AI weather forecasting**  
> arXiv:2511.14218, 2025.  
>
> ¹ Hefei University of Technology, Hefei, China  
> ² Huawei Inc., Shenzhen, China  
> ³ Tsinghua University, Beijing, China  
>
> * Equal contribution. † Corresponding author.

---

## Installation

We recommend using Conda to create the environment.

```bash
conda create -n pangu-bayes python=3.10
conda activate pangu-bayes
pip install -r requirements.txt
```

---

## Data Preparation

Pangu-Bayes uses global atmospheric reanalysis data for training and evaluation.

To help users quickly test the code, this repository provides a small example dataset under:

```text
ERA5_examples/
```

Expected upper-air variables include:

- Geopotential (Z)
- Temperature (T)
- Specific humidity (Q)
- U component of wind (U)
- V component of wind (V)

Expected surface variables include:

- 2-m temperature (T2M)
- 10-m u component of wind (10U)
- 10-m v component of wind (10V)
- Mean sea-level pressure (MSL)

The expected pressure levels are:

```text
50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000 hPa
```

An example directory structure is:

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
└── test/
    ├── 2022_0000.pt
    └── 2022_0001.pt
```

---

## Method Overview

The overall pipeline consists of five stages:

```text
Deterministic pretraining
        ↓
Epistemic uncertainty learning
        ↓
Joint epistemic-aleatoric uncertainty learning
        ↓
Operational-analysis fine-tuning
        ↓
Probabilistic post-processing
```

Pangu-Bayes represents two complementary sources of predictive uncertainty:

1. **Epistemic uncertainty**, represented through Bayesian uncertainty in forecast-model parameters.
2. **Aleatoric uncertainty**, represented through flow-dependent stochastic perturbations of the atmospheric state.

During ensemble inference, posterior-sampled forecast-model realizations are combined with state-dependent perturbations to generate probabilistic global weather forecasts.

---

## Training

### Stage 1: Deterministic Pretraining

The deterministic backbone is trained to predict future global atmospheric fields from historical atmospheric states.

```bash
bash train_scripts/pretrain.sh
```

---

### Stage 2: Epistemic Uncertainty Learning

Epistemic uncertainty is introduced into the forecasting backbone through Bayesian parameter uncertainty.

```bash
bash train_scripts/train_eu.sh
```

---

### Stage 3: Joint Epistemic-Aleatoric Uncertainty Learning

The joint EU-AU model combines epistemic uncertainty from Bayesian model parameters with aleatoric uncertainty from state-dependent atmospheric perturbations.

```bash
bash train_scripts/train_eu_au.sh
```

---

### Stage 4: Operational-Analysis Fine-Tuning

After completing joint epistemic--aleatoric uncertainty learning, we further fine-tune Pangu-Bayes on high-resolution operational-analysis data.

This produces the operational-analysis-initialized **Pangu-Bayes(oper.)** variant used for matched global probabilistic evaluation.

```bash
bash train_scripts/train_eu_au_hres.sh
```

---

### Stage 5: Probabilistic Post-processing

The probabilistic post-processing module is used to calibrate tropical-cyclone intensity forecasts.

Before running the post-processing training script, users should first construct the post-processing training dataset from ensemble forecast outputs.

This step computes task-specific summary statistics from the ensemble forecast fields and pairs them with the corresponding verification targets.

The post-processing workflow is:

```text
Ensemble forecast outputs
        ↓
Compute cyclone-level statistics
        ↓
Construct post-processing training dataset
        ↓
Train probabilistic post-processing model
```

The resulting model provides probabilistic predictions for tropical-cyclone intensity quantities, including maximum sustained wind and minimum sea-level pressure.

---

## Global Weather Forecasting and Evaluation

Pangu-Bayes generates global ensemble weather forecasts from atmospheric initial states.

Example ensemble inference:

```bash
bash inference_ensemble_scripts/inference.sh
```

Global forecast evaluation is performed at the field level.

The principal evaluation metrics include:

- Ensemble-mean root mean squared error (RMSE)
- Continuous ranked probability score (CRPS)
- Ensemble spread
- Spread--skill ratio (SSR)

These metrics are evaluated using latitude-dependent area weighting on the global latitude--longitude grid.

---

## Tropical Cyclone Forecasting and Evaluation

Although Pangu-Bayes is designed as a global ensemble weather forecasting framework, this repository also provides tropical-cyclone ensemble forecasting as an important downstream application.

Pangu-Bayes first generates global ensemble forecast fields. Cyclone-level trajectories and intensity quantities are then extracted and evaluated using external tropical-cyclone analysis tools.

The tropical-cyclone workflow consists of two main steps.

### 1. Cyclone Detection and Tracking

Tropical cyclone candidates and trajectories are identified from global forecast fields using the **TempestExtremes** tracker.

The tracker extracts storm-level quantities from each ensemble member, including:

- Cyclone-centre latitude
- Cyclone-centre longitude
- Maximum sustained wind speed (MSW)
- Minimum sea-level pressure (MSLP)

### 2. Track Matching and Verification

Predicted cyclone tracks are matched with observed best-track records from the **IBTrACS** dataset using the **huracanpy** Python package.

After matching, deterministic and probabilistic verification metrics are computed for cyclone track and intensity forecasts.

Track metrics include:

- Direct positional error (DPE)
- Track CRPS

Intensity metrics include:

- MSW mean absolute error
- MSLP mean absolute error
- MSW CRPS
- MSLP CRPS

The repository also contains scripts used for source-specific uncertainty analyses of tropical-cyclone forecasts.

---

## Model Checkpoints

The Pangu-Bayes model checkpoints are publicly available on Hugging Face:

https://huggingface.co/xionge/pangu-bayes

The checkpoints correspond to the principal stages of the forecasting framework and are organized as:

```text
checkpoint/
├── deterministic_forecast.pt
├── eu_ensemble_forecast.pt
├── eu_au_ensemble_forecast.pt
├── eu_au_ensemble_forecast_hres.pt
└── best_prob_ann.pt
```

The training and inference code, tropical-cyclone tracking and evaluation scripts, and source-specific uncertainty analysis code are publicly available at:

https://github.com/hfutml/pangu-bayes

---

## Preprint

An earlier version of this work is available on arXiv:

**Bridging the Gap Between Bayesian Deep Learning and Ensemble Weather Forecasts**

Xinlei Xiong, Wenbo Hu, Shuxun Zhou, Kaifeng Bi, Lingxi Xie, Ying Liu, Richang Hong, and Qi Tian.

arXiv:2511.14218, 2025.

- **Abstract page:** https://arxiv.org/abs/2511.14218
- **PDF:** https://arxiv.org/pdf/2511.14218
- **DOI:** https://doi.org/10.48550/arXiv.2511.14218

The current repository contains the implementation and resources associated with the continuing development of Pangu-Bayes.

---

## Citation

If you find this repository useful, please cite the current arXiv preprint:

```bibtex
@article{xiong2025bridging,
  title={Bridging the Gap Between Bayesian Deep Learning and Ensemble Weather Forecasts},
  author={Xiong, Xinlei and
          Hu, Wenbo and
          Zhou, Shuxun and
          Bi, Kaifeng and
          Xie, Lingxi and
          Liu, Ying and
          Hong, Richang and
          Tian, Qi},
  journal={arXiv preprint arXiv:2511.14218},
  year={2025}
}
```

The citation information will be updated when a revised preprint or formally published version becomes available.

---

## Code and Model Availability

The training and inference code, tropical-cyclone tracking and evaluation scripts, and source-specific uncertainty analysis code are publicly available at:

https://github.com/hfutml/pangu-bayes

The Pangu-Bayes model checkpoints are publicly available at:

https://huggingface.co/xionge/pangu-bayes

---

## Acknowledgements

This project builds upon recent advances in data-driven global weather forecasting, Bayesian deep learning, ensemble prediction, and probabilistic forecast verification.

We thank the providers of ERA5, HRES and tropical-cyclone best-track datasets for making atmospheric and cyclone records available to the research community.

We also acknowledge the open-source tools and benchmark resources used for global weather forecasting and tropical-cyclone evaluation.

---

## Links

- **GitHub:** https://github.com/hfutml/pangu-bayes
- **Hugging Face:** https://huggingface.co/xionge/pangu-bayes
- **arXiv:** https://arxiv.org/abs/2511.14218
