# 🦠 ASD-Microbiome

> An Interpretable Machine Learning Framework for the Early Screening of Autism Spectrum Disorder (ASD) using Gut Microbial Signatures.

This repository provides a robust, end-to-end machine learning pipeline that leverages gut microbiome profiles to classify children with Autism Spectrum Disorder (ASD) versus neurotypical controls. By combining rigorous feature selection, compositional data augmentation, and explainable AI (SHAP), this project identifies biologically meaningful microbial biomarkers while mitigating the challenges of high dimensionality and dataset heterogeneity.

## 📊 Data Availability

This project utilizes gut microbial profiles merged from 9 distinct cohorts. The raw microbiome abundance files and associated metadata used to train and evaluate this framework are publicly available.

You can access and download the datasets here:
**https://github.com/mortonjt/asd_multiomics_analyses/tree/main/sfari/data/sra**



## 🏗️ Architecture & Pipeline

The framework is structured around four core pillars:
1. **Data Consolidation & Preprocessing:** Merging and harmonizing microbiome profiles from 9 distinct cohorts (total $n=929$).
2. **Feature Selection:** Utilizing the non-parametric Kruskal-Wallis H test to filter the high-dimensional microbial space down to statistically significant features (p < 0.05).
3. **Compositional Data Augmentation:** Employing advanced synthetic data generation techniques (like Aitchison Mixup and Compositional CutMix) to expand the training dataset, reduce overfitting, and handle the compositional nature of microbiome data.
4. **Interpretable Modeling (Nested CV):** Training and evaluating supervised classifiers (Random Forest, XGBoost, SVM, Logistic Regression) within a strict Nested Cross-Validation (NCV) framework, followed by SHAP (SHapley Additive exPlanations) analysis to interpret the biological relevance of the predictions.

---

## 🚀 Getting Started

### Prerequisites
Ensure you have the required dependencies installed. You will generally need:
* `python >= 3.8`
* `scikit-learn`
* `xgboost`
* `shap`
* `pandas`, `numpy`, `scipy`
* `matplotlib`, `seaborn`

*Note: Ensure your raw microbiome abundance files (e.g., `.biom` or `.csv` formats) and metadata are placed in the appropriate data directories before running the pipeline.*

---

## 🛠️ Pipeline Details & Usage

### 1. Data Preprocessing & Merging (`merge_biom_files.py`)
This script handles the ingestion of raw biological data across multiple cohorts. 

**What it does:**
* Parses and merges microbial abundance tables from the 9 independent cohorts.
* Cleans and standardizes the metadata to strictly label samples as either ASD (cases) or neurotypical (controls).

**Execution:**
```bash
python merge_biom_files.py
```

### 2. Feature Selection & Augmentation (`kruskal_wallis.py` & `augmentation_methods.py`)
Microbiome data is notoriously noisy and high-dimensional. This step isolates the signal and safely inflates the training set.

**What it does:**
* **`kruskal_wallis.py`:** Applies the Kruskal-Wallis H test to rank and select microbial features whose abundance profiles differ significantly between ASD and neurotypical groups.
* **`augmentation_methods.py`:** Implements compositional data augmentation strategies (Aitchison Mixup, Compositional CutMix) to synthesize new training samples while respecting the simplex geometry of microbiome abundances.

### 3. Model Training & Nested Cross-Validation (`nested_cv_with_shap_analysis.py`)
The core modeling engine. It ensures unbiased performance estimation by optimizing hyperparameters in an inner loop while evaluating the model in an outer loop.

**What it does:**
* Iterates through a hyperparameter grid (defined in `hyperparameter_grid.py`) for multiple models.
* Integrates data augmentation strictly within the training folds to prevent data leakage.
* Calculates performance metrics (Accuracy, ROC-AUC, F1-Score) and extracts SHAP values for the best-performing models to measure feature importance.

**Execution:**
```bash
python nested_cv_with_shap_analysis.py
```
*(Alternatively, you can run `python run_integrated_rf_shap_analysis.py` if you want to focus specifically on the Random Forest pipeline).*

### 4. Interpretability & Visualization (`shap_plots.py` & `create_and_save_all_plots.py`)
Machine learning in healthcare requires transparency. This module translates black-box predictions into biological insights.

**What it does:**
* Generates SHAP summary plots, dependence plots, and confusion matrices (via `confusion_matrix_util.py`).
* Saves high-resolution figures that map exactly *which* microbial features drove the model to predict ASD.

**Execution:**
```bash
python create_and_save_all_plots.py
```

---

## 📂 Repository Structure

* **Core Execution:** `main_function.py`, `run_analysis.ipynb`
* **Data Processing:** `merge_biom_files.py`, `splitting_analysis.py`
* **Modeling & Tuning:** `hyperparameter_grid.py`, `nested_cv_util.py`, `nested_cv_with_shap_analysis.py`
* **Statistics & Augmentation:** `kruskal_wallis.py`, `bayesian_ranking.py`, `augmentation_methods.py`, `find_best_augmentation_method.py`
* **Explainability (XAI):** `shap_util.py`, `shap_plots.py`, `important_features.txt`
* **Evaluation & Output:** `confusion_matrix_util.py`, `create_and_save_all_plots.py`, `save_all_results.py`
* **Documentation:** `BIBE-2025_paper_115.pdf`

---
## 📝 Citation

If you use this code or our findings in your research, please cite our paper:


### BibTeX
```bibtex
@INPROCEEDINGS{11273536,
  author={Theodorou, Glykeria and Markogiannakis, Aris and Athanasiou, Maria and Mitsis, Konstantinos and Nikita, Konstantina},
  booktitle={2025 IEEE 25th International Conference on Bioinformatics and Bioengineering (BIBE)}, 
  title={Gut Microbial Signatures for Early Screening of Autism Spectrum Disorder: An Interpretable Machine Learning Approach}, 
  year={2025},
  volume={},
  number={},
  pages={346-352},
  keywords={Autism;Microorganisms;Biological system modeling;Pipelines;Machine learning;Predictive models;Biomarkers;Feature extraction;Data augmentation;Data models;ASD;Gut Microbiome;gut-brain axis;ML models;data augmentation},
  doi={10.1109/BIBE66822.2025.00064}}
