# Gut Microbial Signatures for Early Screening of Autism Spectrum Disorder

> Official implementation of the paper:  
> **"Gut Microbial Signatures for Early Screening of Autism Spectrum Disorder: An Interpretable Machine Learning Approach"**  
> Glykeria Theodorou, Konstantinos Mitsis, Aris Markogiannakis, Konstantina Nikita, Maria Athanasiou  
> Accepted at **IEEE BIBE 2025** (Bioinformatics and Bioengineering)

---

## 🧠 Overview
Autism Spectrum Disorder (ASD) is a complex neurodevelopmental disorder with strong evidence linking it to gut microbiome disruptions via the gut–brain axis.  
In this work, we propose an **interpretable machine learning pipeline** for early ASD screening using gut microbiome profiles across **9 cohorts (n = 929 samples)**.

Our contributions:
- 📊 Applied **supervised ML models** (Logistic Regression, SVM, Random Forest, XGBoost) within a **nested cross-validation (NCV)** framework.  
- 🔄 Introduced **compositional data augmentation** methods (Aitchison Mixup, Compositional CutMix, Compositional Feature Dropout) tailored for microbiome data.  
- 🧬 Identified **key microbial species** as candidate biomarkers for ASD using **SHAP values** and **information gain**.  
- 🚀 Achieved best performance with **XGBoost + Feature Dropout**:  
  - Accuracy: **73.3%**  
  - AUC: **82.3%**  
  - F1-score: **73.3%**

---

## 📂 Repository Structure
