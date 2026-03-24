import numpy as np
import pandas as pd

def aitchison_mixup(X, y, n_samples=200, random_state=42):
    """
    Performs Aitchison Mixup: generates synthetic samples using the Aitchison geometry.
    """
    # Set random state for reproducibility
    np.random.seed(random_state)
    
    X_new, y_new = [], []
    
    for _ in range(n_samples):
        # Choose two samples from the same class
        class_c = np.random.choice(np.unique(y))
        indices = np.where(y == class_c)[0]
        idx1, idx2 = np.random.choice(indices, 2, replace=True)
        
        # Draw λ ~ U(0,1)
        lam = np.random.uniform(0, 1)
        
        # Get the two points
        x1 = np.array(X.iloc[idx1]) if isinstance(X, pd.DataFrame) else X[idx1]
        x2 = np.array(X.iloc[idx2]) if isinstance(X, pd.DataFrame) else X[idx2]
        
        # Ensure compositions are valid (sum to 1 and positive)
        x1 = np.maximum(x1, 1e-10)
        x2 = np.maximum(x2, 1e-10)
        x1 = x1 / np.sum(x1)
        x2 = x2 / np.sum(x2)
        
        # Implement Aitchison operations:
        # x_mix = (λ ⊙ x₁) ⊕ ((1 - λ) ⊙ x₂)
        
        # Scalar multiplication (⊙) on x1: power transformation
        term1 = np.power(x1, lam)
        
        # Scalar multiplication (⊙) on x2
        term2 = np.power(x2, 1-lam)
        
        # Aitchison addition (⊕) is component-wise multiplication followed by closure
        x_mix = term1 * term2
        
        # Closure operation to ensure the result is in the simplex
        x_mix = x_mix / np.sum(x_mix)
        
        X_new.append(x_mix)
        y_new.append(class_c)
    
    return pd.DataFrame(X_new, columns=X.columns if isinstance(X, pd.DataFrame) else None), np.array(y_new)

def compositional_cutmix(X, y, n_samples=200, random_state=42):
    """
    Performs Compositional CutMix as described in the paper.

    Creates new samples by taking complementary subcompositions from two training
    points of the same class, then renormalizing.
    """
    # Set random state for reproducibility
    np.random.seed(random_state)
    
    X_new, y_new = [], []
    n_features = X.shape[1]

    for _ in range(n_samples):
        # Draw a class c from the class prior
        class_c = np.random.choice(np.unique(y))

        # Draw λ ~ U(0,1)
        lambda_val = np.random.uniform(0, 1)

        # Draw two training points i₁, i₂ such that y_i₁ = y_i₂ = c
        indices = np.where(y == class_c)[0]
        idx1, idx2 = np.random.choice(indices, 2, replace=True)

        # Get the two points
        x1 = np.array(X.iloc[idx1]) if isinstance(X, pd.DataFrame) else X[idx1]
        x2 = np.array(X.iloc[idx2]) if isinstance(X, pd.DataFrame) else X[idx2]

        # For each feature j, draw I_j ~ Bernoulli(λ)
        mask = np.random.binomial(1, lambda_val, size=n_features)

        # Set x̃_j based on mask (I_j): x̃_j = x_i₁,j if I_j = 0 or x̃_j = x_i₂,j if I_j = 1
        x_mix = np.where(mask == 0, x1, x2)

        # Renormalize: x^aug = x̃/(∑_j x̃_j)
        x_sum = np.sum(x_mix)
        if x_sum > 0:
            x_mix = x_mix / x_sum

        X_new.append(x_mix)
        y_new.append(class_c)

    return pd.DataFrame(X_new, columns=X.columns if isinstance(X, pd.DataFrame) else None), np.array(y_new)

def compositional_feature_dropout(X, y, n_samples=200, random_state=42):
    """
    Performs Compositional Feature Dropout as described in the paper.
    Creates n_samples new augmented samples.
    """
    # Set random state for reproducibility
    np.random.seed(random_state)
    
    X_new, y_new = [], []
    n_original_samples = len(X)
    
    for _ in range(n_samples):  # Loop n_samples times to generate n_samples augmented data points
        # Randomly select an original sample to augment
        original_idx = np.random.randint(0, n_original_samples)
        x_original = np.array(X.iloc[original_idx]) if isinstance(X, pd.DataFrame) else X[original_idx]
        y_original = y[original_idx]
        
        # Draw λ ~ U(0,1)
        lambda_val = np.random.uniform(0, 1)
        
        # For each feature j, draw Ij ~ Bernoulli(λ)
        mask = np.random.binomial(1, lambda_val, size=len(x_original))
        
        # Set x̃_j = 0 if Ij = 0
        x_new = x_original * mask
        
        # Set x^aug = x̃/(∑ x̃_j)
        sum_x_new = np.sum(x_new)
        if sum_x_new > 0:
            x_new = x_new / sum_x_new
        else:
            # If all features were dropped (sum_x_new is 0), keep the original composition
            # as described in some compositional data augmentation papers to avoid division by zero
            # and to ensure a valid composition.
            x_new = x_original # Or you could choose to return a vector of zeros if that's desired, but usually original is preferred.
        
        X_new.append(x_new)
        y_new.append(y_original)
    
    return pd.DataFrame(X_new, columns=X.columns if isinstance(X, pd.DataFrame) else None), np.array(y_new)