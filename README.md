# Ripple Perturbations Through Structure: Likelihood-Constrained Adversarial Attacks on Heterogeneous Tabular Data

[![Conference](https://img.shields.io/badge/ICML-2026-blue.svg)](https://icml.cc/)
[![Paper](https://img.shields.io/badge/Paper-PDF-red.svg)](#)

This repository serves as the official reproducibility hub for the ICML 2026 paper: **"Ripple Perturbations Through Structure: Likelihood-Constrained Adversarial Attacks on Heterogeneous Tabular Data"**. 

## Reproducibility & Open Source Statement

We firmly believe in the importance of open science and reproducibility. Currently, the production-level source code associated with this research is undergoing a strict internal compliance and Intellectual Property (IP) review by **Ant Group**. 

Due to the rigorous nature of this corporate qualification process, the exact timeline for release is currently unknown. We are fully committed to open-sourcing the code immediately upon approval. 

In the meantime, to guarantee full reproducibility, this repository and our paper's Appendix provide an exhaustive technical disclosure. We provide line-by-line pseudo-code, precise hyperparameter configurations, and architectural details to ensure any researcher can implement the LCSA framework from scratch.

---

## Method Overview: LCSA Framework

LCSA (Likelihood-Constrained Structural Attack) is a white-box framework that formulates adversarial generation as an optimization process over structurally admissible perturbations. It operates in two synergistic phases:

1. **Phase I: Heterogeneous Ensemble SCM Learning.** Learns an ensemble of neural Structural Causal Models (SCMs) to capture directed, asymmetric dependencies across both continuous and categorical variables.
2. **Phase II: Structure-Aware Adversarial Optimization.** Introduces a **Ripple Mechanism** that propagates perturbation updates downstream along the learned causal graph, acting as a structural preconditioner to mitigate gradient masking.

---

## Pythonic Pseudo-Code

Below is the detailed, implementable logic for the LCSA framework, corresponding to Algorithm 1 in our paper.
```python

# ==========================================
# Phase I: Heterogeneous Ensemble SCM Learning
# ==========================================
def train_ensemble_scm(D_data, M=10):
    ensemble_models = []
    noise_variances = []
    
    for m in range(M):
        D_m = bootstrap_sample(D_data)
        # Optimize Heterogeneous MLP parameters (Phi) and Adjacency Matrix (W)
        Phi_m, W_m = optimize_scm_heterogeneous(D_m)
        
        # Estimate exogenous noise variance for continuous features
        sigma_m = estimate_variance(D_m, Phi_m, W_m) 
        
        ensemble_models.append((Phi_m, W_m))
        noise_variances.append(sigma_m)
        
    return ensemble_models, noise_variances

# ==========================================
# Phase II: Structure-Aware Adversarial Optimization
# ==========================================
def generate_lcsa_attack(x, y, model, ensemble, budget_Lp, budget_gamma):
    nu = init_zeros()      # \nu^{(0)}
    lambda_param = 0.0     # \lambda^{(0)}
    x_adv = None
    
    for t in range(T_total):
        # 1. Forward pass for current state \nu^{(t)}
        delta_t = project_onto_Omega(ripple_mechanism(nu, ensemble), budget_Lp)
        x_tilde = x + delta_t
        
        # Soft-Embedding: differentiable input for the target model
        z_tilde = apply_soft_embedding(x_tilde, model.embeddings)
        
        # 2. Compute Gradients and Update \nu
        if t < T_warm:
            # Stage 1: Warm Start
            grad = compute_gradient(classification_loss(model(z_tilde), y), nu)
            nu_next = nu + lr_eta * grad
        else:
            # Stage 2: Augmented Lagrangian
            psi_t = calc_scm_loss(x_tilde, ensemble) - calc_scm_loss(x, ensemble) - budget_gamma
            
            aug_loss = classification_loss(model(z_tilde), y) \
                       - lambda_param * psi_t \
                       - (rho / 2) * max(0, psi_t)**2
                       
            grad = compute_gradient(aug_loss, nu)
            nu_next = nu + lr_eta * grad
            
        # 3. Apply state update: \nu^{(t)} -> \nu^{(t+1)}
        nu = nu_next
        
        # 4. Evaluate the NEW state \nu^{(t+1)} (Computed ONLY ONCE)
        delta_next = project_onto_Omega(ripple_mechanism(nu, ensemble), budget_Lp)
        x_tilde_next = x + delta_next
        
        # 5. Update Multiplier (only after warm-up phase)
        if t >= T_warm:
            psi_next = calc_scm_loss(x_tilde_next, ensemble) - calc_scm_loss(x, ensemble) - budget_gamma
            lambda_param = max(0, lambda_param + rho * psi_next)
            
        # 6. Verification
        x_bar = discretize_categorical(x_tilde_next)
        if model(x_bar) != y and (calc_scm_loss(x_bar, ensemble) - calc_scm_loss(x, ensemble) <= budget_gamma):
            x_adv = x_bar
            
    return x_adv
```

## ⚙️ Exact Hyperparameter Configurations

To reproduce the results presented in the paper, please use the following exact configurations.

### Phase I: SCM Learning
| Parameter | Value | Notes |
| :--- | :--- | :--- |
| **Architecture** | 2-layer MLP | Standard multi-layer perceptron per feature |
| **Hidden Units** | 64 | Increased to **128** for high-dimensional datasets (e.g., MALWARE) |
| **Activation** | Leaky ReLU | Applied to hidden layers |
| **Optimizer** | Adam | Standard configuration |
| **Learning Rate** | `5e-4` | |
| **Iterations** | 50 | |
| **Adjacency Threshold** | $90^{th}$ percentile | Adaptive thresholding applied to absolute weight distribution |
| **Sparsity Cap ($k$)** | $k=150$ | Only applied for MALWARE to ensure tractability; $k=d$ for others |

### Phase II: Adversarial Optimization
| Parameter | Value | Definition |
| :--- | :--- | :--- |
| **Ensemble Size ($M$)** | 10 | Number of independent SCMs |
| **Total Iterations ($T_{total}$)**| 20 | Dual-ascent steps |
| **Warm-up Steps ($T_{warm}$)** | 4 | Initial unconstrained exploration steps |
| **Penalty ($\rho$)** | 0.5 | Augmented Lagrangian quadratic penalty |
| **Learning Rate ($\eta$)** | 0.5 | Step size for latent variable $\nu$ |

### Experimental Budgets
*   **Geometric Budget ($\epsilon$)**: $L_2 = 0.5$ or $L_\infty = 0.5$ on $[0, 1]$ normalized continuous features.
*   **Structural Budget ($\gamma$)**: Data-driven, set to the $50^{th}$ percentile ($q=0.5$) of the training set's structural violation scores for main experiments.

---

## Datasets & Models


**Reproducibility Note:** For the real-world datasets and target model architectures, we refer to the codebase provided by the [TabularBench repository](https://github.com/serval-uni-lu/tabularbench). You can directly find the exact data processing pipelines and model implementations there.

---

## Citation

If you use our theoretical framework, pseudo-code, or synthetic benchmarks in your research, please cite our paper:
```bibtex
@inproceedings{zhou2026ripple,
  title={Ripple Perturbations Through Structure: Likelihood-Constrained Adversarial Attacks on Heterogeneous Tabular Data},
  author={Zhou, Zhengjie and Yan, Jiahuan and Ma, Boqun and Feng, Weiwei and Liu, Tengfei and Wang, Weiqiang},
  booktitle={Proceedings of the 43rd International Conference on Machine Learning},
  year={2026},
  organization={PMLR}
}
