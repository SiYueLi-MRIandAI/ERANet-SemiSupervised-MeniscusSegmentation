# ERANet: Semi-Supervised Meniscus Segmentation with Edge Replacement Augmentation and Prototype Consistency Alignment

## ✨ Overview

The **meniscus**, a vital fibrocartilaginous tissue in the knee joint, plays a crucial role in joint stability and osteoarthritis (OA) prevention. However, its segmentation in **MRI** remains challenging due to:

- Anatomical variability
- Partial volume effects
- Low tissue contrast

To overcome these obstacles, we introduce **ERANet**, a novel **semi-supervised segmentation framework** designed to achieve accurate meniscus delineation with minimal manual annotations.

---

## 🧠 Key Contributions

- **Edge Replacement Augmentation (ERA)**  
  A meniscus-specific augmentation that replaces peripheral edge structures with background intensities to simulate realistic anatomical variations.

- **Prototype Consistency Alignment (PCA)**  
  Encourages compact feature representations using class-wise prototypes derived from reliable pseudo-labels.

- **Conditional Self-Training (CST)**  
  Selects temporally stable pseudo-labels for progressive refinement in a two-stage self-training pipeline (U3 and U4).

---

## 📊 Dataset

ERANet was evaluated on:
- **3D DESS MRI (From the **Osteoarthritis Initiative (OAI)** dataset)**
- **3D FSE MRI**

---
