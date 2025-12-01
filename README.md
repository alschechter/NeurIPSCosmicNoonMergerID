# CosmicNoonMergerID
<img width="2788" height="639" alt="MockSteps NSF" src="https://github.com/user-attachments/assets/675e7737-4a0c-45bf-bdd4-383696309b73" />



## Overview
This work uses TNG50, HST CANDELS imaging, and Zoobot to create a CNN to identify galaxy mergers near cosmic noon. This was accepted to NeurIPS ML4PS 2025.

### Data Availability
All datasets used in this project are available on our [Zenodo Page](https://zenodo.org/records/17612012).

---

## Installation

To set up the environment, install the required dependencies using:

```bash
pip install -r requirements.txt
```
or consult appropriate online documentation.

## Code Structure

The repository is organized into the following components:



- **Convolutional Neural Network**:  
  - `BinaryMergerDataset`  
    Custom dataloader.
  - `ResNet_Adam_Zoobot.py`  
    Train Zoobot ResNet18 CNN.

- **Testing Scripts**:  
  - `TestSetAnalysis.ipynb`  
    Standard model evaluation, plus UMAP, tsne, isomap, and calibration curves.
  - `TestSetGradCAM.py`  
    Script for creating GradCAM images.

- Further code to select galaxies, create mock images, and more in depth CNN analysis will be available soon, as part of the repository for our [ApJ paper](https://ui.adsabs.harvard.edu/abs/2025arXiv251012173S/abstract), an expanded version of this work.

### Code Authors

- Aimee Schechter, Becky Nevin, Jacob Shen, with help from Alex Ćiprijanović and Marina Dunn

## Citation

- [ADS](https://ui.adsabs.harvard.edu/abs/2025arXiv251115006S/abstract)
- [arXiv](https://arxiv.org/abs/2511.15006)
