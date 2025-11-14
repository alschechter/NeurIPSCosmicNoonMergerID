# CosmicNoonMergerID
<img width="2788" height="639" alt="MockSteps NSF" src="https://github.com/user-attachments/assets/675e7737-4a0c-45bf-bdd4-383696309b73" />



## Overview
This work uses TNG50, HST CANDELS imaging, and Zoobot to create a CNN to identify galaxy mergers near cosmic noon

### Data Availability
All datasets used in this project are available on our [Zenodo Page](10.5281/zenodo.17612012).

---

## Installation

To set up the environment, install the required dependencies using:

```bash
pip install -r requirements.txt
```
or consult appropriate online documentation.

## Code Structure

The repository is organized into the following components:

- **TNG50 Merger Catalog Creation**:  
  `TNGstuff/build_merger_catalog_TNG_SF.py`
  `matched_nonmergers_TNG.py`
 Walk galaxy merger trees and create merger catalogs and mass-matched nonmerger catalogs through time.


- **Mock Observation Image Pipeline**:  
  -`MockImageFunctions.py`
  `MakeMocks.py`
  Filter, rebin, convolve with PSF, and place mock image in realistic backgrounds.
  -`CutoutBackgroundsLockMethod.py`
  Cutout the realistic backgrounds from CANDELS mosaics, downloadable from Space Telescope Science Institute and MAST.
  
- **Convolutional Neural Network**:  
  - `DivideIntoCNNSets.ipynb`  
    Create 3 color images and divide into training, validation, and test sets.
  - `BinaryMergerDataset`  
    Custom dataloader.
  - `ResNet_Adam_Zoobot.py`  
    Train Zoobot ResNet18 CNN.

- **Testing Scripts**:  
  - `TestSetAnalysis.ipynb`  
    Standard model evaluation, plus UMAP, tsne, isomap, and calibration curves.
  - `TestSetGradCAM.py`  
    Script for creating GradCAM images.



### Code Authors

- Aimee Schechter, Becky Nevin, Jacob Shen, with help from Alex Ćiprijanović and Marina Dunn

## Citation

coming soon
