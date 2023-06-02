# Redscar Dataset Processing

This repository is dedicated to store all the Jupyter notebooks used to generate the baseline methods and their results introduced in the article describing the Redscar Dataset (accessible from the [official Redscar website](http://redscar.uib.es/)), which is:

```
Munar M., González-Hidalgo M, Moyà-Alcover G. 
Developing a Dataset for automatic follow-up of abdominal post-surgical wounds: Redscar
``` 

In that paper we propose some baseline methods to solve the following tasks: wound segmentation, staples segmentation and infection classification. The used notebooks and pre-trained models are described in the following sections.

## Notebooks and models for wound segmentation

To perform the segmentation of the wound present in the image, three different neural network architectures have been trained (available in [this repository](https://github.com/mmunar97/cnn_architectures)), which in turn have been trained with different parameters (number of epochs, input dimension, etc.). Specifically, the experiments are: 

* Segmentation using the U-Net architecture. The notebook is available at `notebooks/cnn_segmentation/Notebook_WoundSegmentation_UNet.ipynb`. All the models have 65 filters, an input size of $512\times 512\times 3$ and the epochs considered were 25, 55 and 100. The models can be found at the following directions:
  - UNet model for wound segmentation trained with 25 epochs. [Download here](https://sourceforge.net/projects/redscar-dataset-processing/files/keras_models/UNet/EXPERIMENT1-WOUND/unet1_epochs%3D25_lr%3D3e-4_res%3D0.h5/download).
  - UNet model for wound segmentation trained with 55 epochs. [Download here](https://sourceforge.net/projects/redscar-dataset-processing/files/keras_models/UNet/EXPERIMENT1-WOUND/unet2_epochs%3D55_lr%3D3e-4_res%3D0.h5/download).
  - UNet model for wound segmentation trained with 100 epochs. [Download here](https://sourceforge.net/projects/redscar-dataset-processing/files/keras_models/UNet/EXPERIMENT1-WOUND/unet3_epochs%3D100_lr%3D3e-4_res%3D0.h5/download).

* Segmentation using the Double U-Net architecture. The notebook is available at `notebooks/cnn_segmentation/Notebook_WoundSegmentation_DoubleUNet.ipynb`. All the models have an input size of $256\times 256\times 3$ and the epochs considered were 25, 55 and 100. The models can be found at the following directions:
  - Double U-Net model for wound segmentation trained with 25 epochs. [Download here](https://sourceforge.net/projects/redscar-dataset-processing/files/keras_models/DoubleUNet/EXPERIMENT1-WOUND/dun1_epochs%3D25_lr%3D3e-5_res%3D0.h5/download).
  - Double U-Net model for wound segmentation trained with 55 epochs. [Download here](https://sourceforge.net/projects/redscar-dataset-processing/files/keras_models/DoubleUNet/EXPERIMENT1-WOUND/dun2_epochs%3D55_lr%3D3e-5_res%3D0.h5/download).
  - Double U-Net model for wound segmentation trained with 100 epochs. [Download here](https://sourceforge.net/projects/redscar-dataset-processing/files/keras_models/DoubleUNet/EXPERIMENT1-WOUND/dun3_epochs%3D100_lr%3D3e-5_res%3D0.h5/download).

* Segmentation using the GSC architecture. The notebook is available at `notebooks/cnn_segmentation/Notebook_WoundSegmentation_GSC.ipynb`. All the models have an input size of $512\times 512$ (grayscale) and the epochs considered were 25, 55 and 100. The models can be found at the following directions:
  - GSC model for wound segmentation trained with 25 epochs. [Download here](https://sourceforge.net/projects/redscar-dataset-processing/files/keras_models/GSC/EXPERIMENT1-WOUND/gsc1_epochs%3D25_lr%3D3e-5_res%3D0.h5/download).
  - GSC model for wound segmentation trained with 55 epochs. [Download here](https://sourceforge.net/projects/redscar-dataset-processing/files/keras_models/GSC/EXPERIMENT1-WOUND/gsc2_epochs%3D55_lr%3D3e-5_res%3D0.h5/download).
  - GSC model for wound segmentation trained with 100 epochs. [Download here](https://sourceforge.net/projects/redscar-dataset-processing/files/keras_models/GSC/EXPERIMENT1-WOUND/gsc3_epochs%3D100_lr%3D3e-5_res%3D0.h5/download).
