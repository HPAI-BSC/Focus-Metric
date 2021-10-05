# Focus metric - Quantitatively Assessing Feature Attribution Methods

We propose in [[1]](#1) a novel evaluation metric -- the _Focus_ -- 
designed to quantify the faithfulness of explanations provided by feature 
attribution methods. This repository contains the mosaics and the code needed to replicate the
experiments in our paper: [Who Explains the Explanation? Quantitatively Assessing Feature AttributionMethods](). 

Three explainability methods have been evaluated:
* Smoothgrad [[7]](#7): the implementation used is based on the [work](https://github.com/wjNam/Relative_Attributing_Propagation) of Nam *et al.* 
* Layer-wise Relevance Propagation (LRP) [[2]](#2): the implementation used is based on the [work](https://github.com/kazuto1011/grad-cam-pytorch) of Nakashima *et al.*
* GradCAM [[6]](#6): The implementation is based on the [work](https://github.com/jacobgil/pytorch-grad-cam)
of Gildenblat *et al.*



### Requirements

This code runs under Python 3.7.1. The python dependencies are defined in `requirements.txt`. 


## Available mosaics

We provide already created mosaics from four different datasets:

* [Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats/overview) mosaics can be downloaded [here](https://storage.hpai.bsc.es/focus-metric/catsdogs_mosaic.zip)
* ILSVRC 2012 [[5]](#5) mosaics can be downloaded [here](https://storage.hpai.bsc.es/focus-metric/ilsvrc2012_mosaic.zip).
* MIT67 [[4]](#4) mosaics can be downloaded [here](https://storage.hpai.bsc.es/focus-metric/mit67_mosaic.zip).
* MAMe [[3]](#3) mosaics can be downloaded [here](https://storage.hpai.bsc.es/focus-metric/mame_mosaic.zip).



## How to run the experiments

We already provide the bash scripts needed to calculate the focus of the 
different settings. Each execution has two steps:

1. First, the explainability method is applied and the relevance matrices
   are obtained and save in: 
   ```$PROJECT_PATH/data/explainability/```

2. Second, the _Focus_ is computed from the relevances obtained in the previous step.
   
To run both steps execute the following bash scripts:

#### Step 1
> `cd $PROJECT_PATH/explainability/scripts/explainability_dataset/`
     
> `sh explainability_dataset_architecture_method.sh`

#### Step 2
> `cd $PROJECT_PATH/evaluation/scripts/evaluation_dataset/`

> `sh evaluation_dataset_architecture_method.sh`

where:
  * <span style="color:#f5b041">**dataset**</span> must be exchanged by 
  one of the following options: <span style="color:#f5b041">**catsdogs**</span>, 
  <span style="color:#f5b041">**ilsvrc2012**</span>,
  <span style="color:#f5b041">**mit67**</span> or
  <span style="color:#f5b041">**mame**</span>. 
  * <span style="color: #48c9b0 ">**architecture**</span> must be exchanged by
  one of the following options: <span style="color: #48c9b0 ">**alexnet**</span>,
  <span style="color: #48c9b0 ">**vgg16**</span> or 
  <span style="color: #48c9b0 ">**resnet18**</span>.
  * And <span style="color: #c39bd3 ">**method**</span> mus be exchanged by 
  <span style="color: #c39bd3 ">**smoothgrad**</span>, 
  <span style="color: #c39bd3 ">**lrp**</span> or 
  <span style="color: #c39bd3 ">**gradcam**</span>.


For example, to get the _Focus_ of the Dogs vs. Cats dataset,
using the ResNet18 architecture and the GradCAM method,
run the following:

#### Step 1
> `cd $PROJECT_PATH/explainability/scripts/explainability_catsdogs/`

> `sh explainability_catsdogs_resnet18_gradcam.sh`

#### Step 2
> `cd $PROJECT_PATH/evaluation/scripts/evaluation_dataset/`

> `sh evaluation_catsdogs_resnet18_gradcam.sh`

## Cite
Please cite our paper when using this code. 
```
@misc{ariasduart2021explains,
      title={Who Explains the Explanation? Quantitatively Assessing Feature Attribution Methods}, 
      author={Anna Arias-Duart and Ferran Parés and Dario Garcia-Gasulla},
      year={2021},
      eprint={2109.15035},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```



## References
<a id="1">[1]</a>
Arias-Duart, A., Parés, F., & Garcia-Gasulla, D. (2021). Who Explains the Explanation? Quantitatively Assessing Feature Attribution Methods. arXiv preprint arXiv:2109.15035.

<a id="2">[2]</a>
Bach, S., Binder, A., Montavon, G., Klauschen, F., Müller, K. R., & Samek,
W. (2015). On pixel-wise explanations for non-linear classifier decisions
by layer-wise relevance propagation. PloS one, 10(7), e0130140.

<a id="3">[3]</a>
Parés, F., Arias-Duart, A., Garcia-Gasulla, D., Campo-Francés, G., Viladrich, N.,
Ayguadé, E., & Labarta, J. (2020). A Closer Look at Art Mediums: 
The MAMe Image Classification Dataset. arXiv preprint arXiv:2007.13693.

<a id="4">[4]</a>
Quattoni, A., & Torralba, A. (2009, June). Recognizing indoor scenes. 
In 2009 IEEE Conference on Computer Vision and Pattern Recognition (pp. 413-420). 
IEEE.

<a id="5">[5]</a>
Russakovsky, O., Deng, J., Su, H., Krause, J., Satheesh, S., Ma, S., ... & Fei-Fei,
L. (2015). Imagenet large scale visual recognition challenge. International journal
of computer vision, 115(3), 211-252.


<a id="6">[6]</a>
Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra,
D. (2017). Grad-cam: Visual explanations from deep networks via gradient-based
localization. In Proceedings of the IEEE international conference on computer
vision (pp. 618-626).

<a id="7">[7]</a>
Smilkov, D., Thorat, N., Kim, B., Viégas, F., & Wattenberg, M. (2017). 
Smoothgrad: removing noise by adding noise. arXiv preprint arXiv:1706.03825.

