# Focus! Rating XAI Methods and Finding Biases with Mosaics

We propose in [[1]](#1) a consistent evaluation metric for feature attribution methods -- the _Focus_ -- 
designed to quantify their coherency to the task. This repository contains the mosaics and the code needed to replicate the
experiments in our paper: [Focus! Rating XAI Methods and Finding Biases with Mosaics](https://arxiv.org/abs/2109.15035). 

Six explainability methods have been evaluated:
* Smoothgrad [[10]](#10): the implementation used is based on the [work](https://github.com/wjNam/Relative_Attributing_Propagation) of Nam *et al.* 
* Layer-wise Relevance Propagation (LRP) [[2]](#2): the implementation used is based on the [work](https://github.com/kazuto1011/grad-cam-pytorch) of Nakashima *et al.*
* GradCAM [[9]](#9): the implementation used is based on the [work](https://github.com/jacobgil/pytorch-grad-cam)
of Gildenblat *et al.*
* LIME [[7]](#7): the implementation used is based on the [work](https://github.com/marcotcr/lime) of Tulio *et al.*
* GradCAM++ [[3]](#3): the implementation used is based on the [work](https://github.com/jacobgil/pytorch-grad-cam) of Gildenblat *et al.*
* Integrated Gradients (IG) [[11]](#11): the implementation used is based on the [work](https://github.com/pytorch/captum) of Kokhlikyan *et al.* [[4]](#4).



### Requirements

This code runs under Python 3.7.1. The python dependencies are defined in `requirements.txt`. 


## Available mosaics

We provide already created mosaics from four different datasets:

* [Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats/overview) mosaics can be downloaded [here](https://storage.hpai.bsc.es/focus-metric/catsdogs_mosaic.zip)
* ILSVRC 2012 [[8]](#8) mosaics can be downloaded [here](https://storage.hpai.bsc.es/focus-metric/ilsvrc2012_mosaic.zip).
* MIT67 [[6]](#6) mosaics can be downloaded [here](https://storage.hpai.bsc.es/focus-metric/mit67_mosaic.zip).
* MAMe [[5]](#5) mosaics can be downloaded [here](https://storage.hpai.bsc.es/focus-metric/mame_mosaic.zip).



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
  * **dataset** must be exchanged by 
  one of the following options: **catsdogs**, **ilsvrc2012**, **mit67** or **mame**. 
  * **architecture** must be exchanged by one of the following options: **alexnet**, **vgg16** or **resnet18**.
  * **method** must be exchanged by **smoothgrad**, **lrp**, **gradcam**, **lime**, **gradcampp** or **intgrad**.


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
@misc{ariasduart2021focus,
      title={Focus! Rating XAI Methods and Finding Biases with Mosaics}, 
      author={Anna Arias-Duart and Ferran Par??s and Dario Garcia-Gasulla},
      year={2021},
      eprint={2109.15035},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```



## References
<a id="1">[1]</a>
Arias-Duart, A., Par??s, F., & Garc??a-Gasulla, D. (2021). Focus! Rating XAI Methods and Finding Biases with Mosaics. arXiv preprint arXiv:2109.15035


<a id="2">[2]</a>
Bach, S., Binder, A., Montavon, G., Klauschen, F., M??ller, K. R., & Samek,
W. (2015). On pixel-wise explanations for non-linear classifier decisions
by layer-wise relevance propagation. PloS one, 10(7), e0130140.

<a id="3">[3]</a>
Chattopadhay, A., Sarkar, A., Howlader, P., & Balasubramanian, V. N. (2018, March). Grad-cam++: Generalized gradient-based visual explanations for deep convolutional networks. In 2018 IEEE winter conference on applications of computer vision (WACV) (pp. 839-847). IEEE.

<a id="4">[4]</a>
Kokhlikyan, N., Miglani, V., Martin, M., Wang, E., Alsallakh, B., Reynolds, J., ... & Reblitz-Richardson, O. (2020). Captum: A unified and generic model interpretability library for pytorch. arXiv preprint arXiv:2009.07896.


<a id="5">[5]</a>
Par??s, F., Arias-Duart, A., Garcia-Gasulla, D., Campo-Franc??s, G., Viladrich, N.,
Ayguad??, E., & Labarta, J. (2020). A Closer Look at Art Mediums: 
The MAMe Image Classification Dataset. arXiv preprint arXiv:2007.13693.

<a id="6">[6]</a>
Quattoni, A., & Torralba, A. (2009, June). Recognizing indoor scenes. 
In 2009 IEEE Conference on Computer Vision and Pattern Recognition (pp. 413-420). 
IEEE.

<a id="7">[7]</a>
Ribeiro, M. T., Singh, S., & Guestrin, C. (2016, August). " Why should i trust you?" Explaining the predictions of any classifier. In Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining (pp. 1135-1144).

<a id="8">[8]</a>
Russakovsky, O., Deng, J., Su, H., Krause, J., Satheesh, S., Ma, S., ... & Fei-Fei,
L. (2015). Imagenet large scale visual recognition challenge. International journal
of computer vision, 115(3), 211-252.


<a id="9">[9]</a>
Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra,
D. (2017). Grad-cam: Visual explanations from deep networks via gradient-based
localization. In Proceedings of the IEEE international conference on computer
vision (pp. 618-626).

<a id="10">[10]</a>
Smilkov, D., Thorat, N., Kim, B., Vi??gas, F., & Wattenberg, M. (2017). 
Smoothgrad: removing noise by adding noise. arXiv preprint arXiv:1706.03825.

<a id="11">[11]</a>
Sundararajan, M., Taly, A., & Yan, Q. (2017, July).
Axiomatic attribution for deep networks. In International Conference on Machine Learning (pp. 3319-3328). PMLR.

