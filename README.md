<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->


<!-- ABOUT THE PROJECT -->
## data-free prototype-based generative classifier with latent distillation

A repository for a **data-free prototype-based generative classifier with latent distillation for time series class-incremental learning**. This repository is built on top of the unified TSCIL (Time Series Class-Incremental Learning) framework.


## What's new
* more datasets are coming


## Requirements
![](https://img.shields.io/badge/python-3.10-green.svg)

![](https://img.shields.io/badge/pytorch-1.13.1-blue.svg)
![](https://img.shields.io/badge/ray-2.3.1-blue.svg)


## Dataset
### Available Datasets
1. [UCI-HAR](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
2. [UWAVE](http://www.timeseriesclassification.com/description.php?Dataset=UWaveGestureLibraryAll)
3. [Dailysports](https://archive.ics.uci.edu/ml/datasets/daily+and+sports+activities) 
4. [WISDM](https://archive.ics.uci.edu/dataset/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset)
5. [GrabMyo](https://physionet.org/content/grabmyo/1.0.2/)


## Continual Learning Algorithms
### Proposed Algorithm
* [GCPP] (data-free prototype-based generative classifier with latent distilation)

### Existing Algorithms
Regularization-based:
* [LwF](https://arxiv.org/abs/1606.09282)
* [EWC](https://arxiv.org/abs/1612.00796)
* [SI](https://arxiv.org/abs/1703.04200)
* [MAS](https://arxiv.org/abs/1711.09601)
* [DT2W](https://ieeexplore.ieee.org/abstract/document/10094960)

Replay-based:
* [ER](https://arxiv.org/abs/1811.11682)
* [DER](https://arxiv.org/abs/2004.07211)
* [Herding](https://arxiv.org/abs/1611.07725)
* [ASER](https://arxiv.org/abs/2009.00093)
* [CLOPS](https://www.nature.com/articles/s41467-021-24483-0)
* [FastICARL](https://arxiv.org/abs/2106.07268)
* [Generative Replay](https://arxiv.org/abs/1705.08690)
* [DeepInversion](https://arxiv.org/abs/1912.08795) (beta)
* [Mnemonics](https://arxiv.org/abs/2002.10211)


## Getting Started


## Acknowledgements
Our implementation uses the source code from the following repositories:
* TSCIL: [Class-incremental Learning for Time Series: Benchmark and Evaluation](https://github.com/zqiao11/TSCIL)
* CIL: [Class-incremental Learning with Generative Classifiers](https://github.com/GMvandeVen/class-incremental-learning)
* Framework & Buffer & LwF & ER & ASER: [Online Continual Learning in Image Classification: An Empirical Survey](https://github.com/RaptorMai/online-continual-learning)
* EWC & SI & MAS: [Avalanche: an End-to-End Library for Continual Learning](https://github.com/ContinualAI/avalanche)
* DER: [Mammoth - An Extendible (General) Continual Learning Framework for Pytorch](https://github.com/aimagelab/mammoth)
* DeepInversion: [Dreaming to Distill: Data-free Knowledge Transfer via DeepInversion](https://github.com/NVlabs/DeepInversion)
* Herding & Mnemonics: [Mnemonics Training: Multi-Class Incremental Learning without Forgetting](https://github.com/yaoyao-liu/class-incremental-learning)
* Soft-DTW: [Soft DTW for PyTorch in CUDA](https://github.com/Maghoumi/pytorch-softdtw-cuda)
* CNN: [AdaTime: A Benchmarking Suite for Domain Adaptation on Time Series Data](https://github.com/emadeldeen24/AdaTime)
* TST & lr scheduler: [PatchTST: A Time Series is Worth 64 Words: Long-term Forecasting with Transformers](https://github.com/yuqinie98/PatchTST)
* Generator: [TimeVAE for Synthetic Timeseries Data Generation](https://github.com/abudesai/timeVAE)


## Contact

<p align="right">(<a href="#readme-top">back to top</a>)</p>




<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/github_username/repo_name.svg?style=for-the-badge
[contributors-url]: https://github.com/github_username/repo_name/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/github_username/repo_name.svg?style=for-the-badge
[forks-url]: https://github.com/github_username/repo_name/network/members
[stars-shield]: https://img.shields.io/github/stars/github_username/repo_name.svg?style=for-the-badge
[stars-url]: https://github.com/github_username/repo_name/stargazers
[issues-shield]: https://img.shields.io/github/issues/github_username/repo_name.svg?style=for-the-badge
[issues-url]: https://github.com/github_username/repo_name/issues
[license-shield]: https://img.shields.io/github/license/github_username/repo_name.svg?style=for-the-badge
[license-url]: https://github.com/github_username/repo_name/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/linkedin_username
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 
