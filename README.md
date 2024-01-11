# PolyNC - the Unified Polymer Design Workflow with Language

<p align="center">
  <img src="https://github.com/HKQiu/Unified_ML4Polymers/assets/73220956/c19349b0-332e-4c9d-aa4b-2ac6d10d1d99" width="50%">
</p>

This is the official repository of the paper entitled as [__PolyNC: a natural and chemical language model for unified polymer properties prediction__](https://pubs.rsc.org/en/Content/ArticleLanding/2023/SC/D3SC05079C) on ___Chemical Science___.
This work features a  a revolutionary model to enable rapid and precise prediction of Polymer properties via the power of Natural language and Chemical language (PolyNC). This work extends
the powerful natural language understanding capabilities of AI to the field of polymer research, marking an impressive step towards the development of expert models and human-level AI for
understanding polymer knowledge.

**All data and code** about model training and polyScreen software are in this repository.

[TMAP](https://github.com/HKQiu/Unified_ML4Polymers/tree/main/TMAP): The HTML files of Polymer Tree of each structure within the train and test dataset of $T_g$.

[data](https://github.com/HKQiu/Unified_ML4Polymers/tree/main/data): The train, test and validation datasets utilized in this work.

[notebooks](https://github.com/HKQiu/Unified_ML4Polymers/tree/main/notebooks): The jupyter notebooks (code history) for training, inference PolyNC and baseline models, and for attention analysis in **.ipynb** format.

[src](https://github.com/HKQiu/Unified_ML4Polymers/tree/main/src): The Python file and configuration file for t-SNE analysis.

By the way, our trained model is stored at this [repo](https://huggingface.co/hkqiu/PolyNC), where you can perform some simple demos with huggingface API.

# Q&A
__How to deploy and use?__
1. Access this [website](https://huggingface.co/hkqiu/PolyNC)🤗, and then you can perform some the demos with huggingface API, such as try this: _Predict the Tg of the following SMILES: c1cc(Oc2ccc(Oc3ccc(-n4c(=O)c5cc6c(c(=O)n()c6=O)cc5c4=O)cc3)cc2)ccc1_ __or__ _Predict the heat resistance class of the following SMILES: c1cc(Oc2ccc(Oc3ccc(-n4c(=O)c5cc6c(c(=O)n()c6=O)cc5c4=O)cc3)cc2)ccc1_. Just enjoy it!
2. Since the development of polymers is an issue of concern to polymer scientists, in order to facilitate everyone's use in a more custom manner (custom task, custom max length, and other parameters), we are developing a [Hugging Face playground](https://huggingface.co/spaces/hkqiu/AI4P) for PolyNC. Stay tuned!🤗

And any issue on this work, please fell free to email [hkqiu@ciac.ac.cn](hkqiu@ciac.ac.cn).


<p align="center">
  <img src="https://github.com/HKQiu/DataAugmentation4SmallData/assets/73220956/d7a243ed-6cd8-42e2-92c3-56a33f4d3c84" width="50%">
</p>


# Cite this:
```
@Article{D3SC05079C,
author ="Qiu, Haoke and Liu, Lunyang and Qiu, Xuepeng and Dai, Xuemin and Ji, Xiangling and Sun, Zhao-Yan",
title  ="PolyNC: a natural and chemical language model for the prediction of unified polymer properties",
journal  ="Chem. Sci.",
year  ="2024",
volume  ="15",
issue  ="2",
pages  ="534-544",
publisher  ="The Royal Society of Chemistry",
doi  ="10.1039/D3SC05079C",
url  ="http://dx.doi.org/10.1039/D3SC05079C"}
