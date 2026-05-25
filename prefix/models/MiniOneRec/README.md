---
license: apache-2.0
language:
- en
base_model:
- Qwen/Qwen2.5-1.5B-Instruct
tags:
- recommendation
- generative
- generative recommendation
- recommendation-system
- llm
- large-language-model
- recommender-system
---
<div align="center">

# 🌟 MiniOneRec · Generative Recommender Checkpoints


<img src="./assets/logo.png" width="500em" ></img> 

**An Open-Source Framework for
Scaling Generative Recommendation**

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/License-Apache--2.0-green.svg)
<a href="https://arxiv.org/abs/2510.24431"><img src="https://img.shields.io/static/v1?label=arXiv&message=Paper&color=red"></a>
</div>



**MiniOneRec** is the first fully open-source **generative recommendation** framework, which provides an end-to-end workflow spanning **SID construction**, **supervised fine-tuning (SFT)**, and recommendation-oriented **reinforcement learning (RL)**. 

These checkpoints accompany the paper:

> **MiniOneRec: An Open-Source Framework for Scaling Generative Recommendation**  
> <a href="https://arxiv.org/abs/2510.24431">📄 Technical Report</a> | <a href="https://github.com/AkaliKong/MiniOneRec">📦 Github</a>|<a href="https://modelscope.cn/models/k925238839/MiniOneRec">🤖  Modelscope</a>

---

## 🗺️ Table of Contents
1. Repository Contents  
2. Key Techniques  
3. Evaluation  
4. Acknowledgements  
5. Institutions  
6. Citation  

---

## 1️⃣ Repository Contents  <a name="repo"></a>

| File / Directory     | Description                                                                                                                    |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `Amazon/`            | **Pre-processed Amazon-2018 dataset** ─ contains both *Industrial* and *Office* splits                                         |
| `Industrial_ckpt/`   | **MiniOneRec full-training checkpoint** for the *Industrial* split (base model: Qwen2.5-1.5B-Instruct)                         |
| `Office_ckpt/`       | **MiniOneRec full-training checkpoint** for the *Office* split (base model: Qwen2.5-1.5B-Instruct)                             |


## 2️⃣ Key Techniques 
<div align="center">
<img src="./assets/minionerec_framework.png" width=100% ></img> 
</div>

- **SID Construction: MiniOneRec begins by transforming every product into a compact, semantically meaningful token.** It concatenates an item’s title and description, feeds this sentence through a frozen text encoder, and then quantises the resulting embedding with a three-level RQ-VAE.

- **SFT: With all items rewritten as SIDs, the model is first trained in a supervised fashion.** It views the chronologically ordered user history as a token sequence and learns, via next-token prediction, to generate the SID of the next product the user is likely to consume. Crucially, this stage is co-trained with a set of language-alignment objectives that map back and forth between natural language and SID space, allowing the recommender to inherit the world knowledge embedded in large language models while grounding that knowledge in discrete item codes.

- **Recommendation-Oriented RL: After SFT, MiniOneRec is further polished with a recommendation-oriented RL phase based on GRPO.** Multiple candidate recommendations are generated for each prompt, their rewards are normalised within the group to stabilise gradients, and a KL penalty keeps the updated policy close to its reference. Because the action space is a closed list of item SIDs, the system switches to constrained beam search, which guarantees that every beam is unique and valid, greatly improving sampling efficiency and diversity. The reward signal itself blends a binary correctness term with a rank-aware component that penalises high-probability yet incorrect items more heavily, and can be augmented with collaborative-filtering scores. Together, this pipeline enables MiniOneRec to couple dense linguistic knowledge, achieving a high-performance, lightweight generative recommendation system.

---

## 3️⃣ Evaluation

<div align="center">
<img src="./assets/minionerec_main_result.png" width=100% ></img> 
</div>

---

## 4️⃣ Acknowledgements

This repository reuses or adapts portions of code from the following open-source projects. We gratefully acknowledge their authors and contributors:

- [ReRe](https://github.com/sober-clever/ReRe)
- [LC-Rec](https://github.com/zhengbw0324/LC-Rec)

---

## 5️⃣ Institutions  <!-- omit in toc -->

This project is developed by the following institutions:

- <img src="assets/lds.png" width="28px"> [LDS](https://data-science.ustc.edu.cn/_upload/tpl/15/04/5380/template5380/index.html)
- <img src="assets/alphalab.jpg" width="28px"> [AlphaLab](https://alphalab-ustc.github.io/index.html)
- <img src="assets/next.jpg" width="28px"> [NExT](https://www.nextcenter.org/)

---

## 6️⃣ Citation

If you find our code/paper/model helpful, please consider citing our papers 📝 and staring us ⭐️！

```bib
@misc{MiniOneRec,
      title={MiniOneRec: An Open-Source Framework for Scaling Generative Recommendation}, 
      author={Xiaoyu Kong and Leheng Sheng and Junfei Tan and Yuxin Chen and Jiancan Wu and An Zhang and Xiang Wang and Xiangnan He},
      year={2025},
      eprint={2510.24431},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
}

@article{ReRe,
      title={Reinforced Preference Optimization for Recommendation}, 
      author={Junfei Tan and Yuxin Chen and An Zhang and Junguang Jiang and Bin Liu and Ziru Xu and Han Zhu and Jian Xu and Bo Zheng and Xiang Wang},
      journal={arXiv preprint arXiv:2510.12211},
      year={2025},
}

@inproceedings{RecZero,
      title={Think before Recommendation: Autonomous Reasoning-enhanced Recommender}, 
      author={Xiaoyu Kong and Junguang Jiang and Bin Liu and Ziru Xu and Han Zhu and Jian Xu and Bo Zheng and Jiancan Wu and Xiang Wang},
      year={2025},
      booktitle={NeurIPS},
}
```
