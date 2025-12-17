
<span><a href="./README.md">📚English </a> | 📚中文阅读   | &nbsp; <a href="https://mp.weixin.qq.com/s/M6yEgQA-iprEPS4LRX5xBg">量子位</a> 
</span>

# <img src="./static/images/logo.png" style="width:auto; height:35px;">[AAAI 2026 Oral] HiMo-CLIP: Modeling Semantic Hierarchy and Monotonicity in Vision-Language Alignment

<div class="is-size-5 publication-authors" align="center">
  <span class="author-block">
    <a href="https://scholar.google.com/citations?user=tG7Y6OQAAAAJ&hl=zh-CN&oi=ao" target="_blank">吴芮葭</a><sup>1,2</sup><sup>†</sup>,&nbsp;
  </span>
  <span class="author-block">
    <a href="https://scholar.google.com/citations?hl=zh-CN&view_op=list_works&user=gpNOW2UAAAAJ" target="_blank">陈平</a><sup>1,2</sup><sup>†</sup>,&nbsp;
  </span>
  <span class="author-block">
    <a href="https://muzishen.github.io/" target="_blank">沈飞</a><sup>3</sup>,&nbsp;
  </span>
  <span class="author-block">
    赵绍安<sup>1,2</sup>,&nbsp;
  </span>
  <span class="author-block">
    <a href="https://github.com/kabutohui/" target="_blank">惠强</a><sup>1,2</sup>,&nbsp;
  </span>
  <span class="author-block">
    <a href="https://github.com/joelulu/" target="_blank">高焕霖</a><sup>1,2</sup>,&nbsp;
  </span>
  <span class="author-block">
    Ting Lu<sup>1,2</sup>,&nbsp;
  </span>
  <span class="author-block">
    <a href="https://scholar.google.com/citations?hl=en&user=L4OXOs0AAAAJ" target="_blank">刘兆祥</a><sup>1,2</sup>
  </span>
  <br>
  <span class="author-block">
    <a href="https://github.com/FangGet" target="_blank">赵放</a><sup>1,2</sup><sup>*</sup>,&nbsp;
  </span>
  <span class="author-block">
    <a href="https://scholar.google.com/citations?user=CFUQLCAAAAAJ&hl=en" target="_blank">王恺</a><sup>1,2</sup>,&nbsp;
  </span>
  <span class="author-block">
    <a href="https://scholar.google.com.hk/citations?user=kCC2oKwAAAAJ&hl=zh-CN&oi=ao" target="_blank">廉士国</a><sup>1,2</sup><sup>*</sup>
  </span>
</div>

<div class="is-size-5 publication-authors" align="center">
  <span class="author-block"><sup>1</sup>中国联通数据科学与人工智能研究院,&nbsp;</span><br>
  <span class="author-block"><sup>2</sup>联通数据智能有限公司</span>
  <span class="author-block"><sup>3</sup>新加坡国立大学</span>
</div>

<div class="is-size-5 publication-authors" align="center">
  († 共同一作. * 通讯作者.)
</div>

<h5 align="center">
<a href="https://unicomai.github.io/HiMo-CLIP/" target="_blank">
  <img src="https://img.shields.io/badge/Project-Website-blue.svg" alt="Project Page">
</a>
<a href="https://arxiv.org/abs/2511.06653" target="_blank">
  <img src="https://img.shields.io/badge/Paper-PDF-critical.svg?logo=adobeacrobatreader" alt="Paper">
</a>
</a>
<a href="https://huggingface.co/5RJ/HiMo-CLIP" target="_blank">
  <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue" alt="Hugging Face Model">
</a>
<a href="./LICENSE" target="_blank">
  <img src="https://img.shields.io/badge/License-Apache%202.0-yellow.svg" alt="License">
</a>
<a href="https://github.com/UnicomAI/HiMo-CLIP/stargazers" target="_blank">
  <img src="https://img.shields.io/github/stars/UnicomAI/HiMo-CLIP.svg?style=social" alt="GitHub Stars">
</a>
</h5>


## 动机

对比视觉-语言模型（如CLIP）在将图像和文本对齐到共享嵌入空间中取得了显著的成果。然而，这些模型通常将文本视为平面标记序列，忽视了语言的组合性和层级结构。这种简化限制了它们处理复杂和长文本描述的能力，而这些描述中包含多个语义层次。

特别是，目前的模型未能捕捉到语言的两个基本属性：
- **(1) 语义层级** — 文本意义的多层次组合结构，
- **(2) 语义单调性** — 更丰富或更完整的描述应当与视觉内容的对齐度更强。

这些限制促使了**HiMo-CLIP**的设计，它明确建模了视觉和语言表示之间的层级和单调关系，同时与标准的CLIP架构兼容。

![Motivation Figure](./static/images/motivation.png)

*(a) 图像的文本描述通常通过添加更多视觉细节逐步丰富语义层次，从短到长。 (b) 然而，现有模型，即使是针对长文本定制的模型，通常未能保持语义单调性，在扩展到更丰富的描述时忽视了这一关键原则。相比之下，HiMo-CLIP 在不同文本粒度之间保持一致的对齐，有效解决了这一被忽视的重要问题。 （注：FineLIP 的相似度超过1，因为其定制的测试时扩展。）*



## 摘要

对比视觉-语言模型如CLIP在图像-文本检索中取得了令人印象深刻的结果，它们通过将图像和文本表示对齐到共享嵌入空间。然而，这些模型通常将文本视为平面序列，限制了它们处理复杂、组合性强且长篇的描述的能力。特别是，它们未能捕捉到语言的两个重要特性：语义层级，反映了文本的多层次组合结构，以及语义单调性，其中更丰富的描述应导致与视觉内容更强的对齐。

为了克服这些局限性，我们提出了**HiMo-CLIP**，这是一种增强CLIP样式模型的表示层框架，不修改编码器架构。HiMo-CLIP引入了两个关键组件：一个层次分解（HiDe）模块，它通过批内PCA从长文本中提取潜在语义组件，从而使得跨不同语义粒度的对齐更加灵活，且对批次敏感；以及一个单调性感知对比损失（MoLo），它联合对齐图像和组件级别的表示，鼓励模型内化语义排序和对齐强度，作为文本完整性的函数。

这些组件共同作用，产生结构化的、认知对齐的跨模态表示。在多个图像-文本检索基准上的实验表明，HiMo-CLIP始终优于强大的基线模型，特别是在长文本或组合描述的情况下。

## 方法

为了解决上述限制，**HiMo-CLIP**引入了两个轻量级的表示层模块，这些模块可以无缝集成到CLIP样式框架中，而无需更改编码器：

*   **层次分解 (HiDe):**
    HiDe 对文本嵌入进行批内主成分分析（PCA），提取出最具判别性的潜在语义组件。这些组件会动态适应批次上下文，揭示每个文本样本的内在语义层次。通过将图像表示与全局和组件级别的嵌入对齐，HiDe实现了细粒度和多粒度的对齐。

*   **单调性感知对比损失 (MoLo):**
    MoLo 联合对齐图像与其完整文本嵌入及其分解的语义组件。该设计强化了语义单调性—确保随着文本变得更加完整或信息量更大，对齐强度也随之增强。该损失函数鼓励模型内化语义排序，从而产生结构化且认知对齐的视觉-语言表示。

这两个模块都仅在表示空间中操作，避免了架构修改和额外监督。它们共同使得HiMo-CLIP能够有效捕捉层次语义和单调对齐特性，在长文本和短文本的检索基准中实现了卓越的表现。

![Framework Figure](./static/images/framework.png)

---

## 性能

**HiMo-CLIP**在所有长文本基准测试中始终优于最先进的方法。在ViT-L/14主干网络下，我们的方法在Urban1k上达到了93.0%/93.1%（I2T/T2I），在Docci上达到了82.4%/84.4%（I2T/T2I），在Long-DCI上达到了62.2%/61.9%（I2T/T2I）。

![Evaluation Table 1](./static/images/eval_tb1.png)
![Evaluation Table 2](./static/images/eval_tb2.png)
![Evaluation Table 3](./static/images/eval_tb3.png)

图3展示了HiMo@5在HiMo-Docci上的趋势，其中HiMo-CLIP始终保持单调的相似度增长，而CLIP和Long-CLIP经常出现不规则下降，验证了我们核心假设，即更丰富的子文本应导致更强的对齐。图4和图5扩展了这个分析，提供了HiMo@2、@3、@4 和 @7 的具体例子，表明HiMo-CLIP即使在更深的层级下也能可靠地保持正确的分数顺序。例如，HiMo-CLIP在HiMo@4（0.93）和HiMo@7（0.97）上达到了最高的定性得分，而FineLIP和TULIP显示出得分逆转，Long-CLIP则产生负的皮尔逊相关系数（$-0.94$, $-0.95$）。在较浅的任务中，HiMo-CLIP在所有步骤中都保持了正确的顺序，而FineLIP和TULIP在HiMo@2和HiMo@3中出现了违背，甚至FG-CLIP在HiMo@3上也未能成功，尽管其在定量评分上表现优秀。这些结果突显了我们在表示层对齐中建模层次语义一致性的鲁棒性和可扩展性。

![Evaluation Figure 3](./static/images/eval_fig3.png)
![Evaluation Figure 4](./static/images/eval_fig4.png)
![Evaluation Figure 5](./static/images/eval_fig5.png)

## 许可证
该项目遵循Apache 2.0许可证进行发布。

## 📖 引用
如果您在研究或应用中发现**HiMo-CLIP**有用，请考虑给我们一个⭐并通过以下BibTeX进行引用：:

```bibtex
@misc{wu2025himoclipmodelingsemantichierarchy,
      title={HiMo-CLIP: Modeling Semantic Hierarchy and Monotonicity in Vision-Language Alignment}, 
      author={Ruijia Wu and Ping Chen and Fei Shen and Shaoan Zhao and Qiang Hui and Huanlin Gao and Ting Lu and Zhaoxiang Liu and Fang Zhao and Kai Wang and Shiguo Lian},
      year={2025},
      eprint={2511.06653},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.06653}, 
}
```
