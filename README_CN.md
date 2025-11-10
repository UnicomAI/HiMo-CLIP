<!-- Open Graph Meta Tags (GitHub README 不支持，已移除) -->

# HiMo-CLIP：建模视觉-语言对齐中的语义层次与单调性

✨ **AAAI 2026 口头报告** ✨

---

## 作者

*   **吴芮葭**<sup>1,2</sup><sup>∗</sup>
*   **陈平**<sup>1,2</sup><sup>∗</sup> ([Google Scholar](https://scholar.google.com/citations?hl=zh-CN&view_op=list_works&user=gpNOW2UAAAAJ))
*   **沈飞**<sup>3</sup> ([Google Scholar](https://scholar.google.com.hk/citations?user=wqvr28MAAAAJ&hl=zh-TW))
*   **赵绍安**<sup>1,2</sup>
*   **惠强**<sup>1,2</sup>
*   **高焕霖**<sup>1,2</sup> ([GitHub](https://github.com/joelulu))
*   **路婷**<sup>1,2</sup>
*   **刘兆祥**<sup>1,2</sup> ([Google Scholar](https://scholar.google.com/citations?hl=en&user=L4OXOs0AAAAJ))
*   **赵放**<sup>1,2</sup><sup>†</sup> ([GitHub](https://github.com/FangGet))
*   **王恺**<sup>1,2</sup> ([Google Scholar](https://scholar.google.com/citations?user=CFUQLCAAAAAJ&hl=en))
*   **廉士国**<sup>1,2</sup> ([Google Scholar](https://scholar.google.com.hk/citations?user=kCC2oKwAAAAJ&hl=zh-CN&oi=ao))

<sup>1</sup>中国联通数据科学与人工智能研究院,  
<sup>2</sup>中国联通数据智能公司,  
<sup>3</sup>新加坡国立大学

(* 共同第一作者。† 通讯作者。)

---

## 链接

[![代码](https://img.shields.io/badge/代码-GitHub-blue?logo=github)](https://github.com/UnicomAI/HiMo-CLIP)
[![论文](https://img.shields.io/badge/论文-PDF-red?logo=adobeacrobat)]()
[![模型](https://img.shields.io/badge/模型-Hugging%20Face-green?logo=huggingface)]()
[![数据集](https://img.shields.io/badge/数据集-Data-orange?logo=googlecloud)]()
[![项目主页](https://img.shields.io/badge/项目主页-Website-lightgrey?logo=firefox)](https://unicomai.github.io/HiMo-CLIP)
[![BibTeX](https://img.shields.io/badge/BibTeX-Cite-blue)](#bibtex)

---

## 动机

CLIP 等对比式视觉-语言模型在图像-文本检索任务中取得了显著成果，通过将图像和文本表示对齐到共享嵌入空间来实现。然而，这些模型通常将文本视为扁平的词元序列，忽略了语言固有的组成性和层次性结构。这种简化限制了它们处理复杂和长文本描述的能力，而这类描述往往包含多个语义层级。

具体而言，当前模型未能捕捉语言的两个基本属性：
- **(1) 语义层次性** —— 文本意义的多层次组成结构。
- **(2) 语义单调性** —— 更丰富或更完整的描述应与视觉内容产生更强的对齐关系。

这些局限性促使我们设计了 **HiMo-CLIP**，该框架明确地对视觉与语言表征之间的层次性和单调性关系进行建模，同时保持与标准 CLIP 架构的兼容性。

![动机图](./static/images/motivation.png)

*(a) 图像的文本描述通常会随着添加更多视觉细节而变得语义更丰富，从简短到冗长。(b) 然而，现有模型（即使是为长文本定制的模型）也常常无法保持语义单调性，在扩展到更丰富的描述时忽视了这一基本原则。相比之下，HiMo-CLIP 能够在不同文本粒度上保持对齐一致性，有效解决了这一被忽视但至关重要的挑战。（注：由于 FineLIP 使用了自定义的测试时缩放，其相似度可能超过1。）*

---

## 摘要

CLIP 等对比式视觉-语言模型通过在共享嵌入空间中对齐图像和文本表示，在图像-文本检索方面取得了令人印象深刻的成果。然而，这些模型通常将文本视为平面序列，限制了它们处理复杂、组合性和长文本描述的能力。特别是，它们未能捕捉语言的两个关键属性：反映文本多层次组成结构的“语义层次性”，以及随着描述更丰富而对齐强度应更强的“语义单调性”。

为解决这些局限性，我们提出了 HiMo-CLIP，这是一个增强 CLIP 风格模型的表征级框架，无需修改编码器架构。HiMo-CLIP 引入了两个关键组件：一个“层次分解”（HiDe）模块，通过对批内文本嵌入执行主成分分析（PCA）来提取潜在语义分量，从而实现跨不同语义粒度的灵活、批感知对齐；以及一个“单调性感知对比损失”（MoLo），它联合对齐全局和分量级表示，鼓励模型将语义排序和对齐强度作为文本完整性的函数来内化。

这两个组件协同工作，以生成结构化、认知对齐的跨模态表示。在多个图像-文本检索基准上的实验表明，HiMo-CLIP 在长文本或组合文本描述下始终优于强大的基线模型。

---

## 方法

为解决上述局限性，**HiMo-CLIP** 引入了两个轻量级的表征级模块，可无缝集成到 CLIP 风格框架中，且无需更改编码器：

*   **层次分解 (HiDe)**:
    HiDe 对文本嵌入执行批内主成分分析（PCA），以提取最具判别力的潜在语义分量。这些分量能动态适应批上下文，揭示每个文本样本的内在语义层次。通过将图像表示与全局和分量级嵌入对齐，HiDe 实现了细粒度和多粒度的对齐。

*   **单调性感知对比损失 (MoLo)**:
    MoLo 同时将图像与其完整文本嵌入及其分解出的语义分量对齐。此设计强制执行 *语义单调性* —— 确保随着文本变得更完整或更具信息量，对齐强度也随之增加。该损失函数鼓励模型内化语义排序，从而产生结构化且符合认知的视觉-语言表示。

这两个模块纯粹在表征空间中运行，避免了架构修改和额外监督。它们共同使 HiMo-CLIP 能够高效地捕捉层次语义和单调对齐特性，在长文本和短文本检索基准上均取得卓越性能。

![框架图](./static/images/framework.png)

---

## 性能

**HiMo-CLIP** 在所有长文本基准测试中始终优于最先进的方法。在 ViT-L/14 主干网络下，我们的方法在 Urban1k 上达到 93.0%/93.1% (I2T/T2I)，在 Docci 上达到 82.4%/84.4% (I2T/T2I)，在 Long-DCI 上达到 62.2%/61.9% (I2T/T2I)，显著超越最强基线（FineLIP）。

![评估表格1](./static/images/eval_tb1.png)
![评估表格2](./static/images/eval_tb2.png)
![评估表格3](./static/images/eval_tb3.png)

图3可视化了 HiMo-Docci 上的 HiMo@5 趋势，其中 HiMo-CLIP 始终保持单调相似度增长，而 CLIP 和 Long-CLIP 则经常出现波动下降，验证了我们的核心假设：更丰富的子文本应产生更强的对齐。图4和图5通过 HiMo@2、@3、@4 和 @7 的具体示例扩展了此分析，显示 HiMo-CLIP 即使在更深的层次结构下也能可靠地保持正确的分数排序。例如，HiMo-CLIP 在 HiMo@4 (0.93) 和 HiMo@7 (0.97) 上获得最高定性分数，而 FineLIP 和 TULIP 出现分数反转，Long-CLIP 甚至产生负皮尔逊相关性 ($-0.94$, $-0.95$)。在较浅的任务中，HiMo-CLIP 在所有步骤都保持正确排序，而 FineLIP 和 TULIP 在 HiMo@2 和 HiMo@3 时出现违规，即使 FG-CLIP 在定量得分上很强，也在 HiMo@3 时失败。这些结果凸显了我们表征级对齐在建模不同深度和内容下的层次语义一致性方面的鲁棒性和可扩展性。

![评估图3](./static/images/eval_fig3.png)
![评估图4](./static/images/eval_fig4.png)
![评估图5](./static/images/eval_fig5.png)

---

