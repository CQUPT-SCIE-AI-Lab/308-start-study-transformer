# 308-start-study-transformer
####
Transformer 是一种深度学习模型架构，首次由 Vaswani 等人在 2017 年的论文《Attention is All You Need》中提出。该模型的核心创新是引入了 自注意力机制（Self-Attention），使得模型能够在处理序列数据时，不依赖于传统的递归神经网络（RNN）或卷积神经网络（CNN）。这一机制使得 Transformer 在处理长距离依赖关系时，比 RNN 和 LSTM 更加高效，并且能够并行计算，大大加快了训练速度。

Transformer 的提出标志着自然语言处理（NLP）领域的一个转折点。最初，Transformer 在机器翻译任务中取得了显著成功，超越了基于 RNN 的模型。随后，Transformer 的变种，如 BERT、GPT、T5 等，相继问世，并推动了 NLP 的研究和应用进入了一个新的时代。这些模型通过大规模预训练和精调（fine-tuning）的策略，能够在多种下游任务中表现出色，如文本生成、情感分析、问答系统等。

<figure>
    <img src="ModalNet-21.png" alt="The Transformer- model architecture.(来源：《Attention is All You Need》)">
    <figcaption>The Transformer- model architecture.(来源：《Attention is All You Need》)</figcaption>
</figure>

随着 Transformer 模型的成功，深度学习的应用领域逐渐扩展到图像处理、语音识别、推荐系统等多个领域。例如， Vision Transformer（ViT）将 Transformer 成功应用于计算机视觉任务，取得了与传统 CNN 模型相媲美的表现。

如今，Transformer 不仅是深度学习研究中的一个重要方向，也是实际应用中不可或缺的基础架构。无论是对自然语言的理解、生成，还是在图像、音频、视频等多模态任务中的广泛应用，Transformer 都在推动人工智能的边界不断拓展，其重要性愈加凸显。

#### 推荐学习资料
- 原论文：[《Attention is All You Need》](https://arxiv.org/abs/1706.03762)
- 原论文解读：[Transformer论文逐段精读【论文精读】](https://www.bilibili.com/video/BV1pu411o7BE/?spm_id_from=333.337.search-card.all.click&vd_source=a78c4419eac6aa8687109c6f6b59f976)
- 理论：[强烈推荐！台大李宏毅自注意力机制和Transformer详解！](https://www.bilibili.com/video/BV1v3411r78R/?spm_id_from=333.1007)
- 代码：
    - [【研1基本功 （真的很简单）注意力机制】手写多头注意力机制](https://www.bilibili.com/video/BV1o2421A7Dr/?spm_id_from=333.788)
    - [PyTorch Transformer Layers](https://pytorch.org/docs/stable/nn.html#transformer-layers)

当前目录的transformer.ipynb文件中有transformer的基本实现方法的代码(参考b站up主 [happy魇](https://space.bilibili.com/478929155) 进行优化)