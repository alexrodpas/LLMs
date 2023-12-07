<h1 align="center">A curated list of papers on Large Language Models (LLMs)</h1>

---

This list aims to help practitioners (including myself!) navigate the vast landscape of research papers on Large Language Models (LLMs). Notice that the various sub-lists are ordered in reverse chronological order (most recent papers first). The reason for this is that although this a relatively new research area (after all, the seminal paper on Transformer architecture, [**"Attention is all you need"**](https://arxiv.org/abs/1706.03762) by Vaswani et al., that we can consider as the starting point for the LLM revolution is as recent as just June 2017), its development pace is astonishing, so I think it's a better idea to list the most recent papers first.

---

Kudos to [RUCAIBox/LLMSurvey](https://github.com/RUCAIBox/LLMSurvey), an amazing GitHub repo with a list that I've used as starting point for this one.

**Table of contents**

- [Models, Corpora and Frameworks](#models-corpora-and-frameworks)
  - [Open-Source Models](#open-source-models)
  - [Closed-source Models](#closed-source-models)
  - [Commonly Used Corpora](#commonly-used-corpora)
    - [Sources](#sources)
  - [Libraries \& Frameworks](#libraries--frameworks)
- [Model Pre-training](#model-pre-training)
  - [Dataset Building](#dataset-building)
  - [Architectures](#architectures)
    - [Mainstream Architectures](#mainstream-architectures)
    - [Detailed Configuration](#detailed-configuration)
    - [Analysis](#analysis)
  - [Training Algorithms](#training-algorithms)
  - [Pre-training on Code](#pre-training-on-code)
    - [LLMs for Program Synthesis](#llms-for-program-synthesis)
    - [NLP Tasks Formatted as Code](#nlp-tasks-formatted-as-code)
- [Model Adaptation Tuning](#model-adaptation-tuning)
  - [Instruction Tuning](#instruction-tuning)
  - [Alignment Tuning](#alignment-tuning)
  - [Parameter-Efficient Model Adaptation](#parameter-efficient-model-adaptation)
  - [Memory-Efficient Model Adaptation](#memory-efficient-model-adaptation)
- [Model usage](#model-usage)
  - [In-Context Learning (ICL)](#in-context-learning-icl)
  - [Chain-of-Thought Reasoning (CoT)](#chain-of-thought-reasoning-cot)
  - [Planning for Complex Task Solving](#planning-for-complex-task-solving)
- [Model Evaluation](#model-evaluation)

# Comprehensive overviews and surveys

1. **"A Comprehensive Overview of Large Language Models"**. *Humza Naveed et al (2023)*. [[Paper](https://arxiv.org/abs/2307.06435)]

# Models, Corpora and Frameworks

## Open-Source Models

1. <u>Mistral-7B</u>: **"Mistral 7B"**. *Albert Q. Jiang et al.*. arXiv 2023. [[Paper](https://arxiv.org/abs/2310.06825)] [[Checkpoint](https://github.com/mistralai/mistral-src)]
1. <u>LLaMA</u>: **"LLaMA: Open and Efficient Foundation Language Models"**. *Hugo Touvron et al.* arXiv 2023. [[Paper](https://arxiv.org/abs/2302.13971v1)] [[Checkpoint](https://github.com/facebookresearch/llama)]
1. <u>Pythia</u>: **"Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling"**. *Stella Biderman et al.* . arXiv 2023. [[Paper](https://arxiv.org/abs/2304.01373)] [[Checkpoint](https://github.com/EleutherAI/pythia)]
1. <u>CodeGeeX</u>: **"CodeGeeX: A Pre-Trained Model for Code Generation with Multilingual Evaluations on HumanEval-X"**. *Qinkai Zheng et al.* . arXiv 2023. [[Paper](https://arxiv.org/abs/2303.17568)] [[Checkpoint](https://github.com/THUDM/CodeGeeX)]
1. <u>OPT-IML</u>: **"OPT-IML: Scaling Language Model Instruction Meta Learning through the Lens of Generalization"**. *Srinivasan et al.* . arXiv 2022. [[Paper](https://arxiv.org/abs/2212.12017)] [[Checkpoint](https://huggingface.co/facebook/opt-iml-30b)]
1. <u>Galactica</u>: **"Galactica: A Large Language Model for Science"**. *Ross Taylor et al.* arXiv 2022. [[Paper](https://arxiv.org/abs/2211.09085)] [[Checkpoint](https://huggingface.co/facebook/galactica-120b)]
1. <u>mT0 && BLOOMZ</u>: **"Crosslingual Generalization through Multitask Finetuning"**. *Niklas Muennighoff et al.* arXiv 2022. [[Paper](https://arxiv.org/abs/2211.01786)] [[Checkpoint](https://github.com/bigscience-workshop/xmtf)]
1. <u>Flan-T5</u>: **"Scaling Instruction-Finetuned Language Models"**. *Hyung Won Chung et al.* arXiv 2022. [[Paper](https://arxiv.org/abs/2210.11416)] [[Checkpoint](https://github.com/google-research/t5x/blob/main/docs/models.md#flan-t5-checkpoints)]
1. <u>GLM</u>: **"GLM-130B: An Open Bilingual Pre-trained Model"**. *Aohan Zeng et al.* arXiv 2022. [[Paper](https://arxiv.org/abs/2210.02414)] [[Checkpoint](https://github.com/THUDM/GLM-130B)]
1. <u>BLOOM</u>: **"BLOOM: A 176B-Parameter Open-Access Multilingual Language Model"**. *BigScience Workshop*. arXiv 2022. [[Paper](https://arxiv.org/abs/2211.05100)] [[Checkpoint](https://huggingface.co/bigscience/bloom)]
1. <u>NLLB</u>: **"No Language Left Behind: Scaling Human-Centered Machine Translation"**. *NLLB Team.* arXiv 2022. [[Paper](https://arxiv.org/abs/2207.04672)] [[Checkpoint](https://github.com/facebookresearch/fairseq/tree/nllb)]
1. <u>OPT</u>: **"OPT: Open Pre-trained Transformer Language Models"**. *Susan Zhang et al.* arXiv 2022. [[Paper](https://arxiv.org/abs/2205.01068)] [[Checkpoint](https://github.com/facebookresearch/metaseq/tree/main/projects/OPT)]
1. <u>UL2</u>: **"UL2: Unifying Language Learning Paradigms"**. *Yi Tay et al.* arXiv 2022. [[Paper](https://arxiv.org/abs/2205.05131)] [[Checkpoint](https://github.com/google-research/google-research/tree/master/ul2)]
1. <u>Tk-Instruct</u>: **"Super-NaturalInstructions: Generalization via Declarative Instructions on 1600+ NLP Tasks"**. *Yizhong Wang et al.* EMNLP 2022. [[Paper](https://arxiv.org/abs/2204.07705)] [[Checkpoint](https://huggingface.co/allenai/tk-instruct-11b-def-pos)]
1. <u>CodeGen</u>: **"CodeGen: An Open Large Language Model for Code with Multi-Turn Program Synthesis"**. *Erik Nijkamp et al.* arXiv 2022. [[Paper](https://arxiv.org/abs/2203.13474)] [[Checkpoint](https://huggingface.co/Salesforce/codegen-16B-nl)]
1. <u>GPT-NeoX-20B</u>: **"GPT-NeoX-20B: An Open-Source Autoregressive Language Model"**. *Sid Black et al.* arXiv 2022. [[Paper](https://arxiv.org/abs/2204.06745)] [[Checkpoint](https://huggingface.co/EleutherAI/gpt-neox-20b/tree/main)]
1. <u>T0</u>: **"Multitask Prompted Training Enables Zero-Shot Task Generalization"**. *Victor Sanh et al.* ICLR 2022. [[Paper](https://arxiv.org/abs/2110.08207)] [[Checkpoint](https://huggingface.co/bigscience/T0)]
1. <u>CPM-2</u>: **"CPM-2: Large-scale Cost-effective Pre-trained Language Models"**. *Zhengyan Zhang et al.* arXiv 2021. [[Paper](https://arxiv.org/abs/2106.10715)] [[Checkpoint](https://github.com/TsinghuaAI/CPM)]
1. <u>PanGu-α</u>: **"PanGu-α: Large-scale Autoregressive Pretrained Chinese Language Models with Auto-parallel Computation"**. *Wei Zeng et al.* arXiv 2021. [[Paper](https://arxiv.org/abs/2104.12369)] [[Checkpoint](https://openi.pcl.ac.cn/PCL-Platform.Intelligence/PanGu-Alpha)]
1. <u>mT5</u>: **"mT5: A massively multilingual pre-trained text-to-text transformer"**. *Linting Xue* et al. NAACL 2021. [[Paper](https://arxiv.org/abs/2010.11934)] [[Checkpoint](https://huggingface.co/google/mt5-xxl/tree/main)]
1. <u>T5</u>: **"Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"**. *Colin Raffel et al.* JMLR 2019. [[Paper](https://arxiv.org/abs/1910.10683)] [[Checkpoint](https://huggingface.co/t5-11b)]

## Closed-source Models

1. <u>PanGu-Σ</u>: **"PanGu-Σ: Towards Trillion Parameter Language Model with Sparse Heterogeneous Computing"**. *Xiaozhe Ren et al.* arXiv 2023. [[Paper](https://arxiv.org/abs/2303.10845)]
1. <u>GPT-4</u>: **"GPT-4 Technical Report"**. *OpenAI*. arXiv 2023. [[Paper](http://arxiv.org/abs/2303.08774v2)]
1. <u>Flan-PaLM && Flan-U-PaLM</u>: **"Scaling Instruction-Finetuned Language Models"**. *Hyung Won Chung et al.* arXiv. [[Paper](https://arxiv.org/abs/2210.11416)]
1. <u>U-PaLM</u>: **"Transcending Scaling Laws with 0.1% Extra Compute"**. *Yi Tay et al.* arXiv 2022. [[Paper](https://arxiv.org/abs/2210.11399)]
1. <u>WeLM</u>: **"WeLM: A Well-Read Pre-trained Language Model for Chinese"**. *Hui Su et al.* . arXiv 2022. [[Paper](https://arxiv.org/abs/2209.10372)]
1. <u>Sparrow</u>: **"Improving alignment of dialogue agents via targeted human judgements"**. *Amelia Glaese et al.* . arXiv 2022. [[Paper](http://arxiv.org/abs/2209.14375v1)]
1. <u>AlexaTM</u>: **"AlexaTM 20B: Few-Shot Learning Using a Large-Scale Multilingual Seq2Seq Model"**. *Saleh Soltan et al.* arXiv 2022. [[Paper](https://arxiv.org/abs/2208.01448)]
1. <u>PaLM</u>: **"PaLM: Scaling Language Modeling with Pathways"**. *Aakanksha Chowdhery et al.* arXiv 2022. [[Paper](https://arxiv.org/abs/2204.02311)]
1. <u>Chinchilla</u>: **"Training Compute-Optimal Large Language Models"**. *Jordan Hoffmann et al.* arXiv. [[Paper](https://arxiv.org/abs/2203.15556)]
1. <u>AlphaCode</u>: **"Competition-Level Code Generation with AlphaCode"**. *Yujia Li et al.* arXiv 2022. [[Paper](http://arxiv.org/abs/2203.07814v1)]
1. <u>InstructGPT</u>: **"Training language models to follow instructions with human feedback"**. *Long Ouyang et al.* arXiv 2022. [[Paper](http://arxiv.org/abs/2203.02155v1)]
1. <u>GLaM</u>: **"GLaM: Efficient Scaling of Language Models with Mixture-of-Experts"**. *Nan Du et al.* ICML 2022. [[Paper](https://arxiv.org/abs/2112.06905)]
1. <u>ERNIE 3.0 Titan</u>: **"ERNIE 3.0 Titan: Exploring Larger-scale Knowledge Enhanced Pre-training for Language Understanding and Generation"**.  *Shuohuan Wang et al.*arXiv 2021. [[Paper](https://arxiv.org/abs/2112.12731)]
1. <u>Gopher</u>: **"Scaling Language Models: Methods, Analysis & Insights from Training Gopher"**.  *Jack W. Rae et al.* arXiv 2021. [[Paper](http://arxiv.org/abs/2112.11446v2)]
1. <u>WebGPT</u>: **"WebGPT: Browser-assisted question-answering with human feedback"** . *Reiichiro Nakano et al.* arXiv 2021. [[Paper](https://arxiv.org/abs/2112.09332)]
1. <u>Anthropic</u>: **"A General Language Assistant as a Laboratory for Alignment"** . *Amanda Askell et al.* arXiv 2021. [[Paper](https://arxiv.org/abs/2112.00861)]
1. <u>Yuan 1.0</u>: **"Yuan 1.0: Large-Scale Pre-trained Language Model in Zero-Shot and Few-Shot Learning"**. *Shaohua Wu et al.* arXiv 2021. [[Paper](https://arxiv.org/abs/2110.04725)]
1. <u>MT-NLG</u>: **"Using DeepSpeed and Megatron to Train Megatron-Turing NLG 530B, A Large-Scale Generative Language Model"**. *Shaden Smith et al.* arXiv 2021. [[Paper](https://arxiv.org/abs/2201.11990)]
1. <u>FLAN</u>: **"Finetuned Language Models Are Zero-Shot Learners"**. *Jason Wei et al.* ICLR 2021. [[Paper](https://arxiv.org/abs/2109.01652)]
1. <u>Jurassic-1</u>: **"Jurassic-1: Technical details and evaluation"**. *Opher Lieber et al.* 2021. [[Paper](https://assets.website-files.com/60fd4503684b466578c0d307/61138924626a6981ee09caf6_jurassic_tech_paper.pdf)]
1. <u>ERNIE 3.0</u>: **"ERNIE 3.0: Large-scale Knowledge Enhanced Pre-training for Language Understanding and Generation"**. *Yu Sun et al.* arXiv 2021. [[Paper](https://arxiv.org/abs/2107.02137)]
1. <u>CodeX</u>: **"Evaluating Large Language Models Trained on Code"**. *Mark Chen et al.* arXiv 2021. [[Paper](https://arxiv.org/abs/2107.03374)]
1. <u>HyperCLOVA</u>: **"What Changes Can Large-scale Language Models Bring? Intensive Study on HyperCLOVA: Billions-scale Korean Generative Pretrained Transformers"**. *Boseop Kim et al.* EMNLP 2021. [[Paper](https://arxiv.org/abs/2109.04650)]
1. <u>LaMDA</u>: **"LaMDA: Language Models for Dialog Applications"**. *Romal Thoppilan et al.* arXiv 2021. [[Paper](https://arxiv.org/abs/2201.08239)]
1. <u>GShard</u>: **"GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding"**. *Dmitry Lepikhin et al.* ICLR 2021. [[Paper](http://arxiv.org/abs/2006.16668v1)]
1. <u>GPT-3</u>: **"Language Models are Few-Shot Learners"**. *Tom B. Brown et al.* NeurIPS 2020. [[Paper](https://arxiv.org/abs/2005.14165)]

## Commonly Used Corpora

1. <u>ROOTS</u>: **"The BigScience ROOTS Corpus: A 1.6TB Composite Multilingual Dataset"**. *Laurençon et al*. NeurIPS 2022 Datasets and Benchmarks Track. [[paper](https://arxiv.org/abs/2303.03915)]
1. <u>The Pile</u>: **"The Pile: An 800GB Dataset of Diverse Text for Language Modeling"**. *Leo Gao et al*. arxiv 2021. [[Paper](http://arxiv.org/abs/2101.00027v1)] [[Source](https://pile.eleuther.ai/)]
1. <u>Pushshift.io</u>: **"The Pushshift Reddit Dataset"**. *Jason Baumgartner et al*. AAAI 2020. [[Paper](http://arxiv.org/abs/2001.08435v1)] [[Source](https://files.pushshift.io/reddit/)]
1. <u>OpenWebText</u>: [[Source](https://skylion007.github.io/OpenWebTextCorpus/)]
1. <u>C4</u>: **"Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"**. *Colin Raffel et al.* JMLR 2019. [[Paper](http://arxiv.org/abs/1910.10683v3)] [[Source](https://www.tensorflow.org/datasets/catalog/c4)]
1. <u>REALNEWs</u>: **"Defending Against Neural Fake News"**. *Rowan Zellers et al.* NeurIPS 2019. [[Paper](http://arxiv.org/abs/1905.12616v3)] [[Source](https://github.com/rowanz/grover/tree/master/realnews)]
1. <u>CC-NEWS</u>: **"RoBERTa: A Robustly Optimized BERT Pretraining Approach"**. *Yinhan Liu et al.* arXiv 2019. [[Paper](http://arxiv.org/abs/1907.11692v1)] [[Source](https://huggingface.co/datasets/cc_news)]
1. <u>CC-stories-R</u>: **"A Simple Method for Commonsense Reasoning"**. *Trieu H. Trinh el al.* arXiv 2018. [[Paper](http://arxiv.org/abs/1806.02847v2)] [[Source](https://huggingface.co/datasets/spacemanidol/cc-stories)]
1. <u>BookCorpus</u>: **"Aligning Books and Movies: Towards Story-like Visual Explanations by Watching Movies and Reading Books"**. *Yukun Zhu et al.*  ICCV 2015. [[Paper](http://arxiv.org/abs/1506.06724v1)] [[Source](https://huggingface.co/datasets/bookcorpus)]

### Sources

1. <u>Google Cloud BigQuery</u>:  [[Source](https://cloud.google.com/bigquery/public-data?hl=en)]
1. <u>Wikipedia</u>: [[Source](https://dumps.wikimedia.org/)]
1. <u>CommonCrawl</u>: [[Source](https://commoncrawl.org/)]
1. <u>Guntenburg</u>: [[Source](https://www.gutenberg.org/)]

## Libraries & Frameworks

1. <u>MindSpore</u>: **"Huawei MindSpore AI Development Framework"** . *Huawei Technologies Co., Ltd.* Artificial Intelligence Technology 2022. [[Paper](https://link.springer.com/chapter/10.1007/978-981-19-2879-6_5)] [[Source](https://github.com/mindspore-ai/mindspore)]
1. <u>OneFlow</u>: **"OneFlow: Redesign the Distributed Deep Learning Framework from Scratch"** . *Jinhui Yuan et al.* arXiv 2021. [[Paper](https://arxiv.org/abs/2110.15032)] [[Source](https://github.com/Oneflow-Inc/oneflow)]
1. <u>FastMoE</u>: **"FastMoE: A Fast Mixture-of-Expert Training System"**.  *Jiaao He et al.* arXiv 2021. [[Paper](https://arxiv.org/abs/2103.13262)] [[Source](https://github.com/laekov/fastmoe)]
1. <u>BMTrain</u>: [[Source](https://github.com/OpenBMB/BMTrain)]
1. <u>Colossal-AI</u>: **"Colossal-AI: A Unified Deep Learning System For Large-Scale Parallel Training"**. *Zhengda Bian et al.* arXiv 2021. [[Paper](http://arxiv.org/abs/2110.14883v2)] [[Source](https://github.com/hpcaitech/ColossalAI)]
1. <u>DeepSpeed</u>: **"Deepspeed: System optimizations enable training deep learning models with over 100 billion parameters"**. *Rasley et al.* KDD 2020. [[Paper](https://dl.acm.org/doi/10.1145/3394486.3406703)] [[Source](https://github.com/microsoft/DeepSpeed)]
1. <u>JAX</u>:  [[Source](https://github.com/google/jax)]
1. <u>Transformers</u>: **"Transformers: State-of-the-Art Natural Language Processing"**. *Thomas Wolf et al.* EMNLP 2020. [[Paper](https://arxiv.org/abs/1910.03771)] [[Source](https://huggingface.co/)]
1. <u>Megatron-LM</u>: **"Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism"**. *Mohammad Shoeybi et al.* arXiv 2019. [[Paper](https://arxiv.org/abs/1909.08053)] [[Source](https://github.com/NVIDIA/Megatron-LM)]
1. <u>Pytorch</u>: **"PyTorch: An Imperative Style, High-Performance Deep Learning Library"**. *Adam Paszke el al.* NeurIPS 2019. [[Paper](https://arxiv.org/abs/1912.01703)] [[Source](https://pytorch.org/)]
1. <u>PaddlePaddle</u>: **"PaddlePaddle: An Open-Source Deep Learning Platform from Industrial Practice"** . *Yanjun Ma et al.* Frontiers of Data and Domputing 2019.  [[Paper](http://www.jfdc.cnic.cn/EN/abstract/abstract2.shtml)] [[Source](https://github.com/PaddlePaddle/Paddle)]
1. <u>TensorFlow</u>: **"TensorFlow: A system for large-scale machine learning"**. *Martín Abadi et al.* OSDI 2016. [[Paper](https://arxiv.org/abs/1605.08695)] [[Source](https://www.tensorflow.org/)]
1. <u>MXNet</u>: **"MXNet: A Flexible and Efficient Machine Learning Library for Heterogeneous Distributed Systems"**. *Tianqi Chen et al.* arXiv 2015. [[Paper](https://arxiv.org/abs/1512.01274)] [[Source](https://github.com/apache/mxnet)]

# Model Pre-training

## Dataset Building

1. **"A Pretrainer's Guide to Training Data: Measuring the Effects of Data Age, Domain Coverage, Quality, & Toxicity"**. *Shayne Longpre et al*. arXiv 2023. [[paper](https://arxiv.org/abs/2305.13169)]
1. **"The BigScience ROOTS Corpus: A 1.6TB Composite Multilingual Dataset"**. *Laurençon et al*. NeurIPS 2022 Datasets and Benchmarks Track. [[paper](https://arxiv.org/abs/2303.03915)]
1. **"Deduplicating Training Data Makes Language Models Better"**. *Katherine Lee et al*. ACL 2022. [[paper](https://arxiv.org/abs/2107.06499)]
1. **"Deduplicating Training Data Mitigates Privacy Risks in Language Models"**. *Nikhil Kandpal et al*. ICML 2022. [[paper](https://arxiv.org/abs/2202.06539)]
1. **"Scaling Laws and Interpretability of Learning from Repeated Data"**. *Danny Hernandez et al*. arXiv 2022. [[paper](https://arxiv.org/abs/2205.10487)]

## Architectures

### Mainstream Architectures

**Causal Decoder**

1. **"OPT: Open Pre-trained Transformer Language Models"**. *Susan Zhang et al*. arXiv 2022. [[paper](http://arxiv.org/abs/2205.01068)]
1. **"BLOOM: A 176B-Parameter Open-Access Multilingual Language Model"**. *Teven Le Scao et al*. arXiv 2022. [[paper](http://arxiv.org/abs/2211.05100)]
1. **"Training Compute-Optimal Large Language Models"**. *Jordan Hoffmann et al*. arXiv 2022. [[paper](http://arxiv.org/abs/2203.15556)]
1. **"Galactica: A Large Language Model for Science"**. *Ross Taylor et al*. arXiv 2022. [[paper](http://arxiv.org/abs/2211.09085)]
1. **"PaLM: Scaling Language Modeling with Pathways"**. *Aakanksha Chowdhery et al*. arXiv 2022. [[paper](http://arxiv.org/abs/2204.02311)]
1. **"Jurassic-1: Technical Details and Evaluation"**. *Opher Lieber et al*. 2022. AI21 Labs. [[paper](https://uploads-ssl.webflow.com/60fd4503684b466578c0d307/61138924626a6981ee09caf6_jurassic_tech_paper.pdf)]
1. **"LaMDA: Language Models for Dialog Applications"**. *Romal Thoppilan et al*. arXiv 2022. [[paper](http://arxiv.org/abs/2201.08239)]
1. **"Scaling Language Models: Methods, Analysis & Insights from Training Gopher"**. *Jack W. Rae et al*. arXiv 2021. [[paper](http://arxiv.org/abs/2112.11446)]
1. **"Language Models are Few-Shot Learners"**. *Tom B. Brown et al*. NeurIPS 2020. [[paper](http://arxiv.org/abs/2005.14165)]

**Prefix Decoder**

1. **"GLM-130B: An Open Bilingual Pre-trained Model"**. *Aohan Zeng et al*. arXiv 2022. [[paper](http://arxiv.org/abs/2210.02414)]
2. **"GLM: General Language Model Pretraining with Autoregressive Blank Infilling"**. *Zhengxiao Du et al*. ACL 2022. [[paper](http://arxiv.org/abs/2103.10360)]
3. **"Transcending Scaling Laws with 0.1% Extra Compute"**. *Yi Tay et al*. arXiv 2022. [[paper](http://arxiv.org/abs/2210.11399)]

**MoE (Mixture of Experts)**

1. **"Unified Scaling Laws for Routed Language Models"**. *Aidan Clark et al*. ICML 2022. [[paper](http://arxiv.org/abs/2202.01169)]
2. **"Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity"**. *William Fedus et al*. JMLR 2021. [[paper](http://arxiv.org/abs/2101.03961)]

**SSM (Space State Models)**

1. **"Hungry Hungry Hippos: Towards Language Modeling with State Space Models"**. *Daniel Y. Fu et al*. ICLR 2023. [[paper](https://arxiv.org/abs/2212.14052)]
2. **"Pretraining Without Attention"**. *Junxiong Wang et al*. arXiv 2022. [[paper](http://arxiv.org/abs/2212.10544)]
3. **"Efficiently Modeling Long Sequences with Structured State Spaces"**. *Albert Gu et al*. ICLR 2022. [[paper](http://arxiv.org/abs/2111.00396)]
4. **"Long Range Language Modeling via Gated State Spaces"**. *Harsh Mehta et al*. arXiv 2022. [[paper](http://arxiv.org/abs/2206.13947)]

### Detailed Configuration

**Layer Normalization**

1. <u>DeepNorm</u>: **"DeepNet: Scaling Transformers to 1,000 Layers"**. *Hongyu Wang et al*. arXiv 2022. [[paper](http://arxiv.org/abs/2203.00555)]
1. <u>Sandwich-LN</u>: **"CogView: Mastering Text-to-Image Generation via Transformers"**. *Ming Ding et al*. NeirIPS 2021. [[paper](https://arxiv.org/abs/2105.13290)]
1. <u>RMSNorm</u>: **"Root Mean Square Layer Normalization"**. *Biao Zhang et al*. NeurIPS 2019. [[paper](http://arxiv.org/abs/1910.07467)]

**Position Encoding**

1. <u>xPos</u>: **"A Length-Extrapolatable Transformer"**. *Yutao Sun et al*. arXiv 2022. [[paper](https://arxiv.org/abs/2212.10554)]
1. <u>ALiBi</u>: **"Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation"**. *Ofir Press et al*. ICLR 2022. [[paper](http://arxiv.org/abs/2108.12409)]
1. <u>RoPE</u>: **"RoFormer: Enhanced Transformer with Rotary Position Embedding"**. *Jianlin Su et al*. arXiv 2021. [[paper](http://arxiv.org/abs/2104.09864)]
1. <u>T5 bias</u>: **"Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"**. *Colin Raffel et al.* JMLR 2019. [[paper](https://arxiv.org/abs/1910.10683)]

**Attention**

1. <u>PagedAttention</u>: **"vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention"**. *Woosuk Kwon et al*.  2023.  paper(Stay Tuned) [[Offical WebSite](https://vllm.ai/)]
1. <u>FlashAttention</u>: **"FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"**. *Tri Dao et al*. NeurIPS 2022. [[paper](https://arxiv.org/abs/2205.14135)]
1. <u>Multi-query attention</u>: **"Fast Transformer Decoding: One Write-Head is All You Need"**. *Noam Shazeer*. arXiv 2019. [[paper](https://arxiv.org/abs/1911.02150)]

### Analysis

1. **"What Language Model Architecture and Pretraining Objective Work Best for Zero-Shot Generalization?"**. *Thomas Wang et al*. ICML 2022. [[paper](http://arxiv.org/abs/2204.05832)]
1. **"What Language Model to Train if You Have One Million GPU Hours?"**. *Teven Le Scao et al*. Findings of EMNLP 2022. [[paper](http://arxiv.org/abs/2210.15424)]
1. **"Examining Scaling and Transfer of Language Model Architectures for Machine Translation"**. *Biao Zhang et al*. ICML 2022. [[paper](http://arxiv.org/abs/2202.00528)]
1. **"Scaling Laws vs Model Architectures: How does Inductive Bias Influence Scaling?"**. *Yi Tay et al*. arXiv 2022. [[paper](http://arxiv.org/abs/2207.10551)]
1. **"Do Transformer Modifications Transfer Across Implementations and Applications?"**. *Sharan Narang et al*. EMNLP 2021. [[paper](http://arxiv.org/abs/2102.11972)]

## Training Algorithms

1. **"Tesseract: Parallelize the Tensor Parallelism Efficiently"**. *Boxiang Wang et al*. ICPP 2022. [[paper](http://arxiv.org/abs/2105.14500)]
1. **"An Efficient 2D Method for Training Super-Large Deep Learning Models"**. *Qifan Xu et al*. arXiv 2021. [[paper](http://arxiv.org/abs/2104.05343)]
1. **"Maximizing Parallelism in Distributed Training for Huge Neural Networks"**. *Zhengda Bian et al*. arXiv 2021. [[paper](http://arxiv.org/abs/2105.14450)]
1. **"ZeRO-Offload: Democratizing Billion-Scale Model Training"**. *Jie Ren et al*. USENIX 2021. [[paper](http://arxiv.org/abs/2101.06840)]
1. **"ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"**. *Samyam Rajbhandari et al*. SC 2020. [[paper](http://arxiv.org/abs/1910.02054)]
1. **"GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism"**. *Yanping Huang et al*. NeurIPS 2019. [[paper](http://arxiv.org/abs/1811.06965)]
1. **"Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism"**. *Mohammad Shoeybi et al*. arXiv 2019. [[paper](http://arxiv.org/abs/1909.08053)]
1. **"PipeDream: Fast and Efficient Pipeline Parallel DNN Training"**. *Aaron Harlap et al*. arXiv 2018. [[paper](http://arxiv.org/abs/1806.03377)]

## Pre-training on Code

### LLMs for Program Synthesis

1. **"Competition-Level Code Generation with AlphaCode"**. *Yujia Li et al*. Science. [[paper](http://arxiv.org/abs/2203.07814)]
1. **"CodeGen: An Open Large Language Model for Code with Multi-Turn Program Synthesis"**. *Erik Nijkamp et al*. ICLR 2023. [[paper](http://arxiv.org/abs/2203.13474)]
1. **"InCoder: A Generative Model for Code Infilling and Synthesis"**. *Daniel Fried et al*. ICLR 2023. [[paper](http://arxiv.org/abs/2204.05999)]
1. **"CodeT: Code Generation with Generated Tests"**. *Bei Chen et al*. ICLR 2023. [[paper](http://arxiv.org/abs/2207.10397)]
1. **"StarCoder: may the source be with you!"**. *Raymond Li et al*. arXiv 2023. [[paper](https://arxiv.org/abs/2305.06161)]
1. **"A Systematic Evaluation of Large Language Models of Code"**. *Frank F. Xu et al*. arXiv 2022. [[paper](http://arxiv.org/abs/2202.13169)]
1. **"Evaluating Large Language Models Trained on Code"**. *Mark Chen et al*. arXiv 2021. [[paper](http://arxiv.org/abs/2107.03374)]
1. **"Program Synthesis with Large Language Models"**. *Jacob Austin et al*. arXiv 2021. [[paper](http://arxiv.org/abs/2108.07732)]
1. **"Show Your Work: Scratchpads for Intermediate Computation with Language Models"**. *Maxwell Nye et al*. arXiv 2021. [[paper](http://arxiv.org/abs/2112.00114)]

### NLP Tasks Formatted as Code

1. **"Language Models of Code are Few-Shot Commonsense Learners"**. *Aman Madaan et al*. EMNLP 2022. [[paper](http://arxiv.org/abs/2210.07128)]
1. **"Autoformalization with Large Language Models"**. *Yuhuai Wu et al*. NeurIPS 2022. [[paper](http://arxiv.org/abs/2205.12615)]

# Model Adaptation Tuning

## Instruction Tuning

1. **"LIMA: Less Is More for Alignment"**. *Chunting Zhou*. arXiv 2023. [[Paper](https://arxiv.org/abs/2305.11206)]
1. **"Maybe Only 0.5% Data is Needed: A Preliminary Exploration of Low Training Data Instruction Tuning"**. *Hao Chen et al*. arXiv 2023. [[Paper](https://arxiv.org/abs/2305.09246)]
1. **"Is Prompt All You Need No. A Comprehensive and Broader View of Instruction Learning"**. *Renze Lou et al*. arXiv 2023. [[Paper](https://arxiv.org/abs/2303.10475)]
1. **"The Flan Collection: Designing Data and Methods for Effective Instruction Tuning"**. *Shayne Longpre et al*. arXiv 2023. [[Paper](https://arxiv.org/abs/2301.13688)] [[Homepage](https://github.com/google-research/FLAN)]
1. **"OPT-IML: Scaling Language Model Instruction Meta Learning through the Lens of Generalization"**. *Srinivasan Iyer et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2212.12017)] [[Checkpoint](https://github.com/facebookresearch/metaseq/tree/main/projects/OPT-IML)]
1. **"Self-Instruct: Aligning Language Model with Self Generated Instructions"**. *Yizhong Wang et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2212.10560)] [[Homepage](https://github.com/yizhongw/self-instruct)]
1. **"Unnatural Instructions: Tuning Language Models with (Almost) No Human Labor"**. *Or Honovich et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2212.09689)] [[Homepage](https://github.com/orhonovich/unnatural-instructions)]
1. **"Scaling Instruction-Finetuned Language Models"**. *Hyung Won Chung et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2210.11416)] [[Homepage](https://github.com/google-research/FLAN)]
1. **"Crosslingual Generalization through Multitask Finetuning"**. *Niklas Muennighoff et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2211.01786)] [[Collection](https://github.com/bigscience-workshop/xmtf#data)] [[Checkpoint](https://github.com/bigscience-workshop/xmtf#models)]
1. **"MVP: Multi-task Supervised Pre-training for Natural Language Generation"**. *Tianyi Tang et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2206.12131)] [[Collection](https://huggingface.co/RUCAIBox)] [[Checkpoint](https://huggingface.co/RUCAIBox)]
1. **"Super-NaturalInstructions: Generalization via Declarative Instructions on 1600+ NLP Tasks"**. *Yizhong Wang et al*. EMNLP 2022. [[Paper](https://arxiv.org/abs/2204.07705)] [[Collection](https://instructions.apps.allenai.org/#data)] [[Checkpoint](https://huggingface.co/models?search=tk-instruct-)]
1. **"Training language models to follow instructions with human feedback"**. *Long Ouyang et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2203.02155)]
1. **"PromptSource: An Integrated Development Environment and Repository for Natural Language Prompts"**. *Stephen H. Bach et al*. ACL 2022. [[Paper](https://arxiv.org/abs/2202.01279)] [[Collection](https://github.com/bigscience-workshop/promptsource)]
1. **"Multitask Prompted Training Enables Zero-Shot Task Generalization"**. *Victor Sanh et al*. ICLR 2022. [[Paper](https://arxiv.org/abs/2110.08207)] [[Checkpoint](https://huggingface.co/bigscience/T0#how-to-use)]
1. **"Finetuned Language Models Are Zero-Shot Learners"**. *Jason Wei et al*. ICLR 2022. [[Paper](https://arxiv.org/abs/2109.01652)] [[Homepage](https://github.com/google-research/FLAN)]
1. **"Cross-Task Generalization via Natural Language Crowdsourcing Instructions"**. *Swaroop Mishra et al*. ACL 2022. [[Paper](https://arxiv.org/abs/2104.08773)] [[Collection](https://instructions.apps.allenai.org/#data)]
1. **"Muppet: Massive Multi-task Representations with Pre-Finetuning"**. *Armen Aghajanyan et al*. EMNLP 2021. [[Paper](https://arxiv.org/abs/2101.11038)] [[Checkpoint](https://huggingface.co/models?other=arxiv:2101.11038)]
1. **"Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"**. *Colin Raffel et al*. JMLR 2020. [[Paper](https://arxiv.org/abs/1910.10683)] [[Checkpoint](https://github.com/google-research/text-to-text-transfer-transformer#released-model-checkpoints)]
1. **"Multi-Task Deep Neural Networks for Natural Language Understanding"**. *Xiaodong Liu et al*. ACL 2019. [[Paper](https://arxiv.org/abs/1901.11504)] [[Homepage](https://github.com/namisan/mt-dnn)]

## Alignment Tuning

1. **"Red-Teaming Large Language Models using Chain of Utterances for Safety-Alignment"**. *Rishabh Bhardwaj et al*. arXiv 2023. [[Paper](https://arxiv.org/abs/2308.09662)]
1. **"RAFT: Reward rAnked FineTuning for Generative Foundation Model Alignment"**. *Hanze Dong et al*. arXiv 2023. [[Paper](https://arxiv.org/abs/2304.06767)]
1. **"The Wisdom of Hindsight Makes Language Models Better Instruction Followers"**. *Tianjun Zhang et al*. arXiv 2023. [[Paper](https://arxiv.org/abs/2302.05206)]
1. **"Scaling Laws for Reward Model Overoptimization"**. *Leo Gao et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2210.10760)]
1. **"Is Reinforcement Learning (Not) for Natural Language Processing: Benchmarks, Baselines, and Building Blocks for Natural Language Policy Optimization"**. *Rajkumar Ramamurthy et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2210.01241)]
1. **"Improving alignment of dialogue agents via targeted human judgements"**. *Amelia Glaese et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2209.14375)]
1. **"Red Teaming Language Models to Reduce Harms: Methods, Scaling Behaviors, and Lessons Learned"**. *Deep Ganguli et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2209.07858)]
1. **"Dynamic Planning in Open-Ended Dialogue using Reinforcement Learning"**. *Deborah Cohen et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2208.02294)]
1. **"Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback"**. *Yuntao Bai et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2204.05862)]
1. **"Teaching language models to support answers with verified quotes"**. *Jacob Menick et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2203.11147)]
1. **"Training language models to follow instructions with human feedback"**. *Long Ouyang et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2203.02155)]
1. **"WebGPT: Browser-assisted question-answering with human feedback"**. *Reiichiro Nakano et al*. arXiv 2021. [[Paper](https://arxiv.org/abs/2112.09332)]
1. **"A General Language Assistant as a Laboratory for Alignment"**. *Amanda Askell et al*. arXiv 2021. [[Paper](https://arxiv.org/abs/2112.00861)]
1. **"Recursively Summarizing Books with Human Feedback"**. *Jeff Wu et al*. arXiv 2021. [[Paper](https://arxiv.org/abs/2109.10862)]
1. **"Alignment of Language Agents"**. *Zachary Kenton et al*. arXiv 2021. [[Paper](https://arxiv.org/abs/2103.14659)]
1. **"Learning to summarize from human feedback"**. *Nisan Stiennon et al*. NeurIPS 2020. [[Paper](https://arxiv.org/abs/2009.01325)]
1. **"Fine-Tuning Language Models from Human Preferences"**. *Daniel M. Ziegler et al*. arXiv 2019. [[Paper](https://arxiv.org/abs/1909.08593)]
1. **"Deep TAMER: Interactive Agent Shaping in High-Dimensional State Spaces"**. *Garrett Warnell et al*. AAAI 2018. [[Paper](https://arxiv.org/abs/1709.10163)]
1. **"Deep Reinforcement Learning from Human Preferences"**. *Paul Christiano et al*. NIPS 2017. [[Paper](https://arxiv.org/abs/1706.03741)]
1. **"Interactive Learning from Policy-Dependent Human Feedback"**. *James MacGlashan et al*. ICML 2017. [[Paper](https://arxiv.org/abs/1701.06049)]
1. **"TAMER: Training an Agent Manually via Evaluative Reinforcement"**. *W. Bradley Knox et al*. ICDL 2008. [[Paper](https://www.cs.utexas.edu/~bradknox/papers/icdl08-knox.pdf)]

## Parameter-Efficient Model Adaptation

1. **"LLM-Adapters: An Adapter Family for Parameter-Efficient Fine-Tuning of Large Language Models"**. *Zhiqiang Hu et al*. arXiv 2023. [[Paper](https://arxiv.org/abs/2304.01933)] [[GitHub](https://github.com/AGI-Edgerunners/LLM-Adapters)]
1. **"LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init Attention"**. *Renrui Zhang et al*. arXiv 2023. [[Paper](https://arxiv.org/abs/2303.16199)] [[GitHub](https://github.com/OpenGVLab/LLaMA-Adapter)]
1. **"Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning"**. *Qingru Zhang et al*. arXiv 2023. [[Paper](https://arxiv.org/abs/2303.10512)] [[GitHub](https://github.com/QingruZhang/AdaLoRA)]
1. **"Parameter-efficient fine-tuning of large-scale pre-trained language models"**. *Ning Ding et al*. Nat Mach Intell. [[Paper](https://www.nature.com/articles/s42256-023-00626-4)] [[GitHub](https://github.com/thunlp/OpenDelta)]
1. **"DyLoRA: Parameter-Efficient Tuning of Pre-trained Models using Dynamic Search-Free Low-Rank Adaptation"**. *Mojtaba Valipour et al*. EACL 2023. [[Paper](https://arxiv.org/abs/2210.07558)] [[GitHub](https://github.com/huawei-noah/KD-NLP/tree/main/DyLoRA)]
1. **"P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks"**. *Xiao Liu et al*. ACL 2022. [[Paper](https://arxiv.org/abs/2110.07602)] [[GitHub](https://github.com/THUDM/P-tuning-v2)]
1. **"Towards a Unified View of Parameter-Efficient Transfer Learning"**. *Junxian He et al*. ICLR 2022. [[Paper](https://arxiv.org/abs/2110.04366)] [[GitHub](https://github.com/jxhe/unify-parameter-efficient-tuning)]
1. **"LoRA: Low-Rank Adaptation of Large Language Models"**. *Edward J. Hu et al*. arXiv 2021. [[Paper](https://arxiv.org/abs/2106.09685)] [[GitHub](https://github.com/microsoft/LoRA)]
1. **"The Power of Scale for Parameter-Efficient Prompt Tuning"**. *Brian Lester et al*. EMNLP 2021. [[Paper](https://arxiv.org/pdf/2104.08691)]
1. **"GPT Understands, Too"**. *Xiao Liu et al*. arXiv 2021. [[Paper](https://arxiv.org/abs/2103.10385)] [[GitHub](https://github.com/THUDM/P-tuning)]
1. **"Prefix-Tuning: Optimizing Continuous Prompts for Generation"**. *Xiang Lisa Li et al*. ACL 2021. [[Paper](https://arxiv.org/abs/2101.00190)] [[GitHub](https://github.com/XiangLi1999/PrefixTuning)]
1. **"AUTOPROMPT: Eliciting Knowledge from Language Models with Automatically Generated Prompts"**. *Taylor Shin et al*. EMNLP 2020. [[Paper](https://arxiv.org/abs/2010.15980)] [[GitHub](https://ucinlp.github.io/autoprompt/)]
1. **"MAD-X: An Adapter-Based Framework for Multi-Task Cross-Lingual Transfer"**. *Jonas Pfeiffer et al*. EMNLP 2020. [[Paper](https://arxiv.org/abs/2005.00052)] [[GitHub](https://github.com/Adapter-Hub/adapter-transformers)]
1. **"Parameter-Efficient Transfer Learning for NLP"**. *Neil Houlsby et al*. ICML 2019. [[Paper](https://arxiv.org/abs/1902.00751)] [[GitHub](https://github.com/google-research/adapter-bert)]

## Memory-Efficient Model Adaptation

1. **"AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration"**. *Ji Lin et al*. arXiv 2023. [[Paper](https://arxiv.org/abs/2306.00978)] [[GitHub](https://github.com/mit-han-lab/llm-awq)]
1. **"LLM-QAT: Data-Free Quantization Aware Training for Large Language Models"**. *Zechun Liu et al*. arXiv 2023. [[Paper](https://arxiv.org/abs/2305.17888)]
1. **"QLoRA: Efficient Finetuning of Quantized LLMs"**. *Tim Dettmers et al*. arXiv 2023. [[Paper](https://arxiv.org/abs/2305.14314)] [[GitHub](https://github.com/artidoro/qlora)]
1. **"ZeroQuant-V2: Exploring Post-training Quantization in LLMs from Comprehensive Study to Low Rank Compensation"**. *Zhewei Yao et al*. arXiv 2023. [[Paper](https://arxiv.org/abs/2303.08302)]
1. **"The case for 4-bit precision: k-bit Inference Scaling Laws"**. *Tim Dettmers et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2212.09720)]
1. **"SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models"**. *Guangxuan Xiao et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2211.10438)] [[GitHub](https://github.com/mit-han-lab/smoothquant)]
1. **"GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"**. *Elias Frantar et al*. ICLR 2023. [[Paper](https://arxiv.org/abs/2210.17323)] [[GitHub](https://github.com/IST-DASLab/gptq)]
1. **"LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale"**. *Tim Dettmers et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2208.07339)] [[GitHub](https://github.com/TimDettmers/bitsandbytes)]
1. **"ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers"**. *Zhewei Yao et al*. NeurIPS 2022. [[Paper](https://arxiv.org/abs/2206.01861)] [[GitHub](https://github.com/microsoft/DeepSpeed)]
1. **"Compression of Generative Pre-trained Language Models via Quantization"**. *Chaofan Tao et al*. ACL 2022. [[Paper](https://arxiv.org/abs/2203.10705)]
1. **"8-bit Optimizers via Block-wise Quantization"**. *Tim Dettmers et al*. arXiv 2021. [[Paper](https://arxiv.org/abs/2110.02861)]
1. **"A Survey of Quantization Methods for Efficient Neural Network Inference"**. *Amir Gholami et al*. arXiv 2021. [[Paper](https://arxiv.org/abs/2103.13630)]

# Model usage

## In-Context Learning (ICL)

1. **Symbol tuning improves in-context learning in language models**. *Jerry Wei*. arXiv 2023. [[Paper](https://arxiv.org/abs/2305.08298)]
1. **Meta-in-context learning in large language models**. *Julian Coda-Forno*. arXiv 2023. [[Paper](https://arxiv.org/abs/2305.12907)]
1. **Larger language models do in-context learning differently**. *Jerry Wei*. arXiv 2023. [[Paper](https://arxiv.org/abs/2303.03846)]
1. **The Learnability of In-Context Learning**. *Noam Wies et al*. arXiv 2023. [[Paper](https://arxiv.org/abs/2303.07895)]
1. **What In-Context Learning "Learns" In-Context: Disentangling Task Recognition and Task Learning**. *Jane Pan et al*. arXiv 2023. [[Paper](https://arxiv.org/abs/2305.09731)]
1. **"A Survey for In-context Learning"**. *Qingxiu Dong et al*. arXiv 2023. [[Paper](https://arxiv.org/abs/2301.00234)]
1. **Do Prompt-Based Models Really Understand the Meaning of Their Prompts?** *Albert Webson et al*. NAACL 2022. [[Paper](https://aclanthology.org/2022.naacl-main.167/)]
1. **"What learning algorithm is in-context learning? investigations with linear models"**. *Ekin Aky{\"{u}}rek et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2211.15661)]
1. **"Transformers learn in-context by gradient descent"**. *Johannes von Oswald et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2212.07677)]
1. **"Transformers as algorithms: Generalization and implicit model selection in in-context learning"**. *Yingcong Li et al*. arXiv 2023. [[Paper](https://arxiv.org/abs/2301.07067)]
1. **"Rethinking the Role of Scale for In-Context Learning: An Interpretability-based Case Study at 66 Billion Scale"**. *Hritik Bansal et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2212.09095)]
1. **"Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?"**. *Sewon Min et al*. EMNLP 2022. [[Paper](https://arxiv.org/abs/2202.12837)]
1. **"On the Effect of Pretraining Corpora on In-context Learning by a Large-scale Language Model"**. *Seongjin Shin et al*. NAACL 2022. [[Paper](https://arxiv.org/abs/2204.13509)]
1. **"In-context Learning and Induction Heads"**. *Catherine Olsson et al*. arXiv 2022. [[Paper](http://arxiv.org/abs/2209.11895)]
1. **"Data distributional properties drive emergent in-context learning in transformers"**. *Stephanie C. Y. Chan et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2205.05055)]
1. **"An Explanation of In-context Learning as Implicit Bayesian Inference"**. S*ang Michael Xie et al*. ICLR 2022. [[Paper](https://arxiv.org/abs/2111.02080)]
1. **"Prompt-Augmented Linear Probing: Scaling Beyond the Limit of Few-shot In-Context Learner"**. *Hyunsoo Cho et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2212.10873)]
1. **"Cross-Task Generalization via Natural Language Crowdsourcing Instructions"**. *Swaroop Mishra et al*. ACL 2022. [[Paper](https://arxiv.org/abs/2104.08773)]
1. **"The Unreliability of Explanations in Few-shot Prompting for Textual Reasoning"**. *Ye, Xi et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2205.03401)]
1. **"Structured Prompting: Scaling In-Context Learning to 1,000 Examples"**. *Hao, Yaru et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2212.06713)]
1. **"Fantastically Ordered Prompts and Where to Find Them: Overcoming Few-Shot Prompt Order Sensitivity"**. *Yao Lu et al*. ACL 2022. [[Paper](https://arxiv.org/abs/2104.08786)]
1. **"Self-adaptive In-context Learning"**. *Zhiyong Wu et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2212.10375)]
1. **"Active Example Selection for In-Context Learning"**. *Yiming Zhang et al*. EMNLP 2022. [[Paper](https://arxiv.org/abs/2211.04486)]
1. **"Demystifying Prompts in Language Models via Perplexity Estimation"**. *Hila Gonen et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2212.04037)]
1. **"Diverse demonstrations improve in-context compositional generalization"**. *Itay Levy et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2212.06800)]
1. **"Learning to retrieve prompts for in-context learning"**. *Ohad Rubin et al*. NAACL 2022. [[Paper](https://arxiv.org/abs/2112.08633)]
1. **"What Makes Good In-Context Examples for GPT-3?"**. *Jiachang Liu et al*. ACL 2022. [[Paper](https://arxiv.org/abs/2101.06804)]
1. **"An Information-theoretic Approach to Prompt Engineering Without Ground Truth Labels"**. *Taylor Sorensen et al*. ACL 2022. [[Paper](https://arxiv.org/abs/2203.11364)]
1. **"Calibrate Before Use: Improving Few-Shot Performance of Language Models"**. *Zihao Zhao et al*. ICML 2021. [[Paper](https://arxiv.org/abs/2102.09690)]

## Chain-of-Thought Reasoning (CoT)

1. **"Multimodal Chain-of-Thought Reasoning in Language Models"**. *Zhuosheng Zhang et al*. arXiv 2023. [[Paper](https://arxiv.org/abs/2302.00923)]
1. **"Automatic Chain of Thought Prompting in Large Language Models"**. *Zhuosheng Zhang et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2210.03493)]
1. **"Chain of Thought Prompting Elicits Reasoning in Large Language Models"**. *Jason Wei et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2201.11903)]
1. **"STaR: Self-Taught Reasoner Bootstrapping Reasoning With Reasoning"**. *Zelikman et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2203.14465)]
1. **"Large language models are zero-shot reasoners"**. *Takeshi Kojima et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2205.11916)]
1. **"Automatic Chain of Thought Prompting in Large Language Models"**. *Zhuosheng Zhang et al*. arXiv. [[Paper](http://arxiv.org/abs/2210.03493)]
1. **"Complexity-Based Prompting for Multi-Step Reasoning"**. *Yao Fu et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2210.00720)]
1. **"Language Models are Multilingual Chain-of-Thought Reasoners"**. *Freda Shi et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2210.03057)]
1. **"Rationale-Augmented Ensembles in Language Models"**. *Xuezhi Wang et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2207.00747)]
1. **"Least-to-Most Prompting Enables Complex Reasoning in Large Language Models"**. *Denny Zhou et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2205.10625)]
1. **"Self-Consistency Improves Chain of Thought Reasoning in Language Models"**. *Xuezhi Wang et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2203.11171)]
1. **"Large Language Models Can Self-Improve"**. *Jiaxin Huang et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2210.11610)]
1. **"Training Verifiers to Solve Math Word Problems"**. *Karl Cobbe et al*. arXiv 2021. [[Paper](https://arxiv.org/abs/2110.14168)]
1. **"On the Advance of Making Language Models Better Reasoners"**. *Yifei Li et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2206.02336)]
1. **"Large Language Models are reasoners with Self-Verification"**. *Yixuan Weng et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2212.09561)]
1. **"Teaching small language models to reason"**. *Lucie Charlotte Magister et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2212.08410)]
1. **"Large language models are reasoning teachers"**. *Namgyu Ho et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2212.10071)]
1. **"The Unreliability of Explanations in Few-shot Prompting for Textual Reasoning"**. *Ye, Xi et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2205.03401)]
1. **"Scaling Instruction-Finetuned Language Models"**. *Hyung Won Chung et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2210.11416)]
1. **"Solving Quantitative Reasoning Problems with Language Models"**. *Aitor Lewkowycz et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2206.14858)]
1. **"Text and patterns: For effective chain of thought, it takes two to tango"**. *Aman Madaan et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2209.07686)]
1. **"Challenging BIG-Bench tasks and whether chain-of-thought can solve them"**. *Mirac Suzgun et al*. arXiv 2022. [[Paper](http://arxiv.org/abs/2210.09261)]
1. **"Reasoning with Language Model Prompting: A Survey"**. *Shuofei Qiao et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2212.09597)]
1. **"Towards Reasoning in Large Language Models: A Survey"**. *Jie Huang et al*. arXiv 2022. [[Paper](https://arxiv.org/abs/2212.10403)]

## Planning for Complex Task Solving

1. **Least-to-Most Prompting Enables Complex Reasoning in Large Language Models**. *Denny Zhou et al*. ICLR 2023. [[Paper](https://openreview.net/forum?id=WZH7099tgfM)]
1. **PAL: Program-aided Language Models**. *Luyu Gao et al*. ICML 2023. [[Paper](https://openreview.net/forum?id=M1fd9Z00sj)]
1. **Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models**. *Lei Wang et al*. ACL 2023. [[Paper](https://arxiv.org/abs/2305.04091)]
1. **Tree of Thoughts: Deliberate Problem Solving with Large Language Models**. *Shunyu Yao et al*. arXiv 2023. [[Paper](https://arxiv.org/abs/2305.10601)]
1. **Voyager: An Open-Ended Embodied Agent with Large Language Models**. *Guanzhi Wang et al*. arXiv 2023. [[Paper](https://arxiv.org/abs/2305.16291)]
1. **Reflexion: Language Agents with Verbal Reinforcement Learning**. *Noah Shinn et al*. arXiv 2023. [[Paper](https://arxiv.org/abs/2303.11366)]
1. **Multimodal Procedural Planning via Dual Text-Image Prompting**. *Yujie Lu et al*. arXiv 2023. [[Paper](https://arxiv.org/abs/2305.01795)]
1. **Self-planning Code Generation with Large Language Model**. *Xue Jiang et al*. arXiv 2023. [[Paper](https://arxiv.org/abs/2303.06689)]
1. **Decomposed Prompting: A Modular Approach for Solving Complex Tasks**. *Tushar Khot et al*. ICLR 2023 [[Paper](https://openreview.net/forum?id=_nGgzQjzaRy)]
1. **Toolformer: Language Models Can Teach Themselves to Use Tools**. *Timo Schick et al*. arXiv 2023. [[Paper](https://arxiv.org/abs/2302.04761)]
1. **HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face**. *Yongliang Shen et al*. arXiv 2023. [[Paper](https://arxiv.org/abs/2303.17580)]
1. **Faithful Chain-of-Thought Reasoning**. *Qing Lyu et al*. arXiv 2023. [[Paper](https://arxiv.org/abs/2301.13379)]
1. **LLM+P: Empowering Large Language Models with Optimal Planning Proficiency**. *Bo Liu et al*. arXiv 2023. [[Paper](https://arxiv.org/abs/2304.11477)]
1. **Reasoning with Language Model is Planning with World Model**. *Shibo Hao et al*. arXiv 2023. [[Paper](https://arxiv.org/abs/2305.14992)]
1. **Generative Agents: Interactive Simulacra of Human Behavior**. *Joon Sung Park et al*. arXiv 2023. [[Paper](https://arxiv.org/abs/2304.03442)]
1. **ReAct: Synergizing Reasoning and Acting in Language Models**. *Shunyu Yao et al*. ICLR 2023. [[Paper](https://openreview.net/forum?id=WE_vluYUL-X)]
1. **ChatCoT: Tool-Augmented Chain-of-Thought Reasoning on Chat-based Large Language Models**. *Zhipeng Chen et al*. arXiv 2023. [[Paper](https://arxiv.org/abs/2305.14323)]
1. **Describe, Explain, Plan and Select: Interactive Planning with Large Language Models Enables Open-World Multi-Task Agents**. *Zihao Wang et al*. arXiv 2023. [[Paper](https://arxiv.org/abs/2302.01560)]
1. **AdaPlanner: Adaptive Planning from Feedback with Language Models**. *Haotian Sun et al*. arXiv 2023. [[Paper](https://arxiv.org/abs/2305.16653)]
1. **ProgPrompt: Generating Situated Robot Task Plans using Large Language Models**. *Ishika Singh et al*. ICRA 2022. [[Paper](https://arxiv.org/abs/2209.11302)]

# Model Evaluation

1. **"Language Models Are Greedy Reasoners: A Systematic Formal Analysis of Chain-of-Thought"**. *Abulhair Saparov et al.* ICLR 2023. [[Paper](http://arxiv.org/abs/2210.01240v4)]
1. **"The End of Programming"**. *Matt Welsh et al.* ACM 2023. [[Paper](https://cacm.acm.org/magazines/2023/1/267976-the-end-of-programming/fulltext)]
1. **"Chatgpt goes to law school"**. *Choi Jonathan H et al.* SSRN 2023. [[Paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4335905)]
1. **"How Close is ChatGPT to Human Experts? Comparison Corpus, Evaluation, and Detection"**. *Biyang Guo et al.* arXiv 2023. [[Paper](https://arxiv.org/abs/2301.07597v1)]
1. **"Is ChatGPT A Good Translator? A Preliminary Study"**. *Wenxiang Jiao et al.* arXiv 2023. [[Paper](https://arxiv.org/abs/2301.08745v3)]
1. **"Could an Artificial-Intelligence agent pass an introductory physics course?"**. *Gerd Kortemeyer et al.* arXiv 2023. [[Paper](https://arxiv.org/abs/2301.12127v2)]
1. **"Mathematical Capabilities of ChatGPT"**. *Simon Frieder et al.* arXiv 2023. [[Paper](https://arxiv.org/abs/2301.13867v1)]
1. **"Synthetic Prompting: Generating Chain-of-Thought Demonstrations for Large Language Models"**. *Zhihong Shao et al.* arXiv 2023. [[Paper](http://arxiv.org/abs/2302.00618v1)]
1. **"Grounding Large Language Models in Interactive Environments with Online Reinforcement Learning"**. *Thomas Carta et al.* arXiv 2023. [[Paper](https://arxiv.org/abs/2302.02662v1)]
1. **"Evaluating ChatGPT as an Adjunct for Radiologic Decision-Making"**. *Arya Yao et al.* medRxiv 2023. [[Paper](https://www.medrxiv.org/content/10.1101/2023.02.02.23285399v1)]
1. **"Theory of Mind May Have Spontaneously Emerged in Large Language Models"**. *Michal Kosinski et al.* arXiv 2023. [[Paper](http://arxiv.org/abs/2302.02083v3)]
1. **"A Categorical Archive of ChatGPT Failures"**. *Ali Borji et al.* arXiv 2023. [[Paper](https://arxiv.org/abs/2302.03494v7)]
1. **"A Multitask, Multilingual, Multimodal Evaluation of ChatGPT on Reasoning, Hallucination, and Interactivity"**. *Yejin Bang et al.* arXiv 2023. [[Paper](http://arxiv.org/abs/2302.04023v2)]
1. **"Toolformer: Language Models Can Teach Themselves to Use Tools"**. *Timo Schick et al.* arXiv 2023. [[Paper](http://arxiv.org/abs/2302.04761v1)]
1. **"Is ChatGPT a General-Purpose Natural Language Processing Task Solver?"**. *Chengwei Qin et al.* arXiv 2023. [[Paper](http://arxiv.org/abs/2302.06476v2)]
1. **"How Good Are GPT Models at Machine Translation? A Comprehensive Evaluation"**. *Hendy Amr et al.* arXiv 2023. [[Paper](http://arxiv.org/abs/2302.09210)]
1. **"Can ChatGPT Understand Too? A Comparative Study on ChatGPT and Fine-tuned BERT"**. *Qihuang Zhong et al.* arXiv 2023. [[Paper](https://arxiv.org/abs/2302.10198v2)]
1. **"Zero-Shot Information Extraction via Chatting with ChatGPT"**. *Xiang Wei et al.* arXiv 2023. [[Paper](https://arxiv.org/abs/2302.10205v1)]
1. **"ChatGPT: Jack of all trades, master of none"**. *Jan Kocon et al.* arXiv 2023. [[Paper](http://arxiv.org/abs/2302.10724v1)]
1. **"On the Robustness of ChatGPT: An Adversarial and Out-of-distribution Perspective"**. *Jindong Wang et al.* arXiv 2023. [[Paper](https://arxiv.org/abs/2302.12095v4)]
1. **"Check Your Facts and Try Again: Improving Large Language Models with External Knowledge and Automated Feedback"**. *Baolin Peng et al.* arXiv 2023. [[Paper](http://arxiv.org/abs/2302.12813v3)]
1. **"An Independent Evaluation of ChatGPT on Mathematical Word Problems (MWP)"**. *Paulo Shakarian et al.* arXiv 2023. [[Paper](https://arxiv.org/abs/2302.13814v2)]
1. **"How Robust is GPT-3.5 to Predecessors? A Comprehensive Study on Language Understanding Tasks"**. *Chen Xuanting et al.* arXiv 2023. [[Paper](http://arxiv.org/abs/2303.00293v1)]
1. **"The utility of ChatGPT for cancer treatment information"**. *Shen Chen et al.* medRxiv 2023. [[Paper](https://www.medrxiv.org/content/10.1101/2023.03.16.23287316v1)]
1. **"Can ChatGPT Assess Human Personalities? A General Evaluation Framework"**. *Haocong Rao et al.* arXiv 2023. [[Paper](https://arxiv.org/abs/2303.01248v2)]
1. **"Will Affective Computing Emerge from Foundation Models and General AI? A First Evaluation on ChatGPT."**. *Mostafa M. Amin et al.* arXiv 2023. [[Paper](https://arxiv.org/abs/2303.03186v1)]
1. **"Exploring the Feasibility of ChatGPT for Event Extraction."**. *Jun Gao et al.* arXiv 2023. [[Paper](https://arxiv.org/abs/2303.03836v2)]
1. **"Does Synthetic Data Generation of LLMs Help Clinical Text Mining?"**. *Tang Ruixiang et al.* arXiv 2023. [[Paper](http://arxiv.org/abs/2303.04360v1)]
1. **"Consistency Analysis of ChatGPT"**. *Myeongjun Jang et al.* arXiv 2023. [[Paper](http://arxiv.org/abs/2303.06273v1)]
1. **"Self-planning Code Generation with Large Language Model"**. *Shun Zhang et al.* ICLR 2023. [[Paper](http://arxiv.org/abs/2303.06689v1)]
1. **"Evaluation of ChatGPT as a Question Answering System for Answering Complex Questions"**. *Yiming Tan et al.* arXiv 2023. [[Paper](https://arxiv.org/abs/2303.07992)]
1. **"GPT-4 Technical Report"**. *OpenAI et al.* OpenAI 2023. [[Paper](http://arxiv.org/abs/2303.08774v3)]
1. **"A Short Survey of Viewing Large Language Models in Legal Aspect"**. *Zhongxiang Sun et al.* arXiv 2023. [[Paper](http://arxiv.org/abs/2303.09136v1)]
1. **"ChatGPT Participates in a Computer Science Exam"**. *Sebastian Bordt et al.* arXiv 2023. [[Paper](https://arxiv.org/abs/2303.09461v2)]
1. **"A Comprehensive Capability Analysis of GPT-3 and GPT-3.5 Series Models"**. *Junjie Ye et al.* arXiv 2023. [[Paper](https://arxiv.org/abs/2303.10420v1)]
1. **"On the Educational Impact of ChatGPT: Is Artificial Intelligence Ready to Obtain a University Degree?"**. *Kamil Malinka et al.* arXiv 2023. [[Paper](http://arxiv.org/abs/2303.11146v1)]
1. **"Sparks of Artificial General Intelligence: Early experiments with GPT-4"**. *S'ebastien Bubeck et al.* arXiv 2023. [[Paper](http://arxiv.org/abs/2303.12712v3)]
1. **"Is ChatGPT A Good Keyphrase Generator? A Preliminary Study"**. *Mingyang Song et al.* arXiv 2023. [[Paper](https://arxiv.org/abs/2303.13001v1)]
1. **"Capabilities of GPT-4 on Medical Challenge Problems"**. *Harsha Nori et al.* arXiv 2023. [[Paper](https://arxiv.org/abs/2303.13375v1)]
1. **"Can we trust the evaluation on ChatGPT?"**. *Rachith Aiyappa et al.* arXiv 2023. [[Paper](https://arxiv.org/abs/2303.12767)]
1. **"ChatGPT Outperforms Crowd-Workers for Text-Annotation Tasks"**. *Fabrizio Gilardi et al.* arXiv 2023. [[Paper](http://arxiv.org/abs/2303.15056v1)]
1. **"Evaluation of ChatGPT for NLP-based Mental Health Applications"**. *Bishal Lamichhane et al.* arXiv 2023. [[Paper](https://arxiv.org/abs/2303.15727v1)]
1. **"ChatGPT is a Knowledgeable but Inexperienced Solver: An Investigation of Commonsense Problem in Large Language Models"**. *Bian Ning et al.* arXiv 2023. [[Paper](http://arxiv.org/abs/2303.16421v1)]
1. **"Evaluating GPT-3.5 and GPT-4 Models on Brazilian University Admission Exams"**. *Desnes Nunes et al.* arXiv 2023. [[Paper](https://arxiv.org/abs/2303.17003v1)]
1. **"Humans in Humans Out: On GPT Converging Toward Common Sense in both Success and Failure"**. *Philipp Koralus et al.* arXiv 2023. [[Paper](https://arxiv.org/abs/2303.17276v1)]
1. **"Yes but.. Can ChatGPT Identify Entities in Historical Documents?"**. *Carlos-Emiliano González-Gallardo et al.* arXiv 2023. [[Paper](https://arxiv.org/abs/2303.17322v1)]
1. **"Uncovering ChatGPT's Capabilities in Recommender Systems"**. *Sunhao Dai et al.* arXiv 2023. [[Paper](https://arxiv.org/abs/2305.02182)]
1. **"Editing Large Language Models: Problems, Methods, and Opportunities"**. *Yunzhi Yao et al.* arXiv 2023. [[Paper](https://arxiv.org/abs/2305.13172)]
1. **"Red teaming ChatGPT via Jailbreaking: Bias, Robustness, Reliability and Toxicity"**. *Terry Yue Zhuo et al.* arXiv 2023. [[Paper](https://arxiv.org/abs/2301.12867)]
1. **"On Robustness of Prompt-based Semantic Parsing with Large Pre-trained Language Model: An Empirical Study on Codex"**. *Terry Yue Zhuo et al.* EACL 2023. [[Paper](https://arxiv.org/abs/2301.12868)]
1. **"A Systematic Study and Comprehensive Evaluation of ChatGPT on Benchmark Datasets"**. Laskar et al.* ACL'23. [[Paper]](https://arxiv.org/abs/2305.18486)
1. **"Red-Teaming Large Language Models using Chain of Utterances for Safety-Alignment"**. *Rishabh Bhardwaj et al*. arXiv 2023. [[Paper](https://arxiv.org/abs/2308.09662)]
1. **"Language Models are Multilingual Chain-of-Thought Reasoners"**. *Freda Shi et al.* ICLR 2023. [[Paper](http://arxiv.org/abs/2210.03057v1)]
1. **"Re3: Generating Longer Stories With Recursive Reprompting and Revision"**. *Kevin Yang et al.* EMNLP 2022. [[Paper](http://arxiv.org/abs/2210.06774v3)]
1. **"Language Models of Code are Few-Shot Commonsense Learners"**. *Aman Madaan et al.* EMNLP 2022. [[Paper](http://arxiv.org/abs/2210.07128v3)]
1. **"Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them"**. *Mirac Suzgun et al.* arXiv 2022. [[Paper](http://arxiv.org/abs/2210.09261v1)]
1. **"Large Language Models Can Self-Improve"**. *Jiaxin Huang et al.* arXiv 2022. [[Paper](https://arxiv.org/abs/2210.11610)]
1. **"Draft, Sketch, and Prove: Guiding Formal Theorem Provers with Informal Proofs"**. *Albert Q. Jiang et al.* ICLR 2023. [[Paper](http://arxiv.org/abs/2210.12283v3)]
1. **"Holistic Evaluation of Language Models"**. *Percy Liang et al.* arXiv 2022. [[Paper](https://arxiv.org/abs/2211.09110)]
1. **"PAL: Program-aided Language Models"**. *Luyu Gao et al.* arXiv 2022. [[Paper](https://arxiv.org/abs/2211.10435)]
1. **"Legal Prompt Engineering for Multilingual Legal Judgement Prediction"**. *Dietrich Trautmann et al.* arXiv 2022. [[Paper](http://arxiv.org/abs/2212.02199v1)]
1. **"How Does ChatGPT Perform on the Medical Licensing Exams? The Implications of Large Language Models for Medical Education and Knowledge Assessment"**. *Aidan Gilson et al.* medRxiv 2022. [[Paper](https://www.medrxiv.org/content/10.1101/2022.12.23.22283901v1)]
1. **"ChatGPT: The End of Online Exam Integrity?"**. *Teo Susnjak et al.* arXiv 2022. [[Paper](http://arxiv.org/abs/2212.09292v1)]
1. **"Large Language Models are reasoners with Self-Verification"**. *Yixuan Weng et al.* arXiv 2022. [[Paper](https://arxiv.org/abs/2212.09561)]
1. **"Self-Instruct: Aligning Language Model with Self Generated Instructions"**. *Yizhong Wang et al.* arXiv 2022. [[Paper](http://arxiv.org/abs/2212.10560v1)]
1. **"ChatGPT Makes Medicine Easy to Swallow: An Exploratory Case Study on Simplified Radiology Reports"**. *Katharina Jeblick et al.* arXiv 2022. [[Paper](http://arxiv.org/abs/2212.14882v1)]
1. **"Language Models as Zero-Shot Planners: Extracting Actionable Knowledge for Embodied Agents"**. *Wenlong Huang et al.* ICML 2022. [[Paper](http://arxiv.org/abs/2201.07207v2)]
1. **"Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"**. *Jason Wei et al.* NeurIPS 2022. [[Paper](http://arxiv.org/abs/2201.11903v6)]
1. **"Training language models to follow instructions with human feedback"**. *Long Ouyang et al.* arXiv 2022. [[Paper](http://arxiv.org/abs/2203.02155v1)]
1. **"Competition-Level Code Generation with AlphaCode"**. *Yujia Li et al.* Science 2022. [[Paper](http://arxiv.org/abs/2203.07814v1)]
1. **"Do As I Can, Not As I Say: Grounding Language in Robotic Affordances"**. *Michael Ahn et al.* arXiv 2022. [[Paper](http://arxiv.org/abs/2204.01691v2)]
1. **"Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback"**. *Yuntao Bai et al.* arXiv 2022. [[Paper](http://arxiv.org/abs/2204.05862v1)]
1. **"Autoformalization with Large Language Models"**. *Yuhuai Wu et al.* NeurIPS 2022. [[Paper](http://arxiv.org/abs/2205.12615v1)]
1. **"Beyond the Imitation Game: Quantifying and extrapolating the capabilities of language models"**. *Aarohi Srivastava et al.* arXiv 2022. [[Paper](https://arxiv.org/abs/2206.04615)]
1. **"Exploring Length Generalization in Large Language Models"**. *Cem Anil et al.* NeurIPS 2022. [[Paper](http://arxiv.org/abs/2207.04901v2)]
1. **"Few-shot Learning with Retrieval Augmented Language Models"**. *Gautier Izacard et al.* arXiv 2022. [[Paper](https://arxiv.org/abs/2208.03299)]
1. **"Limitations of Language Models in Arithmetic and Symbolic Induction"**. *Jing Qian et al.* arXiv 2022. [[Paper](http://arxiv.org/abs/2208.05051v1)]
1. **"Code as Policies: Language Model Programs for Embodied Control"**. *Jacky Liang et al.* arXiv 2022. [[Paper](http://arxiv.org/abs/2209.07753v3)]
1. **"ProgPrompt: Generating Situated Robot Task Plans using Large Language Models"**. *Ishika Singh et al.* arXiv 2022. [[Paper](http://arxiv.org/abs/2209.11302v1)]
1. **"Law Informs Code: A Legal Informatics Approach to Aligning Artificial Intelligence with Humans"**. *John J. Nay et al.* arXiv 2022. [[Paper](http://arxiv.org/abs/2209.13020v13)]
1. **"Measuring Massive Multitask Language Understanding"**. *Dan Hendrycks et al.* ICLR 2021. [[Paper](http://arxiv.org/abs/2009.03300v3)]
1. **"Persistent Anti-Muslim Bias in Large Language Models"**. *Abubakar Abid et al.* AIES 2021. [[Paper](http://arxiv.org/abs/2101.05783v2)]
1. **"Understanding the Capabilities, Limitations, and Societal Impact of Large Language Models"**. *Alex Tamkin et al.* arXiv 2021. [[Paper](http://arxiv.org/abs/2102.02503v1)]
1. **"BEHAVIOR: Benchmark for Everyday Household Activities in Virtual, Interactive, and Ecological Environments"**. *Sanjana Srivastava et al.* CoRL 2021. [[Paper](http://arxiv.org/abs/2108.03332v1)]
1. **"Program Synthesis with Large Language Models"**. *Jacob Austin et al.* arXiv 2021. [[Paper](http://arxiv.org/abs/2108.07732v1)]
1. **"Training Verifiers to Solve Math Word Problems"**. *Karl Cobbe et al.* arXiv 2021. [[Paper](http://arxiv.org/abs/2110.14168v2)]
1. **"Show Your Work: Scratchpads for Intermediate Computation with Language Models"**. *Maxwell I. Nye et al.* arXiv 2021. [[Paper](http://arxiv.org/abs/2112.00114v1)]
