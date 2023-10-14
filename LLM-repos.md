<h1 align="center">A curated list of GitHub repos on Large Language Models (LLMs)</h1>

---

This list of GitHub repos is a compilatiopn of surveys, collections of models and other resources, that I've found useful among the vast amount of GitHub repos on __Large Language Models (LLMs)__ and their applications in __Natural Language Processing (NLP)__ applications.

---

# GitHub repos

## Surveys and lists

- [Mooler0410/LLMsPracticalGuide](https://github.com/Mooler0410/LLMsPracticalGuide): A curated (still actively updated) list of practical guide resources of LLMs. It's based on the survey paper: [Harnessing the Power of LLMs in Practice: A Survey on ChatGPT and Beyond](https://arxiv.org/abs/2304.13712) and efforts from @[xinyadu](https://github.com/xinyadu). The survey is partially based on the second half of this [Blog](https://jingfengyang.github.io/gpt). They also build an evolutionary tree of modern Large Language Models (LLMs) to trace the development of language models in recent years and highlights some of the most well-known models. They also include usage restrictions and data licensing information.
- [eugeneyan/open-llms](https://github.com/eugeneyan/open-llms): It contains a list of open LLMs available for commercial use that I've used to compile my [list of LLM models](LLM-models.md).
- [RUCAIBox/LLMSurvey](https://github.com/RUCAIBox/LLMSurvey): An amazing repo with the original list that I've used to compile my [list of papers on LLMs](LLM-papers.md). It's the official GitHub page for the survey paper [A Survey of Large Language Models](https://arxiv.org/abs/2303.18223).
- [Hannibal046/Awesome-LLM](https://github.com/Hannibal046/Awesome-LLM): A curated list of papers about LLMs, especially relating to ChatGPT. It also contains frameworks for LLM training, tools to deploy LLMs, courses and tutorials about LLMs and all publicly available LLM checkpoints and APIs.
- [FreedomIntelligence/LLMZoo](https://github.com/FreedomIntelligence/LLMZoo): LLM Zoo is a project that provides data, models, and evaluation benchmarks for LLMs.


## LLM Resources

- [imaurer/awesome-decentralized-llm](https://github.com/imaurer/awesome-decentralized-llm): A collection of LLM resources that can be used to build products you can "own" or to perform reproducible research.
- [yrolabs/awesome-langchain](https://github.com/kyrolabs/awesome-langchain): __[LangChain](https://www.langchain.com/)__ is an amazing framework to get LLM projects done in a short time and whose ecosystem is growing fast. This repo is an attempt to keep track of the initiatives around LangChain.
- [langchain-ai/langchain](https://github.com/langchain-ai/langchain): The original __LangChain__ GitHub repo.
- [XiaoxinHe/Awesome-Graph-LLM](https://github.com/XiaoxinHe/Awesome-Graph-LLM): A curated collection of resources about __Graph-Related Large Language Models__.LLMs have shown remarkable progress in natural language processing tasks. However, their integration with graph structures, which are prevalent in real-world applications, remains relatively unexplored. This repository aims to bridge that gap by providing a curated list of research papers that explore the intersection of graph-based techniques with LLMs.
- [huggingface/peft](https://github.com/huggingface/peft): __Parameter-Efficient Fine-Tuning (PEFT)__ methods enable efficient adaptation of pre-trained language models (PLMs) to various downstream applications without fine-tuning all the model's parameters. Fine-tuning large-scale PLMs is often prohibitively costly. In this regard, PEFT methods only fine-tune a small number of (extra) model parameters, thereby greatly decreasing the computational and storage costs. Recent State-of-the-Art PEFT techniques achieve performance comparable to that of full fine-tuning. This 🤗 library seamlessly integrates with 🤗 Accelerate for large scale models leveraging DeepSpeed and Big Model Inference.
- [karpathy/minGPT](https://github.com/karpathy/minGPT): A PyTorch re-implementation of GPT, both training and inference. __minGPT__ tries to be small, clean, interpretable and educational, as most of the currently available GPT model implementations can a bit sprawling. GPT is not a complicated model and this implementation is appropriately about 300 lines of code (see ``mingpt/model.py``). All that's going on is that a sequence of indices feeds into a Transformer, and a probability distribution over the next index in the sequence comes out. The majority of the complexity is just being clever with batching (both across examples and over sequence length) for efficiency.
- [FlowiseAI/Flowise](https://github.com/FlowiseAI/Flowise): Flowise is a drag & drop UI to build your customized LLM flow.
- [microsoft/semantic-kernel](https://github.com/microsoft/semantic-kernel): __[Semantic Kernel](https://learn.microsoft.com/en-us/semantic-kernel/overview/)__ is an SDK from Microsoft that integrates Large Language Models (LLMs) like [OpenAI](https://platform.openai.com/docs/introduction), [Azure OpenAI](https://azure.microsoft.com/en-us/products/ai-services/openai-service), and [Hugging Face](https://huggingface.co/) with conventional programming languages like C#, Python, and Java. Semantic Kernel achieves this by allowing you to define plugins that can be chained together in just a few lines of code. What makes Semantic Kernel special, however, is its ability to automatically orchestrate plugins with AI. With Semantic Kernel planners, you can ask an LLM to generate a plan that achieves a user's unique goal. Afterwards, Semantic Kernel will execute the plan for the user.
-[mlc-ai/mlc-llm](https://github.com/mlc-ai/mlc-llm): Machine Learning Compilation for Large Language Models (MLC LLM) is a high-performance universal deployment solution that allows native deployment of any large language models with native APIs with compiler acceleration. The mission of this project is to enable everyone to develop, optimize and deploy AI models natively on everyone's devices with ML compilation techniques.

