# Dynamic Key-value Memory Enhanced Multi-step Graph Reasoning for Knowledge-based Visual Question Answering
This repo contains the official implementation for the AAAI paper [Dynamic Key-value Memory Enhanced Multi-step Graph Reasoning for Knowledge-based Visual Question Answering](https://arxiv.org/abs/2203.02985)

## Abstract 
Knowledge-based visual question answering (VQA) is a vision-language task that requires an agent to correctly answer image-related questions using knowledge that is not presented in the given image. It is not only a more challenging task than regular VQA but also a vital step towards building a general VQA system. Most existing knowledge-based VQA systems process knowledge and image information similarly and ignore the fact that the knowledge base (KB) contains complete information about a triplet, while the extracted image information might be incomplete as the relations between two objects are missing or wrongly detected. In this paper, we
propose a novel model named dynamic knowledge memory enhanced multi-step graph reasoning (DMMGR), which performs explicit and implicit reasoning over a key-value knowledge memory module and a spatial-aware image graph, respectively. Specifically, the memory module learns a dynamic knowledge representation and generates a knowledge-aware question representation at each reasoning step. Then, this representation is used to guide a graph attention operator over the spatial-aware image graph. Our model achieves new stateof-the-art accuracy on the KRVQR and FVQA datasets. We also conduct ablation experiments to prove the effectiveness of each component of the proposed model. 
## Illustration of Our Method
updating ... 
## Runing

sh run_graph.sh



