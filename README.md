# ⭐️ Compositional Retrieval Challenge
*Contributors: [Ali Nafisi](https://safinal.github.io/), [Hossein Shakibania](https://scholar.google.com/citations?user=huveR90AAAAJ&hl=en&authuser=1)*

## 🔍 Overview
This repository contains our solution for the Compositional Retrieval Challenge, part of the [Rayan International AI Contest](https://ai.rayan.global). The challenge aims to develop a system capable of retrieving the most relevant image from a database by understanding a combination of visual and textual inputs.

## 🎯 Challenge Objective

The task requires building a system that can:

- **Process input consisting of:**
  - A reference image containing a visual scene
  - A text description providing additional context or modifications

- **Identify and return** the single most relevant matching image from a provided database.

The figure below serves as an example of this task:
<p class="row" float="left" align="middle">
<img style="width: 80%; height: auto;" src="assets/task.png"/>
</p>

## 🚀 Our Approach

## 🏆 Results

Our solution for this Challenge achieved outstanding results. The evaluation metric for this challenge is **Accuracy**, with models tested on a private test dataset. Our model achieved the **highest score**, outperforming competitors with a noticeable margin.

The table below presents a summary of the top 10 teams and their respective accuracy scores:

| **Rank** | **Team**                             | **Accuracy (%)** |
|----------|--------------------------------------|------------------|
| 1        | **No Trust Issues Here (Our Team)**  | **95.38**        |
| 2        | Pileh                                | 84.61            |
| 3        | AI Guardians of Trust                | 88.59            |
| 4        | AIUoK                                | 87.30            |
| 5        | red_serotonin                        | 86.90            |
| 6        | GGWP                                 | 85.70            |
| 7        | Persistence                          | 85.20            |
| 8        | AlphaQ                               | 84.50            |
| 9        | Tempest                              | 83.90            |
| 10       | Scientific                           | 82.70            |




## ⚙️ Model Constraints

- **📏 Maximum Model Size:** 4GB  

- **🚫 Not Allowed in Final Model for Inference:**  
  - ❌ Large Language Models (LLMs)  
  - ❌ Object detection models  
  - ❌ Pre-trained models that directly solve the task without modifications  

- **✅ Allowed:**  
  - ✔️ Pre-trained Vision-Language Models (e.g., CLIP), if fine-tuned for this task  
