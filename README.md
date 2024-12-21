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

## ⚙️ Model Constraints

- **📏 Maximum Model Size:** 4GB  

- **🚫 Not Allowed in Final Model for Inference:**  
  - ❌ Large Language Models (LLMs)  
  - ❌ Object detection models  
  - ❌ Pre-trained models that directly solve the task without modifications  

- **✅ Allowed:**  
  - ✔️ Pre-trained Vision-Language Models (e.g., CLIP), **provided they are fine-tuned specifically for this task**  
