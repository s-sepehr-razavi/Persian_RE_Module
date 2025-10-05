# 🇮🇷 ParsRelEx: Persian Relation Extraction

## 🌟 Overview

**ParsRelEx** is a project focused on **Relation Extraction (RE)** for the **Persian (Farsi) language**. It leverages the state-of-the-art **ATLOP** (Adaptive Thresholding and Ordering for Link Prediction) model architecture, fine-tuned using the **ParsBERT** model as its backbone, to accurately identify semantic relationships between entities in Persian text.

This repository provides:

1. The **training code and methodology** based on the ATLOP architecture.
2. A standalone **Inference Module** for practical, zero-shot relation extraction.
3. A simple **Graphical User Interface (GUI)** for interactive usage and demonstration.

---

## 🚀 Features

- **High Accuracy:** Uses the robust **ATLOP** model for effective relation extraction.
- **Persian-Native:** Employs **ParsBERT** to capture the nuances of the Persian language.
- **Easy Inference:** A dedicated Python module for quick and simple predictions.
- **Interactive UI:** A user-friendly interface for non-programmatic testing.

---

## ⚙️ Installation and Setup

To use the inference module, follow these steps:

### 1. Prerequisites

Ensure you have **Python 3.8+** installed.

### 2. Clone the Repository

```bash
git clone https://github.com/s-sepehr-razavi/Persian_RE_Module.git
```

### 3. Install Required Libraries

Install the necessary Python packages using the provided requirements.txt file

```bash
pip install -r requirements.txt
```

## 💾 Model Weights

To run the model, you must download the **pre-trained model weights**.

### Download Link

| File                         | Description                         |
| :--------------------------- | :---------------------------------- |
| `parsrelex_atlop_weights.pt` | The final fine-tuned model weights. |

**Download the weights from [https://drive.google.com/file/d/1GZVEWd0XuZ_tsvsqUi01eDJ14UpONXJh/view?usp=sharing]** and place the file in the directory named as models_weight.

---

## 🔮 Usage: Interactive UI

For the quickest, most interactive way to test the model without writing any Python code, execute the UI script in your terminal:

```bash
streamlit run re_ui.py
```

---

## 📄 Model Details

- **Base Model:** **ParsBERT** (The Persian-native BERT model)
- **Architecture:** **ATLOP** (Adaptive Thresholding and Ordering for Link Prediction)
- **Task:** **Relation Extraction**
- **Language:** **Persian (Farsi)**

### References

- **ATLOP Paper:** _[https://arxiv.org/abs/2010.11304]_
- **ParsBERT:** _[https://huggingface.co/HooshvareLab/bert-base-parsbert-uncased]_
