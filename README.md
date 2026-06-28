<!-- ---
title: Swahili LLM Scratch
emoji: 🏃
colorFrom: yellow
colorTo: yellow
sdk: gradio
sdk_version: 6.19.0
python_version: '3.13'
app_file: app.py
pinned: false
license: apache-2.0
short_description: Swahili Swaiba LLM From Scratch
--- -->

# Swahili Swahiba LLM | SwahiliGpt from Scratch. 

<p align="center">
    <img src="/MDBANNER/swahili_banner.png" alt="screenshot" width="800">
</p>

This project presents a complete implementation of a language model built entirely from scratch, designed specifically for the mixed‑language communication style used across East Africa commonly referred to as **Kiswaenglish**. This form of speech naturally blends Kiswahili and English in daily conversation, a pattern rarely supported well by standard global language models.

Unlike most existing solutions, this model was developed without relying on pre‑trained model weights. It combines **custom synthetic data** and **publicly available Swahili corpora** to reflect local language use, culture, and context. The entire system is optimized to run efficiently on standard consumer hardware, making it accessible to students, developers, and researchers without access to specialized infrastructure.

Currently the core training and inference scripts are built for **Silicon macOS using MLX**, with support for **Windows and Linux** planned contributions from the community to extend compatibility are welcome.

> **Please, Refer to this repo for the WebApp Interface of this project**  
> https://github.com/zuck30/SWAHIBA

# Inspiration & Reference
This work is inspired by the approach, teaching, endless tutorials and open contributions of **Andrej Karpathy** and **Alec Radford**, whose work has shown what is possible when building language models from first principles. The architecture follows the Transformer design introduced in the foundational paper:
> **Attention Is All You Need**  
> https://papers.neurips.cc/paper/7181-attention-is-all-you-need.pdf


# How to Run

Follow these steps in order from start to finish to set up, build, and run the model:

# 1. Install Dependencies
First, install all required libraries:
```bash
pip install -r requirements.txt
```
> Note: For Windows and Linux, use the PyTorch‑compatible requirements when they become available. The equivalent versions using PyTorch (which runs on Windows and Linux) haven’t been added to the repository yet; they will be added later by other researchers or contributors. Other contributors will translate and adapt the existing MLX‑based scripts to PyTorch, which will run on these operating systems.

# 2. Get the Swahili Corpus
Download the public corpus from Mendeley Data and place all `.txt` files into a folder named `Swahili_Corpus` in your project root, delete all other files and use the one file that says Swahili_Corpus_Combined.txt:
> https://data.mendeley.com/research-data/?query=swahili

# 3. Build the Full Training Dataset
Synthetic data is generated automatically during this step. Combine it with the Swahili Corpus into one balanced dataset following the **30% synthetic, 70% real** rule:
```bash
python build_full_dataset.py
```
Works across all operating systems.

# 4. Train the Custom Tokenizer
Train a SentencePiece tokenizer optimized for your language mix:
```bash
python build_tokenizer.py
```
Works across all operating systems.

# 5. Start Training the Model
Train the Transformer model from scratch:
- **macOS**: Use MLX for best performance
```bash
python train_scratch.py
```
- **Windows / Linux**: Compatible versions look at pytorch dir
>pytorch/

# 6. Generate Text / Test the Model
Once training completes, use the matching script:
- **macOS**:
```bash
python swahili_gpt.py
```
- **Windows / Linux**: Compatible versions look at pytorch dir
>pytorch/


# Key Features

- Built completely from scratch, no external model dependencies
- Native support for pure Kiswahili, pure English, and natural mixed‑language text
- Balanced training data: **30% synthetic / 70% real data** for best quality and consistency
- Real data sourced from trusted public corpora
- Lightweight architecture optimized for laptops and personal devices
- Capable of conversation, instruction following, explanation, and basic reasoning
- Fully independent, customizable, and easy to extend
- Cross‑platform support in progress, with community contributions for Windows and Linux


# Getting the Data

All data is openly available or can be generated locally:

- **Original Swahili Corpus**: Download from **Mendeley Data** → https://data.mendeley.com/research-data/?query=swahili
- **Synthetic data**: Created automatically within `build_full_dataset.py` using functions from `synthesize_all.py`
- **Final dataset**: Built using `python build_full_dataset.py` following the 30|70 ratio

*Note: Data folders are excluded from Git to keep the repository lightweight; you can generate or download them yourself whenever needed.*


# Dataset License and Acknowledgment

The files associated with this dataset are licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license.

What does this mean? You are free to share, copy, and modify this dataset for any purpose, provided you adhere to the following terms:

Attribution: You must give appropriate credit, provide a link to the CC BY license, and clearly indicate if any changes were made.

No Endorsement: You may not use the dataset in any way that suggests the rights holder endorses you or your specific use of the data.

Third-Party Content Notice: Further permission may be required for any specific content within this dataset that is explicitly identified as belonging to a third party.


# Technical Details

| Attribute | Value |
|-----------|-------|
| **Model Size** | Approximately 120 million parameters |
| **Architecture** | Custom Transformer‑based design with 12 layers and 8 attention heads |
| **Context Window** | 2048 tokens |
| **Tokenizer** | Custom‑built using SentencePiece, trained on full dataset (3,500 vocabulary size) |
| **Training Data** | Mixed set: synthetic + original Swahili corpus (~100–110 million tokens total) |
| **Data Split** | 30% synthetic, 70% real data |
| **Supported Systems** | macOS (fully supported), Windows and Linux (coming via community contributions) |
| **Hardware Used** | MacBook Pro M3  |
| **Framework** | MLX for efficient performance on Apple hardware; PyTorch versions planned for other systems |
| **Training Progress** | Loss reduced from above 10.0 to ~2.5 over 8 epochs |


# Project Structure

All components were developed specifically for this work:
```
swahili-llm-scratch/
├── .gitignore
├── LICENSE
├── README.md
├── MDBANNER/
│   ├── loss_curve.png
│   ├── swahili_banner.png
│   └── training_loss_curve.png
├── pytorch/
│   ├── .gitignore
│   ├── README.md
│   ├── build_corpus.py
│   ├── build_instruct.py
│   ├── chat.py
│   ├── finetune.py
│   ├── generate.py
│   ├── model.py
│   ├── model_config.json
│   ├── requirements.txt
│   └── train.py
├── build_full_dataset.py
├── build_tokenizer.py
├── model_config.json
├── plot_figure.py
├── requirements.txt
├── swahili_gpt.py
├── synthesize_all.py
└── train_scratch.py

```
- **`synthesize_all.py`** → Generates structured synthetic text in Swahili, English, and Kiswaenglish, imported automatically
- **`build_full_dataset.py`** → Combines synthetic data and Swahili Corpus into one balanced dataset following the 30|70 rule
- **`build_tokenizer.py`** → Trains custom tokenizer to handle mixed language patterns correctly
- **`model_config.json`** → Stores all model and training settings in one place


# How to Contribute

Contributions are welcome and easy to follow:

- **Code, ideas, documentation, and new data sources** → submit via Pull Requests or open an Issue
- **Porting to Windows and Linux**: Other contributors are welcome to provide compatible versions using PyTorch to expand support
- **Do NOT commit large data files or model weights** → these are too big for version control
- **To run locally**: Install dependencies, download or prepare your corpus, and build the dataset using the scripts provided
- **To share new data**: Upload to a public hosting service and share the link instead of pushing files directly


# Results

The model successfully produces natural, grammatically correct text in all supported modes:

- **Pure Kiswahili**: Clear explanations, advice, and conversation
- **Pure English**: Accurate answers and structured content
- **Kiswaenglish**: Natural code‑switching matching how people actually speak

Training was stable and effective, confirming that high‑quality language models can be built independently using widely available resources.


# Use Cases

- Offline AI assistant for education and daily use
- Localized customer service tools
- Educational content generation aligned with regional languages
- Research platform for language modeling and low‑resource language technology
- Foundation for further development of African language AI


# Future Work

- Expand vocabulary to include regional dialects, slang, and more specialized terms
- Scale model size and capacity as hardware allows
- Add features such as translation, summarization, and document understanding
- Build a simple web interface to allow public use and collect feedback
- Optimize deployment for mobile and web platforms
- Share methodology to support development for other African languages

<br>

# First Training Loss Curve on MLX 
<p align="center">
    <img src="/MDBANNER/training_loss_curve.png" alt="training loss curve" width="800">
</p>


# Ongoing Training and Finetuning

> **To keep up with what's going on Please refer and visit hugging face**
>https://huggingface.co/Benjamin-png/swahili-gpt-71m-instruct

# Loss trajectory for ongoing finetuning and learning

<p align="center">
    <img src="/MDBANNER/loss_curve.png" alt="training loss curve" width="800">
</p>


# License

This project is open for research, education, and non‑commercial use.

![technologies](https://skillicons.dev/icons?i=git,github,vscode,apple&perline=10)

<p align="center">
    <a href="https://sheddydev.netlify.app"><img src="https://img.shields.io/badge/Blog-sheddydev.netlify.app-purple.svg"></a>
    <a href="https://sheddysilicon.netlify.app"><img src="https://img.shields.io/badge/Author-sheddysilicon.netlify.app-green.svg"></a>
    <a href="mailto:mwalyangashadrack@gmail.com"><img src="https://img.shields.io/badge/Email-mwalyangashadrack%40gmail.com-red.svg"></a>
</p>