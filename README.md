<p align="center">
    <a href="#"><img src="https://img.shields.io/badge/status-active-brightgreen.svg"></a>
    <a href="#"><img src="https://img.shields.io/badge/MLX-Framework-orange.svg"></a>
    <a href="#"><img src="https://img.shields.io/badge/Model-120M--params-blue.svg"></a>
    <a href="#"><img src="https://img.shields.io/badge/License-Research--NonCommercial-red.svg"></a>
</p>

![Banner](https://capsule-render.vercel.app/api?type=venom&height=200&color=0:FA520F,100:000000&text=KiswaEnglish%20LM&textBg=false&desc=Language%20Model%20Built%20From%20Scratch&descAlign=75&fontAlign=50&descAlignY=70&fontColor=ffffff)

<p align="center">
    <strong>Kiswahili-English Language Model Built From Scratch</strong><br>
</p>


<p align="center">
    <img src="training_loss_curve.png" alt="training loss curve" width="800">
</p>


This project presents a complete implementation of a language model built entirely from scratch, designed specifically for the mixed-language communication style used across East Africa commonly referred to as KiswaEnglish. This form of speech naturally blends Kiswahili and English in daily conversation, a pattern rarely supported well by standard global language models.

Unlike most existing solutions, this model was developed without using any pre-trained weights or external datasets. All vocabulary, grammar rules, and training examples were created manually to reflect local language use, culture, and context. The entire system is optimized to run efficiently on standard consumer hardware, making it accessible to students, developers, and researchers without access to specialized infrastructure.


## Key Features

- Built completely from scratch, no external model dependencies
- Native support for pure Kiswahili, pure English, and natural mixed-language text
- Trained on custom synthetic data designed to match real local communication
- Lightweight architecture optimized for laptops and personal devices
- Capable of conversation, instruction following, explanation, and basic reasoning
- Fully independent and customizable


## Technical Details

| Attribute | Value |
|-----------|-------|
| **Model Size** | Approximately 120 million parameters |
| **Architecture** | Custom Transformer-based design with 12 layers and 8 attention heads |
| **Context Window** | 2048 tokens |
| **Tokenizer** | Custom-built using SentencePiece, trained exclusively on project data (1,400 vocabulary size) |
| **Training Data** | 200,000 synthetic samples (~30 million tokens) |
| **Hardware Used** | MacBook M3 with 8GB unified memory |
| **Framework** | MLX for efficient performance on Apple hardware |
| **Training Progress** | Loss reduced from above 10.0 to 2.5 over 15 epochs |


## Project Structure

All components were developed specifically for this work:

- **Data generation module**  creates structured, clean, and relevant training text
- **Custom tokenizer** handles Kiswahili morphology and mixed-language patterns correctly
- **Neural network definition** lightweight design balanced for performance and memory use
- **Training pipeline**  optimized for stable learning on consumer hardware
- **Inference engine**  generates responses efficiently in all supported language forms


## Results

The model successfully produces natural, grammatically correct text in all supported modes:

- **Pure Kiswahili**: Clear explanations, advice, and conversation
- **Pure English**: Accurate answers and structured content
- **KiswaEnglish**: Natural code-switching matching how people actually speak

Training was stable and effective, confirming that high-quality language models can be built independently using widely available resources.


## Use Cases

- Offline AI assistant for education and daily use
- Localized customer service tools
- Educational content generation aligned with regional languages
- Research platform for language modeling and low-resource language technology
- Foundation for further development of African language AI

## Future Work

- Expand vocabulary to include regional dialects, slang, and more specialized terms
- Scale model size and capacity as hardware allows
- Add features such as translation, summarization, and document understanding
- Optimize deployment for mobile and web platforms
- Share methodology to support development for other African languages

## License
This project is open for research, education, and non-commercial use.

