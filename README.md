# ELMPA: Evolving LLM-Based Memory and Personality Agents

ELMPA is an AI framework integrating **exemplar memory**, **personality adaptation**, and **evolutionary optimization** to enhance financial analysis report generation.

This repository contains the official implementation and dataset for the paper:  
**"Evolutionary LLM-Based AI Agents: Refining Exemplar Memory and Personality for Enhanced Financial Analysis Report Generation."**


## 🚀 Features
- Multi-agent evolutionary training for continuous optimization
- Exemplar memory retrieval for enhanced decision-making and consistency
- Personality-based adaptation for industry specialization and analytical refinement

## 🔧 Setup
To use ELMPA with OpenAI or DeepSeek APIs, ensure the API keys are properly configured:

- For **OpenAI API**, set your key in `utils/text_generation.py`
- For **DeepSeek API**, set your key in `utils/text_generation_deepseek.py`

Both files require inserting your respective API key in the designated configuration section.

## 📂 Dataset
The processed dataset used in this study is available in: data/processed_data.json
A dataset in a similar format can be used to provide input for the agent to generate financial analysis reports.


