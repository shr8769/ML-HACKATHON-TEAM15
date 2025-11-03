# ML-HACKATHON-TEAM15: Hangman Solver & RL Agent

This repository contains code and resources for building an automated Hangman solver using statistical modeling and reinforcement learning. The primary implementation is in the form of a Google Colab notebook (`mlhackathonteam15.ipynb`), which can be run interactively for experimenting, training, and evaluating models.

## Features

- **Corpus Cleaning & Bucketing**: Data preprocessing for Hangman word lists, including cleaning and bucketing by word length.
- **NGram Hidden Markov Model (HMM)**: Learns letter and trigram patterns from the training corpus to guess efficiently.
- **Hangman Environment**: Modular classes for simulating the Hangman game and tracking guesses, lives, and rewards.
- **Greedy HMM Policy**: Baseline agent for Hangman using blended letter probabilities from the HMM.
- **Performance Evaluation**: Tools and functions for automated evaluation and reporting on accuracy and win rate.
- **Reinforcement Learning (DQN)**: Deep Q-Network agent with curriculum training across word length stages ("short", "medium", "long").
- **Replay Buffer/Masked Actions**: RL infrastructure tailored for the Hangman environment with valid action masking.
- **Model Saving & Exporting**: Persistent storage of trained assets for further fine-tuning or contest entry.

## How to Use

1. **Google Colab Execution**

   Open [this notebook in Colab](https://colab.research.google.com/github/shr8769/ML-HACKATHON-TEAM15/blob/main/mlhackathonteam15.ipynb)  
   *(alternatively, download and upload the notebook to Colab or Jupyter)*

2. **Data Preparation**

   - You will need two files:  
     `corpus.txt` (word list for training)  
     `test.txt` (word list for evaluation)
   - When prompted in the notebook (Cell 2), upload these files using the widget.

3. **Training and Evaluation**

   - Run cells sequentially.
   - The notebook will train a statistical HMM, perform greedy policy evaluations, and then train a DQN agent using deep RL.
   - Curriculum training proceeds from short to medium to long words.

4. **Results & Assets**

   - Performance metrics (accuracy, success rate) will be printed as output in respective cells.
   - Final model assets for RL are saved as `rl_curriculum_final.pkl`.

## Requirements

- Python 3  
- PyTorch  
- `tqdm`, `numpy`, `pickle`  
- Google Colab or Jupyter Notebook (GPU recommended for RL training)

All required libraries are installed automatically in Colab. For local Jupyter use, please manually install any missing dependencies.

## File Overview

- `mlhackathonteam15.ipynb`: Main notebook with all code, explanations, and evaluation.
- `corpus.txt`: Training word list (user provides).
- `test.txt`: Evaluation word list (user provides).
- `rl_curriculum_final.pkl`: Trained RL model asset (generated).
- `rl_assets.pkl`: Saved HMM/index/alphabet for RL (generated).

## Key Sections in the Notebook

- **Imports & Constants**: Helper libraries and alphabet definitions.
- **Corpus Loading**: Upload and clean word lists.
- **Corpus Indexing**: For fast filtering by length and mask.
- **NGramHMM**: Statistical model for letter guesses.
- **HangmanEnv & RLHangmanEnv**: Environments for both greedy and RL approaches.
- **Greedy Policy**: Baseline performance.
- **DQN RL Agent**: Deep learning agent and training loop.
- **Curriculum Training**: Staged RL learning.
- **Model Saving**: For contest submission or further use.

## References

- [Google Colab Documentation](https://colab.research.google.com/)
- [PyTorch Documentation](https://pytorch.org/)
- [Standard Hangman Game Rules](https://en.wikipedia.org/wiki/Hangman_(game))

## License

MIT License (see accompanying file if present).

---

**Contact:** [shr8769](https://github.com/shr8769)
