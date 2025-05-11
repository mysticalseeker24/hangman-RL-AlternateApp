# Hangman RL - TrexQuant Interview Project

## Overview
This repository contains a solution for the TrexQuant Hangman Game interview assignment. The objective is to develop an algorithm that plays Hangman through an API and significantly outperforms the provided benchmark (18% win rate) using only the provided training dictionary. The project explores both supervised (Bi-LSTM) and reinforcement learning (Deep Q-Learning/DQN) approaches, with detailed rationale and implementation for each.

---

## Problem Statement
- **Goal:** Build an algorithm to play Hangman via API, maximizing win rate.
- **Constraints:**
  - Only the provided training dictionary (~250,000 words) may be used.
  - The test set is disjoint from the training set.
  - No external word lists or data sources are permitted.
- **Benchmark:** Provided algorithm achieves ~18% win rate using masked pattern matching and letter frequency.

---

## Approaches

### 1. Bi-LSTM (Supervised Learning)
#### Rationale
- **Sequential modeling:** Hangman word patterns are sequential; Bi-LSTM captures bidirectional dependencies.
- **Generalization:** Learns to predict likely letters from masked patterns, supporting generalization to unseen words.

#### Implementation
- **Input Representation:**
  - Word pattern as one-hot (27-dim: 26 letters + underscore), padded to fixed length.
  - Guessed letters as a 26-dim binary vector.
- **Architecture:**
  - Embedding layer → 2-layer Bi-LSTM (hidden_dim=128) → concatenate with guessed vector → Dense layers → 26-way sigmoid output.
- **Training:**
  - Generate ~100,000 simulated game states (random words, partial reveals).
  - Binary cross-entropy loss; Adam optimizer; batch size 64; 10–20 epochs.
- **Inference:**
  - For a given pattern and guess history, select the unguessed letter with highest probability.
- **Expected Performance:**
  - 65–75% win rate (substantially above benchmark).

#### Reference
- [YAPhoa/HangmanKeras](https://github.com/YAPhoa/HangmanKeras)

### 2. Deep Q-Learning (DQN, Reinforcement Learning)
#### Rationale
- **Sequential decision-making:** RL is suited for maximizing long-term rewards in games.
- **State Representation:**
  - Masked word (one-hot), guessed letters, remaining attempts.
- **Action Space:**
  - 26 possible actions (letters a-z).
- **Reward Structure:**
  - Positive for correct guesses, negative for incorrect, large reward for win, penalty for loss.

#### Implementation
- **Model:**
  - Standard LSTM (input_size=27, hidden_size=32, 1 layer) → concatenate with action history → Dense layers → 26 Q-values.
- **Training:**
  - Experience replay, epsilon-greedy exploration, target network, Huber loss, Adam optimizer.
  - Simulate thousands of games for training.
- **Action Selection:**
  - Epsilon-greedy: random exploration decaying to exploitation of Q-values.
- **Expected Performance:**
  - 60–70%+ win rate possible with tuning and sufficient training.

---

## File Structure
- `hangman_api_user.ipynb` – Main notebook: API interface, data prep, training, evaluation, and integration.
- `dqn.py` – DQN model definition (standard, non-memory-optimized version).
- `cuda_utils.py` – Basic CUDA setup and cleanup utilities.
- `word_lists.py` – Utility for loading and filtering word lists.
- `words_250000_train.txt` – Provided training dictionary.

---

## Usage Instructions
1. **Clone this repository:**
   ```bash
   git clone https://github.com/mysticalseeker24/hangman-RL-AlternateApp.git
   cd hangman-RL-AlternateApp
   ```
2. **Install dependencies:**
   - Python 3.7+
   - `pip install torch numpy pandas matplotlib requests`
3. **Run the notebook:**
   - Open `hangman_api_user.ipynb` in Jupyter and follow the instructions to train/evaluate the model.
4. **API Integration:**
   - The notebook includes the `HangmanAPI` class for playing games via the provided API server.

---

## Notes
- **No memory-optimized code:** This submission uses the original, non-memory-optimized model implementations as required.
- **License:** See `LICENSE` file in the repository.

---

## Author
- Submission for TrexQuant Interview Assignment
- [GitHub: mysticalseeker24](https://github.com/mysticalseeker24)
