Here is a `README.md` file tailored for your GitHub repository based on the provided Jupyter Notebook. 

---

# Deep Q-Network (DQN) for Atari Breakout 🕹️

This repository contains an implementation of a Deep Q-Network (DQN) to play the classic Atari Breakout game using OpenAI's `gym`. The algorithm relies on approximate Q-learning, leveraging experience replay and target networks to stabilize training. 

This code was originally developed as part of the **Practical Reinforcement Learning** course on Coursera.

## 🚀 Overview

The agent learns to play Breakout directly from raw pixel inputs. Instead of manually extracting features, a Convolutional Neural Network (CNN) maps the visual state of the game to optimal Q-values for each possible action. 

The project includes video monitoring capabilities to export and evaluate the agent's gameplay over time.

## 🧠 Core Architecture

This implementation utilizes several standard techniques to make Deep Reinforcement Learning stable and efficient:

* **Image Preprocessing:** Raw Atari frames (210x160 RGB) are cropped to remove the UI, resized to 84x84, and converted to grayscale to reduce the computational load.
* **Frame Buffering:** The agent requires a sense of motion to determine object velocity. The environment is wrapped in a buffer that stacks the last 4 frames into a single observation state.
* **Experience Replay:** Transitions $(s, a, r, s', \text{done})$ are stored in a replay buffer of size 70,000. Mini-batches are sampled randomly during training to break the correlation between consecutive samples and smooth out learning.
* **Target Network:** Two identical CNNs are used: an active network and a target network. The target network provides stable Q-value targets and its weights are updated from the active network every 500 steps.

### Q-Learning Formulation

The algorithm minimizes the TD (Temporal Difference) error using the Huber loss function. The reference Q-value is computed as:

$$Q_{reference}(s,a) = r + \gamma \max_{a'} Q_{target}(s', a')$$

Where:
* $r$ is the immediate reward.
* $\gamma$ is the discount factor (set to `0.99`).
* $Q_{target}$ is the Q-value predicted by the frozen target network for the next state $s'$.

## 🛠️ Dependencies

**Important:** This code is written for **TensorFlow 1.x** (uses `tf.InteractiveSession`, `tf.placeholder`, etc.) and Keras. 

* Python 3.x
* TensorFlow 1.15.x
* Keras
* OpenAI Gym (`gym`, `atari_py`)
* OpenCV (`cv2`)
* NumPy, Pandas, Matplotlib

If you are running this on a headless server, you will also need `xvfb` to render the environment frames.

## 📈 Training Details

* **Exploration vs. Exploitation:** The agent uses an $\epsilon$-greedy strategy. Epsilon starts at `1.0` (pure exploration) and decays by a factor of `0.999` every 500 steps, capping at a minimum of `0.01`.
* **Optimizer:** Adam Optimizer with a learning rate of `1e-5`.
* **Warm-up:** The first few thousand steps are required to fill the replay buffer and allow the agent to "warm up" before noticeable improvements in the mean reward occur.
* **Patience:** Training RL from raw pixels takes a significant amount of time. An optimistic estimate is reaching an average reward $> 10$ after roughly 10,000 to 20,000 training steps.

## 📊 Monitoring Progress

During training, the script outputs live charts displaying:
1. **Mean Reward per Game:** Expected to oscillate heavily but trend upward over thousands of iterations.
2. **TD Loss History (Moving Average):** The Mean Squared Error between current and target Q-values. It is normal for this to slowly increase or decrease, as long as it does not drop immediately to exactly zero or explode to `NaN`.

## 🎬 Evaluating and Recording

To evaluate a trained agent, you can set `agent.epsilon = 0` to disable random exploration and rely entirely on the learned policy. The repository includes code to record the agent's gameplay using `gym.wrappers.Monitor`.

If you have a pre-trained weights file (`dqn_model_atari_weights.h5`), you can load it directly into the active network to skip training:

```python
agent.network.load_weights('dqn_model_atari_weights.h5')
```

