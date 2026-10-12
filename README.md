# CartPole-v1 with Double DQN

A Double Deep Q-Network (DDQN) agent that learns to balance the pole in Gymnasium's `CartPole-v1`, implemented from scratch in PyTorch. The project includes a hyperparameter study (learning rate, network width, exploration schedule) and a final trained agent.

![Trained agent](assets/agent.gif)

## Results

<!-- TODO: fill in with your own numbers. Evaluate the final model greedily (epsilon = 0) over 100 episodes. -->

| Model | Mean reward (100 greedy episodes) | Std | Episodes to reach avg ≥ 475 |
|-------|-----------------------------------|-----|-----------------------------|
| Final DDQN (lr=1e-4, hidden=128, ε-decay=300) | TODO | TODO | TODO |

![Final training curve](assets/training_curve.png)

## Method

- **Double DQN target**: the policy network selects the next action and the target network evaluates it, which reduces the overestimation bias of vanilla DQN.
- **Experience replay**: buffer of 10,000 transitions, mini-batches of 64, training starts after 1,000 transitions.
- **Target network**: hard update every 1,000 environment steps.
- **Exploration**: exponentially decaying ε-greedy, from 1.0 to 0.05.
- **Stability**: Huber loss and gradient clipping (max norm 10).
- **Network**: MLP, 4 → hidden → hidden → 2, ReLU activations.
- **Other**: γ = 0.99, Adam optimizer, seed 42.

## Hyperparameter Study

Each configuration was trained for 300 episodes, changing one parameter at a time (defaults: lr=1e-3, hidden=128, ε-decay=600).

<!-- TODO: fill in, e.g. mean reward of the last 50 training episodes -->

| Parameter | Values tested | Best | Observation |
|-----------|---------------|------|-------------|
| Learning rate | 1e-2, 1e-3, 1e-4 | TODO | TODO |
| Hidden size | 64, 128, 256 | TODO | TODO |
| ε-decay (steps) | 100, 300, 600 | TODO | TODO |

![Hyperparameter study](assets/hyperparameter_study.png)

## Project Structure

```
main.py                  Training and hyperparameter experiments
play.py                  Visual demonstration of the trained agent
assets/                  Figures used in this README
results_cartpole_dqn/    Saved models, reward logs (CSV) and plots (created by main.py)
report.pdf               Full project report
requirements.txt
```

## How to Run

```bash
pip install -r requirements.txt
python main.py   # runs the hyperparameter study and trains the final model
python play.py   # watch the trained agent
```

## Possible Extensions

- Comparison with vanilla DQN under the same settings
- Results averaged over multiple seeds, with confidence intervals
- Dueling architecture and prioritized experience replay

## Academic Context

Developed for the course *Computational Intelligence – Deep Reinforcement Learning*, Department of Informatics, Aristotle University of Thessaloniki.

## Author

**Grigoris Liatsos** · [GitHub](https://github.com/grigorisliatsos)
