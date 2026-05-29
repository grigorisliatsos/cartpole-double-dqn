# =========================================================
# DEEP RL PROJECT - CARTPOLE WITH DQN
# Μάθημα: Υπολογιστική Νοημοσύνη – Βαθιά Ενισχυτική Μάθηση
# =========================================================

import os
import csv
import math
import random
from collections import deque, namedtuple

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

import gymnasium as gym

# =========================================================
# REPRODUCIBILITY (Αναπαραγωγιμότητα)
# Ορισμός σταθερών seeds για να είναι τα αποτελέσματα συγκρίσιμα
# =========================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Αυτόματη επιλογή GPU (CUDA) αν υπάρχει, αλλιώς χρήση CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESULTS_DIR = "results_cartpole_dqn"

# =========================================================
# MEMORY (Μνήμη Αναπαραγωγής - Experience Replay)
# =========================================================
Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        """Προσθήκη νέας εμπειρίας στη μνήμη."""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """Τυχαία επιλογή δείγματος εμπειριών για εκπαίδευση."""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# =========================================================
# MODEL (Αρχιτεκτονική Deep Q-Network)
# =========================================================
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.net(x)


# =========================================================
# EPSILON-GREEDY STRATEGY (Εξερεύνηση vs Εκμετάλλευση)
# =========================================================
def get_epsilon(step, start, end, decay):
    """Υπολογίζει το epsilon που μειώνεται εκθετικά με το χρόνο."""
    return end + (start - end) * math.exp(-step / decay)


def select_action(state, model, action_dim, step, eps_start, eps_end, eps_decay):
    epsilon = get_epsilon(step, eps_start, eps_end, eps_decay)

    if random.random() < epsilon:
        return random.randrange(action_dim), epsilon

    with torch.no_grad():
        state = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        return int(torch.argmax(model(state))), epsilon


# =========================================================
# OPTIMIZE (Double DQN Logic + Huber Loss)
# =========================================================
def optimize(memory, policy, target, optimizer, batch_size, gamma, min_replay):
    # Περιμένουμε να μαζευτεί αρκετή εμπειρία στη μνήμη πριν ξεκινήσουμε
    if len(memory) < max(batch_size, min_replay):
        return

    # Προετοιμασία του batch δεδομένων
    batch = Transition(*zip(*memory.sample(batch_size)))

    # ΔΙΟΡΘΩΣΗ: Μετατροπή με χρήση np.stack και torch.from_numpy για αποφυγή warnings
    states = torch.from_numpy(np.stack(batch.state)).float().to(DEVICE)
    actions = torch.tensor(batch.action, dtype=torch.long, device=DEVICE).unsqueeze(1)
    rewards = torch.tensor(batch.reward, dtype=torch.float32, device=DEVICE).unsqueeze(1)
    next_states = torch.from_numpy(np.stack(batch.next_state)).float().to(DEVICE)
    dones = torch.tensor(batch.done, dtype=torch.float32, device=DEVICE).unsqueeze(1)

    # Υπολογισμός Q(s,a) από το τρέχον δίκτυο (Policy Network)
    q_values = policy(states).gather(1, actions)

    # DOUBLE DQN LOGIC: Επιλογή από το policy, αξιολόγηση από το target
    with torch.no_grad():
        next_actions = policy(next_states).argmax(1, keepdim=True)
        max_next = target(next_states).gather(1, next_actions)
        target_q = rewards + (1 - dones) * gamma * max_next

    # Huber Loss
    loss = nn.SmoothL1Loss()(q_values, target_q)

    # Gradient Descent
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 10)
    optimizer.step()


# =========================================================
# PLOTS & DATA STORAGE (Οπτικοποίηση & Αποθήκευση)
# =========================================================
def moving_avg(x, w=20):
    return [np.mean(x[max(0, i - w):i + 1]) for i in range(len(x))]


def plot_rewards(rewards, path):
    plt.figure()
    plt.plot(rewards, label="Reward per Episode", alpha=0.5)
    plt.plot(moving_avg(rewards), label="Moving Average (20)", linewidth=2)
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.savefig(path)
    plt.close()


def save_rewards_csv(rewards, path):
    with open(path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "Reward"])
        for i, r in enumerate(rewards):
            writer.writerow([i, r])


# =========================================================
# TRAIN (Κύριος Βρόχος Εκπαίδευσης)
# =========================================================
def train(run_name, lr, hidden, eps_decay, episodes):
    run_dir = os.path.join(RESULTS_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)

    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    BATCH_SIZE = 64
    GAMMA = 0.99
    MIN_REPLAY = 1000
    TARGET_UPDATE_STEPS = 1000

    policy = DQN(state_dim, action_dim, hidden).to(DEVICE)
    target = DQN(state_dim, action_dim, hidden).to(DEVICE)
    target.load_state_dict(policy.state_dict())

    optimizer = optim.Adam(policy.parameters(), lr=lr)
    memory = ReplayMemory(10000)

    rewards_all = []
    best_reward = -float("inf")
    step = 0

    for ep in range(episodes):
        state, _ = env.reset()
        total = 0

        for t in range(500):
            action, _ = select_action(state, policy, action_dim, step, 1.0, 0.05, eps_decay)
            next_state, reward, done, trunc, _ = env.step(action)

            memory.push(state, action, reward, next_state, float(done or trunc))

            state = next_state
            total += reward
            step += 1

            optimize(memory, policy, target, optimizer, BATCH_SIZE, GAMMA, MIN_REPLAY)

            # ΔΙΟΡΘΩΣΗ 1: Το target update μετράει βήματα ΜΟΝΟ αφού ξεκινήσει η πραγματική εκπαίδευση
            if len(memory) >= MIN_REPLAY and step % TARGET_UPDATE_STEPS == 0:
                target.load_state_dict(policy.state_dict())

            if done or trunc:
                break

        rewards_all.append(total)

        if total > best_reward:
            best_reward = total
            torch.save(policy.state_dict(), os.path.join(run_dir, "best_model.pth"))

        if ep % 20 == 0:
            print(f"{run_name} | Episode {ep} | Reward {total:.1f}")

    torch.save(policy.state_dict(), os.path.join(run_dir, "last_model.pth"))
    plot_rewards(rewards_all, os.path.join(run_dir, "rewards.png"))
    save_rewards_csv(rewards_all, os.path.join(run_dir, "rewards.csv"))

    env.close()
    return rewards_all


# =========================================================
# EXPERIMENTS & FINAL TRAINING
# =========================================================
def run_experiments():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Έλεγχος Learning Rate
    for lr in [1e-2, 1e-3, 1e-4]:
        train(f"lr/lr_{lr}", lr, 128, 600, 300)

    # 2. Έλεγχος μεγέθους κρυφών επιπέδων (Hidden Size)
    for h in [64, 128, 256]:
        train(f"hidden/h_{h}", 1e-3, h, 600, 300)

    # 3. Έλεγχος ταχύτητας μείωσης του Epsilon (Epsilon Decay)
    for d in [100, 300, 600]:
        train(f"eps/decay_{d}", 1e-3, 128, d, 300)


def train_final_model():
    """Εκπαίδευση του τελικού μοντέλου βάσει των βέλτιστων υπερπαραμέτρων."""
    train(
        run_name="final_model",
        lr=1e-4,
        hidden=128,
        eps_decay=300,
        episodes=800
    )


if __name__ == "__main__":
    run_experiments()
    train_final_model()