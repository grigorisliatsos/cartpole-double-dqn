# =========================================================
# PLAY SCRIPT - Επίδειξη του εκπαιδευμένου πράκτορα
# =========================================================

import os
import time
import torch
import torch.nn as nn
import gymnasium as gym



class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
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



env = gym.make("CartPole-v1", render_mode="human")

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

model = DQN(state_dim, action_dim, hidden_dim=128)
device = torch.device("cpu")
model.to(device)
MODEL_PATH = os.path.join("results_cartpole_dqn", "final_model", "best_model.pth")

try:
    # ΔΙΟΡΘΩΣΗ: Προσθήκη weights_only=True για συμβατότητα με PyTorch 2.x+
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    )
    print(f"✓ Μοντέλο φορτώθηκε επιτυχώς από: {MODEL_PATH}")
except FileNotFoundError:
    print(
        f"[ΣΦΑΛΜΑ] Το αρχείο δεν βρέθηκε: {MODEL_PATH}\n"
        "Τρέξε πρώτα το main.py για να ολοκληρωθεί η εκπαίδευση."
    )
    env.close()
    raise SystemExit(1)

model.eval()

for episode in range(5):
    state, _ = env.reset()
    total_reward = 0

    while True:
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            q_values = model(state_tensor)
            action = int(torch.argmax(q_values))

        state, reward, done, truncated, _ = env.step(action)
        total_reward += reward

        time.sleep(0.02)

        if done or truncated:
            break

    print(f"Episode {episode + 1} | Total Reward: {total_reward}")

env.close()