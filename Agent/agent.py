import os
import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim


class DQNAgent:
    def __init__(self, state_size=49, action_size=49):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = nn.Sequential(
            nn.Linear(49, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 49),
        ).to(self.device)

        self.target_model = nn.Sequential(
            nn.Linear(49, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 49),
        ).to(self.device)

        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()

    def act(self, state, valid_moves):
        state = torch.FloatTensor(state.flatten()).to(self.device)
        if np.random.rand() <= self.epsilon:
            return random.choice(valid_moves)
        else:
            with torch.no_grad():
                q_values = self.model(state)
            q_values = q_values.cpu().numpy().reshape(7, 7)
            valid_q_values = q_values[tuple(valid_moves.T)]
            max_idx = np.argmax(valid_q_values)
            return valid_moves[max_idx]

    def remember(self, state, action, reward, next_state, done):
        self.memory.append(
            (state.flatten(), action, reward, next_state.flatten(), done)
        )

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        targets = self.model(states).detach().clone()

        for i in range(self.batch_size):
            if dones[i]:
                targets[i, actions[i][0] * 7 + actions[i][1]] = rewards[i]
            else:
                q_next = torch.max(self.target_model(next_states[i]))
                targets[i, actions[i][0] * 7 + actions[i][1]] = (
                    rewards[i] + self.gamma * q_next
                )

        current_q = self.model(states)
        loss = self.loss_fn(current_q, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    @staticmethod
    def save_model(agent, episode, save_dir="saved_models"):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(
            {
                "episode": episode,
                "model_state_dict": agent.model.state_dict(),
                "target_model_state_dict": agent.target_model.state_dict(),
                "optimizer_state_dict": agent.optimizer.state_dict(),
                "epsilon": agent.epsilon,
            },
            os.path.join(save_dir, f"checkpoint_episode_{episode}.pt"),
        )

        print(f"Model saved at episode {episode} to {save_dir}")

    @staticmethod
    def load_model(agent, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        agent.model.load_state_dict(checkpoint["model_state_dict"])
        agent.target_model.load_state_dict(checkpoint["target_model_state_dict"])
        agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        agent.epsilon = checkpoint["epsilon"]
        start_episode = checkpoint["episode"] + 1
        print(f"Loaded from {checkpoint_path}. Resuming from {start_episode}")
        return start_episode
