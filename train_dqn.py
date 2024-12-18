import os
import yaml
import json
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
)
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

from datasets import get_dataset_from_name
from eval_utils import load_config, save_logs

class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        return self.layers(x)

class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

def train_and_evaluate(config, log_dir):
    # Load dataset
    dataset_name = config['dataset']
    dataset = get_dataset_from_name(dataset_name)
    X_train, X_test, y_train, y_test = dataset.get_train_test_split()

    # Keep only numeric attributes
    X_train = X_train.select_dtypes(include=['number'])
    X_test = X_test.select_dtypes(include=['number'])

    state_size = X_train.shape[1]
    action_size = 2  # Binary classification (0 and 1)
    gamma = config['algorithm_params']['gamma']
    epsilon = config['algorithm_params']['epsilon']
    epsilon_min = config['algorithm_params']['epsilon_min']
    epsilon_decay = config['algorithm_params']['epsilon_decay']
    batch_size = config['algorithm_params']['batch_size']
    target_update_freq = config['algorithm_params']['target_update_freq']
    replay_buffer = ReplayBuffer(max_size=10000)

    # Initialize Q-Networks
    q_network = QNetwork(state_size, action_size)
    target_network = QNetwork(state_size, action_size)
    target_network.load_state_dict(q_network.state_dict())
    target_network.eval()

    optimizer = optim.Adam(q_network.parameters(), lr=config['algorithm_params']['learning_rate'])
    criterion = nn.MSELoss()

    # Training loop
    episodes = config['algorithm_params']['episodes']
    for episode in range(episodes):
        total_reward = 0
        for idx in range(len(X_train)):
            state = torch.FloatTensor(X_train.iloc[idx].values)
            true_label = int(y_train.iloc[idx])

            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.randint(0, 1)
            else:
                with torch.no_grad():
                    q_values = q_network(state)
                    action = torch.argmax(q_values).item()

            # Calculate reward based on confusion matrix
            reward = 1 if action == true_label else -1
            next_state = state  # Static environment
            done = idx == len(X_train) - 1

            replay_buffer.add(state, action, reward, next_state, done)
            total_reward += reward

            # Update Q-network
            if len(replay_buffer) > batch_size:
                minibatch = replay_buffer.sample(batch_size)
                states, actions, rewards, next_states, dones = zip(*minibatch)
                states = torch.stack(states)
                next_states = torch.stack(next_states)
                actions = torch.LongTensor(actions)
                rewards = torch.FloatTensor(rewards)
                dones = torch.BoolTensor(dones)

                # Compute targets
                with torch.no_grad():
                    target_q_values = target_network(next_states)
                    max_next_q_values = torch.max(target_q_values, dim=1)[0]
                    targets = rewards + gamma * max_next_q_values * ~dones

                # Compute current Q-values
                q_values = q_network(states)
                q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

                # Update Q-network
                loss = criterion(q_values, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Update target network
        if episode % target_update_freq == 0:
            target_network.load_state_dict(q_network.state_dict())

        # Decay epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        print(f"Episode: {episode+1}, Total Reward: {total_reward}, Epsilon: {epsilon:.4f}")

    # Evaluate the model
    y_pred_proba = []
    with torch.no_grad():
        for idx in range(len(X_test)):
            state = torch.FloatTensor(X_test.iloc[idx].values)
            q_values = q_network(state)
            proba = torch.softmax(q_values, dim=0)[1].item()
            y_pred_proba.append(proba)

    y_pred = [1 if p > 0.5 else 0 for p in y_pred_proba]
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_pred_proba)
    }

    return q_network, metrics, y_test, y_pred_proba

def main(config_path):
    config = load_config(config_path)
    log_dir = save_logs(config, {}, config['dataset'], [], [])
    model, metrics, y_test, y_pred_proba = train_and_evaluate(config, log_dir)
    save_logs(config, metrics, config['dataset'], y_test, y_pred_proba, log_dir)
    print(f"Training complete. Logs saved in {os.path.join('logs', config['dataset'])}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train_dqn.py <config_path>")
    else:
        main(sys.argv[1])
