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


def preprocess_data(X, categorical_columns):
    """Preprocess numerical and categorical features."""
    cat_indices = {col: X[col].astype('category').cat.codes for col in categorical_columns}
    for col in categorical_columns:
        X[col] = cat_indices[col]
    return X, {col: len(X[col].unique()) for col in categorical_columns}


class QNetwork(nn.Module):
    def __init__(self, input_size, output_size, cat_dims, embed_dim=4):
        super(QNetwork, self).__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(cat_dim, embed_dim) for cat_dim in cat_dims
        ])
        self.embed_output_size = len(cat_dims) * embed_dim
        self.fc_input_size = input_size - len(cat_dims) + self.embed_output_size

        self.layers = nn.Sequential(
            nn.Linear(self.fc_input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )
    
    def forward(self, x, cat_features):
        if len(cat_features.shape) == 1:
            cat_features = cat_features.unsqueeze(0)
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        embedded = [emb(cat_features[:, i]) for i, emb in enumerate(self.embeddings)]
        embedded = torch.cat(embedded, dim=1)
        x = torch.cat([x, embedded], dim=1)
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
    categorical_columns = dataset.CATEGORICAL_COLUMNS
    X_train, X_test, y_train, y_test = dataset.get_train_test_split()
    X_train, cat_dims = preprocess_data(X_train, categorical_columns)
    X_test, _ = preprocess_data(X_test, categorical_columns)

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
    q_network = QNetwork(state_size, action_size, list(cat_dims.values()))
    target_network = QNetwork(state_size, action_size, list(cat_dims.values()))
    target_network.load_state_dict(q_network.state_dict())
    target_network.eval()

    optimizer = optim.Adam(q_network.parameters(), lr=config['algorithm_params']['learning_rate'])
    criterion = nn.MSELoss()

    # Training loop
    episodes = config['algorithm_params']['episodes']
    for episode in range(episodes):
        total_reward = 0
        for idx in range(len(X_train)):
            num_features = torch.FloatTensor(X_train.iloc[idx, ~X_train.columns.isin(categorical_columns)].values)
            cat_features = torch.LongTensor(X_train.iloc[idx, X_train.columns.isin(categorical_columns)].values)
            true_label = int(y_train.iloc[idx])

            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.randint(0, 1)
            else:
                with torch.no_grad():
                    q_values = q_network(num_features, cat_features)
                    action = torch.argmax(q_values).item()

            # Calculate reward based on confusion matrix
            reward = 1 if action == true_label else -1
            next_state = num_features, cat_features  # Static environment
            done = idx == len(X_train) - 1

            replay_buffer.add((num_features, cat_features), action, reward, next_state, done)
            total_reward += reward

            # Update Q-network
            if len(replay_buffer) > batch_size:
                minibatch = replay_buffer.sample(batch_size)
                states, actions, rewards, next_states, dones = zip(*minibatch)
                num_states = torch.stack([s[0] for s in states])
                cat_states = torch.stack([s[1] for s in states])
                num_next_states = torch.stack([s[0] for s in next_states])
                cat_next_states = torch.stack([s[1] for s in next_states])
                actions = torch.LongTensor(actions)
                rewards = torch.FloatTensor(rewards)
                dones = torch.BoolTensor(dones)

                # Compute targets
                q_values = q_network(num_states, cat_states).gather(1, torch.LongTensor(actions).unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    target_q_values = target_network(num_next_states, cat_next_states)
                    max_next_q_values = torch.max(target_q_values, dim=1)[0]
                    targets = rewards + gamma * max_next_q_values * ~dones

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
            num_features = torch.FloatTensor(X_test.iloc[idx, ~X_test.columns.isin(categorical_columns)].values)
            cat_features = torch.LongTensor(X_test.iloc[idx, X_test.columns.isin(categorical_columns)].values)
            q_values = q_network(num_features, cat_features).squeeze(0)
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
