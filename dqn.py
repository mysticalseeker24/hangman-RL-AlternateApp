import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
import gc
import random

logger = logging.getLogger('root')

class DQN(nn.Module):
    def __init__(self, device=None):
        super(DQN, self).__init__()
        num_classes = 26
        num_layers = 1
        input_size = 27
        hidden_size = 32  # original size
        seq_length = 27

        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length

        # Standard LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc_1 = nn.Linear(hidden_size+26, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        self.to(self.device)

    def forward(self, word, actions):
        word = word.to(self.device) if not isinstance(word, torch.Tensor) or word.device != self.device else word
        actions = actions.to(self.device) if not isinstance(actions, torch.Tensor) or actions.device != self.device else actions
        word = word.float() if word.dtype != torch.float32 else word
        output, (hn, cn) = self.lstm(word)
        hn = hn.view(-1, self.hidden_size)
        combined = torch.cat((hn, actions), 1)
        out = self.relu(self.fc_1(combined))
        out = self.fc(out)
        return out

    def select_action(self, state, available_actions=None, epsilon=0.1):
        # Standard epsilon-greedy action selection
        if available_actions is None:
            available_actions = list(range(26))
        if np.random.rand() < epsilon:
            return np.random.choice(available_actions)
        with torch.no_grad():
            q_values = self.forward(state[0], state[1])
            q_values = q_values.cpu().numpy().flatten()
            q_values = [q_values[a] for a in available_actions]
            return available_actions[int(np.argmax(q_values))]
        with torch.no_grad():
            # Handle various input types and move to device
            if isinstance(state, np.ndarray):
                state = torch.from_numpy(state).to(self.device, non_blocking=True)
            elif isinstance(state, list):
                state = torch.tensor(state, dtype=torch.float).to(self.device, non_blocking=True)
            
            # Use half precision when on GPU to save memory
            if self.device.type == 'cuda':
                state = state.half()
                
            # Process small batches if needed to save memory
            if hasattr(state, 'shape') and len(state.shape) > 0 and state.shape[0] > 16:  # Process in chunks if batch is large
                results = []
                for i in range(0, state.shape[0], 16):
                    batch = state[i:i+16].unsqueeze(0) if state[i:i+16].dim() == 1 else state[i:i+16]
                    results.append(self(batch))
                q_values = torch.cat(results, 0)
            else:
                # Add batch dimension if needed
                if state.dim() == 1:
                    state = state.unsqueeze(0)
                # Process state through model and select best action
                q_values = self(state)
            
            # If available_actions is provided, filter out unavailable actions
            if available_actions is not None:
                # Set q-values of unavailable actions to negative infinity
                mask = torch.ones(self.num_classes, device=self.device) * float('-inf')
                mask[available_actions] = 0
                q_values = q_values + mask
            
            # Clean up before returning
            result = q_values.max(1)[1].item()
            del q_values, state
            
            # Return action with highest Q-value
            return result
    
    def get_action(self, state, guessed_letters):
        """Helper method to get action prediction with proper device handling"""
        # Convert inputs to tensors and move to device
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        guessed_tensor = torch.tensor(guessed_letters, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            self.eval()
            q_values = self(state_tensor, guessed_tensor)
            action = q_values.argmax(dim=1).item()
        
        self.train()
        return action
