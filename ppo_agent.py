import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

class ActorCritic(nn.Module):
    """
    Actor-Critic network for PPO with multiple output heads for different action components.
    """
    def __init__(self, state_size, num_territories):
        super(ActorCritic, self).__init__()
        self.num_territories = num_territories

        # Shared layers for both Actor and Critic
        self.fc1 = nn.Linear(state_size, 256)
        self.relu = nn.ReLU() # Using ReLU for hidden layers
        self.fc2 = nn.Linear(256, 128)

        # --- Actor Heads ---
        # Reinforcement Phase Actions:
        # 1. Choose a territory to reinforce (42 options)
        self.actor_reinforce_territory = nn.Linear(128, num_territories)
        # 2. Choose number of armies to place (indices 0-4 for 1-5 armies)
        self.actor_reinforce_armies = nn.Linear(128, 5)

        # Attack Phase Actions:
        # Attack from (42) x Attack to (42) x Number of dice (3)
        # This is a flattened output, we'll reshape it for sampling
        self.actor_attack = nn.Linear(128, num_territories * num_territories * 3)

        # Fortify Phase Actions:
        # Fortify from (42) x Fortify to (42) x Number of armies (indices 0-9 for 1-10 armies)
        self.actor_fortify = nn.Linear(128, num_territories * num_territories * 10)

        # General Actions (can be chosen in specific phases):
        # 1. Trade Cards: 0 (No Trade), 1 (Trade)
        self.actor_trade_cards = nn.Linear(128, 2)
        # 2. Phase Transition: 0 (End Reinforcement), 1 (End Attack), 2 (End Fortify)
        self.actor_phase_transition = nn.Linear(128, 3)

        # --- Critic Head ---
        self.critic = nn.Linear(128, 1)

    def forward(self, state):
        """
        Forward pass through the network.
        Returns a dictionary of logits for each action component and the critic value.
        """
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))

        logits = {
            "reinforce_t": self.actor_reinforce_territory(x),
            "reinforce_a": self.actor_reinforce_armies(x),
            "attack": self.actor_attack(x),
            "fortify": self.actor_fortify(x),
            "trade_cards": self.actor_trade_cards(x),
            "phase_transition": self.actor_phase_transition(x)
        }
        critic_value = self.critic(x)

        return logits, critic_value

class PPOAgent:
    """
    Proximal Policy Optimization (PPO) Agent implementation.
    Handles multi-headed action spaces and GAE for advantage calculation.
    """
    def __init__(self, state_size, num_territories, learning_rate=0.0003, gamma=0.99, clip_epsilon=0.2, gae_lambda=0.95):
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.gae_lambda = gae_lambda
        self.num_territories = num_territories

        self.actor_critic = ActorCritic(state_size, num_territories)
        
        # --- Device Selection for AMD GPU (DirectML) ---
        try:
            import torch_directml
            self.device = torch_directml.device()
            print(f"PPOAgent using device: {self.device} (DirectML)")
        except ImportError:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                print("PPOAgent using device: cuda")
            else:
                self.device = torch.device("cpu")
                print("PPOAgent using device: cpu")

        self.actor_critic.to(self.device)
        
        # Use SGD to avoid DirectML CPU fallback with Adam/AdamW (aten::lerp)
        self.optimizer = optim.SGD(self.actor_critic.parameters(), lr=learning_rate, momentum=0.9)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.9) # Learning rate scheduler
        
        # Buffer to store experiences: (state, action_dict, reward, next_state, log_prob_dict, value, done, env_phase)
        self.buffer = [] 

    def select_action(self, state, env_phase, action_masks=None):
        """
        Selects an action based on the current state and game phase.
        Samples from relevant action distributions.
        Returns the chosen action (as a dictionary), its log probabilities, and the state value.
        """
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits_dict, value = self.actor_critic(state_tensor)

        action_dict = {}
        log_prob_dict = {}
        
        # Helper to sample and store action/log_prob for a given logits tensor
        def _sample_and_store(logits_tensor, key_prefix, mask=None):
            if mask is not None:
                mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).to(self.device)
                # Set invalid actions to a very large negative number
                logits_tensor = logits_tensor + (mask_tensor - 1) * 1e9
            
            probs = torch.softmax(logits_tensor, dim=-1)
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            action_dict[key_prefix] = action.item()
            log_prob_dict[key_prefix] = log_prob.item()
            return action, log_prob, dist # Return dist for entropy calculation

        # Phase-specific action selection
        if env_phase == "reinforcement":
            # Trade cards decision
            mask_trade = action_masks["trade_cards"] if action_masks else None
            _sample_and_store(logits_dict["trade_cards"], "trade_cards", mask_trade)

            # Reinforce territory and armies
            mask_reinforce_t = action_masks["reinforce_t"] if action_masks else None
            _sample_and_store(logits_dict["reinforce_t"], "reinforce_t", mask_reinforce_t)
            
            mask_reinforce_a = action_masks["reinforce_a"] if action_masks else None
            _sample_and_store(logits_dict["reinforce_a"], "reinforce_a", mask_reinforce_a)

            # Phase transition (end reinforcement)
            # Mask phase transition (index 0 is end_reinforcement)
            # The mask for phase_transition is size 3: [end_reinf, end_attack, end_fortify]
            # But the actor head outputs 3 values. We should mask all 3 based on the phase?
            # Actually, the head is generic for "phase transition". 
            # But in reinforcement, only index 0 makes sense?
            # The original code didn't distinguish indices for phases in the output, just 3 outputs.
            # Let's assume index 0=end_reinf, 1=end_attack, 2=end_fortify.
            # So we should apply the full mask.
            mask_phase = action_masks["phase_transition"] if action_masks else None
            _sample_and_store(logits_dict["phase_transition"], "phase_transition", mask_phase)

        elif env_phase == "attack":
            # Attack Action (from_t, to_t, num_dice)
            # Flatten logits for sampling, then unflatten sampled index
            attack_logits_flat = logits_dict["attack"].view(-1)
            
            # Construct flat mask
            if action_masks:
                # Mask is (attacker, defender) matrix + dice mask?
                # The output head is (num_t * num_t * 3).
                # We have masks for attacker (num_t), defender (num_t, num_t), dice (3).
                # We need to combine them into a flat mask of size (num_t * num_t * 3).
                
                # Expand masks to match dimensions
                mask_attacker = torch.from_numpy(action_masks["attack_attacker"]).float().to(self.device) # (T)
                mask_defender = torch.from_numpy(action_masks["attack_defender"]).float().to(self.device) # (T, T)
                mask_dice = torch.from_numpy(action_masks["attack_dice"]).float().to(self.device)         # (3)
                
                # Combine: Mask is valid if attacker is valid AND defender is valid AND dice is valid
                # (T, 1, 1) * (T, T, 1) * (1, 1, 3) -> (T, T, 3)
                full_mask = mask_attacker.view(-1, 1, 1) * mask_defender.unsqueeze(2) * mask_dice.view(1, 1, -1)
                full_mask_flat = full_mask.view(-1).unsqueeze(0) # (1, T*T*3)
                
                # Apply mask
                attack_logits_flat = attack_logits_flat + (full_mask_flat - 1) * 1e9
            
            probs_attack = torch.softmax(attack_logits_flat, dim=-1)
            dist_attack = Categorical(probs_attack)
            action_attack_flat = dist_attack.sample()
            log_prob_attack = dist_attack.log_prob(action_attack_flat)
            
            # Convert flat index back to (attacker_idx, defender_idx, dice_idx) tuple
            action_attack_indices = np.unravel_index(action_attack_flat.item(), 
                                                     (self.num_territories, self.num_territories, 3))
            action_dict["attack"] = action_attack_indices
            log_prob_dict["attack"] = log_prob_attack.item()

            # Phase transition (end attack)
            mask_phase = action_masks["phase_transition"] if action_masks else None
            _sample_and_store(logits_dict["phase_transition"], "phase_transition", mask_phase)

        elif env_phase == "fortify":
            # Fortify action (from_t, to_t, num_armies)
            # Flatten logits for sampling, then unflatten sampled index
            fortify_logits_flat = logits_dict["fortify"].view(-1)
            
            # Construct flat mask
            if action_masks:
                mask_from = torch.from_numpy(action_masks["fortify_from"]).float().to(self.device) # (T)
                mask_to = torch.from_numpy(action_masks["fortify_to"]).float().to(self.device)     # (T, T)
                mask_armies = torch.from_numpy(action_masks["fortify_armies"]).float().to(self.device) # (10)
                
                # Combine: (T, 1, 1) * (T, T, 1) * (1, 1, 10) -> (T, T, 10)
                full_mask = mask_from.view(-1, 1, 1) * mask_to.unsqueeze(2) * mask_armies.view(1, 1, -1)
                full_mask_flat = full_mask.view(-1).unsqueeze(0)
                
                # Apply mask
                fortify_logits_flat = fortify_logits_flat + (full_mask_flat - 1) * 1e9

            probs_fortify = torch.softmax(fortify_logits_flat, dim=-1)
            dist_fortify = Categorical(probs_fortify)
            action_fortify_flat = dist_fortify.sample()
            log_prob_fortify = dist_fortify.log_prob(action_fortify_flat)
            
            # Convert flat index back to (from_idx, to_idx, armies_idx) tuple
            action_fortify_indices = np.unravel_index(action_fortify_flat.item(), 
                                                      (self.num_territories, self.num_territories, 10))
            action_dict["fortify"] = action_fortify_indices
            log_prob_dict["fortify"] = log_prob_fortify.item()

            # Phase transition (end fortify)
            mask_phase = action_masks["phase_transition"] if action_masks else None
            _sample_and_store(logits_dict["phase_transition"], "phase_transition", mask_phase)

        return action_dict, log_prob_dict, value.item()

    def store_transition(self, state, action_dict, reward, next_state, log_prob_dict, value, done, env_phase):
        """
        Stores a single transition in the agent's buffer, including the environment phase.
        """
        self.buffer.append((state, action_dict, reward, next_state, log_prob_dict, value, done, env_phase))

    def learn(self):
        """
        Performs a PPO learning update using the collected experiences in the buffer.
        Calculates policy and value losses for multi-headed actions.
        """
        if not self.buffer:
            return

        # Unpack experiences
        # Note: actions_dicts and log_probs_dicts are lists of dictionaries
        states, actions_dicts, rewards, next_states, log_probs_dicts, values, dones, env_phases = zip(*self.buffer)

        # Convert to tensors
        states = torch.tensor(np.array(states), dtype=torch.float).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float).to(self.device)
        values = torch.tensor(values, dtype=torch.float).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float).to(self.device) # True (1.0) if episode ended, False (0.0) otherwise

        # Calculate advantages (Generalized Advantage Estimation - GAE)
        advantages = torch.zeros_like(rewards)
        last_gae_lambda = 0
        
        with torch.no_grad():
            # Get values for next states to compute TD errors
            next_logits_dummy, next_values = self.actor_critic(next_states)
            next_values = next_values.squeeze()

        for t in reversed(range(len(rewards))):
            # If done, next_value is 0 as there's no future state
            next_value = next_values[t] * (1 - dones[t]) 
            delta = rewards[t] + self.gamma * next_value - values[t]
            advantages[t] = last_gae_lambda = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae_lambda

        # Normalize advantages (optional but common practice)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Perform PPO updates (multiple epochs over the collected data)
        for _ in range(5): # PPO epochs (hyperparameter)
            # Get new logits and values from the current policy
            new_logits_dict, new_values = self.actor_critic(states)
            new_values = new_values.squeeze()

            # Calculate policy loss (actor_loss)
            actor_loss = 0
            entropy_loss = 0

            for i in range(len(states)): # Iterate through each experience in the batch
                action_dict = actions_dicts[i]
                old_log_prob_dict = log_probs_dicts[i]
                current_env_phase = env_phases[i]

                # --- Calculate Actor Loss for each relevant action component ---
                if current_env_phase == "reinforcement":
                    # Trade Cards
                    new_probs_trade = torch.softmax(new_logits_dict["trade_cards"][i], dim=-1)
                    new_dist_trade = Categorical(new_probs_trade)
                    new_log_prob_trade = new_dist_trade.log_prob(torch.tensor(action_dict["trade_cards"]).to(self.device))
                    ratio_trade = torch.exp(new_log_prob_trade - torch.tensor(old_log_prob_dict["trade_cards"]).to(self.device))
                    surr1_trade = ratio_trade * advantages[i]
                    surr2_trade = torch.clamp(ratio_trade, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages[i]
                    actor_loss -= torch.min(surr1_trade, surr2_trade)
                    entropy_loss += new_dist_trade.entropy()

                    # Reinforce Territory
                    new_probs_reinforce_t = torch.softmax(new_logits_dict["reinforce_t"][i], dim=-1)
                    new_dist_reinforce_t = Categorical(new_probs_reinforce_t)
                    new_log_prob_reinforce_t = new_dist_reinforce_t.log_prob(torch.tensor(action_dict["reinforce_t"]).to(self.device))
                    ratio_reinforce_t = torch.exp(new_log_prob_reinforce_t - torch.tensor(old_log_prob_dict["reinforce_t"]).to(self.device))
                    surr1_reinforce_t = ratio_reinforce_t * advantages[i]
                    surr2_reinforce_t = torch.clamp(ratio_reinforce_t, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages[i]
                    actor_loss -= torch.min(surr1_reinforce_t, surr2_reinforce_t)
                    entropy_loss += new_dist_reinforce_t.entropy()

                    # Reinforce Armies
                    new_probs_reinforce_a = torch.softmax(new_logits_dict["reinforce_a"][i], dim=-1)
                    new_dist_reinforce_a = Categorical(new_probs_reinforce_a)
                    new_log_prob_reinforce_a = new_dist_reinforce_a.log_prob(torch.tensor(action_dict["reinforce_a"]).to(self.device))
                    ratio_reinforce_a = torch.exp(new_log_prob_reinforce_a - torch.tensor(old_log_prob_dict["reinforce_a"]).to(self.device))
                    surr1_reinforce_a = ratio_reinforce_a * advantages[i]
                    surr2_reinforce_a = torch.clamp(ratio_reinforce_a, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages[i]
                    actor_loss -= torch.min(surr1_reinforce_a, surr2_reinforce_a)
                    entropy_loss += new_dist_reinforce_a.entropy()

                elif current_env_phase == "attack":
                    # Attack Action (flattened)
                    attack_logits_flat = new_logits_dict["attack"][i].view(-1)
                    new_probs_attack = torch.softmax(attack_logits_flat, dim=-1)
                    new_dist_attack = Categorical(new_probs_attack)
                    # Convert sampled action indices back to flat index for log_prob
                    sampled_attack_flat_idx = np.ravel_multi_index(action_dict["attack"], (self.num_territories, self.num_territories, 3))
                    new_log_prob_attack = new_dist_attack.log_prob(torch.tensor(sampled_attack_flat_idx).to(self.device))
                    ratio_attack = torch.exp(new_log_prob_attack - torch.tensor(old_log_prob_dict["attack"]).to(self.device))
                    surr1_attack = ratio_attack * advantages[i]
                    surr2_attack = torch.clamp(ratio_attack, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages[i]
                    actor_loss -= torch.min(surr1_attack, surr2_attack)
                    entropy_loss += new_dist_attack.entropy()

                elif current_env_phase == "fortify":
                    # Fortify Action (flattened)
                    fortify_logits_flat = new_logits_dict["fortify"][i].view(-1)
                    new_probs_fortify = torch.softmax(fortify_logits_flat, dim=-1)
                    new_dist_fortify = Categorical(new_probs_fortify)
                    # Convert sampled action indices back to flat index for log_prob
                    sampled_fortify_flat_idx = np.ravel_multi_index(action_dict["fortify"], (self.num_territories, self.num_territories, 10))
                    new_log_prob_fortify = new_dist_fortify.log_prob(torch.tensor(sampled_fortify_flat_idx).to(self.device))
                    ratio_fortify = torch.exp(new_log_prob_fortify - torch.tensor(old_log_prob_dict["fortify"]).to(self.device))
                    surr1_fortify = ratio_fortify * advantages[i]
                    surr2_fortify = torch.clamp(ratio_fortify, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages[i]
                    actor_loss -= torch.min(surr1_fortify, surr2_fortify)
                    entropy_loss += new_dist_fortify.entropy()

                # Phase Transition (always present)
                new_probs_phase_trans = torch.softmax(new_logits_dict["phase_transition"][i], dim=-1)
                new_dist_phase_trans = Categorical(new_probs_phase_trans)
                new_log_prob_phase_trans = new_dist_phase_trans.log_prob(torch.tensor(action_dict["phase_transition"]).to(self.device))
                ratio_phase_trans = torch.exp(new_log_prob_phase_trans - torch.tensor(old_log_prob_dict["phase_transition"]).to(self.device))
                surr1_phase_trans = ratio_phase_trans * advantages[i]
                surr2_phase_trans = torch.clamp(ratio_phase_trans, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages[i]
                actor_loss -= torch.min(surr1_phase_trans, surr2_phase_trans)
                entropy_loss += new_dist_phase_trans.entropy()

            # Average actor loss and entropy loss over the batch
            actor_loss /= len(states)
            entropy_loss /= len(states)

            # Critic loss (Huber Loss / Smooth L1 Loss)
            critic_loss = nn.functional.smooth_l1_loss(new_values, values)

            # Total loss: sum of actor loss, scaled critic loss, and entropy regularization
            # Entropy regularization encourages exploration
            self.optimizer.zero_grad()
            loss.backward()
            # Gradient clipping to prevent exploding gradients
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5) 
            self.optimizer.step()
            self.scheduler.step() # Update learning rate

        self.buffer = [] # Clear buffer after learning

    def pretrain_step(self, state, target_action_dict, env_phase):
        """
        Perform a supervised learning step to mimic a target action (Behavior Cloning).
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.actor_critic.train()

        logits, _ = self.actor_critic(state_tensor)
        loss = 0
        
        # Calculate Cross Entropy Loss for each action component relevant to the phase
        if env_phase == "reinforcement":
            # Reinforce Territory
            target_t = torch.tensor([target_action_dict["reinforce_t"]], dtype=torch.long).to(self.device)
            loss += nn.CrossEntropyLoss()(logits["reinforce_t"], target_t)
            
            # Reinforce Armies
            target_a = torch.tensor([target_action_dict["reinforce_a"]], dtype=torch.long).to(self.device)
            loss += nn.CrossEntropyLoss()(logits["reinforce_a"], target_a)
            
            # Trade Cards
            target_trade = torch.tensor([target_action_dict["trade_cards"]], dtype=torch.long).to(self.device)
            loss += nn.CrossEntropyLoss()(logits["trade_cards"], target_trade)
            
        elif env_phase == "attack":
            # Attack (flattened index)
            if "attack" in target_action_dict:
                attacker, defender, dice = target_action_dict["attack"]
                # Convert tuple to flattened index: attacker * (num_t * 3) + defender * 3 + dice
                flat_idx = attacker * (self.num_territories * 3) + defender * 3 + dice
                target_attack = torch.tensor([flat_idx], dtype=torch.long).to(self.device)
                loss += nn.CrossEntropyLoss()(logits["attack"], target_attack)
            
            # Phase Transition (End Attack?)
            target_phase = torch.tensor([target_action_dict["phase_transition"]], dtype=torch.long).to(self.device)
            loss += nn.CrossEntropyLoss()(logits["phase_transition"], target_phase)

        elif env_phase == "fortify":
            # Fortify (flattened index)
            if "fortify" in target_action_dict:
                from_t, to_t, armies = target_action_dict["fortify"]
                # Convert tuple to flattened index: from * (num_t * 10) + to * 10 + armies
                flat_idx = from_t * (self.num_territories * 10) + to_t * 10 + armies
                target_fortify = torch.tensor([flat_idx], dtype=torch.long).to(self.device)
                loss += nn.CrossEntropyLoss()(logits["fortify"], target_fortify)
            
            # Phase Transition (End Fortify?)
            target_phase = torch.tensor([target_action_dict["phase_transition"]], dtype=torch.long).to(self.device)
            loss += nn.CrossEntropyLoss()(logits["phase_transition"], target_phase)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def pretrain_batch(self, states, target_actions, env_phases):
        """
        Perform a supervised learning step on a batch of data.
        """
        # Convert lists to tensors
        state_tensor = torch.FloatTensor(np.array(states)).to(self.device)
        self.actor_critic.train()

        logits_dict, _ = self.actor_critic(state_tensor)
        loss = 0
        
        # We need to handle mixed phases in a batch, or assume the batch is sorted/masked.
        # Simplest approach: Iterate through the batch indices and calculate loss.
        # Vectorized approach is harder with mixed phases. Let's stick to iteration for loss calculation 
        # but forward pass is batched.
        
        for i in range(len(states)):
            env_phase = env_phases[i]
            target_action_dict = target_actions[i]
            
            if env_phase == "reinforcement":
                loss += nn.CrossEntropyLoss()(logits_dict["reinforce_t"][i].unsqueeze(0), torch.tensor([target_action_dict["reinforce_t"]], dtype=torch.long).to(self.device))
                loss += nn.CrossEntropyLoss()(logits_dict["reinforce_a"][i].unsqueeze(0), torch.tensor([target_action_dict["reinforce_a"]], dtype=torch.long).to(self.device))
                loss += nn.CrossEntropyLoss()(logits_dict["trade_cards"][i].unsqueeze(0), torch.tensor([target_action_dict["trade_cards"]], dtype=torch.long).to(self.device))
                
            elif env_phase == "attack":
                if "attack" in target_action_dict:
                    attacker, defender, dice = target_action_dict["attack"]
                    flat_idx = attacker * (self.num_territories * 3) + defender * 3 + dice
                    loss += nn.CrossEntropyLoss()(logits_dict["attack"][i].unsqueeze(0), torch.tensor([flat_idx], dtype=torch.long).to(self.device))
                
                loss += nn.CrossEntropyLoss()(logits_dict["phase_transition"][i].unsqueeze(0), torch.tensor([target_action_dict["phase_transition"]], dtype=torch.long).to(self.device))

            elif env_phase == "fortify":
                if "fortify" in target_action_dict:
                    from_t, to_t, armies = target_action_dict["fortify"]
                    flat_idx = from_t * (self.num_territories * 10) + to_t * 10 + armies
                    loss += nn.CrossEntropyLoss()(logits_dict["fortify"][i].unsqueeze(0), torch.tensor([flat_idx], dtype=torch.long).to(self.device))
                
                loss += nn.CrossEntropyLoss()(logits_dict["phase_transition"][i].unsqueeze(0), torch.tensor([target_action_dict["phase_transition"]], dtype=torch.long).to(self.device))

        # Average loss over batch
        loss = loss / len(states)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def save_model(self, path):
        """Saves the actor-critic model's state dictionary."""
        torch.save(self.actor_critic.state_dict(), path)

    def load_model(self, path):
        """Loads the actor-critic model's state dictionary."""
        self.actor_critic.load_state_dict(torch.load(path))
        # Ensure the model is in evaluation mode after loading if not training
        self.actor_critic.eval() 
