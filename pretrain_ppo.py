import numpy as np
from risk_game_environment import RiskEnv
from ppo_agent import PPOAgent
from rule_based_agents import StrategicAgent, RandomAgent
import torch
import os

def pretrain(num_episodes=50, batch_size=64):
    env = RiskEnv(num_players=4)
    state_size = len(env.reset())
    num_territories = len(env.territory_names)
    
    # Initialize PPO Agent (Student)
    ppo_agent = PPOAgent(state_size, num_territories)
    
    # Initialize Strategic Agent (Teacher)
    teacher = StrategicAgent(player_id=0, env=env) 
    
    print(f"Starting Pre-training (Imitation Learning) for {num_episodes} episodes with batch size {batch_size}...", flush=True)
    
    total_loss = 0
    steps = 0
    
    # Batch buffers
    batch_states = []
    batch_actions = []
    batch_phases = []
    
    try:
        for episode in range(num_episodes):
            state = env.reset()
            done = False
            
            while not done:
                current_player_idx = env.current_player
                
                # Skip eliminated players
                if current_player_idx not in env.active_players:
                    env._switch_player()
                    continue
                
                # Update teacher's player_id
                teacher.player_id = current_player_idx
                
                # Get Expert Action
                try:
                    target_action_dict, _, _ = teacher.select_action(state, env.turn_phase)
                except Exception as e:
                    env._switch_player()
                    continue

                # Collect data for batch
                batch_states.append(state)
                batch_actions.append(target_action_dict)
                batch_phases.append(env.turn_phase)
                
                # Update if batch is full
                if len(batch_states) >= batch_size:
                    loss = ppo_agent.pretrain_batch(batch_states, batch_actions, batch_phases)
                    total_loss += loss
                    steps += 1
                    
                    # Clear buffers
                    batch_states = []
                    batch_actions = []
                    batch_phases = []
                
                # Step Environment
                env_action = env.decode_action(target_action_dict)
                next_state, reward, done, info = env.step(env_action)
                state = next_state
            
            # Process remaining items in batch at end of episode (optional, but good for completeness)
            if len(batch_states) > 0:
                loss = ppo_agent.pretrain_batch(batch_states, batch_actions, batch_phases)
                total_loss += loss
                steps += 1
                batch_states = []
                batch_actions = []
                batch_phases = []
                
            if (episode + 1) % 1 == 0: # Print every episode
                avg_loss = total_loss / steps if steps > 0 else 0
                print(f"Episode {episode + 1}/{num_episodes} | Avg Loss: {avg_loss:.4f}", flush=True)
                total_loss = 0
                steps = 0
                
    except KeyboardInterrupt:
        print("Pre-training interrupted.", flush=True)
    finally:
        print("Pre-training complete/stopped.", flush=True)
        save_path = "risk_ppo_agent_pretrained.pth"
        ppo_agent.save_model(save_path)
        print(f"Pre-trained model saved to {save_path}", flush=True)

if __name__ == "__main__":
    pretrain()
