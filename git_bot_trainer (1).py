import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import torch
import signal
import sys
import os

# Import classes from our internal environment and PPO agent
from risk_game_environment import RiskEnv
from ppo_agent import PPOAgent
# Import the new GitBot
from git_bot import GitBot

def save_and_exit(signum, frame):
    print("\nTraining interrupted. Saving model...")
    for i, agent in enumerate(agents_global):
        # Only save PPO agents, as rule-based agents don't have a learnable model
        if isinstance(agent, PPOAgent):
            agent.save_model(f"risk_ppo_agent_player{i}_interrupted_training_gitbot.pth")
    sys.exit(0)


# --- Main Training Loop ---
if __name__ == "__main__":
    num_players = 4
    env = RiskEnv(num_players=num_players)
    sample_state = env.reset()
    state_size = len(sample_state)
    num_territories = len(env.territory_names)

    # Initialize agents for TRAINING against GitBots
    # Player 0: Our Aggressive PPO learning agent
    # Players 1, 2, 3: GitBot instances (mimicking external bot's greedy strategy)
    agents = [
        PPOAgent(state_size, num_territories),
        GitBot(player_id=1, env=env),
        GitBot(player_id=2, env=env),
        GitBot(player_id=3, env=env)
    ]
    agents_global = agents # Make accessible to signal handler

    # --- Load Existing Model for PPO Agent if Present ---
    start_episode = 0
    # Only load for the PPO agent (Player 0)
    load_path_ppo = f"risk_ppo_agent_player0_interrupted_training_gitbot.pth"
    if os.path.exists(load_path_ppo):
        agents[0].load_model(load_path_ppo)
        print(f"Resuming training for PPO Agent (Player 0) from saved model.")
        # Optionally, you could try to infer the episode number from the filename
        # to continue logging correctly. For simplicity here, we restart the episode counter.

    num_episodes = 50000
    max_steps_per_episode = 2000
    batch_size = 2048
    save_interval = 5000 # Save model every N episodes
    summary_interval = 1000 # Print action summary every N episodes

    all_episode_rewards = []
    win_rates = deque(maxlen=100) # Track win rate over last 100 episodes for PPO Agent (Player 0)
    episode_lengths = deque(maxlen=100)
    ppo_territories_controlled = deque(maxlen=100)
    ppo_total_armies = deque(maxlen=100)
    gitbot_territories_controlled = [deque(maxlen=100) for _ in range(num_players - 1)]
    gitbot_total_armies = [deque(maxlen=100) for _ in range(num_players - 1)]

    # Initialize total action counts for summary
    total_action_counts = {
        "attack": 0, "reinforce": 0, "fortify": 0, "trade_cards": 0,
        "conquer": 0, "end_reinforcement": 0, "end_attack": 0, "end_fortify": 0,
        "skip_turn": 0 # For eliminated players' turns
    }

    print(f"Starting TRAINING with {num_players} AI agents (PPO vs GitBots)...")
    print(f"State Size: {state_size}, Num Territories: {num_territories}")

    # --- Register Signal Handler ---
    signal.signal(signal.SIGINT, save_and_exit)

    for episode in range(start_episode, num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        steps_in_episode = 0
        
        # Reset PPO agent's buffer at the start of each episode
        # Rule-based agents don't have buffers
        for agent in agents:
            if isinstance(agent, PPOAgent):
                agent.buffer = []

        episode_action_counts = {
            "attack": 0, "reinforce": 0, "fortify": 0, "trade_cards": 0,
            "conquer": 0, "end_reinforcement": 0, "end_attack": 0, "end_fortify": 0,
            "skip_turn": 0
        }

        # Track previous state for reward shaping (PPO Agent - Player 0)
        prev_territories = len(env.player_states[0]['territories'])
        prev_armies = sum(env.player_states[0]['armies'].values())

        while not done and steps_in_episode < max_steps_per_episode:
            current_player_agent_idx = env.current_player
            current_agent = agents[current_player_agent_idx]

            # If the current player has been eliminated, skip their turn
            if current_player_agent_idx not in env.active_players:
                env._switch_player()
                steps_in_episode += 1
                episode_action_counts["skip_turn"] += 1
                continue

            # Agent selects an action based on its type
            if isinstance(current_agent, PPOAgent):
                agent_action_dict, log_prob_dict, value = current_agent.select_action(state, env.turn_phase)
                store_transition = True
            else: # GitBot (rule-based)
                agent_action_dict, log_prob_dict, value = current_agent.select_action(state, env.turn_phase)
                store_transition = False # GitBot does not store transitions

            # Translate agent's action to environment's action format
            env_action = env.decode_action(agent_action_dict)

            # Step the environment
            next_state, reward, done, info = env.step(env_action)

            # Check if PPO agent (Player 0) is eliminated
            if 0 not in env.active_players:
                done = True
                # Optional: Add a large penalty for elimination if not already handled by the environment
                # reward -= 1.0 


            # Accumulate action counts for the episode (Removed - handled at end of episode)
            # for action_type, count in info["action_counts"].items():
            #     episode_action_counts[action_type] += count

            # --- Adjust Rewards to Encourage Actions (only for PPO Agent) ---
            # --- Adjust Rewards to Encourage Actions (only for PPO Agent) ---
            if isinstance(current_agent, PPOAgent):
                # 1. Survival Reward
                reward += 0.001

                # 2. Territory Delta Reward
                current_territories = len(env.player_states[0]['territories'])
                territory_delta = current_territories - prev_territories
                reward += territory_delta * 0.5 # Significant reward/penalty for gaining/losing ground
                prev_territories = current_territories

                # 3. Army Delta Reward
                current_armies = sum(env.player_states[0]['armies'].values())
                army_delta = current_armies - prev_armies
                reward += army_delta * 0.005 # Reward for growing army count, penalty for losing
                prev_armies = current_armies

                # 4. Damage Dealt Reward
                defenders_lost = info.get("defenders_lost", 0)
                if defenders_lost > 0:
                    reward += defenders_lost * 0.05 # Reward for killing enemy armies

                # 5. Action Validity/Effectiveness
                # Removed "participation award" to prevent action milking.
                # if env_action.get("phase") not in ["end_attack", "end_fortify", "end_reinforcement"]:
                #     reward += 0.005 
                
                # if env_action.get("phase") == "attack" and reward > 0.0: 
                #     pass # Already rewarded via damage/territory delta
                # elif env_action.get("phase") not in ["end_attack", "end_fortify", "end_reinforcement"] and reward == 0.006: # 0.001 (survival) + 0.005 (valid)
                #     reward -= 0.002 # Slight penalty if action didn't change state (besides survival/validity)

                # Store transition for the current PPO agent
                if store_transition:
                    current_agent.store_transition(
                        state, agent_action_dict, reward, next_state, log_prob_dict, value, done, env.turn_phase
                    )

            state = next_state
            total_reward += reward
            steps_in_episode += 1

            # Perform learning update if PPO agent's buffer is full
            if isinstance(current_agent, PPOAgent) and len(current_agent.buffer) >= batch_size:
                current_agent.learn()

        # Episode ends
        all_episode_rewards.append(total_reward)
        episode_lengths.append(steps_in_episode)

        # Track win rates for Player 0 (our PPO Agent)
        if env.winner is not None:
            if env.winner == 0: # Assuming Player 0 is the PPO AI we are tracking
                win_rates.append(1)
            else:
                win_rates.append(0)
        else:
            win_rates.append(0) # No winner (e.g., max steps reached)

        # Track territories and armies at the end of the episode
        ppo_territories = len(env.player_states[0]['territories'])
        ppo_armies = sum(env.player_states[0]['armies'].values())
        ppo_territories_controlled.append(ppo_territories)
        ppo_total_armies.append(ppo_armies)

        for i in range(1, num_players):
            gitbot_territories = len(env.player_states[i]['territories'])
            gitbot_armies = sum(env.player_states[i]['armies'].values())
            gitbot_territories_controlled[i-1].append(gitbot_territories)
            gitbot_total_armies[i-1].append(gitbot_armies)

        # Accumulate action counts (using the final counts from the environment for this episode)
        # Note: RiskEnv resets action_counts on reset(), so info['action_counts'] at the end of episode
        # contains the total counts for that episode.
        # However, we need to get it from the last 'info' returned by step(), or access env.action_counts directly.
        for action_type, count in env.action_counts.items():
            total_action_counts[action_type] += count

        # Log progress and metrics
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(all_episode_rewards[-100:]) if all_episode_rewards else 0
            avg_win_rate = np.mean(win_rates) if win_rates else 0
            avg_episode_length = np.mean(episode_lengths) if episode_lengths else 0
            # Convert deques to numpy arrays before slicing to ensure compatibility
            avg_ppo_territories = np.mean(np.array(list(ppo_territories_controlled))[-100:]) if ppo_territories_controlled else 0
            avg_ppo_armies = np.mean(np.array(list(ppo_total_armies))[-100:]) if ppo_total_armies else 0
            
            # Calculate average metrics for each GitBot, ensuring the deque is not empty
            avg_gitbot_territories = [
                np.mean(np.array(list(gitbot_territories_controlled[i]))[-100:]) if gitbot_territories_controlled[i] else 0
                for i in range(num_players - 1)
            ]
            avg_gitbot_armies = [
                np.mean(np.array(list(gitbot_total_armies[i]))[-100:]) if gitbot_total_armies[i] else 0
                for i in range(num_players - 1)
            ]

            winner_info = f"Winner: {env.winner if env.winner is not None else 'None'} (Player {env.winner})" if env.winner is not None else "Winner: None"
            
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Avg Reward (last 100): {avg_reward:.2f} | "
                  f"Win Rate (last 100, PPO Agent): {avg_win_rate:.2%} | "
                  f"Avg Length (last 100): {avg_episode_length:.1f} | "
                  f"{winner_info}")
            print(f"  PPO Agent (P0) - Avg Territories: {avg_ppo_territories:.1f}, Avg Armies: {avg_ppo_armies:.1f}")
            for i in range(num_players - 1):
                print(f"  GitBot (P{i+1}) - Avg Territories: {avg_gitbot_territories[i]:.1f}, Avg Armies: {avg_gitbot_armies[i]:.1f}")


        # Print total action summary every summary_interval episodes
        if (episode + 1) % summary_interval == 0:
            print(f"\n--- Action Summary (Episodes {episode + 1 - summary_interval + 1} to {episode + 1}) ---")
            for action_type, count in total_action_counts.items():
                print(f"  {action_type.replace('_', ' ').title()}: {count}")
            print("--------------------------------------------------\n")
            # Reset counts for the next interval
            total_action_counts = {k: 0 for k in total_action_counts}


        # Save PPO agent model periodically
        if (episode + 1) % save_interval == 0 and episode > 0:
            agents[0].save_model(f"risk_ppo_agent_player0_ep{(episode + 1):06d}_training_gitbot.pth")
            print(f"Model saved for PPO Agent (Player 0) at episode {episode + 1}")

    print("\nTraining complete!")
    # Save final PPO agent model
    agents[0].save_model("risk_ppo_agent_player0_final_training_gitbot.pth")
    print("Final PPO Agent model saved.")

    # --- Visualization (No Changes) ---
    plt.figure(figsize=(12, 6))

    # Plot Rewards
    plt.subplot(1, 2, 1)
    plt.plot(np.convolve(all_episode_rewards, np.ones(100)/100, mode='valid')) # Moving average
    plt.xlabel("Episode")
    plt.ylabel("Total Reward (Moving Average)")
    plt.title("AI Agent Total Reward Over Training")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(np.convolve(np.array(win_rates), np.ones(100)/100, mode='valid') * 100) # Moving average %
    plt.xlabel("Episode")
    plt.ylabel("Win Rate (%) (Moving Average)")
    plt.title("PPO Agent Win Rate Over Training (vs. GitBots)")
    plt.ylim(0, 100)
    plt.grid(True)

    plt.tight_layout()
    plt.show()
