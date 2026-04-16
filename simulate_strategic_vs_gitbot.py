import numpy as np
from risk_game_environment import RiskEnv
from rule_based_agents import StrategicAgent
from git_bot import GitBot
import time

def simulate(num_games=100):
    env = RiskEnv(num_players=4)
    
    # Player 0: StrategicAgent (The new logic)
    # Players 1-3: GitBots (The hard opponents)
    agents = [
        StrategicAgent(player_id=0, env=env),
        GitBot(player_id=1, env=env),
        GitBot(player_id=2, env=env),
        GitBot(player_id=3, env=env)
    ]
    
    wins = {0: 0, 1: 0, 2: 0, 3: 0}
    strategic_wins = 0
    
    print(f"Starting simulation: StrategicAgent (P0) vs 3 GitBots (P1-P3) for {num_games} games...")
    
    for game in range(num_games):
        state = env.reset()
        done = False
        steps = 0
        
        while not done and steps < 2000:
            current_player_idx = env.current_player
            current_agent = agents[current_player_idx]
            
            # Skip eliminated players
            if current_player_idx not in env.active_players:
                env._switch_player()
                steps += 1
                continue
                
            # Get action
            # Note: GitBot and StrategicAgent might have different signatures if not standardized.
            # StrategicAgent uses (state, env_phase)
            # GitBot uses (state, env_phase) ? Let's check git_bot_trainer.py usage.
            # In git_bot_trainer: action = agent.select_action(state, env.turn_phase)
            # So it should be compatible.
            
            try:
                if isinstance(current_agent, StrategicAgent):
                    action, _, _ = current_agent.select_action(state, env.turn_phase)
                else:
                    action, _, _ = current_agent.select_action(state, env.turn_phase)
                    
                # Decode action
                env_action = env.decode_action(action)
                
                # Step
                next_state, reward, done, info = env.step(env_action)
                state = next_state
                steps += 1
            except Exception as e:
                print(f"Error in game {game}, step {steps}, player {current_player_idx}: {e}")
                break
                
        if env.winner is not None:
            wins[env.winner] += 1
            if env.winner == 0:
                strategic_wins += 1
        
        if (game + 1) % 10 == 0:
            print(f"Game {game + 1}/{num_games} | Strategic Win Rate: {strategic_wins / (game + 1):.2%} | Wins: {wins}")

    print("\nSimulation Complete!")
    print(f"Total Games: {num_games}")
    print(f"StrategicAgent Wins: {strategic_wins} ({strategic_wins/num_games:.2%})")
    print(f"GitBot Wins: {sum(wins.values()) - strategic_wins}")
    print(f"Detailed Wins: {wins}")

if __name__ == "__main__":
    simulate()
