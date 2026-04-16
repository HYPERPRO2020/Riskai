# risk_ai_trainer.py
# Train PPO to play Risk against a rotating pool of rule-based agents.
# Autosaves checkpoints and supports resume/eval.

import os
import sys
import csv
import time
import math
import random
import argparse
from pathlib import Path
from collections import deque, defaultdict

import numpy as np
import torch

from risk_game_environment import RiskEnv                               # :contentReference[oaicite:4]{index=4}
from ppo_agent import PPOAgent                                          # :contentReference[oaicite:5]{index=5}
from rule_based_agents import RandomAgent, DefensiveAgent, BalancedAgent # :contentReference[oaicite:6]{index=6}
from git_bot import GitBot                                              # :contentReference[oaicite:7]{index=7}


# ------------- Utilities

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def softmax_masked(logits: np.ndarray, mask: np.ndarray):
    """Numerically stable softmax over a masked domain (True = keep)."""
    masked = np.where(mask, logits, -1e9)
    m = masked.max()
    ex = np.exp(masked - m)
    ex *= mask
    s = ex.sum()
    return (ex / s) if s > 0 else mask.astype(float) / (mask.sum() + 1e-9)


# ------------- Action translation (PPO & rule-based -> env.step format)

class ActionTranslator:
    """
    Bridges policy outputs (indices) to RiskEnv.step(...) dictionaries.
    Works for both PPO (multi-headed) and the provided rule-based agents.
    """

    def __init__(self, env: RiskEnv):
        self.env = env
        self.T = self.env.num_territories
        self.territory_names = self.env.territory_names

    def idx_to_name(self, idx: int) -> str:
        idx = int(max(0, min(self.T - 1, idx)))
        return self.territory_names[idx]

    def decode_attack_tuple(self, tup):
        # From PPO: (att_idx, def_idx, dice_idx) where dice_idx in {0,1,2} -> dice 1..3
        att_i, def_i, dice_i = int(tup[0]), int(tup[1]), int(tup[2])
        return self.idx_to_name(att_i), self.idx_to_name(def_i), int(dice_i) + 1

    def decode_fortify_tuple(self, tup):
        # From PPO: (from_idx, to_idx, armies_idx) where armies_idx in {0..9} -> armies 1..10
        f_i, t_i, a_i = int(tup[0]), int(tup[1]), int(tup[2])
        return self.idx_to_name(f_i), self.idx_to_name(t_i), int(a_i) + 1

    def translate(self, agent_action_dict: dict, env_phase: str):
        """
        Convert agent_action_dict into a valid env.step action.
        Be conservative: if invalid, send the appropriate *end_* action for the phase.
        """
        phase = env_phase

        # --- Reinforcement phase ---
        if phase == "reinforcement":
            # Optional: trade cards
            if agent_action_dict.get("trade_cards", 0) == 1 and \
               len(self.env.player_states[self.env.current_player]['cards']) >= 3:
                # Simplest heuristic: trade the first 3 cards
                return {"phase": "trade_cards", "card_indices": [0, 1, 2]}

            # Place armies
            t_idx = agent_action_dict.get("reinforce_t", None)
            a_idx = agent_action_dict.get("reinforce_a", None)

            if t_idx is not None and a_idx is not None:
                territory = self.idx_to_name(t_idx)
                armies = int(a_idx) + 1  # 1..5
                # Only place up to available reinforcements (env enforces)
                return {"phase": "reinforce", "placements": {territory: armies}}

            # Or explicitly end
            if agent_action_dict.get("phase_transition", 0) == 0:
                return {"phase": "end_reinforcement"}

            # Fallback (try to end if nothing else makes sense)
            return {"phase": "end_reinforcement"}

        # --- Attack phase ---
        if phase == "attack":
            # End attack requested?
            if agent_action_dict.get("phase_transition", 1) == 1:
                return {"phase": "end_attack"}

            # Attempt an attack
            if "attack" in agent_action_dict:
                attacker_t, defender_t, dice = self.decode_attack_tuple(agent_action_dict["attack"])
                # Validate quickly; env also validates
                if defender_t in self.env.adjacency_list.get(attacker_t, []):
                    return {"phase": "attack", "attacker": attacker_t, "defender": defender_t, "dice": dice}

            # If invalid, end attack
            return {"phase": "end_attack"}

        # --- Fortify phase ---
        if phase == "fortify":
            # End fortify requested?
            if agent_action_dict.get("phase_transition", 2) == 2:
                return {"phase": "end_fortify"}

            if "fortify" in agent_action_dict:
                from_t, to_t, armies = self.decode_fortify_tuple(agent_action_dict["fortify"])
                if to_t in self.env.adjacency_list.get(from_t, []):
                    return {"phase": "fortify", "from": from_t, "to": to_t, "armies": armies}

            # If invalid, end fortify
            return {"phase": "end_fortify"}

        # Unknown phase fallback (shouldn't happen)
        return {"phase": "end_fortify"}


# ------------- Opponent factory

def make_opponent(agent_name: str, player_id: int, env: RiskEnv):
    name = agent_name.lower()
    if name == "random":
        return RandomAgent(player_id, env)        # :contentReference[oaicite:8]{index=8}
    if name == "defensive":
        return DefensiveAgent(player_id, env)     # :contentReference[oaicite:9]{index=9}
    if name == "balanced":
        return BalancedAgent(player_id, env)      # :contentReference[oaicite:10]{index=10}
    if name == "gitbot":
        return GitBot(player_id, env)             # :contentReference[oaicite:11]{index=11}
    raise ValueError(f"Unknown opponent agent: {agent_name}")


# ------------- Episode runner (self-play vs pool)

def seat_players(env: RiskEnv, learner_seat: int, opponent_names):
    """
    Instantiate opponents and place PPO learner in the chosen seat.
    Returns:
        ppo_id (int), opponents (dict: seat -> rule-based agent)
    """
    opponents = {}
    for pid in range(env.num_players):
        if pid == learner_seat:
            continue
        name = random.choice(opponent_names)
        opponents[pid] = make_opponent(name, pid, env)
    return learner_seat, opponents


def run_episode(env: RiskEnv, ppo: PPOAgent, learner_seat: int, opponent_names, device="cpu",
                max_steps=20000):
    """
    Runs a single game episode. The PPO agent only learns from its own actions.
    Returns stats and fills ppo.buffer with learner transitions.
    """
    translator = ActionTranslator(env)
    ppo.actor_critic.to(device)

    ppo_id, opponents = seat_players(env, learner_seat, opponent_names)

    state = env._get_state()  # env.reset() already called by caller
    done = False
    step_count = 0
    winner = None

    while not done and step_count < max_steps:
        current_pid = env.current_player
        env_phase = env.turn_phase  # "reinforcement" | "attack" | "fortify"

        if current_pid == ppo_id:
            # PPO selects action
            action_dict, log_prob_dict, value = ppo.select_action(state, env_phase)
            env_action = translator.translate(action_dict, env_phase)

            next_state, reward, done, info = env.step(env_action)

            # Store transition only for our actions
            ppo.store_transition(state, action_dict, reward, next_state, log_prob_dict, value, done, env_phase)
        else:
            # Rule-based opponent
            rb_agent = opponents[current_pid]
            rb_action_dict, _, _ = rb_agent.select_action(state, env_phase)
            env_action = translator.translate(rb_action_dict, env_phase)
            next_state, _, done, info = env.step(env_action)

        state = next_state
        step_count += 1

        if done:
            winner = env.winner

    return {
        "steps": step_count,
        "winner": winner,
        "ppo_won": (winner == ppo_id)
    }


# ------------- Training loop

def train(
    episodes=500,
    num_players=4,
    learner_seat="random",           # "random" or int seat id
    lr=3e-4,
    gamma=0.99,
    clip_eps=0.2,
    gae_lambda=0.95,
    save_every=50,
    eval_every=100,
    log_path="training_log.csv",
    ckpt_dir="checkpoints",
    resume=True,
    seed=42,
    opponent_pool=("random", "defensive", "balanced", "gitbot"),  # pool to sample from
    device="cpu"
):
    set_seed(seed)
    ensure_dir(Path(ckpt_dir))

    # Build env to infer state size
    env = RiskEnv(num_players=num_players)  # reset inside __init__  :contentReference[oaicite:12]{index=12}
    state_size = env._get_state().shape[0]  # uses the full feature vector  :contentReference[oaicite:13]{index=13}
    num_territories = env.num_territories

    ppo = PPOAgent(
        state_size=state_size,
        num_territories=num_territories,
        learning_rate=lr,
        gamma=gamma,
        clip_epsilon=clip_eps,
        gae_lambda=gae_lambda
    )  # :contentReference[oaicite:14]{index=14}

    latest_path = Path(ckpt_dir) / "latest.pt"
    best_path = Path(ckpt_dir) / "best.pt"

    if resume and latest_path.exists():
        try:
            ppo.load_model(str(latest_path))
            print(f"[Resume] Loaded model from {latest_path}")
        except Exception as e:
            print(f"[Resume] Failed to load {latest_path}: {e}")

    # Logging
    if not Path(log_path).exists():
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "steps", "ppo_win", "rolling_win_rate"])

    rolling = deque(maxlen=100)
    best_rolling = -1.0

    for ep in range(1, episodes + 1):
        # New episode
        env = RiskEnv(num_players=num_players)  # fresh game  :contentReference[oaicite:15]{index=15}
        if learner_seat == "random":
            learner_pid = random.randrange(num_players)
        else:
            learner_pid = int(learner_seat)

        stats = run_episode(env, ppo, learner_pid, opponent_pool, device=device)
        # Learn once per episode from collected transitions
        ppo.learn()  # performs multiple PPO epochs over buffer  :contentReference[oaicite:16]{index=16}

        # Track
        rolling.append(1.0 if stats["ppo_won"] else 0.0)
        rolling_rate = sum(rolling) / max(1, len(rolling))

        # Save CSV row
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([ep, stats["steps"], int(stats["ppo_won"]), f"{rolling_rate:.4f}"])

        # Periodic autosave
        if ep % save_every == 0:
            ppo.save_model(str(latest_path))       # :contentReference[oaicite:17]{index=17}
            print(f"[Save] Episode {ep}: wrote {latest_path}")

        # Best rolling win-rate save
        if rolling_rate > best_rolling and len(rolling) >= rolling.maxlen // 2:
            best_rolling = rolling_rate
            ppo.save_model(str(best_path))
            print(f"[Best] Episode {ep}: rolling win-rate {best_rolling:.3f} -> saved {best_path}")

        # Optional eval
        if eval_every and ep % eval_every == 0:
            wr = evaluate(ppo, num_matches=20, num_players=num_players, opponent_pool=opponent_pool, device=device)
            print(f"[Eval] Episode {ep}: eval win-rate over 20 matches = {wr:.3f}")

    # Final autosave
    ppo.save_model(str(latest_path))
    print(f"[Done] Training complete. Latest saved to {latest_path} | Best to {best_path if best_path.exists() else 'n/a'}")


# ------------- Evaluation

def evaluate(ppo: PPOAgent, num_matches=50, num_players=4, opponent_pool=("random","defensive","balanced","gitbot"), device="cpu"):
    wins = 0
    for _ in range(num_matches):
        env = RiskEnv(num_players=num_players)
        learner_pid = random.randrange(num_players)
        stats = run_episode(env, ppo, learner_pid, opponent_pool, device=device)
        wins += 1 if stats["ppo_won"] else 0
    return wins / num_matches


# ------------- CLI

def main():
    parser = argparse.ArgumentParser(description="Train PPO to play Risk vs rule-based opponents with autosave.")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--players", type=int, default=4)
    parser.add_argument("--seat", type=str, default="random", help='"random" or integer seat id [0..players-1]')
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--gae", type=float, default=0.95)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--log", type=str, default="training_log.csv")
    parser.add_argument("--ckpt", type=str, default="checkpoints")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu","cuda"])
    parser.add_argument("--eval", action="store_true", help="Run evaluation only (load latest checkpoint if present).")

    args = parser.parse_args()

    resume = not args.no_resume
    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"

    if args.eval:
        # Build env to get sizes
        env = RiskEnv(num_players=args.players)
        state_size = env._get_state().shape[0]
        ppo = PPOAgent(state_size=state_size, num_territories=env.num_territories,
                       learning_rate=args.lr, gamma=args.gamma, clip_epsilon=args.clip, gae_lambda=args.gae)
        latest_path = Path(args.ckpt) / "latest.pt"
        if latest_path.exists():
            ppo.load_model(str(latest_path))
            print(f"[Eval] Loaded {latest_path}")
        wr = evaluate(ppo, num_matches=50, num_players=args.players, opponent_pool=("random","defensive","balanced","gitbot"), device=device)
        print(f"[Eval] Win-rate over 50 matches: {wr:.3f}")
        return

    train(
        episodes=args.episodes,
        num_players=args.players,
        learner_seat=args.seat,
        lr=args.lr,
        gamma=args.gamma,
        clip_eps=args.clip,
        gae_lambda=args.gae,
        save_every=args.save_every,
        eval_every=args.eval_every,
        log_path=args.log,
        ckpt_dir=args.ckpt,
        resume=resume,
        seed=args.seed,
        opponent_pool=("random","defensive","balanced","gitbot"),
        device=device
    )


if __name__ == "__main__":
    main()
