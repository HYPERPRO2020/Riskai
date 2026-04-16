"""
Single-file RiskAI trainer.

You pick the opponents, you pick the size of the game, you pick how long it
trains for, and you can play against the AI yourself.

Depends on:
    risk_game_environment.py   (RiskEnv)
    ppo_agent.py               (PPOAgent)
    rule_based_agents.py       (RandomAgent, DefensiveAgent, BalancedAgent, StrategicAgent)

Usage examples:
    # Train 100k games against a mix of bots
    python train.py --episodes 100000 --opponents random,defensive,balanced,strategic

    # Train against the strongest bot only
    python train.py --episodes 50000 --opponents strategic

    # Self-play (opponents are frozen snapshots of the current policy)
    python train.py --episodes 50000 --opponents self --players 4

    # Mix of self-play and bots (curriculum-like)
    python train.py --episodes 100000 --opponents self,strategic,balanced

    # Evaluate a trained checkpoint
    python train.py --eval --opponents strategic --matches 100

    # Play against your trained AI (you get seat 0)
    python train.py --play --players 3 --opponents strategic
"""

import argparse
import copy
import csv
import os
import random
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

from risk_game_environment import RiskEnv
from ppo_agent import PPOAgent
from rule_based_agents import (
    RandomAgent, DefensiveAgent, BalancedAgent, StrategicAgent,
    get_territory_idx_from_name,
)


# ============================================================================
# Opponent registry
# ============================================================================

RULE_BASED = {
    "random": RandomAgent,
    "defensive": DefensiveAgent,
    "balanced": BalancedAgent,
    "strategic": StrategicAgent,
}

SPECIAL = {"self", "human", "frozen"}
VALID_OPPONENTS = set(RULE_BASED.keys()) | SPECIAL


# ============================================================================
# Action-mask builder
#
# The existing PPOAgent.select_action() already supports masks but was never
# fed any. Without masks the agent samples territories it doesn't own, fights
# non-adjacent territories, etc., which is why 100k games of training does
# basically nothing. We compute a proper mask every step.
# ============================================================================

class MaskBuilder:
    def __init__(self, env: RiskEnv):
        self.env = env
        self.T = env.num_territories

    def build(self, phase: str):
        T = self.T
        env = self.env
        pid = env.current_player
        owned = env.player_states[pid]['territories']
        armies = env.player_states[pid]['armies']

        masks = {}

        if phase == "reinforcement":
            # trade cards: 1 (trade) is valid only with >=3 cards
            num_cards = len(env.player_states[pid]['cards'])
            trade = np.zeros(2, dtype=np.float32)
            trade[0] = 1.0
            if num_cards >= 3:
                trade[1] = 1.0
            masks["trade_cards"] = trade

            # reinforce_t: only owned territories
            rt = np.zeros(T, dtype=np.float32)
            for name in owned:
                rt[env.territory_names.index(name)] = 1.0
            if rt.sum() == 0:  # dead - place anywhere to avoid crash
                rt[:] = 1.0
            masks["reinforce_t"] = rt

            # reinforce_a: 1..5 armies. Cap at available reinforcements.
            avail = env.player_states[pid].get('reinforcements_available', 0)
            if avail <= 0:
                avail = 1  # avoid all-zero mask; env will no-op
            ra = np.zeros(5, dtype=np.float32)
            ra[:min(5, avail)] = 1.0
            masks["reinforce_a"] = ra

            # phase transition: only "end reinforcement" (index 0) is meaningful here
            pt = np.zeros(3, dtype=np.float32)
            pt[0] = 1.0
            masks["phase_transition"] = pt

        elif phase == "attack":
            # attacker: owned territory with >=2 armies AND has an enemy neighbour
            attacker = np.zeros(T, dtype=np.float32)
            defender = np.zeros((T, T), dtype=np.float32)
            for name in owned:
                if armies[name] < 2:
                    continue
                ai = env.territory_names.index(name)
                enemy_neighbours = [
                    n for n in env.adjacency_list.get(name, [])
                    if env.territories.get(n) != pid
                ]
                if not enemy_neighbours:
                    continue
                attacker[ai] = 1.0
                for en in enemy_neighbours:
                    defender[ai, env.territory_names.index(en)] = 1.0
            masks["attack_attacker"] = attacker
            masks["attack_defender"] = defender
            # dice 1..3 always fine; env clamps to armies-1
            masks["attack_dice"] = np.ones(3, dtype=np.float32)

            # phase transition: agent may either attack (0) or end attack (1).
            pt = np.zeros(3, dtype=np.float32)
            pt[0] = 1.0  # attack again
            pt[1] = 1.0  # end attack
            masks["phase_transition"] = pt

        elif phase == "fortify":
            # from: owned with >=2 armies
            ffrom = np.zeros(T, dtype=np.float32)
            fto = np.zeros((T, T), dtype=np.float32)
            for name in owned:
                if armies[name] < 2:
                    continue
                fi = env.territory_names.index(name)
                friendly_neighbours = [
                    n for n in env.adjacency_list.get(name, [])
                    if env.territories.get(n) == pid
                ]
                if not friendly_neighbours:
                    continue
                ffrom[fi] = 1.0
                for fn in friendly_neighbours:
                    fto[fi, env.territory_names.index(fn)] = 1.0
            masks["fortify_from"] = ffrom
            masks["fortify_to"] = fto
            masks["fortify_armies"] = np.ones(10, dtype=np.float32)

            pt = np.zeros(3, dtype=np.float32)
            pt[0] = 1.0  # fortify
            pt[2] = 1.0  # end fortify
            masks["phase_transition"] = pt

        return masks

    def any_attack_possible(self):
        env = self.env
        pid = env.current_player
        for name in env.player_states[pid]['territories']:
            if env.player_states[pid]['armies'][name] < 2:
                continue
            for n in env.adjacency_list.get(name, []):
                if env.territories.get(n) != pid:
                    return True
        return False

    def any_fortify_possible(self):
        env = self.env
        pid = env.current_player
        owned = env.player_states[pid]['territories']
        for name in owned:
            if env.player_states[pid]['armies'][name] < 2:
                continue
            for n in env.adjacency_list.get(name, []):
                if env.territories.get(n) == pid:
                    return True
        return False


# ============================================================================
# Action translation: agent output dict -> RiskEnv.step() dict
# ============================================================================

def _card_type(env, card_name):
    if card_name in env.continents["North America"] or card_name in env.continents["South America"]:
        return "A"
    if card_name in env.continents["Europe"] or card_name in env.continents["Africa"]:
        return "B"
    return "C"


def _find_valid_trade_set(env, pid):
    """Return 3 indices into the player's hand that form a valid Risk set
    (all same type or all different), or None if no valid combo exists.
    The env expects unique indices; the first valid 3-combo is returned."""
    cards = env.player_states[pid]['cards']
    n = len(cards)
    if n < 3:
        return None
    types = [_card_type(env, c) for c in cards]
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                s = {types[i], types[j], types[k]}
                if len(s) == 1 or len(s) == 3:
                    return (i, j, k)
    return None


def translate_action(env, action_dict, phase):
    T = env.num_territories
    names = env.territory_names
    pid = env.current_player

    def nm(idx):
        return names[max(0, min(T - 1, int(idx)))]

    if phase == "reinforcement":
        if action_dict.get("trade_cards", 0) == 1 and \
                len(env.player_states[pid]['cards']) >= 3:
            valid_indices = _find_valid_trade_set(env, pid)
            if valid_indices is not None:
                return {"phase": "trade_cards", "card_indices": list(valid_indices)}
            # No valid set exists right now - fall through to placement so we
            # don't get stuck re-trying the same invalid trade.
        t_idx = action_dict.get("reinforce_t")
        a_idx = action_dict.get("reinforce_a")
        if t_idx is not None and a_idx is not None:
            target = nm(t_idx)
            # If the chosen territory isn't owned, pick any owned one so we
            # don't loop forever. Opponents like RandomAgent regularly send
            # non-owned indices; the env would otherwise keep rejecting them.
            if target not in env.player_states[pid]['territories']:
                owned = list(env.player_states[pid]['territories'])
                if not owned:
                    return {"phase": "end_reinforcement"}
                target = random.choice(owned)
            avail = env.player_states[pid].get('reinforcements_available', 0)
            armies = int(a_idx) + 1
            if avail > 0:
                armies = max(1, min(armies, avail))
            return {"phase": "reinforce", "placements": {target: armies}}
        return {"phase": "end_reinforcement"}

    if phase == "attack":
        if action_dict.get("phase_transition", 1) == 1:
            return {"phase": "end_attack"}
        if "attack" in action_dict:
            att_i, def_i, dice_i = action_dict["attack"]
            att = nm(att_i); dfn = nm(def_i)
            owned = env.player_states[pid]['territories']
            if (att in owned
                    and dfn not in owned
                    and dfn in env.adjacency_list.get(att, [])
                    and env.player_states[pid]['armies'][att] >= 2):
                dice = max(1, min(3, int(dice_i) + 1,
                                  env.player_states[pid]['armies'][att] - 1))
                return {"phase": "attack", "attacker": att,
                        "defender": dfn, "dice": dice}
        return {"phase": "end_attack"}

    if phase == "fortify":
        if action_dict.get("phase_transition", 2) == 2:
            return {"phase": "end_fortify"}
        if "fortify" in action_dict:
            f_i, t_i, a_i = action_dict["fortify"]
            fr = nm(f_i); to = nm(t_i)
            owned = env.player_states[pid]['territories']
            if (fr in owned and to in owned
                    and to in env.adjacency_list.get(fr, [])
                    and env.player_states[pid]['armies'][fr] >= 2):
                armies = max(1, min(int(a_i) + 1,
                                    env.player_states[pid]['armies'][fr] - 1))
                return {"phase": "fortify", "from": fr, "to": to,
                        "armies": armies}
        return {"phase": "end_fortify"}

    return {"phase": "end_fortify"}


# ============================================================================
# Human opponent - prompts on stdin
# ============================================================================

class HumanAgent:
    def __init__(self, player_id, env):
        self.player_id = player_id
        self.env = env

    def _prompt_int(self, msg, lo, hi, default=None):
        while True:
            raw = input(msg).strip()
            if raw == "" and default is not None:
                return default
            try:
                v = int(raw)
                if lo <= v <= hi:
                    return v
            except ValueError:
                pass
            print(f"  (enter an integer in [{lo}, {hi}])")

    def _print_state(self):
        env = self.env
        pid = self.player_id
        print(f"\n--- Your turn (player {pid}), phase: {env.turn_phase} ---")
        owned = sorted(env.player_states[pid]['territories'])
        for t in owned:
            a = env.player_states[pid]['armies'][t]
            enemies = [n for n in env.adjacency_list.get(t, [])
                       if env.territories.get(n) != pid]
            print(f"  {t:25s} armies={a}  enemy-borders={len(enemies)}")

    def select_action(self, state, env_phase):
        env = self.env
        pid = self.player_id
        self._print_state()
        T = env.num_territories
        act = {"reinforce_t": 0, "reinforce_a": 0, "trade_cards": 0,
               "phase_transition": 0, "attack": (0, 0, 0), "fortify": (0, 0, 0)}

        if env_phase == "reinforcement":
            avail = env.player_states[pid].get('reinforcements_available', 0)
            if avail == 0:
                env.player_states[pid]['reinforcements_available'] = \
                    env._calculate_reinforcements(pid)
                avail = env.player_states[pid]['reinforcements_available']
            ncards = len(env.player_states[pid]['cards'])
            print(f"  armies to place: {avail}    cards: {ncards}")
            if ncards >= 3:
                t = self._prompt_int("Trade cards? (0=no, 1=yes) [0]: ", 0, 1, 0)
                act["trade_cards"] = t
                if t == 1:
                    return act, {}, 0.0
            owned = sorted(env.player_states[pid]['territories'])
            for i, n in enumerate(owned):
                print(f"   [{i}] {n}")
            i = self._prompt_int("Place in which territory? ", 0, len(owned) - 1)
            target = owned[i]
            n = self._prompt_int(f"How many armies (1..{min(5, avail)})? ",
                                 1, min(5, avail))
            act["reinforce_t"] = env.territory_names.index(target)
            act["reinforce_a"] = n - 1
            return act, {}, 0.0

        if env_phase == "attack":
            valid = []
            for name in env.player_states[pid]['territories']:
                if env.player_states[pid]['armies'][name] < 2:
                    continue
                for n in env.adjacency_list.get(name, []):
                    if env.territories.get(n) != pid:
                        valid.append((name, n))
            if not valid:
                print("  No valid attacks. Ending attack phase.")
                act["phase_transition"] = 1
                return act, {}, 0.0
            for i, (a, d) in enumerate(valid):
                aa = env.player_states[pid]['armies'][a]
                dd = env.player_states[env.territories[d]]['armies'][d]
                print(f"   [{i}] {a}({aa}) -> {d}({dd})")
            print(f"   [{len(valid)}] end attack phase")
            i = self._prompt_int("Choose attack: ", 0, len(valid))
            if i == len(valid):
                act["phase_transition"] = 1
                return act, {}, 0.0
            att, dfn = valid[i]
            maxd = min(3, env.player_states[pid]['armies'][att] - 1)
            d = self._prompt_int(f"How many dice (1..{maxd})? ", 1, maxd)
            act["phase_transition"] = 0
            act["attack"] = (env.territory_names.index(att),
                             env.territory_names.index(dfn), d - 1)
            return act, {}, 0.0

        if env_phase == "fortify":
            valid = []
            for name in env.player_states[pid]['territories']:
                if env.player_states[pid]['armies'][name] < 2:
                    continue
                for n in env.adjacency_list.get(name, []):
                    if env.territories.get(n) == pid:
                        valid.append((name, n))
            if not valid:
                act["phase_transition"] = 2
                return act, {}, 0.0
            for i, (a, b) in enumerate(valid):
                aa = env.player_states[pid]['armies'][a]
                print(f"   [{i}] {a}({aa}) -> {b}")
            print(f"   [{len(valid)}] end fortify (skip)")
            i = self._prompt_int("Choose fortify: ", 0, len(valid))
            if i == len(valid):
                act["phase_transition"] = 2
                return act, {}, 0.0
            fr, to = valid[i]
            maxm = min(10, env.player_states[pid]['armies'][fr] - 1)
            m = self._prompt_int(f"How many armies (1..{maxm})? ", 1, maxm)
            act["phase_transition"] = 0
            act["fortify"] = (env.territory_names.index(fr),
                              env.territory_names.index(to), m - 1)
            return act, {}, 0.0

        return act, {}, 0.0


# ============================================================================
# Frozen PPO opponent (self-play)
# ============================================================================

class FrozenPPO:
    """Wraps a snapshot of PPOAgent.actor_critic to act as an opponent."""

    def __init__(self, player_id, env, snapshot_state_dict, state_size,
                 num_territories, device):
        from ppo_agent import ActorCritic
        self.player_id = player_id
        self.env = env
        self.device = device
        self.net = ActorCritic(state_size, num_territories).to(device)
        self.net.load_state_dict(snapshot_state_dict)
        self.net.eval()
        self.T = num_territories
        self.mask_builder = MaskBuilder(env)

    def select_action(self, state, env_phase):
        masks = self.mask_builder.build(env_phase)
        s = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, _ = self.net(s)
        action_dict = {}

        def sample_masked(logits_tensor, mask_np):
            mask_t = torch.from_numpy(mask_np).float().to(self.device)
            if mask_t.sum() < 1:
                mask_t = torch.ones_like(mask_t)
            adj = logits_tensor + (mask_t.view_as(logits_tensor) - 1) * 1e9
            probs = torch.softmax(adj, dim=-1)
            return int(torch.distributions.Categorical(probs).sample().item())

        if env_phase == "reinforcement":
            action_dict["trade_cards"] = sample_masked(
                logits["trade_cards"][0], masks["trade_cards"])
            action_dict["reinforce_t"] = sample_masked(
                logits["reinforce_t"][0], masks["reinforce_t"])
            action_dict["reinforce_a"] = sample_masked(
                logits["reinforce_a"][0], masks["reinforce_a"])
            action_dict["phase_transition"] = sample_masked(
                logits["phase_transition"][0], masks["phase_transition"])
        elif env_phase == "attack":
            full = (masks["attack_attacker"].reshape(-1, 1, 1) *
                    masks["attack_defender"].reshape(self.T, self.T, 1) *
                    masks["attack_dice"].reshape(1, 1, -1))
            if full.sum() < 1:
                action_dict["phase_transition"] = 1
                action_dict["attack"] = (0, 0, 0)
            else:
                flat = sample_masked(logits["attack"][0].view(-1), full.reshape(-1))
                action_dict["attack"] = np.unravel_index(flat, (self.T, self.T, 3))
                action_dict["phase_transition"] = sample_masked(
                    logits["phase_transition"][0], masks["phase_transition"])
        elif env_phase == "fortify":
            full = (masks["fortify_from"].reshape(-1, 1, 1) *
                    masks["fortify_to"].reshape(self.T, self.T, 1) *
                    masks["fortify_armies"].reshape(1, 1, -1))
            if full.sum() < 1:
                action_dict["phase_transition"] = 2
                action_dict["fortify"] = (0, 0, 0)
            else:
                flat = sample_masked(logits["fortify"][0].view(-1), full.reshape(-1))
                action_dict["fortify"] = np.unravel_index(flat, (self.T, self.T, 10))
                action_dict["phase_transition"] = sample_masked(
                    logits["phase_transition"][0], masks["phase_transition"])
        return action_dict, {}, 0.0


# ============================================================================
# Opponent factory
# ============================================================================

def make_opponent(kind, pid, env, self_snapshot=None, ppo_shape=None, device="cpu"):
    kind = kind.lower()
    if kind in RULE_BASED:
        return RULE_BASED[kind](pid, env)
    if kind == "human":
        return HumanAgent(pid, env)
    if kind in ("self", "frozen"):
        if self_snapshot is None:
            return RandomAgent(pid, env)  # fallback early in training
        state_size, num_territories = ppo_shape
        return FrozenPPO(pid, env, self_snapshot, state_size,
                         num_territories, device)
    raise ValueError(f"Unknown opponent '{kind}'. "
                     f"Valid: {sorted(VALID_OPPONENTS)}")


# ============================================================================
# Reward shaping
#
# Base env gives tiny per-step rewards and +10 on win. We add a dense signal
# proportional to the change in owned territories, so the agent has something
# to credit mid-game.
# ============================================================================

def shaped_reward(env, pid, base_reward, prev_terr, prev_armies, done, won):
    r = float(base_reward)
    cur_terr = len(env.player_states[pid]['territories'])
    cur_armies = sum(env.player_states[pid]['armies'].values())
    r += 0.02 * (cur_terr - prev_terr)
    r += 0.002 * (cur_armies - prev_armies)
    if done:
        r += 20.0 if won else -5.0
    return r, cur_terr, cur_armies


# ============================================================================
# Episode runner
# ============================================================================

def run_episode(env, ppo, learner_pid, opponent_kinds,
                self_snapshot=None, ppo_shape=None,
                train=True, max_steps=8000, device="cpu", verbose=False):
    mb = MaskBuilder(env)
    opponents = {}
    for pid in range(env.num_players):
        if pid == learner_pid:
            continue
        kind = random.choice(opponent_kinds)
        opponents[pid] = make_opponent(kind, pid, env, self_snapshot,
                                       ppo_shape, device)

    state = env.reset() if env.current_player is None else env._get_state()
    prev_terr = len(env.player_states[learner_pid]['territories'])
    prev_armies = sum(env.player_states[learner_pid]['armies'].values())
    steps = 0
    done = False

    while not done and steps < max_steps:
        pid = env.current_player
        phase = env.turn_phase

        if pid == learner_pid:
            masks = mb.build(phase)
            # Short-circuit "nothing to do" phases to avoid wasted stored
            # transitions.
            if phase == "attack" and masks["attack_attacker"].sum() == 0:
                action_dict, logp, val = {"phase_transition": 1,
                                          "attack": (0, 0, 0)}, None, 0.0
                env_action = {"phase": "end_attack"}
                _, r, done, _ = env.step(env_action)
                state = env._get_state()
                steps += 1
                continue
            if phase == "fortify" and masks["fortify_from"].sum() == 0:
                env_action = {"phase": "end_fortify"}
                _, r, done, _ = env.step(env_action)
                state = env._get_state()
                steps += 1
                continue

            action_dict, log_prob_dict, value = ppo.select_action(
                state, phase, action_masks=masks)
            env_action = translate_action(env, action_dict, phase)
            next_state, raw_reward, done, info = env.step(env_action)
            won = done and env.winner == learner_pid
            r, prev_terr, prev_armies = shaped_reward(
                env, learner_pid, raw_reward, prev_terr, prev_armies, done, won)
            if train:
                ppo.store_transition(state, action_dict, r, next_state,
                                     log_prob_dict, value, done, phase)
            state = next_state
        else:
            rb_action, _, _ = opponents[pid].select_action(state, phase)
            env_action = translate_action(env, rb_action, phase)
            state, _, done, _ = env.step(env_action)
            # If the base env tracks the learner's lost territory etc., our
            # next shaped reward correctly picks it up on the learner's turn.

        steps += 1

    return {
        "steps": steps,
        "winner": env.winner,
        "won": env.winner == learner_pid,
    }


# ============================================================================
# Training
# ============================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_ppo(num_players, lr, gamma, clip, gae_lambda):
    env = RiskEnv(num_players=num_players)
    state = env._get_state()
    state_size = state.shape[0]
    num_territories = env.num_territories
    ppo = PPOAgent(state_size=state_size, num_territories=num_territories,
                   learning_rate=lr, gamma=gamma,
                   clip_epsilon=clip, gae_lambda=gae_lambda)
    # PPO trains FAR better with Adam than SGD; upgrade unless DirectML forces SGD.
    using_dml = "privateuseone" in str(ppo.device).lower()
    if not using_dml:
        ppo.optimizer = optim.Adam(ppo.actor_critic.parameters(), lr=lr)
        ppo.scheduler = optim.lr_scheduler.StepLR(ppo.optimizer,
                                                  step_size=2000, gamma=0.95)
    return ppo, state_size, num_territories


def train_loop(args):
    set_seed(args.seed)
    opp_list = [o.strip().lower() for o in args.opponents.split(",")]
    for o in opp_list:
        if o not in VALID_OPPONENTS:
            sys.exit(f"Unknown opponent '{o}'. "
                     f"Valid: {sorted(VALID_OPPONENTS)}")

    ckpt_dir = Path(args.ckpt)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest = ckpt_dir / "latest.pt"
    best = ckpt_dir / "best.pt"

    ppo, state_size, num_territories = build_ppo(
        args.players, args.lr, args.gamma, args.clip, args.gae)

    if args.resume and latest.exists():
        ppo.load_model(str(latest))
        print(f"[resume] loaded {latest}")

    log_new = not Path(args.log).exists()
    log_f = open(args.log, "a", newline="")
    writer = csv.writer(log_f)
    if log_new:
        writer.writerow(["episode", "steps", "won", "rolling_winrate",
                         "mean_reward", "opponents"])

    rolling = deque(maxlen=200)
    best_wr = -1.0
    self_snapshot = None
    ppo_shape = (state_size, num_territories)

    for ep in range(1, args.episodes + 1):
        env = RiskEnv(num_players=args.players)
        learner_pid = random.randrange(args.players)
        stats = run_episode(env, ppo, learner_pid, opp_list,
                            self_snapshot=self_snapshot, ppo_shape=ppo_shape,
                            train=True, max_steps=args.max_steps,
                            device=str(ppo.device))

        total_r = sum(t[2] for t in ppo.buffer) if ppo.buffer else 0.0
        ppo.learn()

        rolling.append(1.0 if stats["won"] else 0.0)
        wr = sum(rolling) / len(rolling)
        writer.writerow([ep, stats["steps"], int(stats["won"]),
                         f"{wr:.4f}", f"{total_r:.3f}", "|".join(opp_list)])
        log_f.flush()

        if ep % max(1, args.print_every) == 0:
            print(f"ep {ep:>6d} | steps {stats['steps']:>4d} | "
                  f"won={int(stats['won'])} | rolling wr={wr:.3f} | "
                  f"reward={total_r:+.2f}")

        if ep % args.save_every == 0:
            ppo.save_model(str(latest))

        # Snapshot for self-play: refresh every N episodes so "self" isn't
        # a stale early-random policy forever.
        if "self" in opp_list and (ep == 1 or ep % args.snapshot_every == 0):
            self_snapshot = copy.deepcopy(ppo.actor_critic.state_dict())
            print(f"[self-play] refreshed snapshot at episode {ep}")

        if len(rolling) == rolling.maxlen and wr > best_wr:
            best_wr = wr
            ppo.save_model(str(best))
            print(f"[best] ep {ep}: rolling wr={wr:.3f} -> {best}")

    ppo.save_model(str(latest))
    log_f.close()
    print(f"[done] saved {latest}, best wr={best_wr:.3f}")


def eval_loop(args):
    set_seed(args.seed)
    opp_list = [o.strip().lower() for o in args.opponents.split(",")]
    ppo, state_size, num_territories = build_ppo(
        args.players, args.lr, args.gamma, args.clip, args.gae)
    latest = Path(args.ckpt) / "latest.pt"
    if latest.exists():
        ppo.load_model(str(latest))
        print(f"[eval] loaded {latest}")
    else:
        print(f"[eval] no checkpoint at {latest}, evaluating untrained model")

    wins = 0
    for i in range(1, args.matches + 1):
        env = RiskEnv(num_players=args.players)
        learner_pid = random.randrange(args.players)
        stats = run_episode(env, ppo, learner_pid, opp_list,
                            train=False, max_steps=args.max_steps,
                            device=str(ppo.device))
        wins += int(stats["won"])
        print(f"  match {i:>3d}: won={int(stats['won'])} steps={stats['steps']}")
    print(f"[eval] win rate over {args.matches} matches vs "
          f"{opp_list}: {wins / args.matches:.3f}")


def play_loop(args):
    """One-off game where you play seat 0 and the AI fills the other seats."""
    opp_list = [o.strip().lower() for o in args.opponents.split(",")]
    # Make sure at least one non-human opponent kind exists; we manually
    # place the human at seat 0 and let the AI (if a checkpoint exists) be
    # one seat, and rule-based bots fill the rest.
    ppo, state_size, num_territories = build_ppo(
        args.players, args.lr, args.gamma, args.clip, args.gae)
    latest = Path(args.ckpt) / "latest.pt"
    if latest.exists():
        ppo.load_model(str(latest))
        print(f"[play] loaded trained AI from {latest}")
    else:
        print(f"[play] no checkpoint at {latest}; AI will be untrained.")

    env = RiskEnv(num_players=args.players)
    env.reset()
    mb = MaskBuilder(env)
    human_pid = 0
    ai_pid = 1
    human = HumanAgent(human_pid, env)
    others = {ai_pid: "ai"}
    for pid in range(args.players):
        if pid in (human_pid, ai_pid):
            continue
        others[pid] = make_opponent(random.choice(opp_list), pid, env)

    state = env._get_state()
    done = False
    steps = 0
    while not done and steps < args.max_steps:
        pid = env.current_player
        phase = env.turn_phase
        if pid == human_pid:
            act, _, _ = human.select_action(state, phase)
            env_action = translate_action(env, act, phase)
        elif pid == ai_pid:
            masks = mb.build(phase)
            if phase == "attack" and masks["attack_attacker"].sum() == 0:
                env_action = {"phase": "end_attack"}
            elif phase == "fortify" and masks["fortify_from"].sum() == 0:
                env_action = {"phase": "end_fortify"}
            else:
                act, _, _ = ppo.select_action(state, phase, action_masks=masks)
                env_action = translate_action(env, act, phase)
            print(f"[AI seat {ai_pid}, {phase}] -> {env_action}")
        else:
            act, _, _ = others[pid].select_action(state, phase)
            env_action = translate_action(env, act, phase)
        state, _, done, _ = env.step(env_action)
        steps += 1

    print(f"\nGame over. Winner: player {env.winner}. "
          f"{'YOU WIN!' if env.winner == human_pid else 'AI or bot wins.'}")


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--opponents", type=str, default="random,defensive,balanced,strategic",
                   help="Comma-separated list of opponent kinds. "
                        f"Valid: {sorted(VALID_OPPONENTS)}. "
                        "'self' = frozen snapshot of current policy (self-play). "
                        "'human' = prompt on stdin.")
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--players", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=8000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--gae", type=float, default=0.95)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--snapshot-every", type=int, default=500,
                   help="Refresh frozen self-play snapshot every N episodes.")
    p.add_argument("--print-every", type=int, default=10)
    p.add_argument("--ckpt", type=str, default="checkpoints")
    p.add_argument("--log", type=str, default="training_log.csv")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--eval", action="store_true",
                   help="Evaluate an existing checkpoint instead of training.")
    p.add_argument("--matches", type=int, default=50)
    p.add_argument("--play", action="store_true",
                   help="Play one interactive game against the trained AI.")
    return p.parse_args()


def main():
    args = parse_args()
    if args.play:
        play_loop(args)
    elif args.eval:
        eval_loop(args)
    else:
        train_loop(args)


if __name__ == "__main__":
    main()
