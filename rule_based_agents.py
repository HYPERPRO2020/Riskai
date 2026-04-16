import numpy as np
import random
from collections import defaultdict

# Helper function to convert territory name to index
def get_territory_idx_from_name(name, env):
    return env.territory_names.index(name)

# Helper function to convert territory index to name
def get_territory_name_from_idx(idx, env):
    return env.territory_names[idx]

class BaseRuleBasedAgent:
    """
    Base class for rule-based agents. Provides common utility methods.
    """
    def __init__(self, player_id, env):
        self.player_id = player_id
        self.env = env # Reference to the game environment

    def select_action(self, state, env_phase):
        """
        Abstract method to be implemented by subclasses.
        Returns an agent_action_dict, dummy log_prob_dict, and dummy value.
        """
        raise NotImplementedError("select_action must be implemented by subclasses")

    def _get_owned_territories_info(self):
        """Returns a dict of owned territories and their army counts."""
        return {t: self.env.player_states[self.player_id]['armies'][t]
                for t in self.env.player_states[self.player_id]['territories']}

    def _get_border_territories(self):
        """Returns owned territories adjacent to enemy territories."""
        border_territories = set()
        owned_territories = self.env.player_states[self.player_id]['territories']
        for t in owned_territories:
            for adj_t in self.env.adjacency_list.get(t, []):
                if self.env.territories[adj_t] != self.player_id:
                    border_territories.add(t)
                    break
        return list(border_territories)

    def _get_valid_attack_targets(self, attacker_territory):
        """Returns a list of valid enemy territories that can be attacked from attacker_territory."""
        targets = []
        for adj_t in self.env.adjacency_list.get(attacker_territory, []):
            if self.env.territories[adj_t] != self.player_id:
                targets.append(adj_t)
        return targets

    def _get_valid_fortify_paths(self, from_territory):
        """Returns a list of valid owned territories to fortify to from from_territory."""
        paths = []
        owned_territories = self.env.player_states[self.player_id]['territories']
        # Simple BFS to find connected owned territories
        queue = [from_territory]
        visited = {from_territory}
        while queue:
            current_t = queue.pop(0)
            for neighbor in self.env.adjacency_list.get(current_t, []):
                if neighbor in owned_territories and neighbor not in visited:
                    paths.append(neighbor)
                    visited.add(neighbor)
                    queue.append(neighbor)
        return paths


class RandomAgent(BaseRuleBasedAgent):
    """
    An AI agent that makes purely random valid moves.
    """
    def select_action(self, state, env_phase):
        agent_action_dict = {}
        log_prob_dict = {} # Dummy values for rule-based agents
        value = 0.0        # Dummy value

        owned_territories_info = self._get_owned_territories_info()
        owned_territories = list(owned_territories_info.keys())
        all_territory_indices = list(range(len(self.env.territory_names)))

        if env_phase == "reinforcement":
            # Randomly decide to trade cards (if possible)
            if random.random() < 0.5 and len(self.env.player_states[self.player_id]['cards']) >= 3:
                agent_action_dict["trade_cards"] = 1
            else:
                agent_action_dict["trade_cards"] = 0 # Don't trade

            # Randomly pick a territory to reinforce
            reinforce_t_idx = random.choice(all_territory_indices)
            agent_action_dict["reinforce_t"] = reinforce_t_idx

            # Randomly pick number of armies (1-5)
            reinforce_a_idx = random.randint(0, 4) # 0-4 for 1-5 armies
            agent_action_dict["reinforce_a"] = reinforce_a_idx

            # Randomly decide to end reinforcement phase (0: end, 1: continue)
            agent_action_dict["phase_transition"] = random.choice([0, 1]) # 0 for end, 1 for continue (dummy for now)

        elif env_phase == "attack":
            # Randomly decide to end attack phase
            if random.random() < 0.3: # 30% chance to end attack phase
                agent_action_dict["phase_transition"] = 1 # End attack
            else:
                agent_action_dict["phase_transition"] = random.choice([0, 1]) # 0 for attack, 1 for end (dummy for now)

                # Random attack
                if owned_territories:
                    attacker_t = random.choice(owned_territories)
                    valid_targets = self._get_valid_attack_targets(attacker_t)
                    if valid_targets:
                        num_armies_on_attacker = owned_territories_info[attacker_t]
                        if num_armies_on_attacker > 1: # Must have more than 1 army to attack
                            defender_t = random.choice(valid_targets)
                            num_dice = random.randint(1, min(3, num_armies_on_attacker - 1))
                            agent_action_dict["attack"] = (get_territory_idx_from_name(attacker_t, self.env),
                                                          get_territory_idx_from_name(defender_t, self.env),
                                                          num_dice - 1) # Convert to 0-indexed dice
                        else:
                            agent_action_dict["phase_transition"] = 1 # End attack
                    else:
                        # No valid attack from chosen territory, force end attack
                        agent_action_dict["phase_transition"] = 1 # End attack
                else:
                    agent_action_dict["phase_transition"] = 1 # End attack

        elif env_phase == "fortify":
            # Randomly decide to end fortify phase
            if random.random() < 0.5: # 50% chance to end fortify phase
                agent_action_dict["phase_transition"] = 2 # End fortify
            else:
                agent_action_dict["phase_transition"] = random.choice([0, 2]) # 0 for fortify, 2 for end (dummy for now)

                # Random fortify
                if owned_territories:
                    from_t = random.choice(owned_territories)
                    valid_paths = self._get_valid_fortify_paths(from_t)
                    # Ensure there are enough armies to move (more than 1 on the 'from' territory)
                    if valid_paths and owned_territories_info[from_t] > 1:
                        to_t = random.choice(valid_paths)
                        num_armies = random.randint(1, owned_territories_info[from_t] - 1)
                        agent_action_dict["fortify"] = (get_territory_idx_from_name(from_t, self.env),
                                                       get_territory_idx_from_name(to_t, self.env),
                                                       num_armies - 1) # Convert to 0-indexed armies
                    else:
                        agent_action_dict["phase_transition"] = 2 # End fortify
                else:
                    agent_action_dict["phase_transition"] = 2 # End fortify

        # Fill in dummy log_prob_dict and value
        for key in ["reinforce_t", "reinforce_a", "attack", "fortify", "trade_cards", "phase_transition"]:
            log_prob_dict[key] = 0.0 # Dummy
        return agent_action_dict, log_prob_dict, value


class DefensiveAgent(BaseRuleBasedAgent):
    """
    An AI agent that prioritizes defense and only attacks when highly advantageous.
    """
    def select_action(self, state, env_phase):
        agent_action_dict = {}
        log_prob_dict = {}
        value = 0.0

        owned_territories_info = self._get_owned_territories_info()
        owned_territories = list(owned_territories_info.keys())
        all_territory_indices = list(range(len(self.env.territory_names)))

        if env_phase == "reinforcement":
            # Trade cards if possible and has 5+ cards (to get more armies)
            if len(self.env.player_states[self.player_id]['cards']) >= 5: # Try to save for bigger bonus
                agent_action_dict["trade_cards"] = 1
            else:
                agent_action_dict["trade_cards"] = 0

            # Prioritize reinforcing border territories with fewest armies
            border_territories = self._get_border_territories()
            if border_territories:
                target_territory = min(border_territories, key=lambda t: owned_territories_info[t])
            elif owned_territories:
                target_territory = max(owned_territories, key=lambda t: owned_territories_info[t]) # Reinforce strongest if no borders
            else:
                target_territory = random.choice(self.env.territory_names) # Fallback

            agent_action_dict["reinforce_t"] = get_territory_idx_from_name(target_territory, self.env)
            agent_action_dict["reinforce_a"] = 4 # Place max armies (5)

            agent_action_dict["phase_transition"] = 0 # Always try to place armies, then end phase

        elif env_phase == "attack":
            # Very defensive: only attack if overwhelming advantage
            best_attack = None
            max_advantage = 3.0 # Attacker armies / Defender armies
            for attacker_t in owned_territories:
                if owned_territories_info[attacker_t] > 1: # Ensure enough armies to attack
                    for defender_t in self._get_valid_attack_targets(attacker_t):
                        defender_armies = self.env.player_states[self.env.territories[defender_t]]['armies'][defender_t]
                        if defender_armies > 0: # Avoid division by zero
                            advantage = owned_territories_info[attacker_t] / defender_armies
                            if advantage > max_advantage:
                                max_advantage = advantage
                                best_attack = (attacker_t, defender_t, min(3, owned_territories_info[attacker_t] - 1))

            if best_attack:
                attacker_t, defender_t, num_dice = best_attack
                agent_action_dict["attack"] = (get_territory_idx_from_name(attacker_t, self.env),
                                              get_territory_idx_from_name(defender_t, self.env),
                                              num_dice - 1)
                agent_action_dict["phase_transition"] = 0 # Attack
            else:
                agent_action_dict["phase_transition"] = 1 # End attack

        elif env_phase == "fortify":
            # Consolidate armies to border territories or weakest points
            border_territories = self._get_border_territories()
            if border_territories:
                # Move armies from interior to weakest border
                interior_territories = [t for t in owned_territories if t not in border_territories]
                if interior_territories:
                    from_t = max(interior_territories, key=lambda t: owned_territories_info[t]) # Strongest interior
                    to_t = min(border_territories, key=lambda t: owned_territories_info[t]) # Weakest border
                    # Ensure there are enough armies to move
                    if owned_territories_info[from_t] > 1 and to_t in self._get_valid_fortify_paths(from_t):
                        num_armies = owned_territories_info[from_t] - 1 # Move all but one
                        agent_action_dict["fortify"] = (get_territory_idx_from_name(from_t, self.env),
                                                       get_territory_idx_from_name(to_t, self.env),
                                                       num_armies - 1)
                        agent_action_dict["phase_transition"] = 0 # Fortify
                    else:
                        agent_action_dict["phase_transition"] = 2 # End fortify
                else:
                    agent_action_dict["phase_transition"] = 2 # End fortify
            else:
                agent_action_dict["phase_transition"] = 2 # End fortify

        for key in ["reinforce_t", "reinforce_a", "attack", "fortify", "trade_cards", "phase_transition"]:
            log_prob_dict[key] = 0.0
        return agent_action_dict, log_prob_dict, value


class BalancedAgent(BaseRuleBasedAgent):
    """
    An AI agent that tries to balance offensive and defensive strategies,
    aiming for continent control.
    """
    def select_action(self, state, env_phase):
        agent_action_dict = {}
        log_prob_dict = {}
        value = 0.0

        owned_territories_info = self._get_owned_territories_info()
        owned_territories = list(owned_territories_info.keys())
        all_territory_indices = list(range(len(self.env.territory_names)))

        if env_phase == "reinforcement":
            # Trade cards if possible and has 3+ cards
            if len(self.env.player_states[self.player_id]['cards']) >= 3:
                agent_action_dict["trade_cards"] = 1
            else:
                agent_action_dict["trade_cards"] = 0

            # Prioritize reinforcing territories that help secure a continent or border territories
            reinforce_target = None
            for continent_name, territories_in_continent in self.env.continents.items():
                # Check if close to controlling a continent
                player_continent_territories = [t for t in territories_in_continent if t in owned_territories]
                enemy_continent_territories = [t for t in territories_in_continent if t not in owned_territories]
                if len(player_continent_territories) > 0 and len(enemy_continent_territories) <= 1: # Close to controlling
                    if enemy_continent_territories:
                        # Reinforce a territory that can attack the last enemy territory in the continent
                        for t in owned_territories:
                            if t in self.env.adjacency_list.get(t, []) and enemy_continent_territories[0] in self.env.adjacency_list.get(t, []):
                                reinforce_target = t
                                break
                    if not reinforce_target: # If no direct attacker, reinforce strongest in continent
                        reinforce_target = max(player_continent_territories, key=lambda t: owned_territories_info[t])
                    break # Found a continent to focus on

            if not reinforce_target:
                # If no continent focus, reinforce border territories
                border_territories = self._get_border_territories()
                if border_territories:
                    reinforce_target = max(border_territories, key=lambda t: owned_territories_info[t]) # Reinforce strongest border
                elif owned_territories:
                    reinforce_target = max(owned_territories, key=lambda t: owned_territories_info[t]) # Reinforce strongest overall
                else:
                    reinforce_target = random.choice(self.env.territory_names) # Fallback

            agent_action_dict["reinforce_t"] = get_territory_idx_from_name(reinforce_target, self.env)
            agent_action_dict["reinforce_a"] = 4 # Place max armies (5)

            agent_action_dict["phase_transition"] = 0 # Always try to place armies, then end phase

        elif env_phase == "attack":
            # Balanced attack: look for good opportunities, but not overly aggressive
            best_attack = None
            max_value_attack = -1.0 # Value based on army ratio and strategic importance
            for attacker_t in owned_territories:
                if owned_territories_info[attacker_t] > 1: # Ensure enough armies to attack
                    for defender_t in self._get_valid_attack_targets(attacker_t):
                        defender_armies = self.env.player_states[self.env.territories[defender_t]]['armies'][defender_t]
                        if defender_armies > 0:
                            advantage = owned_territories_info[attacker_t] / defender_armies
                            # Simple heuristic: prioritize weaker enemies, or enemies that complete continents
                            strategic_value = advantage
                            if defender_t in self.env.continents.get(self.env.territories[defender_t], []): # If attacking a key continent territory
                                strategic_value *= 1.5 # Boost value

                            if strategic_value > max_value_attack:
                                max_value_attack = strategic_value
                                best_attack = (attacker_t, defender_t, min(3, owned_territories_info[attacker_t] - 1))

            if best_attack and max_value_attack > 1.5: # Only attack if reasonable advantage
                attacker_t, defender_t, num_dice = best_attack
                agent_action_dict["attack"] = (get_territory_idx_from_name(attacker_t, self.env),
                                              get_territory_idx_from_name(defender_t, self.env),
                                              num_dice - 1)
                agent_action_dict["phase_transition"] = 0 # Attack
            else:
                agent_action_dict["phase_transition"] = 1 # End attack

        elif env_phase == "fortify":
            # Consolidate armies to attack points or defend weak borders
            # Find strongest interior territory and weakest border territory
            interior_territories = [t for t in owned_territories if t not in self._get_border_territories()] # Fixed typo here
            border_territories = self._get_border_territories()

            if interior_territories and border_territories:
                from_t = max(interior_territories, key=lambda t: owned_territories_info[t])
                to_t = min(border_territories, key=lambda t: owned_territories_info[t])
                # Ensure there are enough armies to move
                if owned_territories_info[from_t] > 1 and to_t in self._get_valid_fortify_paths(from_t):
                    num_armies = owned_territories_info[from_t] - 1
                    agent_action_dict["fortify"] = (get_territory_idx_from_name(from_t, self.env),
                                                   get_territory_idx_from_name(to_t, self.env),
                                                   num_armies - 1)
                    agent_action_dict["phase_transition"] = 0 # Fortify
                else:
                    agent_action_dict["phase_transition"] = 2 # End fortify
            else:
                agent_action_dict["phase_transition"] = 2 # End fortify

        for key in ["reinforce_t", "reinforce_a", "attack", "fortify", "trade_cards", "phase_transition"]:
            log_prob_dict[key] = 0.0
        return agent_action_dict, log_prob_dict, value


class StrategicAgent(BaseRuleBasedAgent):
    """
    An AI agent that implements a specific strategic logic:
    - Capture smallest nearby continent.
    - Keep borders tight (stack on fewest entry points).
    - Attack with numerical advantage.
    - Break opponent bonuses.
    - Finish off weak players.
    - Fortify after successful attacks.
    - Trade cards strategically.
    - Keep a mobile stack.
    """
    def select_action(self, state, env_phase):
        agent_action_dict = {}
        log_prob_dict = {}
        value = 0.0

        owned_territories_info = self._get_owned_territories_info()
        owned_territories = list(owned_territories_info.keys())
        
        # --- Helper: Identify Continents and Borders ---
        continent_status = {} # continent_name: {owned_count, total_count, enemy_count}
        for continent, territories in self.env.continents.items():
            owned = [t for t in territories if t in owned_territories]
            continent_status[continent] = {
                "owned_count": len(owned),
                "total_count": len(territories),
                "owned_territories": owned,
                "enemy_territories": [t for t in territories if t not in owned_territories]
            }

        # --- Helper: Identify Weak Players ---
        weak_players = []
        for pid in self.env.active_players:
            if pid != self.player_id:
                p_territories = len(self.env.player_states[pid]['territories'])
                p_cards = len(self.env.player_states[pid]['cards'])
                if p_territories < 5 and p_cards >= 3: # Weak but juicy
                    weak_players.append(pid)

        if env_phase == "reinforcement":
            # 1. Card Trades: Trade if it swings a turn (get bonus armies) or forced (5+ cards)
            # "Swing a turn" -> if we have a good attack opportunity or need defense.
            # Simplified: Trade if >= 5 cards OR (>=3 cards AND we have a target continent)
            should_trade = False
            if len(self.env.player_states[self.player_id]['cards']) >= 5:
                should_trade = True
            elif len(self.env.player_states[self.player_id]['cards']) >= 3:
                # Check if we are close to a continent (1-2 territories away)
                for c, status in continent_status.items():
                    if status["total_count"] - status["owned_count"] <= 2:
                        should_trade = True
                        break
            
            agent_action_dict["trade_cards"] = 1 if should_trade else 0

            # 2. Reinforcement Logic
            # Priority 1: Smallest nearby continent to get early bonus
            target_continent = None
            best_score = -1
            
            for c, status in continent_status.items():
                if status["owned_count"] == status["total_count"]: continue # Already owned
                if status["owned_count"] == 0: continue # Ignore if we have no presence
                
                # Score = (Owned / Total) * (1 / Total) -> Prefer smaller, mostly owned continents
                score = (status["owned_count"] / status["total_count"]) * (10 / status["total_count"])
                if score > best_score:
                    best_score = score
                    target_continent = c

            reinforce_target = None
            
            # If we have a target continent, reinforce its borders or attack points
            if target_continent:
                # Find territory in continent that borders an enemy in the continent
                candidates = []
                for t in continent_status[target_continent]["owned_territories"]:
                    # Check neighbors
                    for adj in self.env.adjacency_list.get(t, []):
                        if adj in continent_status[target_continent]["enemy_territories"]:
                            candidates.append(t)
                            break
                if candidates:
                    reinforce_target = max(candidates, key=lambda t: owned_territories_info[t]) # Reinforce strongest candidate
            
            # Priority 2: Keep borders tight (stack on fewest entry points)
            if not reinforce_target:
                border_territories = self._get_border_territories()
                if border_territories:
                    # Find border with most enemy neighbors (choke point)
                    best_border = None
                    max_threat = -1
                    for t in border_territories:
                        threat = 0
                        for adj in self.env.adjacency_list.get(t, []):
                            if self.env.territories[adj] != self.player_id:
                                threat += self.env.player_states[self.env.territories[adj]]['armies'][adj]
                        if threat > max_threat:
                            max_threat = threat
                            best_border = t
                    reinforce_target = best_border

            # Priority 3: Mobile Stack (Reinforce strongest stack if nothing else)
            if not reinforce_target and owned_territories:
                reinforce_target = max(owned_territories, key=lambda t: owned_territories_info[t])

            # Fallback
            if not reinforce_target:
                reinforce_target = random.choice(owned_territories) if owned_territories else self.env.territory_names[0]

            agent_action_dict["reinforce_t"] = get_territory_idx_from_name(reinforce_target, self.env)
            agent_action_dict["reinforce_a"] = 4 # Max armies
            agent_action_dict["phase_transition"] = 0 

        elif env_phase == "attack":
            best_attack = None
            
            # 1. Finish off weak players
            for pid in weak_players:
                # Find if we can attack any territory of this player
                for t in self.env.player_states[pid]['territories']:
                    # Find our neighbors
                    for adj in self.env.adjacency_list.get(t, []):
                        if adj in owned_territories and owned_territories_info[adj] > 3:
                            # Attack!
                            best_attack = (adj, t, min(3, owned_territories_info[adj] - 1))
                            break
                if best_attack: break

            # 2. Break opponent continent bonuses
            if not best_attack:
                for c, status in continent_status.items():
                    # Check if an enemy owns this continent (or close to it)
                    # Simplified: Check if any player owns all territories
                    # Actually, we can just check if we can attack a territory that breaks a bonus
                    # This requires knowing who owns what continent. 
                    # For now, just attack if we can take a territory in a continent we want.
                    pass 

            # 3. Attack with clear numerical advantage (Advantage > 1.5)
            if not best_attack:
                possible_attacks = []
                for attacker_t in owned_territories:
                    if owned_territories_info[attacker_t] > 1:
                        for defender_t in self._get_valid_attack_targets(attacker_t):
                            defender_armies = self.env.player_states[self.env.territories[defender_t]]['armies'][defender_t]
                            if defender_armies > 0:
                                advantage = owned_territories_info[attacker_t] / defender_armies
                                if advantage > 1.5:
                                    possible_attacks.append((attacker_t, defender_t, advantage))
                
                if possible_attacks:
                    # Sort by advantage
                    possible_attacks.sort(key=lambda x: x[2], reverse=True)
                    attacker_t, defender_t, _ = possible_attacks[0]
                    best_attack = (attacker_t, defender_t, min(3, owned_territories_info[attacker_t] - 1))

            if best_attack:
                attacker_t, defender_t, num_dice = best_attack
                agent_action_dict["attack"] = (get_territory_idx_from_name(attacker_t, self.env),
                                              get_territory_idx_from_name(defender_t, self.env),
                                              num_dice - 1)
                agent_action_dict["phase_transition"] = 0
            else:
                agent_action_dict["phase_transition"] = 1 # End attack

        elif env_phase == "fortify":
            # After successful attack, stop and fortify.
            # Keep borders tight.
            # Use card trades only when they swing a turn... (handled in reinforce)
            # Keep one large mobile troop stack.
            
            # Strategy: Move armies from interior to border, OR combine stacks.
            border_territories = self._get_border_territories()
            interior_territories = [t for t in owned_territories if t not in border_territories]
            
            from_t = None
            to_t = None
            
            # Move from interior to border
            if interior_territories and border_territories:
                # Find strongest interior
                candidate_from = max(interior_territories, key=lambda t: owned_territories_info[t])
                if owned_territories_info[candidate_from] > 1:
                    from_t = candidate_from
                    # Find closest/weakest border
                    # Simplified: just random border reachable
                    valid_paths = self._get_valid_fortify_paths(from_t)
                    valid_borders = [t for t in valid_paths if t in border_territories]
                    if valid_borders:
                        to_t = min(valid_borders, key=lambda t: owned_territories_info[t]) # Reinforce weakest border
            
            # If no interior move, try to consolidate borders (mobile stack)
            if not from_t and len(border_territories) > 1:
                # Find strongest border stack
                strongest_border = max(border_territories, key=lambda t: owned_territories_info[t])
                if owned_territories_info[strongest_border] > 1:
                    # Move to a neighbor border if possible to make a "deathball"?
                    # Or move TO the strongest from a neighbor?
                    # User said "Keep one large mobile troop stack".
                    # So move TO the strongest.
                    valid_paths = self._get_valid_fortify_paths(strongest_border) # Wait, this is 'from'.
                    # We want to move TO strongest_border FROM somewhere else.
                    # Find a neighbor of strongest_border that is also owned
                    neighbors = [n for n in self.env.adjacency_list.get(strongest_border, []) if n in owned_territories]
                    if neighbors:
                        # Pick neighbor with most armies
                        best_neighbor = max(neighbors, key=lambda t: owned_territories_info[t])
                        if owned_territories_info[best_neighbor] > 1:
                            from_t = best_neighbor
                            to_t = strongest_border

            if from_t and to_t:
                num_armies = owned_territories_info[from_t] - 1
                agent_action_dict["fortify"] = (get_territory_idx_from_name(from_t, self.env),
                                               get_territory_idx_from_name(to_t, self.env),
                                               num_armies - 1)
                agent_action_dict["phase_transition"] = 0
            else:
                agent_action_dict["phase_transition"] = 2 # End fortify

        for key in ["reinforce_t", "reinforce_a", "attack", "fortify", "trade_cards", "phase_transition"]:
            log_prob_dict[key] = 0.0
        return agent_action_dict, log_prob_dict, value
