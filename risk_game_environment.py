import numpy as np
import random
from collections import defaultdict

class RiskEnv:
    """
    A simplified environment for the classic Risk board game.
    Supports 4 players and basic game mechanics.
    """
    def __init__(self, num_players=4):
        if not (2 <= num_players <= 6):
            raise ValueError("Number of players must be between 2 and 6.")
        self.num_players = num_players

        # Define all 42 territories
        # Ensure territory_names is consistent with adjacency list and continents
        self.territory_names = [
            "Alaska", "Northwest Territory", "Greenland", "Alberta", "Ontario", "Quebec",
            "Western United States", "Eastern United States", "Central America",
            "Venezuela", "Peru", "Brazil", "Argentina",
            "Iceland", "Scandinavia", "Ukraine", "Great Britain", "Northern Europe",
            "Western Europe", "Southern Europe",
            "North Africa", "Egypt", "East Africa", "Congo", "South Africa", "Madagascar", # Congo is Central Africa from earlier
            "Ural", "Siberia", "Yakutsk", "Kamchatka", "Irkutsk", "Mongolia",
            "Japan", "Afghanistan", "China", "Middle East", "India", "Siam",
            "Indonesia", "New Guinea", "Western Australia", "Eastern Australia"
        ]
        self.num_territories = len(self.territory_names)

        # Define adjacency list for territories
        self.adjacency_list = {
            "Alaska": ["Northwest Territory", "Alberta", "Kamchatka"],
            "Northwest Territory": ["Alaska", "Alberta", "Ontario", "Greenland"],
            "Greenland": ["Northwest Territory", "Ontario", "Quebec", "Iceland"],
            "Alberta": ["Alaska", "Northwest Territory", "Ontario", "Western United States"],
            "Ontario": ["Northwest Territory", "Alberta", "Western United States", "Eastern United States", "Quebec"],
            "Quebec": ["Ontario", "Eastern United States", "Greenland"],
            "Western United States": ["Alberta", "Ontario", "Eastern United States", "Central America"],
            "Eastern United States": ["Ontario", "Quebec", "Western United States", "Central America"],
            "Central America": ["Western United States", "Eastern United States", "Venezuela"],
            "Venezuela": ["Central America", "Brazil", "Peru"],
            "Peru": ["Venezuela", "Brazil", "Argentina"],
            "Brazil": ["Venezuela", "Peru", "Argentina", "North Africa"],
            "Argentina": ["Peru", "Brazil"],
            "Iceland": ["Greenland", "Great Britain", "Scandinavia"],
            "Scandinavia": ["Iceland", "Ukraine", "Northern Europe", "Great Britain"],
            "Ukraine": ["Scandinavia", "Northern Europe", "Southern Europe", "Middle East", "Afghanistan", "Ural"],
            "Great Britain": ["Iceland", "Scandinavia", "Northern Europe", "Western Europe"],
            "Northern Europe": ["Great Britain", "Scandinavia", "Ukraine", "Southern Europe", "Western Europe"],
            "Western Europe": ["Great Britain", "Northern Europe", "Southern Europe", "North Africa"],
            "Southern Europe": ["Western Europe", "Northern Europe", "Ukraine", "Middle East", "Egypt", "North Africa"],
            "North Africa": ["Western Europe", "Southern Europe", "Egypt", "East Africa", "Congo", "Brazil"],
            "Egypt": ["Southern Europe", "Middle East", "East Africa", "North Africa"],
            "East Africa": ["Egypt", "Middle East", "Madagascar", "South Africa", "Congo", "North Africa"],
            "Congo": ["North Africa", "East Africa", "South Africa"], # Renamed from Central Africa to Congo
            "South Africa": ["Congo", "East Africa", "Madagascar"],
            "Madagascar": ["South Africa", "East Africa"],
            "Ural": ["Ukraine", "Siberia", "China", "Afghanistan"],
            "Siberia": ["Ural", "Yakutsk", "Irkutsk", "Mongolia", "China"],
            "Yakutsk": ["Siberia", "Kamchatka", "Irkutsk"],
            "Kamchatka": ["Yakutsk", "Alaska", "Irkutsk", "Mongolia", "Japan"],
            "Irkutsk": ["Siberia", "Yakutsk", "Kamchatka", "Mongolia"],
            "Mongolia": ["Irkutsk", "Kamchatka", "Japan", "China", "Siberia"],
            "Japan": ["Kamchatka", "Mongolia"],
            "Afghanistan": ["Ukraine", "Ural", "China", "India", "Middle East"],
            "China": ["Ural", "Siberia", "Mongolia", "India", "Siam", "Afghanistan"],
            "Middle East": ["Ukraine", "Afghanistan", "India", "Southern Europe", "Egypt", "East Africa"],
            "India": ["Middle East", "Afghanistan", "China", "Siam"],
            "Siam": ["India", "China", "Indonesia"],
            "Indonesia": ["Siam", "New Guinea", "Western Australia"],
            "New Guinea": ["Indonesia", "Eastern Australia", "Western Australia"],
            "Western Australia": ["Indonesia", "New Guinea", "Eastern Australia"],
            "Eastern Australia": ["Western Australia", "New Guinea"]
        }

        # Define continents and their bonus armies
        self.continents = {
            "North America": ["Alaska", "Northwest Territory", "Greenland", "Alberta", "Ontario", "Quebec",
                              "Western United States", "Eastern United States", "Central America"],
            "South America": ["Venezuela", "Peru", "Brazil", "Argentina"],
            "Europe": ["Iceland", "Scandinavia", "Ukraine", "Great Britain", "Northern Europe", "Western Europe",
                       "Southern Europe"],
            "Africa": ["North Africa", "Egypt", "East Africa", "Congo", "South Africa", "Madagascar"],
            "Asia": ["Ural", "Siberia", "Yakutsk", "Kamchatka", "Irkutsk", "Mongolia", "Japan", "Afghanistan",
                     "China", "Middle East", "India", "Siam"],
            "Australia": ["Indonesia", "New Guinea", "Western Australia", "Eastern Australia"]
        }
        self.continent_bonuses = {
            "North America": 5, "South America": 2, "Europe": 5, "Africa": 3, "Asia": 7, "Australia": 2
        }

        # Territory cards (simplified: one card per territory)
        self.card_deck = list(self.territory_names) # Each territory has a corresponding card
        self.trade_in_bonus = [4, 6, 8, 10, 12, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60] # Bonus armies for trading sets
        self.trade_index = 0 # Tracks which bonus to give next

        # Game state variables
        self.player_states = defaultdict(lambda: {'territories': set(), 'armies': defaultdict(int), 'cards': []})
        self.territories = {} # Territory name: owner (player_id)
        self.current_player = None
        self.game_over = False
        self.winner = None
        self.turn_phase = None # "reinforcement", "attack", "fortify"
        self.attack_conquered_this_turn = False # Flag to track if a territory was conquered for card purposes
        self.active_players = set() # Keep track of players still in the game
        self.num_turns = 0 # Track total turns in an episode

        # Action tracking for logging
        self.action_counts = {
            "attack": 0,
            "reinforce": 0,
            "fortify": 0,
            "trade_cards": 0,
            "conquer": 0,
            "end_reinforcement": 0,
            "end_attack": 0,
            "end_fortify": 0,
        }
        
        self.MAX_ARMIES = 50 # Standardize normalization constant

        # Pre-allocate masks for performance
        self.mask_buffers = {
            "reinforce_t": np.zeros(self.num_territories, dtype=np.float32),
            "reinforce_a": np.zeros(5, dtype=np.float32),
            "attack_attacker": np.zeros(self.num_territories, dtype=np.float32),
            "attack_defender": np.zeros((self.num_territories, self.num_territories), dtype=np.float32),
            "attack_dice": np.zeros(3, dtype=np.float32),
            "fortify_from": np.zeros(self.num_territories, dtype=np.float32),
            "fortify_to": np.zeros((self.num_territories, self.num_territories), dtype=np.float32),
            "fortify_armies": np.zeros(10, dtype=np.float32),
            "trade_cards": np.zeros(2, dtype=np.float32),
            "phase_transition": np.zeros(3, dtype=np.float32)
        }

    def reset(self):
        """
        Resets the game environment to its initial state for a new episode.
        """
        self.player_states = defaultdict(lambda: {'territories': set(), 'armies': defaultdict(int), 'cards': []})
        self.territories = {} # Territory name: owner (player_id)
        self.current_player = random.choice(list(range(self.num_players)))
        self.game_over = False
        self.winner = None
        self.turn_phase = "reinforcement"
        self.attack_conquered_this_turn = False
        self.trade_index = 0
        self.active_players = set(range(self.num_players))
        self.num_turns = 0

        # Reset action counts for the new episode
        self.action_counts = {
            "attack": 0, "reinforce": 0, "fortify": 0, "trade_cards": 0,
            "conquer": 0, "end_reinforcement": 0, "end_attack": 0, "end_fortify": 0,
        }

        self._initialize_board()
        return self._get_state()

    def _initialize_board(self):
        """
        Distributes territories and initial armies among players.
        """
        all_territories_shuffled = list(self.territory_names)
        random.shuffle(all_territories_shuffled)

        # Assign territories
        for i, territory in enumerate(all_territories_shuffled):
            player_id = i % self.num_players
            self.player_states[player_id]['territories'].add(territory)
            self.player_states[player_id]['armies'][territory] = 1 # Each territory starts with 1 army
            self.territories[territory] = player_id # Update global territory owner map

        # Distribute initial remaining armies
        initial_armies_per_player = {2: 40, 3: 35, 4: 30, 5: 25, 6: 20}[self.num_players]
        for player_id in range(self.num_players):
            armies_to_place = initial_armies_per_player - len(self.player_states[player_id]['territories'])
            self._place_initial_remaining_armies(player_id, armies_to_place)

        # Shuffle and prepare the card deck
        random.shuffle(self.card_deck)


    def _place_initial_remaining_armies(self, player_id, num_armies):
        """
        Places remaining initial armies on a player's owned territories.
        Simple strategy: distribute evenly, then randomly for remainder.
        """
        owned_territories = list(self.player_states[player_id]['territories'])
        if not owned_territories:
            return # Player has no territories (shouldn't happen at start)

        armies_per_territory = num_armies // len(owned_territories)
        remainder = num_armies % len(owned_territories)

        for territory in owned_territories:
            self.player_states[player_id]['armies'][territory] += armies_per_territory

        # Distribute remainder randomly
        random.shuffle(owned_territories)
        for i in range(remainder):
            self.player_states[player_id]['armies'][owned_territories[i]] += 1

    def _get_state(self):
        """
        Generates a numerical representation of the current game state for the AI.
        State vector components:
        - Territory owners (42 elements: player ID)
        - Army counts (42 elements: number of armies)
        - Current player ID (1 element)
        - Turn phase (3 elements: one-hot encoded for reinforcement, attack, fortify)
        - Current player's hand size (1 element)
        - Current player's continent control (6 elements: 1 if controlled, 0 otherwise)
        - Opponent territory counts (num_players elements: count for each player)
        """
        # 1. Territory owners (42 elements) - one-hot encoded for each player
        territory_owner_one_hot = np.zeros((self.num_territories, self.num_players), dtype=int)
        # 2. Army counts (42 elements)
        army_counts_vec = np.zeros(self.num_territories, dtype=int)

        for i, name in enumerate(self.territory_names):
            owner = self.territories.get(name, -1) # Use .get() for safety
            if owner != -1:
                territory_owner_one_hot[i, owner] = 1
                army_counts_vec[i] = self.player_states[owner]['armies'].get(name, 0) # Use .get() for safety

        # Flatten territory owners one-hot
        territory_owners_vec = territory_owner_one_hot.flatten()

        # 3. Current player ID (1 element) - one-hot encoded
        current_player_one_hot = np.zeros(self.num_players)
        if self.current_player is not None:
            current_player_one_hot[self.current_player] = 1

        # 4. Turn phase (3 elements)
        turn_phase_one_hot = np.zeros(3)
        if self.turn_phase == "reinforcement":
            turn_phase_one_hot[0] = 1
        elif self.turn_phase == "attack":
            turn_phase_one_hot[1] = 1
        else: # fortify
            turn_phase_one_hot[2] = 1

        # 5. Current player's hand size (1 element)
        hand_size_vec = np.array([len(self.player_states[self.current_player]['cards'])]) if self.current_player is not None else np.array([0])

        # 6. Current player's continent control (6 elements)
        continent_control_vec = np.zeros(len(self.continents))
        continent_names = list(self.continents.keys())
        if self.current_player is not None:
            for i, continent_name in enumerate(continent_names):
                controls_continent = True
                for territory in self.continents[continent_name]:
                    if self.territories.get(territory) != self.current_player:
                        controls_continent = False
                        break
                if controls_continent:
                    continent_control_vec[i] = 1

        # 7. All player territory counts (num_players elements)
        all_player_territory_counts_vec = np.zeros(self.num_players)
        for i in range(self.num_players):
            all_player_territory_counts_vec[i] = len(self.player_states[i]['territories'])

        # Concatenate all parts to form the final state vector
        state = np.concatenate([
            territory_owners_vec, # 42 * num_players
            army_counts_vec,      # 42
            current_player_one_hot, # num_players
            turn_phase_one_hot,   # 3
            hand_size_vec,        # 1
            continent_control_vec,# 6
            all_player_territory_counts_vec # num_players
        ]).astype(np.float32) # Ensure float32 for PyTorch

        # Normalize army counts (important for neural networks)
        state[self.num_territories * self.num_players : self.num_territories * self.num_players + self.num_territories] /= 50.0 # Assuming max 50 armies on a territory

        return state

    def _calculate_reinforcements(self, player_id):
        """
        Calculates the number of new armies a player receives at the start of their turn.
        """
        num_territories = len(self.player_states[player_id]['territories'])
        reinforcements = max(3, num_territories // 3) # Minimum 3 armies

        # Add continent bonuses
        for continent, territories in self.continents.items():
            controls_continent = True
            for territory in territories:
                if self.territories.get(territory) != player_id: # Use .get() for safety
                    controls_continent = False
                    break
            if controls_continent:
                reinforcements += self.continent_bonuses[continent]
        return reinforcements

    def _can_attack(self, attacker_territory, defender_territory):
        """
        Checks if an attack is valid according to Risk rules.
        """
        current_player_id = self.current_player
        if attacker_territory not in self.player_states[current_player_id]['territories']:
            return False # Attacker doesn't own the territory
        if defender_territory not in self.adjacency_list.get(attacker_territory, []):
            return False # Territories are not adjacent
        if self.territories.get(defender_territory) == current_player_id: # Use .get() for safety
            return False # Cannot attack own territory
        if self.player_states[current_player_id]['armies'][attacker_territory] < 2:
            return False # Must have at least 2 armies to attack (1 stays behind)
        return True

    def _resolve_combat(self, attacker_id, attacker_territory, defender_id, defender_territory, num_attacker_dice):
        """
        Resolves a single combat round using dice rolls.
        Returns casualties for attacker and defender, and if territory was conquered.
        """
        attacker_armies_on_t = self.player_states[attacker_id]['armies'][attacker_territory]
        defender_armies_on_t = self.player_states[defender_id]['armies'][defender_territory]

        attacker_dice = min(attacker_armies_on_t - 1, num_attacker_dice) # Max 3 dice, min 1 army left
        defender_dice = min(defender_armies_on_t, 2) # Max 2 dice

        # Ensure at least 1 die is rolled if possible
        if attacker_dice < 1: attacker_dice = 1 if attacker_armies_on_t > 1 else 0
        if defender_dice < 1: defender_dice = 1 if defender_armies_on_t > 0 else 0

        if attacker_dice == 0 and defender_dice == 0: # No dice to roll, battle can't happen
             return {'attackers_lost': 0, 'defenders_lost': 0, 'conquered': False}

        attacker_rolls = sorted(random.sample(range(1, 7), attacker_dice), reverse=True)
        defender_rolls = sorted(random.sample(range(1, 7), defender_dice), reverse=True)

        casualties = {'attackers_lost': 0, 'defenders_lost': 0}
        
        # Compare dice rolls
        for i in range(min(len(attacker_rolls), len(defender_rolls))):
            if attacker_rolls[i] > defender_rolls[i]:
                casualties['defenders_lost'] += 1
            else: # Defender wins ties
                casualties['attackers_lost'] += 1
        
        self.player_states[attacker_id]['armies'][attacker_territory] -= casualties['attackers_lost']
        self.player_states[defender_id]['armies'][defender_territory] -= casualties['defenders_lost']

        conquered = False
        if self.player_states[defender_id]['armies'][defender_territory] <= 0:
            conquered = True

        return {'attackers_lost': casualties['attackers_lost'], 
                'defenders_lost': casualties['defenders_lost'], 
                'conquered': conquered}

    def _trade_cards(self, player_id, card_indices):
        """
        Attempts to trade in a set of 3 cards for bonus armies.
        Returns (bonus_armies, success_flag).
        """
        if len(card_indices) != 3:
            return 0, False

        # Ensure unique indices and valid range
        unique_indices = sorted(list(set(card_indices)), reverse=True) # Sort in reverse to pop safely
        if len(unique_indices) != 3 or any(idx < 0 or idx >= len(self.player_states[player_id]['cards']) for idx in unique_indices):
            return 0, False

        cards_to_trade = [self.player_states[player_id]['cards'][i] for i in unique_indices]

        # Check for valid set (3 of a kind or 1 of each kind)
        # Simplified card types (e.g., based on continent for demonstration)
        # In a real game, cards have types (infantry, cavalry, artillery) or wildcards
        card_types = set()
        for card_name in cards_to_trade:
            # Assign a 'type' based on some arbitrary grouping for demonstration
            # This simplification may not match exact Risk card types (infantry, cavalry, artillery, wildcard)
            # A more robust system would map each territory card to its type, and handle wildcards.
            if card_name in self.continents["North America"] or card_name in self.continents["South America"]:
                card_types.add("typeA")
            elif card_name in self.continents["Europe"] or card_name in self.continents["Africa"]:
                card_types.add("typeB")
            else: # Asia, Australia
                card_types.add("typeC")

        is_valid_set = (len(card_types) == 1) or (len(card_types) == 3) # All same type or all different types

        if not is_valid_set:
            return 0, False

        # If valid, calculate bonus armies
        bonus_armies = self.trade_in_bonus[min(self.trade_index, len(self.trade_in_bonus) - 1)]
        self.trade_index += 1

        # Remove cards from player's hand and return to deck
        for idx in unique_indices:
            card = self.player_states[player_id]['cards'].pop(idx)
            self.card_deck.append(card) # Return to bottom of deck (simplified)
        random.shuffle(self.card_deck) # Reshuffle deck

        # Add 2 bonus armies for each matching territory card traded
        territory_matches_bonus = 0
        for card_name in cards_to_trade:
            if card_name in self.player_states[player_id]['territories']:
                territory_matches_bonus += 2 # 2 extra armies
                # self.player_states[player_id]['armies'][card_name] += 2 # Place armies directly on territory (not in general pool)
        
        return bonus_armies + territory_matches_bonus, True


    def decode_action(self, agent_action_dict):
        """
        Translates the AI agent's chosen action (numerical indices)
        into the dictionary format expected by the RiskEnv.step() method.
        Includes validation to ensure actions are legal.
        """
        env_action = {}
        current_player_id = self.current_player
        owned_territories = list(self.player_states[current_player_id]['territories'])

        # --- Reinforcement Phase ---
        if self.turn_phase == "reinforcement":
            # Check for trade cards action first
            if agent_action_dict.get("trade_cards") == 1: # Agent chose to trade cards
                if len(self.player_states[current_player_id]['cards']) >= 3:
                    # For simplicity, pick 3 random cards to trade.
                    # A more advanced AI would learn which cards to trade.
                    card_indices_to_trade = random.sample(range(len(self.player_states[current_player_id]['cards'])), 3)
                    env_action = {"phase": "trade_cards", "card_indices": card_indices_to_trade}
                    return env_action
                else:
                    # If agent tries to trade without enough cards, default to end reinforcement
                    pass # Fall through to reinforcement placement or end phase

            # Check for reinforcement placement
            reinforce_t_idx = agent_action_dict.get("reinforce_t")
            reinforce_a_idx = agent_action_dict.get("reinforce_a") # 0-4 for 1-5 armies

            # Ensure indices are within valid range before converting
            if reinforce_t_idx is not None and reinforce_a_idx is not None and \
               0 <= reinforce_t_idx < len(self.territory_names) and \
               0 <= reinforce_a_idx < 5: # Max 5 armies (index 4)
                num_armies_to_place = reinforce_a_idx + 1 # Convert index to actual army count
                territory_to_reinforce = self.territory_names[reinforce_t_idx]
                reinforcements_available = self.player_states[current_player_id].get('reinforcements_available', 0)

                # If agent chose to place armies and it's a valid territory and armies are available
                if territory_to_reinforce in owned_territories and num_armies_to_place > 0 and \
                   num_armies_to_place <= reinforcements_available:
                    env_action = {"phase": "reinforce", "placements": {territory_to_reinforce: num_armies_to_place}}
                    return env_action

            # If no valid reinforcement or trade action, check for phase transition
            if agent_action_dict.get("phase_transition") == 0: # Agent chose to end reinforcement phase
                env_action = {"phase": "end_reinforcement"}
                return env_action

            # Default if no valid action chosen by agent (e.g., agent picked invalid reinforce, or no armies to place)
            # In this case, we effectively force an end_reinforcement to avoid infinite loops.
            return {"phase": "end_reinforcement"}

        # --- Attack Phase ---
        elif self.turn_phase == "attack":
            # Check for phase transition first
            if agent_action_dict.get("phase_transition") == 1: # Agent chose to end attack phase
                env_action = {"phase": "end_attack"}
                return env_action

            # Otherwise, try to interpret an attack action
            attack_tuple = agent_action_dict.get("attack")
            if attack_tuple is not None and len(attack_tuple) == 3:
                attacker_idx, defender_idx, dice_idx = attack_tuple
                num_dice = dice_idx + 1 # Convert index to actual dice count (1-3)

                # Ensure indices are within valid range
                if 0 <= attacker_idx < len(self.territory_names) and \
                   0 <= defender_idx < len(self.territory_names) and \
                   0 <= dice_idx < 3: # Max 3 dice (index 2)

                    attacker_t = self.territory_names[attacker_idx]
                    defender_t = self.territory_names[defender_idx]

                    # Validate attack action
                    if self._can_attack(attacker_t, defender_t) and \
                       self.player_states[current_player_id]['armies'][attacker_t] > num_dice: # Must have more armies than dice
                        return {"phase": "attack", "attacker": attacker_t, "defender": defender_t, "dice": num_dice}
            
            # If invalid attack action or no attack chosen, force end attack phase
            return {"phase": "end_attack"}

        # --- Fortify Phase ---
        elif self.turn_phase == "fortify":
            # Check for phase transition first
            if agent_action_dict.get("phase_transition") == 2: # Agent chose to end fortify phase
                env_action = {"phase": "end_fortify"}
                return env_action

            # Otherwise, try to interpret a fortify action
            fortify_tuple = agent_action_dict.get("fortify")
            if fortify_tuple is not None and len(fortify_tuple) == 3:
                from_idx, to_idx, armies_idx = fortify_tuple
                num_armies_to_move = armies_idx + 1 # Convert index to actual army count (1-10)

                # Ensure indices are within valid range
                if 0 <= from_idx < len(self.territory_names) and \
                   0 <= to_idx < len(self.territory_names) and \
                   0 <= armies_idx < 10: # Max 10 armies (index 9)

                    from_t = self.territory_names[from_idx]
                    to_t = self.territory_names[to_idx]

                    # Validate fortify action
                    if from_t in owned_territories and to_t in owned_territories and \
                       to_t in self.adjacency_list.get(from_t, []) and \
                       self.player_states[current_player_id]['armies'][from_t] > num_armies_to_move: # Must leave at least 1 army
                        return {"phase": "fortify", "from": from_t, "to": to_t, "armies": num_armies_to_move}
            
            # If invalid fortify action or no fortify chosen, force end fortify phase
            return {"phase": "end_fortify"}
        
        # Default fallback (should not be reached if phases are handled correctly)
        return {"phase": "end_fortify"}


    def step(self, action):
        """
        Applies a chosen action to the environment and advances the game state.
        Args:
            action (dict): A dictionary representing the action for the current phase:
                - {"phase": "reinforce", "placements": {"territory_name": num_armies, ...}}
                - {"phase": "attack", "attacker": "t1", "defender": "t2", "dice": num_dice}
                - {"phase": "fortify", "from": "t1", "to": "t2", "armies": num_armies}
                - {"phase": "trade_cards", "card_indices": [idx1, idx2, idx3]}
                - {"phase": "end_attack"} (to move from attack to fortify phase)
                - {"phase": "end_fortify"} (to end turn)
                - {"phase": "end_reinforcement"} (to move from reinforce to attack phase)

        Returns:
            (state, reward, done, info):
            - state (np.array): New state vector.
            - reward (float): Reward received for the action.
            - done (bool): True if the game is over, False otherwise.
            - info (dict): Additional information (e.g., winner, action_counts).
        """
        reward = 0.0
        done = False
        info = {}
        current_player_id = self.current_player
        
        # Capture current number of active players before any eliminations
        original_num_active_players = len(self.active_players) 

        # --- Handle player elimination early ---
        if current_player_id not in self.active_players:
            # This player has been eliminated. Switch to the next active player
            # and indicate that no meaningful action was taken for this step.
            self._switch_player()
            # Return current state, 0 reward, and not done (unless game is over)
            info["action_counts"] = self.action_counts.copy() # Ensure counts are always returned
            return self._get_state(), 0.0, self.game_over, info


        # --- Reinforcement Phase ---
        if self.turn_phase == "reinforcement":
            # If entering reinforcement for the first time this turn, calculate armies
            if 'reinforcements_available' not in self.player_states[current_player_id] or \
               self.player_states[current_player_id]['reinforcements_available'] == 0:
                self.player_states[current_player_id]['reinforcements_available'] = self._calculate_reinforcements(current_player_id)
            
            # Handle card trading first if requested
            if action.get("phase") == "trade_cards":
                self.action_counts["trade_cards"] += 1
                card_indices = action.get("card_indices", [])
                bonus_armies, success = self._trade_cards(current_player_id, card_indices)
                if success:
                    self.player_states[current_player_id]['reinforcements_available'] += bonus_armies
                    reward += 0.05 # A small positive reward for successful card trade
                else:
                    reward -= 0.01 # Penalize invalid card trade attempts
                # Stay in reinforcement phase after trade attempt, allowing placement
                info["action_counts"] = self.action_counts.copy()
                return self._get_state(), reward, done, info 

            # Handle army placement
            elif action.get("phase") == "reinforce":
                self.action_counts["reinforce"] += 1
                placements = action.get("placements", {})
                placed_count = 0

                for territory, num_armies in placements.items():
                    # Validate territory ownership and army count
                    if territory in self.player_states[current_player_id]['territories'] and num_armies > 0:
                        if placed_count + num_armies <= self.player_states[current_player_id]['reinforcements_available']:
                            self.player_states[current_player_id]['armies'][territory] += num_armies
                            placed_count += num_armies
                        else:
                            reward -= 0.05 # Penalize attempting to overplace
                            break # Stop processing further placements if invalid
                    else:
                        reward -= 0.05 # Penalize placing on non-owned or invalid territory
                        break
                
                self.player_states[current_player_id]['reinforcements_available'] -= placed_count

                # If all armies placed or no more to place, move to attack phase
                if self.player_states[current_player_id]['reinforcements_available'] <= 0:
                    self.turn_phase = "attack"
                # Else, stay in reinforcement to allow more placements/trades
                info["action_counts"] = self.action_counts.copy()
                return self._get_state(), reward, done, info

            # Explicitly end reinforcement phase
            elif action.get("phase") == "end_reinforcement":
                self.action_counts["end_reinforcement"] += 1
                # Any remaining reinforcements are lost (or placed here automatically)
                self.player_states[current_player_id]['reinforcements_available'] = 0 
                self.turn_phase = "attack"
                info["action_counts"] = self.action_counts.copy()
                return self._get_state(), reward, done, info

            # If no specific action for reinforcement, but phase is still reinforcement
            # This should generally not be reached if translate_agent_action_to_env_action works correctly
            # and ensures an explicit action is always sent.
            else:
                reward -= 0.05 # Penalize invalid or unhandled action in phase
                info["action_counts"] = self.action_counts.copy()
                return self._get_state(), reward, done, info

        # --- Attack Phase ---
        elif self.turn_phase == "attack":
            if action.get("phase") == "attack":
                self.action_counts["attack"] += 1
                attacker_t = action.get("attacker")
                defender_t = action.get("defender")
                num_dice = action.get("dice")

                if self._can_attack(attacker_t, defender_t) and 1 <= num_dice <= 3 \
                   and self.player_states[current_player_id]['armies'][attacker_t] > num_dice: # Ensure enough armies to roll dice
                    
                    defender_owner = self.territories[defender_t]
                    combat_result = self._resolve_combat(
                        current_player_id, attacker_t, 
                        defender_owner, defender_t, num_dice
                    )
                    
                    reward -= combat_result['attackers_lost'] * 0.01 # Small penalty for losing armies

                    # Check if defender territory is conquered
                    if combat_result['conquered']:
                        self.action_counts["conquer"] += 1
                        reward += 0.1 # Reward for conquering a territory
                        self.attack_conquered_this_turn = True # Flag for card drawing

                        # Transfer ownership
                        self.player_states[current_player_id]['territories'].add(defender_t)
                        self.player_states[defender_owner]['territories'].discard(defender_t) # Remove from old owner
                        self.territories[defender_t] = current_player_id # Update global map

                        # Move armies to conquered territory (must move at least num_dice)
                        # The player must move at least the number of dice they rolled.
                        # It is common to move all but one army. Let's simplify this for AI for now.
                        armies_to_move = min(num_dice, self.player_states[current_player_id]['armies'][attacker_t] - 1)
                        if armies_to_move < 1: armies_to_move = 1 # Must move at least 1 army
                        
                        self.player_states[current_player_id]['armies'][defender_t] = armies_to_move
                        self.player_states[current_player_id]['armies'][attacker_t] -= armies_to_move
                        
                        # Check for elimination
                        if not self.player_states[defender_owner]['territories']:
                            reward += 1.0 # Reward for eliminating a player
                            info['eliminated_player'] = defender_owner
                            self.active_players.discard(defender_owner) # Remove from active players

                            # Transfer eliminated player's cards to conqueror
                            self.player_states[current_player_id]['cards'].extend(self.player_states[defender_owner]['cards'])
                            self.player_states[defender_owner]['cards'] = [] # Clear eliminated player's hand

                            # Check for overall win condition (only 1 player left)
                            if len(self.active_players) == 1:
                                done = True
                                self.winner = current_player_id
                                reward += 10.0 # Big reward for winning
                    
                    # After an attack, the player can choose to attack again or end attack phase.
                    # The phase remains "attack" until an "end_attack" action is received.
                    info["action_counts"] = self.action_counts.copy()
                    info["defenders_lost"] = combat_result['defenders_lost'] # Expose damage dealt for reward shaping
                    return self._get_state(), reward, done, info

                else: # Invalid attack action
                    reward -= 0.05
                    info["action_counts"] = self.action_counts.copy()
                    return self._get_state(), reward, done, info

            elif action.get("phase") == "end_attack":
                self.action_counts["end_attack"] += 1
                # If a territory was conquered, give a card
                if self.attack_conquered_this_turn and self.card_deck:
                    card = self.card_deck.pop(0) # Draw top card
                    self.player_states[current_player_id]['cards'].append(card)
                    self.attack_conquered_this_turn = False # Reset flag
                self.turn_phase = "fortify"
                info["action_counts"] = self.action_counts.copy()
                return self._get_state(), reward, done, info

            else: # Invalid action for attack phase
                reward -= 0.05
                info["action_counts"] = self.action_counts.copy()
                return self._get_state(), reward, done, info

        # --- Fortify Phase ---
        elif self.turn_phase == "fortify":
            if action.get("phase") == "fortify":
                self.action_counts["fortify"] += 1
                from_t = action.get("from")
                to_t = action.get("to")
                num_armies = action.get("armies")

                # Check if territories are owned by current player, adjacent, and enough armies
                if from_t in self.player_states[current_player_id]['territories'] and \
                   to_t in self.player_states[current_player_id]['territories'] and \
                   to_t in self.adjacency_list.get(from_t, []) and \
                   self.player_states[current_player_id]['armies'][from_t] > num_armies and \
                   num_armies > 0:
                    self.player_states[current_player_id]['armies'][from_t] -= num_armies
                    self.player_states[current_player_id]['armies'][to_t] += num_armies
                    reward += 0.01 # Small positive reward for fortifying
                else:
                    reward -= 0.05 # Invalid fortify action
                
                # After fortify, the player's turn ends (only one fortify move allowed by rule)
                self._switch_player()
                self.turn_phase = "reinforcement" # Next player starts reinforcement
                # Calculate reinforcements for next player immediately
                if self.current_player in self.active_players:
                    self.player_states[self.current_player]['reinforcements_available'] = self._calculate_reinforcements(self.current_player)
                self.num_turns += 1 # Increment turn count only when a player's full turn ends
                info["action_counts"] = self.action_counts.copy()
                return self._get_state(), reward, done, info

            elif action.get("phase") == "end_fortify":
                self.action_counts["end_fortify"] += 1
                # Player chose not to fortify or finished fortifying
                self._switch_player()
                self.turn_phase = "reinforcement" # Next player starts reinforcement
                # Calculate reinforcements for next player immediately
                if self.current_player in self.active_players:
                    self.player_states[self.current_player]['reinforcements_available'] = self._calculate_reinforcements(self.current_player)
                self.num_turns += 1 # Increment turn count only when a player's full turn ends
                info["action_counts"] = self.action_counts.copy()
                return self._get_state(), reward, done, info
            else: # Invalid action for fortify phase
                reward -= 0.05
                info["action_counts"] = self.action_counts.copy()
                return self._get_state(), reward, done, info
        else:
            # Should not happen: unknown phase
            reward -= 0.1
            info["action_counts"] = self.action_counts.copy()
            return self._get_state(), reward, done, info

        # --- End of step: Check for game over conditions and reward ---
        # This part ensures that if a player wins or max turns are reached, 'done' is set.
        done = False
        if len(self.active_players) == 1:
            self.winner = list(self.active_players)[0]
            reward += 10.0 # Big reward for winning
            done = True
        elif self.num_turns >= 2000: # Max turns reached
            done = True
            # No winner, possibly a draw or stalemate
            self.winner = None 
            # You might want to penalize for not winning if this happens
            # reward -= 5.0 

        info["action_counts"] = self.action_counts.copy() # Ensure counts are always returned
        return self._get_state(), reward, done, info


    def _switch_player(self):
        """
        Switches to the next active player.
        """
        if not self.active_players: # No active players left (game over)
            self.game_over = True
            self.winner = None # No winner if all eliminated somehow
            return

        # Create a sorted list of active player IDs to ensure consistent cycling
        sorted_active_players = sorted(list(self.active_players))
        
        try:
            current_player_list_idx = sorted_active_players.index(self.current_player)
        except ValueError:
            # Current player was eliminated and is no longer in active_players
            # This should ideally be caught by the calling loop, but handle as fallback
            # Just pick the first active player for the next turn
            self.current_player = sorted_active_players[0]
            return

        next_player_list_idx = (current_player_list_idx + 1) % len(sorted_active_players)
        self.current_player = sorted_active_players[next_player_list_idx]

        # The loop in risk_ai_trainer will keep calling step if the player is not active.
        # This _switch_player ensures current_player always points to an active player.
        # No need for a while loop here as the active_players set is updated
        # by eliminations, and we only cycle through the current active set.

    def render(self):
        """
        Prints a simplified text-based representation of the board state.
        """
        print("\n--- Current Game State ---")
        print(f"Current Player: Player {self.current_player}")
        print(f"Turn Phase: {self.turn_phase}")
        print(f"Active Players: {sorted(list(self.active_players))}")
        print(f"Current Turn: {self.num_turns}")
        print("-" * 25)

        for player_id in range(self.num_players):
            if player_id in self.active_players:
                print(f"Player {player_id} (Armies: {sum(self.player_states[player_id]['armies'].values())}, "
                      f"Territories: {len(self.player_states[player_id]['territories'])}, "
                      f"Cards: {len(self.player_states[player_id]['cards'])}):")
                sorted_territories = sorted(list(self.player_states[player_id]['territories']))
                for t in sorted_territories:
                    print(f"  - {t}: {self.player_states[player_id]['armies'][t]} armies")
            else:
                print(f"Player {player_id} (Eliminated)")
        print("-" * 25)

    def get_possible_actions(self):
        """
        Returns a list of all valid actions for the current player in the current phase.
        Used by rule-based agents (Random, Defensive, Balanced).
        """
        current_player_id = self.current_player
        possible_actions = []

        if current_player_id not in self.active_players:
            return [{"phase": "skip_turn"}] # Player eliminated, skip turn

        # Get current player's owned territories and their army counts
        owned_territories = list(self.player_states[current_player_id]['territories'])
        owned_territories_with_armies = {t: self.player_states[current_player_id]['armies'][t] for t in owned_territories}

        # --- Reinforcement Phase Actions ---
        if self.turn_phase == "reinforcement":
            # Option to trade cards
            if len(self.player_states[current_player_id]['cards']) >= 3:
                possible_actions.append({"phase": "trade_cards"})

            # Option to place armies (if reinforcements are available)
            reinforcements_available = self.player_states[current_player_id].get('reinforcements_available', 0)
            if reinforcements_available > 0 and owned_territories:
                for territory in owned_territories:
                    # Allow placing 1 army up to available reinforcements
                    max_place = min(reinforcements_available, 5) # Capped at 5 for simplicity with AI action space
                    for num_armies in range(1, max_place + 1):
                        possible_actions.append({"phase": "reinforce", "placements": {territory: num_armies}})
            
            # Always allow ending the reinforcement phase
            possible_actions.append({"phase": "end_reinforcement"}) 

        # --- Attack Phase Actions ---
        elif self.turn_phase == "attack":
            # Always allow ending the attack phase
            possible_actions.append({"phase": "end_attack"})

            # Iterate through all owned territories that can attack
            for attacker_t in owned_territories:
                if owned_territories_with_armies[attacker_t] > 1: # Must have at least 2 armies to attack
                    for defender_t in self.adjacency_list.get(attacker_t, []):
                        if self.territories.get(defender_t) != current_player_id: # Must be an enemy territory
                            # Allow attacking with 1, 2, or 3 dice
                            for num_dice in range(1, min(owned_territories_with_armies[attacker_t], 4)): # Max 3 dice
                                possible_actions.append({
                                    "phase": "attack",
                                    "attacker": attacker_t,
                                    "defender": defender_t,
                                    "dice": num_dice
                                })

        # --- Fortify Phase Actions ---
        elif self.turn_phase == "fortify":
            # Always allow ending the fortify phase (and turn)
            possible_actions.append({"phase": "end_fortify"})

            # Iterate through all owned territories for fortifying
            for from_t in owned_territories:
                if owned_territories_with_armies[from_t] > 1: # Must leave at least 1 army behind
                    for to_t in self.adjacency_list.get(from_t, []):
                        if self.territories.get(to_t) == current_player_id: # Must fortify to an owned, adjacent territory
                            # Allow moving 1 army up to (armies_on_from_t - 1)
                            for num_armies in range(1, min(owned_territories_with_armies[from_t], 11)): # Capped at 10 for AI action space
                                possible_actions.append({
                                    "phase": "fortify",
                                    "from": from_t,
                                    "to": to_t,
                                    "armies": num_armies
                                })
        return possible_actions

    def get_action_mask(self):
        """
        Returns a dictionary of boolean masks for each action head.
        1 indicates a valid action, 0 indicates an invalid action.
        """
        current_player_id = self.current_player
        
        # Reset all masks to 0 (invalid) using pre-allocated buffers
        for key in self.mask_buffers:
            self.mask_buffers[key].fill(0)
        masks = self.mask_buffers

        if current_player_id not in self.active_players:
            # If eliminated, only allow "end" actions to pass turn quickly
            masks["phase_transition"][:] = 1
            return masks

        owned_territories = list(self.player_states[current_player_id]['territories'])
        owned_territories_set = set(owned_territories)
        
        # --- Reinforcement Phase ---
        if self.turn_phase == "reinforcement":
            # Trade Cards
            # Check if player has a valid set (simplified: just check count >= 3 for now)
            # In a real implementation, we'd check for specific combinations.
            if len(self.player_states[current_player_id]['cards']) >= 3:
                masks["trade_cards"][1] = 1 # Allow trade
            masks["trade_cards"][0] = 1 # Always allow NOT trading

            # Reinforce Territory
            reinforcements_available = self.player_states[current_player_id].get('reinforcements_available', 0)
            if reinforcements_available > 0:
                for t_name in owned_territories:
                    t_idx = self.territory_names.index(t_name)
                    masks["reinforce_t"][t_idx] = 1
                
                # Reinforce Armies
                # Allow placing 1 up to min(5, reinforcements_available)
                max_place = min(5, reinforcements_available)
                masks["reinforce_a"][:max_place] = 1
            
            # Phase Transition
            # Can only end reinforcement if no reinforcements left (or if we want to force placement)
            # But usually, you MUST place all reinforcements.
            # For this simplified env, let's say you can end if you have 0 left.
            if reinforcements_available <= 0:
                 masks["phase_transition"][0] = 1 # End Reinforcement
            else:
                 # Must place armies, cannot end phase yet
                 pass 

        # --- Attack Phase ---
        elif self.turn_phase == "attack":
            # Phase Transition: Always allow ending attack phase
            masks["phase_transition"][1] = 1 

            # Attack Actions
            for t_name in owned_territories:
                t_idx = self.territory_names.index(t_name)
                armies = self.player_states[current_player_id]['armies'][t_name]
                
                if armies >= 2: # Need at least 2 armies to attack
                    # This territory is a valid attacker
                    
                    valid_targets = []
                    for adj_name in self.adjacency_list[t_name]:
                        if self.territories[adj_name] != current_player_id:
                            valid_targets.append(adj_name)
                    
                    if valid_targets:
                        masks["attack_attacker"][t_idx] = 1
                        for target_name in valid_targets:
                            target_idx = self.territory_names.index(target_name)
                            masks["attack_defender"][t_idx, target_idx] = 1
            
            # Attack Dice
            # We can't easily mask dice per specific attack pair in a single static mask output
            # without a huge action space.
            # Strategy: Allow all dice 1-3 generally, and the env/agent will clamp or penalty if invalid for specific pair.
            # OR: The agent picks dice *after* picking pair? No, it's simultaneous.
            # Let's just mask based on the *maximum* possible dice anyone can roll (which is 3).
            masks["attack_dice"][:] = 1 

        # --- Fortify Phase ---
        elif self.turn_phase == "fortify":
            # Phase Transition: Always allow ending fortify phase
            masks["phase_transition"][2] = 1

            # Fortify Actions
            for t_name in owned_territories:
                t_idx = self.territory_names.index(t_name)
                armies = self.player_states[current_player_id]['armies'][t_name]
                
                if armies > 1: # Need > 1 army to move
                    # This territory is a valid source
                    
                    # Find valid destinations (BFS for connected owned territories)
                    # For simplicity/speed, let's just check adjacent owned territories for now.
                    # Full pathfinding every step might be slow, but let's do adjacent first.
                    # If the rules allow chaining, we need BFS. The rules say "connected path of owned territories".
                    # Let's stick to adjacent for this specific implementation to keep it fast, 
                    # or do a quick BFS if needed. Let's do adjacent for now as per original code logic often implies.
                    # Wait, original code `_get_valid_fortify_paths` in rule_based_agents uses BFS.
                    # Let's replicate that BFS logic here for correctness.
                    
                    valid_destinations = []
                    queue = [t_name]
                    visited = {t_name}
                    connected_owned = []
                    
                    # BFS to find all connected owned territories
                    # Optimization: Pre-calculate connected components for the player? 
                    # For 42 territories, BFS is fast enough.
                    
                    # Actually, to construct the mask efficiently:
                    # We need to know for EACH 'from' territory, what are valid 'to' territories.
                    
                    # Let's just do adjacent owned for the mask to encourage simple moves first, 
                    # or do full BFS if we want perfect play.
                    # Let's do adjacent owned for simplicity of the mask matrix.
                    # (The agent can learn to chain moves if we allowed multiple fortifies, but Risk is usually 1 fortify).
                    
                    for adj_name in self.adjacency_list[t_name]:
                        if self.territories[adj_name] == current_player_id:
                            valid_destinations.append(adj_name)

                    if valid_destinations:
                        masks["fortify_from"][t_idx] = 1
                        for dest_name in valid_destinations:
                            dest_idx = self.territory_names.index(dest_name)
                            masks["fortify_to"][t_idx, dest_idx] = 1
            
            # Fortify Armies
            # Similar to dice, hard to mask per pair.
            # Allow all 1-10 generally.
            masks["fortify_armies"][:] = 1

        return masks
