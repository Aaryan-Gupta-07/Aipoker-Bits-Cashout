from enum import Enum
from typing import List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from card import Card

try:
    from tensorflow.keras.models import load_model
    MODEL_LOADING_AVAILABLE = True
except ImportError:
    MODEL_LOADING_AVAILABLE = False
    print("Warning: TensorFlow not available, will use fallback strategy")

class PlayerAction(Enum):
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"
    ALL_IN = "all-in"


class PlayerStatus(Enum):
    ACTIVE = "active"
    FOLDED = "folded"
    ALL_IN = "all-in"
    OUT = "out"


@dataclass
class Player:
    name: str
    stack: int
    uuid: int = 0
    status: PlayerStatus = PlayerStatus.ACTIVE
    hole_cards: List[Card] = None
    bet_amount: int = 0
    model_path: str = None  # Make model path optional

    def __post_init__(self):
        if self.hole_cards is None:
            self.hole_cards = []
        self.model = None
        if self.model_path and MODEL_LOADING_AVAILABLE:
            try:
                self.model = load_model(self.model_path)
                print(f"Successfully loaded model from {self.model_path}")
            except Exception as e:
                print(f"Failed to load model: {e}")
                self.model = None

    def reset_for_new_hand(self):
        self.hole_cards = []
        self.status = PlayerStatus.ACTIVE if self.stack > 0 else PlayerStatus.OUT
        self.bet_amount = 0

    def can_make_action(self) -> bool:
        return self.status in [PlayerStatus.ACTIVE]

    def take_action(self, action: PlayerAction, amount: int = 0) -> Tuple[PlayerAction, int]:
        if action == PlayerAction.FOLD:
            self.status = PlayerStatus.FOLDED
            return action, 0

        if action == PlayerAction.CALL:
            max_bet = min(amount, self.stack)
            self.stack -= max_bet
            self.bet_amount += max_bet
            if self.stack == 0:
                self.status = PlayerStatus.ALL_IN
                return PlayerAction.ALL_IN, max_bet
            return PlayerAction.CALL, max_bet

        if action in [PlayerAction.BET, PlayerAction.RAISE]:
            max_bet = min(amount, self.stack)
            delta = max_bet - self.bet_amount

            if action == PlayerAction.RAISE:
                max_bet = min(amount - self.bet_amount, self.stack)

            if max_bet == self.stack:
                self.stack -= max_bet
                self.bet_amount += max_bet
            else:
                self.stack -= delta
                self.bet_amount += delta

            if self.stack == 0:
                self.status = PlayerStatus.ALL_IN
                return PlayerAction.ALL_IN, max_bet

            return action, delta

        if action == PlayerAction.ALL_IN:
            actual = self.stack
            self.bet_amount += self.stack
            self.stack = 0
            self.status = PlayerStatus.ALL_IN
            return PlayerAction.ALL_IN, actual

        return action, 0

    def _extract_features(self, game_state: list[int]) -> np.ndarray:
        """Extracts and formats the required parameters for the model"""
        # Default values if model isn't available
        if not MODEL_LOADING_AVAILABLE or not self.model:
            return np.zeros((1, 17))  # Return dummy features

        try:
            # Extract from game_state (based on PokerGame.get_game_state() structure)
            hole_cards = game_state[:2]  # First two elements are hole cards
            community_cards = game_state[2:7]  # Next five are community cards
            pot_size = game_state[7]  # Pot size
            current_bet = game_state[8]  # Current bet amount
            big_blind = game_state[9]  # Big blind amount
            player_position = game_state[10]  # Active player index
            num_players = game_state[11]  # Total players
            stacks = game_state[12:12+num_players]  # Player stacks
            
            # Calculate derived features
            bankroll = self.stack
            round_num = self._get_round_num(game_state)
            num_players_active = sum(1 for s in stacks if s > 0)
            
            # Opponents features (simplified)
            opponents = []
            for i, stack in enumerate(stacks):
                if i != player_position and stack > 0:
                    opponents.append(stack)
            
            # Convert to numpy array
            features = np.array([
                *hole_cards,
                *community_cards,
                pot_size,
                current_bet,
                bankroll,
                round_num,
                player_position,
                num_players_active,
                *opponents[:5]  # Take up to 5 opponents
            ], dtype=np.float32)
            
            # Pad to expected length (17 features)
            expected_length = 17
            if len(features) < expected_length:
                features = np.pad(features, (0, expected_length - len(features)))
            elif len(features) > expected_length:
                features = features[:expected_length]
                
            return features.reshape(1, -1)
            
        except Exception as e:
            print(f"Feature extraction failed: {e}")
            return np.zeros((1, 17))

    def _get_round_num(self, game_state: list[int]) -> int:
        """Convert game phase to numerical round number"""
        community_cards = game_state[2:7]
        num_community_cards = sum(1 for c in community_cards if c > 0)
        
        return {
            0: 0,  # Pre-flop
            3: 1,  # Flop
            4: 2,  # Turn
            5: 3   # River
        }.get(num_community_cards, 0)

    def action(self, game_state: list[int], action_history: list) -> Tuple[PlayerAction, int]:
        """
        Uses either the trained model or fallback strategy to make decisions.
        """
        if self.model and MODEL_LOADING_AVAILABLE:
            try:
                features = self._extract_features(game_state)
                prediction = self.model.predict(features, verbose=0)
                action_idx = np.argmax(prediction[0][:6])
                actions = list(PlayerAction)
                action = actions[action_idx]
                
                if action in [PlayerAction.BET, PlayerAction.RAISE]:
                    big_blind = game_state[9]
                    amount = min(self.stack, int(big_blind * (1 + 2 * prediction[0][6])))
                    return action, max(amount, big_blind)
                elif action == PlayerAction.ALL_IN:
                    return action, self.stack
                else:
                    return action, 0
                    
            except Exception as e:
                print(f"Model prediction failed: {e}")
        
        # Fallback strategy when model isn't available
        if self.stack > game_state[8] * 3:  # If we have more than 3x current bet
            return PlayerAction.RAISE, min(self.stack, game_state[8] * 2)
        elif self.stack > game_state[8]:
            return PlayerAction.CALL, 0
        else:
            return PlayerAction.FOLD, 0
    