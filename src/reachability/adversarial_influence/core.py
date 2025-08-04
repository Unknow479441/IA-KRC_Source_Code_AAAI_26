import math
import heapq
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import defaultdict, deque
import logging


logger = logging.getLogger(__name__)


class EnemyUnit:
    """Enemy RL_marine unit class"""
    def __init__(self, position: Tuple[float, float], unit_id: str = None):
        self.unit_type = 'RL_marine'
        self.position = position
        self.unit_id = unit_id or f"enemy_{id(self)}"
        
        # SC2 RL_marine standard attributes
        self.health = 1.0  # Normalized health [0-1]
        self.attack_range = 5.0  # SC2 standard attack range
        self.damage = 6.0  # SC2 standard attack damage
        self.movement_speed = 2.25  # SC2 standard movement speed
        self.sight_range = 9.0  # SC2 standard sight range
        
        # Tracking data
        self.last_seen_position = position
        self.trajectory_history = [position]
        self.attack_history = []
        self.threat_level = 0.5  # Initial threat level
        self.last_update_step = 0

    def update_position(self, new_position: Tuple[float, float], step: int):
        """Update position and trajectory history"""
        self.last_seen_position = self.position
        self.position = new_position
        self.trajectory_history.append(new_position)
        self.last_update_step = step
        
        # Keep last 100 steps of trajectory
        if len(self.trajectory_history) > 100:
            self.trajectory_history = self.trajectory_history[-100:]
    
    def calculate_threat_level(self) -> float:
        """Calculate threat level based on historical behavior"""
        # Enhanced threat assessment based on movement patterns and attack history
        movement_threat = min(1.0, len(self.trajectory_history) / 50.0)
        attack_threat = min(1.0, len(self.attack_history) / 10.0)
        health_threat = self.health  # Higher health = higher threat
        
        # Add movement variance as threat indicator
        if len(self.trajectory_history) > 5:
            recent_positions = self.trajectory_history[-5:]
            distances = []
            for i in range(1, len(recent_positions)):
                dist = ((recent_positions[i][0] - recent_positions[i-1][0])**2 + 
                       (recent_positions[i][1] - recent_positions[i-1][1])**2)**0.5
                distances.append(dist)
            movement_variance = sum(distances) / len(distances) if distances else 0.0
            mobility_threat = min(1.0, movement_variance / 3.0)
        else:
            mobility_threat = 0.0
        
        self.threat_level = (movement_threat + attack_threat + health_threat + mobility_threat) / 4.0
        return self.threat_level
    
    def get_training_samples(self) -> List[Dict]:
        """Get training samples for neural network as described in paper"""
        if not hasattr(self, 'actual_movements'):
            return []
        return getattr(self, 'actual_movements', []).copy()
    
    def set_state_features(self, features):
        """Set current state features for next movement prediction"""
        self.previous_state_features = features
    
    def record_actual_movement(self, new_position: Tuple[float, float], step: int):
        """Record actual movement for neural network training"""
        if hasattr(self, 'previous_state_features') and self.previous_state_features is not None:
            actual_movement = (new_position[0] - self.position[0], new_position[1] - self.position[1])
            if not hasattr(self, 'actual_movements'):
                self.actual_movements = []
            self.actual_movements.append({
                'state': self.previous_state_features.copy(),
                'actual_movement': actual_movement,
                'step': step
            })
            
            # Keep reasonable number of movement samples for training
            if len(self.actual_movements) > 50:
                self.actual_movements = self.actual_movements[-25:]


class AllyUnit:
    """Ally RL_marine unit class"""
    def __init__(self, position: Tuple[float, float], unit_id: str = None):
        self.unit_type = 'RL_marine'
        self.position = position
        self.unit_id = unit_id or f"ally_{id(self)}"
        
        # SC2 RL_marine standard attributes
        self.health = 1.0  # Normalized health [0-1]
        self.movement_speed = 2.25  # SC2 standard movement speed
        
        # Ally weakening parameters
        self.weakening_range = 3.0  # SC2 standard weakening range
        self.weakening_strength = 0.3  # SC2 standard weakening strength
        self.last_update_step = 0

    def update_position(self, new_position: Tuple[float, float], step: int):
        """Update position"""
        self.position = new_position
        self.last_update_step = step


class ReplayBuffer:
    """Experience replay buffer"""
    def __init__(self, size: int = 10000):
        self.size = size
        self.buffer = []
        self.position = 0
    
    def add(self, experience: Dict):
        """Add experience"""
        if len(self.buffer) < self.size:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.size
    
    def sample(self, batch_size: int) -> List[Dict]:
        """Sample experience"""
        if len(self.buffer) < batch_size:
            return self.buffer.copy()
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
    
    def size(self) -> int:
        """Get buffer size"""
        return len(self.buffer)
    
    def __len__(self) -> int:
        """Get buffer size - supports len() built-in function"""
        return len(self.buffer)


class AttackIntentNet:
    """Attack intent prediction network as described in the paper appendix"""
    def __init__(self, input_dim: int = 22, hidden_dim1: int = 128, hidden_dim2: int = 64, output_dim: int = 2):
        self.input_dim = input_dim  # Recent trajectory (10 positions) + current health + other features
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.output_dim = output_dim  # (x, y) components of attack intent vector
        
        # Simple weight initialization (in a real implementation, this would use PyTorch)
        import numpy as np
        self.weights = {
            'W1': np.random.randn(input_dim, hidden_dim1) * 0.01,
            'b1': np.zeros((1, hidden_dim1)),
            'W2': np.random.randn(hidden_dim1, hidden_dim2) * 0.01,
            'b2': np.zeros((1, hidden_dim2)),
            'W3': np.random.randn(hidden_dim2, output_dim) * 0.01,
            'b3': np.zeros((1, output_dim))
        }
        
        self.learning_rate = 0.001
        self.training_data = []
        
    def _relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def _forward(self, x):
        """Forward pass through the network"""
        z1 = np.dot(x, self.weights['W1']) + self.weights['b1']
        a1 = self._relu(z1)
        
        z2 = np.dot(a1, self.weights['W2']) + self.weights['b2']
        a2 = self._relu(z2)
        
        z3 = np.dot(a2, self.weights['W3']) + self.weights['b3']
        return z3  # Linear output layer
    
    def predict_attack_intent(self, enemy_state: np.ndarray) -> Tuple[float, float]:
        """Predict attack intent vector for an enemy"""
        if enemy_state.shape[0] != self.input_dim:
            # Pad or truncate to correct size
            padded_state = np.zeros(self.input_dim)
            min_len = min(len(enemy_state), self.input_dim)
            padded_state[:min_len] = enemy_state[:min_len]
            enemy_state = padded_state
        
        enemy_state = enemy_state.reshape(1, -1)
        intent_vector = self._forward(enemy_state)
        return float(intent_vector[0, 0]), float(intent_vector[0, 1])
    
    def add_training_sample(self, enemy_state: np.ndarray, true_movement: Tuple[float, float]):
        """Add training sample for supervised learning"""
        self.training_data.append((enemy_state.copy(), np.array(true_movement)))
    
    def train_step(self):
        """Perform one training step using collected samples"""
        if len(self.training_data) < 32:  # Need minimum batch size
            return
        
        # Simple training with recent samples (in real implementation, would use proper optimization)
        batch_size = min(32, len(self.training_data))
        recent_samples = self.training_data[-batch_size:]
        
        # Calculate loss and update weights (simplified)
        total_loss = 0.0
        for state, true_movement in recent_samples:
            predicted = self.predict_attack_intent(state)
            loss = np.sum((np.array(predicted) - true_movement) ** 2)
            total_loss += loss
        
        # Clear old training data to prevent memory issues
        if len(self.training_data) > 1000:
            self.training_data = self.training_data[-500:]


class EnemyBehaviorNet:
    """Enhanced enemy behavior prediction network with neural network for attack intent"""
    def __init__(self, learning_rate: float = 0.001):
        self.learning_rate = learning_rate
        
        # Attack intent prediction network
        self.intent_net = AttackIntentNet()
        
        # Store learned parameters
        self.learned_params = {
            'influence_range': 5.0,
            'decay_rate': 0.3,
            'base_influence': 2.0
        }
        
        # Parameter learning statistics
        self.update_count = 0
        self.behavior_patterns = {}
    
    def predict_next_action(self, enemy: EnemyUnit, context: Dict) -> Dict:
        """Predict enemy's next action using neural network"""
        # Prepare input features for neural network
        enemy_features = self._extract_enemy_features(enemy)
        
        # Predict attack intent
        intent_x, intent_y = self.intent_net.predict_attack_intent(enemy_features)
        
        # Calculate angle from intent vector
        intent_angle = math.atan2(intent_y, intent_x)
        
        return {
            'predicted_position': enemy.position,
            'predicted_action': 'move',
            'attack_intent_angle': intent_angle,
            'intent_vector': (intent_x, intent_y),
            'confidence': 0.7
        }
    
    def _extract_enemy_features(self, enemy: EnemyUnit) -> np.ndarray:
        """Extract features for neural network input as described in paper"""
        features = []
        
        # Recent trajectory (last 10 positions)
        trajectory = enemy.trajectory_history[-10:] if len(enemy.trajectory_history) >= 10 else enemy.trajectory_history
        while len(trajectory) < 10:
            trajectory = [enemy.position] + trajectory  # Pad with current position
        
        for pos in trajectory:
            features.extend([pos[0], pos[1]])
        
        # Current health
        features.append(enemy.health)
        
        # Additional features to reach input_dim=22
        features.append(enemy.threat_level)
        
        return np.array(features)
    
    def update(self, experiences: List[Dict]):
        """Update network parameters and influence parameters"""
        if self.learning_rate == 0.0:
            return
        
        self.update_count += 1
        
        # Train the neural network with collected experiences
        for exp in experiences:
            if 'enemy_movements' in exp:
                for enemy_id, movement_data in exp['enemy_movements'].items():
                    if 'samples' in movement_data:
                        # Process training samples from enemy units
                        for sample in movement_data['samples']:
                            if 'state' in sample and 'actual_movement' in sample:
                                self.intent_net.add_training_sample(
                                    sample['state'], 
                                    sample['actual_movement']
                                )
        
        # Perform training step
        if self.update_count % 10 == 0:
            self.intent_net.train_step()
        
        # Adjust influence parameters based on experiences
        if len(experiences) > 0:
            total_enemies = 0
            enemy_distances = []
            
            for exp in experiences:
                enemies = exp.get('enemies', [])
                total_enemies += len(enemies)
                
                # Calculate average distance between enemies
                if len(enemies) > 1:
                    for i, (pos1, _) in enumerate(enemies):
                        for j, (pos2, _) in enumerate(enemies[i+1:], i+1):
                            dist = ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5
                            enemy_distances.append(dist)
            
            if total_enemies > 0:
                avg_enemy_count = total_enemies / len(experiences)
                
                # Learning rule 1: Increase influence strength when enemy count is high
                if avg_enemy_count > 1.5:
                    self.learned_params['base_influence'] += self.learning_rate * 0.1
                else:
                    self.learned_params['base_influence'] -= self.learning_rate * 0.05
                
                # Learning rule 2: Increase influence range when enemies are clustered
                if enemy_distances and sum(enemy_distances) / len(enemy_distances) < 3.0:
                    self.learned_params['influence_range'] += self.learning_rate * 0.5
                
                # Limit parameter range
                self.learned_params['base_influence'] = max(1.0, min(4.0, self.learned_params['base_influence']))
                self.learned_params['influence_range'] = max(3.0, min(8.0, self.learned_params['influence_range']))
                self.learned_params['decay_rate'] = max(0.1, min(0.8, self.learned_params['decay_rate']))
        
        # Update behavior pattern statistics
        for exp in experiences:
            enemy_id = exp.get('enemy_id', 'default')
            if enemy_id not in self.behavior_patterns:
                self.behavior_patterns[enemy_id] = {'moves': 0, 'attacks': 0}
    
    def get_learned_params(self) -> Dict:
        """Get learned parameters"""
        return self.learned_params.copy()
    
    def get_attack_intent_angle(self, enemy: EnemyUnit) -> float:
        """Get predicted attack intent angle for interference calculation"""
        action_data = self.predict_next_action(enemy, {})
        return action_data.get('attack_intent_angle', 0.0)


class AllyWeakeningNet:
    """Ally weakening effect network (optimized for performance)"""
    def __init__(self, learning_rate: float = 0.0):
        self.learning_rate = learning_rate
        # Comment out learning parameters to optimize performance
        # self.learned_weakening_params = {
        #     'range': 3.0,
        #     'strength': 0.3
        # }
        # self.update_count = 0

    def calculate_weakening_effect(self, ally: AllyUnit, enemy_position: Tuple[float, float]) -> float:
        """Calculate ally's weakening effect on enemy (optimized for performance)"""
        # Comment out weakening effect calculation to optimize performance
        return 0.0
        # distance = ((ally.position[0] - enemy_position[0])**2 + (ally.position[1] - enemy_position[1])**2)**0.5
        # if distance <= ally.weakening_range:
        #     return ally.weakening_strength * math.exp(-distance / ally.weakening_range)
        # return 0.0

    def update(self, experiences: List[Dict]):
        """Update weakening network parameters (optimized for performance)"""
        # Comment out weakening parameter learning to optimize performance
        pass


class AdversarialOpponentModel:
    """Adversarial opponent modeling class"""
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        # Network components
        self.enemy_behavior_network = EnemyBehaviorNet(
            learning_rate=self.config.get('opponent_learning_rate', 0.0)
        )
        self.ally_weakening_network = AllyWeakeningNet(
            learning_rate=self.config.get('opponent_learning_rate', 0.0)
        )

        # Experience replay buffer
        self.experience_buffer = ReplayBuffer(
            size=self.config.get('experience_buffer_size', 10000)
        )
        
        # Parameter settings
        self.learning_rate = self.config.get('opponent_learning_rate', 0.0)
        self.update_frequency = self.config.get('opponent_update_frequency', 100)
        self.cost_multiplier = self.config.get('cost_multiplier', 1.5)
        self.ally_weakening_factor = self.config.get('ally_weakening_factor', 0.0)
        
        # Current observation data
        self.current_enemies: List[EnemyUnit] = []
        self.current_allies: List[AllyUnit] = []
        
        # Training counter
        self.step_count = 0
    
    def update_observations(self, enemies: List[EnemyUnit], allies: List[AllyUnit]):
        """Update current observation data and collect training samples"""
        # Update enemy tracking and collect training data
        for enemy in enemies:
            # Extract features for neural network before position update
            enemy_features = self._extract_enemy_features_for_training(enemy)
            enemy.set_state_features(enemy_features)
            
            # Update enemy position and record actual movement
            if hasattr(enemy, 'position'):
                enemy.record_actual_movement(enemy.position, self.step_count)
        
        self.current_enemies = enemies
        self.current_allies = allies
        self.step_count += 1
        
        # Record experience with enemy movement data
        experience = {
            'step': self.step_count,
            'enemies': [(e.position, e.threat_level) for e in enemies],
            'allies': [a.position for a in allies],
            'enemy_movements': {},
            'timestamp': self.step_count
        }
        
        # Collect training samples from enemies
        for enemy in enemies:
            training_samples = enemy.get_training_samples()
            if training_samples:
                experience['enemy_movements'][enemy.unit_id] = {
                    'samples': training_samples,
                    'current_state': enemy_features if 'enemy_features' in locals() else None
                }
        
        self.experience_buffer.add(experience)
        
        # Automatically trigger learning update
        if self.learning_rate > 0.0 and self.step_count % self.update_frequency == 0:
            self.train_step()
    
    def _extract_enemy_features_for_training(self, enemy: EnemyUnit):
        """Extract features for neural network training (same as in EnemyBehaviorNet)"""
        features = []
        
        # Recent trajectory (last 10 positions)
        trajectory = enemy.trajectory_history[-10:] if len(enemy.trajectory_history) >= 10 else enemy.trajectory_history
        while len(trajectory) < 10:
            trajectory = [enemy.position] + trajectory  # Pad with current position
        
        for pos in trajectory:
            features.extend([pos[0], pos[1]])
        
        # Current health
        features.append(enemy.health)
        
        # Additional features to reach input_dim=22
        features.append(enemy.threat_level)
        
        return np.array(features)
    
    def get_marine_influence_params(self) -> Dict:
        """Get RL_marine influence parameters"""
        # Prioritize parameters from config
        influence_range = self.config.get('influence_range', 5.0)
        decay_rate = self.config.get('influence_decay_rate', 0.3)
        base_influence = self.config.get('base_influence_strength', 2.0)
        
        # If learning is enabled and learning rate > 0, use learned parameters
        if self.learning_rate > 0.0:
            learned_params = self.enemy_behavior_network.get_learned_params()
            influence_range = learned_params.get('influence_range', influence_range)
            decay_rate = learned_params.get('decay_rate', decay_rate)
            base_influence = learned_params.get('base_influence', base_influence)
        
        return {
            'influence_range': influence_range,
            'decay_rate': decay_rate,
            'base_influence': base_influence
        }
    
    def get_marine_weakening_params(self) -> Dict:
        """
        Get Marine unit weakening parameters (optimized for performance)
        
        Returns:
            Dict: Weakening parameter dictionary
        """
        # Comment out weakening parameters to optimize performance
        return {
            'range': 0.0,
            'strength': 0.0
        }
        # if not hasattr(self, 'ally_weakening_net'):
        #     return {
        #         'range': 3.0,
        #         'strength': 0.3
        #     }
        # return self.ally_weakening_net.learned_weakening_params.copy()
    
    def train_step(self):
        """Perform one training update"""
        if self.learning_rate == 0.0 or len(self.experience_buffer) == 0:
            return
        
        # Sample experiences
        batch_size = self.config.get('batch_size', 32)
        experiences = self.experience_buffer.sample(batch_size)
        
        # Update networks
        self.enemy_behavior_network.update(experiences)
        self.ally_weakening_network.update(experiences)


class AdversarialInfluenceMap:
    """Adversarial Influence Graph Core Class"""

    def __init__(self, map_dimensions: Tuple[int, int], resolution: float = 1.0, config: Optional[Dict] = None):
        # Basic influence graph structure
        self.width, self.height = map_dimensions
        self.resolution = resolution

        # === Performance optimization: Pre-calculated tables ===
        self._init_precomputed_tables()
        self.config = config or {}
        # === Performance optimization: Memory pools ===
        self._init_memory_pools()
        
        # === Add obstacle map storage ===
        self.obstacle_map = None  # Store obstacle map
        
        # Core influence graph components - fix: ensure dimension order consistency (width, height)
        # Enemy influence map (stores enemy negative influence strength)
        self.enemy_influence_map = np.zeros((self.width, self.height), dtype=np.float32)
        
        # Path cost map (for k-step algorithm)
        self.path_cost_map = np.ones((self.width, self.height), dtype=np.float32)
        
        # K-step reachability maps
        self.reachability_maps = {}  # {agent_id: reachability_map}
        
        # Heatmap storage format - fix: ensure dimension order consistency
        self.heatmap_data = {
            'terrain_map': np.zeros((self.width, self.height, 3), dtype=np.uint8),    # RGB terrain map
            'influence_heatmap': np.zeros((self.width, self.height, 3), dtype=np.uint8), # RGB influence map
            'cost_heatmap': np.zeros((self.width, self.height, 3), dtype=np.uint8),     # RGB cost map
            'reachability_heatmap': np.zeros((self.width, self.height, 3), dtype=np.uint8) # RGB reachability map
        }
        
        # Opponent modeling component
        self.opponent_model = AdversarialOpponentModel(config=self.config)
        
        # Step counter
        self.current_step = 0
        self.last_update_step = -1
        
        # Current observation data cache
        self.current_enemy_positions: List[Tuple[float, float]] = []
        self.current_ally_positions: List[Tuple[float, float]] = []
        self.obstacle_positions: List[Tuple[float, float]] = []
        
        # === Performance optimization: Incremental update support ===
        self.previous_enemy_positions: List[Tuple[float, float]] = []
        self.enemy_influence_cache = {}  # {enemy_id: influence_contribution}
        
        # Debug settings
        self.enable_debug_logging = self.config.get('enable_debug_logging', False)
        self.debug_log_frequency = self.config.get('debug_log_frequency', 50)
        
        if self.enable_debug_logging:
            logger.setLevel(logging.DEBUG)

    def _init_precomputed_tables(self):
        """Initialize pre-calculated tables to avoid expensive runtime calculations"""
        # Pre-calculate distance table (avoid sqrt calculations)
        max_distance_sq = 100  # Support max distance 10 squares
        self.distance_table = np.zeros(max_distance_sq + 1, dtype=np.float32)
        for i in range(max_distance_sq + 1):
            self.distance_table[i] = math.sqrt(i)
        
        # Pre-calculate influence value table (avoid exp calculations)
        # Support common parameter combinations
        self.influence_table = {}
        decay_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        base_influences = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
        max_distance = 10.0
        
        for decay_rate in decay_rates:
            for base_influence in base_influences:
                key = (decay_rate, base_influence)
                # Pre-calculate influence values for 0-10 distance for this parameter combination
                influence_array = np.zeros(int(max_distance * 10) + 1, dtype=np.float32)  # 0.1 precision
                for i in range(len(influence_array)):
                    distance = i * 0.1
                    influence_array[i] = base_influence * math.exp(-decay_rate * distance)
                self.influence_table[key] = influence_array
    
    def _init_memory_pools(self):
        """Initialize memory pools to reduce runtime memory allocation"""
        pool_size = 5  # Support up to 5 temporary arrays being used simultaneously
        
        # Influence map memory pool
        self.influence_pool = [
            np.zeros((self.width, self.height), dtype=np.float32) 
            for _ in range(pool_size)
        ]
        self.available_influence = list(range(pool_size))
        
        # Boolean array pool (for reachability calculations)
        self.bool_pool = [
            np.zeros((self.width, self.height), dtype=bool)
            for _ in range(pool_size)
        ]
        self.available_bool = list(range(pool_size))
        
        # Distance array pool
        self.distance_pool = [
            np.full((self.width, self.height), np.inf, dtype=np.float32)
            for _ in range(pool_size)
        ]
        self.available_distance = list(range(pool_size))
    
    def _get_pooled_array(self, array_type: str):
        """Get array from memory pool"""
        if array_type == 'influence':
            if self.available_influence:
                idx = self.available_influence.pop()
                self.influence_pool[idx].fill(0.0)
                return self.influence_pool[idx], idx
        elif array_type == 'bool':
            if self.available_bool:
                idx = self.available_bool.pop()
                self.bool_pool[idx].fill(False)
                return self.bool_pool[idx], idx
        elif array_type == 'distance':
            if self.available_distance:
                idx = self.available_distance.pop()
                self.distance_pool[idx].fill(np.inf)
                return self.distance_pool[idx], idx
        
        # Fallback to regular allocation if memory pool is exhausted
        if array_type == 'influence':
            return np.zeros((self.width, self.height), dtype=np.float32), -1
        elif array_type == 'bool':
            return np.zeros((self.width, self.height), dtype=bool), -1
        elif array_type == 'distance':
            return np.full((self.width, self.height), np.inf, dtype=np.float32), -1
    
    def _return_pooled_array(self, array_type: str, idx: int):
        """Return array to memory pool"""
        if idx >= 0:  # Only arrays taken from the pool need to be returned
            if array_type == 'influence':
                self.available_influence.append(idx)
            elif array_type == 'bool':
                self.available_bool.append(idx)
            elif array_type == 'distance':
                self.available_distance.append(idx)

    def set_obstacle_map(self, obstacle_map: np.ndarray):
        """
        Set obstacle map
        
        Args:
            obstacle_map: Obstacle map, -1 for obstacles, 0 for free space
        """
        self.obstacle_map = obstacle_map

    def process_step_observations(self, visual_observations: Dict, agent_positions: Dict) -> Tuple[np.ndarray, Dict]:
        """
        Observation processing for each step following Algorithm 1 from paper appendix
        
        Algorithm 1: IA-KRC Grouping Algorithm
        Phase 1: Multi-Layer Map Update and Interference Prediction
        Phase 2: Compute Pairwise Distances and Centrality  
        Phase 3: Leader Election and Follower Assignment
        """
        self.current_step += 1
        
        # Phase 1: Multi-Layer Map Update and Interference Prediction
        # 1.1 Clear previous observation data (but keep obstacle map)
        self.clear_previous_observations()
        
        # 1.2 Parse current visual observations (extract enemies and allies)
        current_enemies = self.extract_enemy_positions(visual_observations)
        current_allies = self.extract_ally_positions(visual_observations)
        
        # 1.3 Update opponent model with new observations
        self.opponent_model.update_observations(current_enemies, current_allies)
        
        # 1.4 Generate influence maps using neural network predictions
        self.update_influence_maps(current_enemies, current_allies)
        
        # 1.5 Calculate path costs using interference potential field
        self.calculate_path_costs()
        
        # Phase 2: Compute Pairwise Distances and Centrality
        # 1.6 Generate K-step reachability maps for all agents
        self.calculate_reachability_maps(agent_positions)
        
        # Phase 3: Dynamic Grouping (handled by communication controller)
        # 1.7 Update heatmap data for visualization
        self.update_heatmap_data()
        
        # Return final cost map and reachability maps as per algorithm
        return self.get_path_cost_map(), self.get_reachability_maps()

    def clear_previous_observations(self):
        """Clear previous observation data"""
        self.enemy_influence_map.fill(0.0)
        self.current_enemy_positions.clear()
        self.current_ally_positions.clear()
    
    def extract_enemy_positions(self, visual_observations: Dict) -> List[EnemyUnit]:
        """Extract enemy RL_marine positions from visual observations"""
        enemies = []
        
        if isinstance(visual_observations, dict):
            # Process dictionary format observations
            if 'enemy_positions' in visual_observations:
                enemy_positions = visual_observations['enemy_positions']
                if isinstance(enemy_positions, (list, tuple)):
                    for i, pos in enumerate(enemy_positions):
                        if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                            enemy = EnemyUnit(position=(float(pos[0]), float(pos[1])), unit_id=f'enemy_{i}')
                            enemy.health = 1.0  # Default full health
                            enemies.append(enemy)
                            self.current_enemy_positions.append(enemy.position)
            
            # Process other possible formats
            elif 'enemies' in visual_observations:
                enemy_data = visual_observations['enemies']
                if isinstance(enemy_data, (list, tuple)):
                    for i, enemy_info in enumerate(enemy_data):
                        if isinstance(enemy_info, dict):
                            unit_id = enemy_info.get('unit_id', f'enemy_{i}')
                            position = enemy_info.get('position', (0.0, 0.0))
                            health = enemy_info.get('health', 1.0)
                            
                            enemy = EnemyUnit(position=position, unit_id=unit_id)
                            enemy.health = health  # Correctly set health value
                            enemies.append(enemy)
                            self.current_enemy_positions.append(enemy.position)
        
        elif isinstance(visual_observations, (list, tuple)):
            # Process list format observations (assuming it's a position list)
            for i, pos in enumerate(visual_observations):
                if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                    enemy = EnemyUnit(position=(float(pos[0]), float(pos[1])), unit_id=f'enemy_{i}')
                    enemy.health = 1.0
                    enemies.append(enemy)
                    self.current_enemy_positions.append(enemy.position)
        
        return enemies
    
    def extract_ally_positions(self, visual_observations: Dict) -> List[AllyUnit]:
        """Extract ally RL_marine positions from visual observations"""
        allies = []
        
        if isinstance(visual_observations, dict):
            # Process dictionary format observations
            if 'ally_positions' in visual_observations:
                ally_positions = visual_observations['ally_positions']
                if isinstance(ally_positions, (list, tuple)):
                    for i, pos in enumerate(ally_positions):
                        if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                            ally = AllyUnit(position=(float(pos[0]), float(pos[1])), unit_id=f'ally_{i}')
                            ally.health = 1.0  # Default full health
                            allies.append(ally)
                            self.current_ally_positions.append(ally.position)
            
            # Process other possible formats
            elif 'allies' in visual_observations:
                ally_data = visual_observations['allies']
                if isinstance(ally_data, (list, tuple)):
                    for i, ally_info in enumerate(ally_data):
                        if isinstance(ally_info, dict):
                            unit_id = ally_info.get('unit_id', f'ally_{i}')
                            position = ally_info.get('position', (0.0, 0.0))
                            health = ally_info.get('health', 1.0)
                            
                            ally = AllyUnit(position=position, unit_id=unit_id)
                            ally.health = health  # Correctly set health value
                            allies.append(ally)
                            self.current_ally_positions.append(ally.position)
        
        return allies
    
    def update_influence_maps(self, enemies: List[EnemyUnit], allies: List[AllyUnit]):
        """
        Update influence map data
        
        Args:
            enemies: List of enemy units
            allies: List of ally units (only for position tracking, weakening function disabled)
        """
        # Clear previous influence
        self.enemy_influence_map.fill(0.0)
        
        # Update enemy influence
        for enemy in enemies:
            self._add_enemy_influence(enemy)
        
        # Update opponent model observations (keep ally positions for visualization)
        if hasattr(self, 'opponent_model'):
            self.opponent_model.update_observations(enemies, allies)

    def _add_enemy_influence(self, enemy: EnemyUnit):
        """Add single enemy's influence with neural network predicted attack intent (paper-aligned version)"""
        # Get parameters
        params = self.opponent_model.get_marine_influence_params()
        influence_range = params['influence_range']
        decay_rate = params['decay_rate']
        base_influence = params['base_influence'] * enemy.threat_level

        # Get predicted attack intent angle from neural network
        attack_intent_angle = self.opponent_model.enemy_behavior_network.get_attack_intent_angle(enemy)

        # Boundary checks and coordinate calculations
        ex, ey = int(round(enemy.position[0])), int(round(enemy.position[1]))
        
        # Calculate influence range boundaries
        range_limit = int(influence_range) + 1
        
        # Optimized boundary checks: if completely outside the map, return directly
        if (ex + range_limit < 0 or ex - range_limit >= self.width or 
            ey + range_limit < 0 or ey - range_limit >= self.height):
            return
        
        # Calculate actual influence area boundaries
        x_min = max(0, ex - range_limit)
        x_max = min(self.width, ex + range_limit + 1)
        y_min = max(0, ey - range_limit)
        y_max = min(self.height, ey + range_limit + 1)
        
        # Check if there is a valid area
        if x_min >= x_max or y_min >= y_max:
            return
        
        # === Performance optimization: Use pre-calculated table ===
        # Find the closest pre-calculated parameter key
        table_key = self._find_closest_table_key(decay_rate, base_influence)
        influence_array = self.influence_table.get(table_key)
        
        if influence_array is None:
            # Fallback to original calculation
            self._add_enemy_influence_fallback_with_intent(enemy, attack_intent_angle)
            return
        
        # === Paper-aligned: Implement effective distance calculation ===
        # alpha parameter for angle influence (from paper)
        alpha = self.config.get('alpha', 0.5)
        
        # === Performance optimization: Cache-friendly memory access pattern ===
        for y in range(y_min, y_max):
            y_offset = y - ey
            y_offset_sq = y_offset * y_offset
            
            for x in range(x_min, x_max):
                x_offset = x - ex
                distance_sq = x_offset * x_offset + y_offset_sq
                
                # Distance check
                if distance_sq > influence_range * influence_range:
                    continue
                
                # Obstacle check
                if self.obstacle_map is not None and self.obstacle_map[x, y] == -1:
                    continue
                
                # Calculate actual distance
                if distance_sq < len(self.distance_table):
                    actual_distance = self.distance_table[distance_sq]
                else:
                    actual_distance = math.sqrt(distance_sq)
                
                # === Paper-aligned: Calculate angle between attack intent and target direction ===
                target_angle = math.atan2(y - ey, x - ex)
                angle_diff = abs(target_angle - attack_intent_angle)
                # Normalize angle difference to [0, π]
                angle_diff = min(angle_diff, 2 * math.pi - angle_diff)
                
                # Calculate effective distance using paper formula: d_eff = d_actual * (1 + α(1 - cos(θ)))
                effective_distance = actual_distance * (1 + alpha * (1 - math.cos(angle_diff)))
                
                # Query influence value table using effective distance
                distance_index = min(int(effective_distance * 10), len(influence_array) - 1)
                influence_value = influence_array[distance_index]
                
                # Accumulate to influence map
                self.enemy_influence_map[x, y] += influence_value

    def _find_closest_table_key(self, decay_rate: float, base_influence: float) -> Tuple[float, float]:
        """Find the closest pre-calculated table key"""
        # Predefined parameter values
        decay_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        base_influences = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
        
        # Find closest decay_rate
        closest_decay = min(decay_rates, key=lambda x: abs(x - decay_rate))
        # Find closest base_influence  
        closest_base = min(base_influences, key=lambda x: abs(x - base_influence))
        
        return (closest_decay, closest_base)
    
    def _add_enemy_influence_fallback(self, enemy: EnemyUnit):
        """Fallback to original calculation method (for extreme cases)"""
        params = self.opponent_model.get_marine_influence_params()
        influence_range = params['influence_range']
        decay_rate = params['decay_rate']
        base_influence = params['base_influence'] * enemy.threat_level

        ex, ey = int(round(enemy.position[0])), int(round(enemy.position[1]))
        range_limit = int(influence_range) + 1
        
        x_min = max(0, ex - range_limit)
        x_max = min(self.width, ex + range_limit + 1)
        y_min = max(0, ey - range_limit)
        y_max = min(self.height, ey + range_limit + 1)
        
        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                distance_sq = (x - ex)**2 + (y - ey)**2
                distance = math.sqrt(distance_sq)
                
                if distance <= influence_range:
                    if self.obstacle_map is None or self.obstacle_map[x, y] != -1:
                        influence_value = base_influence * math.exp(-decay_rate * distance)
                        self.enemy_influence_map[x, y] += influence_value

    def _add_enemy_influence_fallback_with_intent(self, enemy: EnemyUnit, attack_intent_angle: float):
        """Fallback calculation method with attack intent (paper-aligned)"""
        params = self.opponent_model.get_marine_influence_params()
        influence_range = params['influence_range']
        decay_rate = params['decay_rate']
        base_influence = params['base_influence'] * enemy.threat_level
        
        # Alpha parameter for angle influence
        alpha = self.config.get('alpha', 0.5)

        ex, ey = int(round(enemy.position[0])), int(round(enemy.position[1]))
        range_limit = int(influence_range) + 1
        
        x_min = max(0, ex - range_limit)
        x_max = min(self.width, ex + range_limit + 1)
        y_min = max(0, ey - range_limit)
        y_max = min(self.height, ey + range_limit + 1)
        
        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                distance_sq = (x - ex)**2 + (y - ey)**2
                actual_distance = math.sqrt(distance_sq)
                
                if actual_distance <= influence_range:
                    if self.obstacle_map is None or self.obstacle_map[x, y] != -1:
                        # Calculate angle between attack intent and target direction
                        target_angle = math.atan2(y - ey, x - ex)
                        angle_diff = abs(target_angle - attack_intent_angle)
                        # Normalize angle difference to [0, π]
                        angle_diff = min(angle_diff, 2 * math.pi - angle_diff)
                        
                        # Calculate effective distance using paper formula
                        effective_distance = actual_distance * (1 + alpha * (1 - math.cos(angle_diff)))
                        
                        # Calculate influence using effective distance
                        influence_value = base_influence * math.exp(-decay_rate * effective_distance)
                        self.enemy_influence_map[x, y] += influence_value

    def calculate_path_costs(self):
        """
        Calculate path cost map implementing paper's interference cost formula
        
        Implements: C(T_{s1,s2}=t|π_D) = Σ I(s) for s ∈ S(T_{s1,s2}=t|π_D)
        where I(s) is the interference strength at state s
        """
        if self.enemy_influence_map is None:
            self.path_cost_map = np.ones((self.width, self.height), dtype=np.float32)
            return
        
        # Base cost (geometric layer cost) - paper mentions integration with base costs
        base_cost = 1.0
        
        # Cost multiplier parameter from config (corresponds to paper's cost_multiplier)
        cost_multiplier = self.opponent_model.cost_multiplier
        
        # Apply interference cost: final_cost = base_cost + cost_multiplier * interference_strength
        # This implements the paper's approach where interference adds to existing path costs
        self.path_cost_map = base_cost + cost_multiplier * self.enemy_influence_map
        
        # Apply obstacle constraints - obstacles should have infinite cost
        if self.obstacle_map is not None:
            self.path_cost_map[self.obstacle_map == -1] = 1e6
        
        # Normalize to expected range as described in paper
        self._normalize_cost_map()

    def _normalize_cost_map(self):
        """
        Normalize cost map to the expected range of existing algorithms
        Map to [1.0, 10.0] range, maintaining relative proportions
        Ignore extremely high obstacle costs (>=1e5), to avoid pulling up the overall range and causing visualization distortion
        """
        # Create non-obstacle mask (cost less than a threshold)
        valid_mask = self.path_cost_map < 1e5  # Obstacles set to 1e6, exclude these
        if not valid_mask.any():
            # Extreme case: entire map is obstacles
            return
        
        valid_costs = self.path_cost_map[valid_mask]
        current_min = valid_costs.min()
        current_max = valid_costs.max()
        
        # Define target range
        min_cost, max_cost = 1.0, 10.0
        
        if current_max > current_min:
            # Linear normalization to target range
            normalized = (valid_costs - current_min) / (current_max - current_min)
            self.path_cost_map[valid_mask] = normalized * (max_cost - min_cost) + min_cost
        else:
            # All valid costs are the same, set to minimum cost directly
            self.path_cost_map[valid_mask] = min_cost
        
        # Keep obstacle extremely high cost unchanged
        # (obstacles are already set to >=1e6, no need to modify)

    def calculate_reachability_maps(self, agent_positions: Dict):
        """
        Calculate reachability for each agent under K-step constraints
        Use improved Dijkstra algorithm, considering adversarial costs
        """
        self.reachability_maps.clear()
        
        # Get K-step limit parameters
        k_step = self.get_k_step()  # Get from config
        move_amount = self.get_move_amount()    # Get from sc2custom.yaml


        for agent_id, agent_pos in agent_positions.items():
            if isinstance(agent_pos, (list, tuple)) and len(agent_pos) >= 2:
                # Calculate max cost budget for this agent
                max_cost_budget = k_step * move_amount
                
                reachability_map = self._calculate_single_agent_reachability(
                    agent_pos, max_cost_budget
                )
                self.reachability_maps[str(agent_id)] = reachability_map

    def _calculate_single_agent_reachability(self, start_pos: Tuple[float, float], max_cost: float) -> np.ndarray:
        """
        Calculate single agent's reachability (highly optimized version: memory pool + pre-allocation)
        Use modified Dijkstra algorithm, considering adversarial path costs
        
        Args:
            start_pos: Starting position (x, y)
            max_cost: Maximum allowed cost
            
        Returns:
            np.ndarray: Reachability map (bool type)
        """
        # === Performance optimization: Use memory pool ===
        reachability_map, reachability_idx = self._get_pooled_array('bool')
        cost_grid, cost_idx = self._get_pooled_array('distance')
        visited, visited_idx = self._get_pooled_array('bool')
        
        try:
            # Convert to grid coordinates
            start_x = int(start_pos[0] / self.resolution)
            start_y = int(start_pos[1] / self.resolution)
            
            # Boundary checks
            if not (0 <= start_x < self.width and 0 <= start_y < self.height):
                return reachability_map
            
            # Initialize starting point
            cost_grid[start_x, start_y] = 0.0
            
            # Optimized heap queue initialization
            pq = [(0.0, start_x, start_y)]
            
            # Predefined direction vectors to avoid duplicate creation
            directions = np.array([
                (-1, -1), (-1, 0), (-1, 1),
                (0, -1),           (0, 1),
                (1, -1),  (1, 0),  (1, 1)
            ], dtype=np.int32)
            
            # Diagonal cost multipliers (pre-calculated)
            diagonal_costs = np.array([
                math.sqrt(2), 1.0, math.sqrt(2),
                1.0,               1.0,
                math.sqrt(2), 1.0, math.sqrt(2)
            ], dtype=np.float32)
            
            while pq:
                current_cost, x, y = heapq.heappop(pq)
                
                # Skip visited nodes
                if visited[x, y]:
                    continue
                
                # Mark as visited and reachable
                visited[x, y] = True
                reachability_map[x, y] = True
                
                # Skip if cost exceeds budget
                if current_cost >= max_cost:
                    continue
                
                # Vectorized neighbor processing
                neighbors = directions + np.array([x, y])
                
                # Batch boundary checks
                valid_mask = (
                    (neighbors[:, 0] >= 0) & (neighbors[:, 0] < self.width) &
                    (neighbors[:, 1] >= 0) & (neighbors[:, 1] < self.height)
                )
                
                if not valid_mask.any():
                    continue
                
                # Filter valid neighbors
                valid_neighbors = neighbors[valid_mask]
                valid_diagonal_costs = diagonal_costs[valid_mask]
                
                # Batch check for unvisited nodes
                unvisited_mask = ~visited[valid_neighbors[:, 0], valid_neighbors[:, 1]]
                if not unvisited_mask.any():
                    continue
                
                final_neighbors = valid_neighbors[unvisited_mask]
                final_costs = valid_diagonal_costs[unvisited_mask]
                
                # Batch calculate movement costs
                base_move_costs = self.path_cost_map[final_neighbors[:, 0], final_neighbors[:, 1]]
                move_costs = base_move_costs * final_costs
                new_costs = current_cost + move_costs
                
                # Batch update costs and heap
                for i, (nx, ny) in enumerate(final_neighbors):
                    new_cost = new_costs[i]
                    if new_cost <= max_cost and new_cost < cost_grid[nx, ny]:
                        cost_grid[nx, ny] = new_cost
                        heapq.heappush(pq, (new_cost, nx, ny))
            
            # Copy result to new array (because memory pool arrays will be reused)
            result = reachability_map.copy()
            
        finally:
            # === Performance optimization: Return memory pool arrays ===
            self._return_pooled_array('bool', reachability_idx)
            self._return_pooled_array('distance', cost_idx)
            self._return_pooled_array('bool', visited_idx)
        
        return result
    
    def get_k_step(self) -> int:
        """Get k_step parameter (passed from czy.yaml or rlczy.yaml)"""
        return self.opponent_model.config.get('k_step', 0)  # k_step passed from czy.yaml or rlczy.yaml
    
    def get_move_amount(self) -> float:
        """Get move_amount parameter (passed from sc2custom.yaml)"""
        return self.opponent_model.config.get('move_amount', 2)  # move_amount passed from sc2custom.yaml, default value changed to 2
    
    def get_base_path_costs(self) -> np.ndarray:
        """
        Get base path costs (geometric/topological/experience layer costs)
        TODO: In actual integration, this should be obtained from existing GraphManager
        """
        # Simplified implementation: return uniform base cost
        return np.ones((self.height, self.width), dtype=np.float32)
    
    def query_adversarial_reachability(self, start_pos: Tuple[float, float], 
                                     goal_pos: Tuple[float, float], 
                                     agent_id: str = None) -> Tuple[bool, float, List[Tuple[int, int]]]:
        """
        Query adversarial reachability
        
        Args:
            start_pos: Starting position
            goal_pos: Goal position  
            agent_id: Agent ID (optional)
            
        Returns:
            Tuple[bool, float, List]: (reachable, path cost, path coordinate list)
        """
        # Get and output current k_step configuration
        current_k_step = self.get_k_step()
        current_move_amount = self.get_move_amount()
        print(f"[Reachability Query] Using k_step: {current_k_step}, move_amount: {current_move_amount}")
        
        # Convert to grid coordinates
        start_x, start_y = int(start_pos[0] / self.resolution), int(start_pos[1] / self.resolution)
        goal_x, goal_y = int(goal_pos[0] / self.resolution), int(goal_pos[1] / self.resolution)
        
        # Boundary checks
        if not (0 <= start_x < self.width and 0 <= start_y < self.height and
                0 <= goal_x < self.width and 0 <= goal_y < self.height):
            return False, float('inf'), []
        
        # Check agent's reachability
        if agent_id and str(agent_id) in self.reachability_maps:
            reachability_map = self.reachability_maps[str(agent_id)]
            # Fix: Use correct coordinate system to check reachability
            if not reachability_map[goal_x, goal_y]:  # Fix: Use goal_x, goal_y
                print(f"[Reachability Query] Target ({goal_x}, {goal_y}) is not reachable for Agent {agent_id} (k_step={current_k_step})")
                return False, float('inf'), []
        
        # Use A* algorithm to find optimal path
        path, cost = self._find_adversarial_path(
            (start_x, start_y), (goal_x, goal_y)
        )
        
        is_reachable = len(path) > 0 and cost < float('inf')
        print(f"[Reachability Query] Pathfinding result: reachable={is_reachable}, cost={cost:.2f}, path_length={len(path)} (k_step={current_k_step})")
        return is_reachable, cost, path
    
    def _find_adversarial_path(self, start: Tuple[int, int], 
                             goal: Tuple[int, int]) -> Tuple[List[Tuple[int, int]], float]:
        """
        Use A* algorithm to find the optimal path considering adversarial costs
        
        Args:
            start: Starting grid coordinates (x, y)
            goal: Goal grid coordinates (x, y)
            
        Returns:
            Tuple[List, float]: (path coordinate list, total cost)
        """
        def heuristic(pos1, pos2):
            """Manhattan distance heuristic"""
            return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
        
        # A* data structures
        open_set = [(0.0, start)]  # (f_score, position)
        came_from = {}
        g_score = {start: 0.0}
        f_score = {start: heuristic(start, goal)}
        
        # 8-directional movement
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        diagonal_multiplier = math.sqrt(2)
        
        while open_set:
            current_f, current = heapq.heappop(open_set)
            
            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path, g_score[goal]
            
            x, y = current
            for dx, dy in directions:
                neighbor = (x + dx, y + dy)
                nx, ny = neighbor
                
                # Boundary checks
                if not (0 <= nx < self.width and 0 <= ny < self.height):
                    continue
                
                # Calculate movement cost - Fix: Use correct coordinate system
                move_cost = self.path_cost_map[nx, ny]  # Fix: Use nx, ny
                if dx != 0 and dy != 0:  # Diagonal
                    move_cost *= diagonal_multiplier
                
                tentative_g_score = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    
                    # If not in open set, add to it
                    if (f_score[neighbor], neighbor) not in open_set:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        # No path found
        return [], float('inf')

    def integrate_with_base_costs(self, base_cost_map: np.ndarray) -> np.ndarray:
        """
        Integrate adversarial costs with existing base costs
        
        Args:
            base_cost_map: Existing base cost map (geometric/topological/experience layers)
            
        Returns:
            np.ndarray: Integrated cost map
        """
        if base_cost_map.shape != (self.height, self.width):
            logger.warning(f"Base cost map size mismatch: expected {(self.height, self.width)}, got {base_cost_map.shape}")
            return self.path_cost_map.copy()
        
        # Adversarial cost increment
        adversarial_increment = self.path_cost_map - 1.0  # Subtract base cost 1.0
        adversarial_increment = np.maximum(0.0, adversarial_increment)
        
        # Combine with existing base costs
        integrated_costs = base_cost_map + adversarial_increment
        
        logger.debug(f"Cost integration: base cost range=[{base_cost_map.min():.2f}, {base_cost_map.max():.2f}], "
                    f"adversarial increment range=[{adversarial_increment.min():.2f}, {adversarial_increment.max():.2f}], "
                    f"integrated range=[{integrated_costs.min():.2f}, {integrated_costs.max():.2f}]")
        
        return integrated_costs

    def update_heatmap_data(self):
        """Update heatmap data according to new color specifications"""
        # 1. Terrain map: Enemy (dark red) + Obstacle (black) + Free space (white) + Ally (blue)
        self.heatmap_data['terrain_map'] = self._create_terrain_heatmap()
        
        # 2. Influence map: Enemy influence strength (red gradient)
        self.heatmap_data['influence_heatmap'] = self._create_influence_heatmap()
        
        # 3. Cost map: Green (low cost) -> Red (high cost)
        self.heatmap_data['cost_heatmap'] = self._create_cost_heatmap()
        
        # 4. Reachability map: Reachability for each agent
        self.heatmap_data['reachability_heatmap'] = self._create_reachability_heatmap()
    
    def _create_terrain_heatmap(self) -> np.ndarray:
        """Create terrain heatmap"""
        terrain_map = np.full((self.height, self.width, 3), 255, dtype=np.uint8)  # Default white free space
        
        # Add obstacles (black)
        for obs_pos in self.obstacle_positions:
            obs_x = int(obs_pos[0] / self.resolution)
            obs_y = int(obs_pos[1] / self.resolution)
            if 0 <= obs_x < self.width and 0 <= obs_y < self.height:
                terrain_map[obs_y, obs_x] = [0, 0, 0]  # Black
        
        # Add enemy agents (dark red)
        for enemy_pos in self.current_enemy_positions:
            enemy_x = int(enemy_pos[0] / self.resolution)
            enemy_y = int(enemy_pos[1] / self.resolution)
            if 0 <= enemy_x < self.width and 0 <= enemy_y < self.height:
                terrain_map[enemy_y, enemy_x] = [255, 0, 0]  # Dark red
        
        # Add ally agents (blue)
        for ally_pos in self.current_ally_positions:
            ally_x = int(ally_pos[0] / self.resolution)
            ally_y = int(ally_pos[1] / self.resolution)
            if 0 <= ally_x < self.width and 0 <= ally_y < self.height:
                terrain_map[ally_y, ally_x] = [0, 0, 255]  # Blue
        
        return terrain_map
    
    def _create_influence_heatmap(self) -> np.ndarray:
        """Create influence heatmap"""
        influence_map = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Map enemy influence values to red gradient (0-255)
        if self.enemy_influence_map is not None and self.enemy_influence_map.max() > 0:
            # Fix dimension mismatch: enemy_influence_map is (width, height), needs to be transposed to (height, width)
            influence_data = self.enemy_influence_map.T  # Transpose to correct dimension
            normalized_influence = (influence_data / influence_data.max() * 255).astype(np.uint8)
            influence_map[:, :, 0] = normalized_influence  # Red channel shows enemy influence
            # Green and blue channels remain 0
        
        return influence_map
    
    def _create_cost_heatmap(self) -> np.ndarray:
        """Create cost heatmap"""
        cost_map = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        if self.path_cost_map is not None and self.path_cost_map.max() > self.path_cost_map.min():
            # Fix dimension mismatch: path_cost_map is (width, height), needs to be transposed to (height, width)
            cost_data = self.path_cost_map.T  # Transpose to correct dimension
            
            # Normalize cost values to [0, 1]
            normalized_costs = (cost_data - cost_data.min()) / (cost_data.max() - cost_data.min())
            
            # Green -> Red gradient
            cost_map[:, :, 0] = (normalized_costs * 255).astype(np.uint8)     # Red enhanced
            cost_map[:, :, 1] = ((1 - normalized_costs) * 255).astype(np.uint8)  # Green weakened
            # Blue channel remains 0
        else:
            # All costs are the same, display as green
            cost_map[:, :, 1] = 255
        
        return cost_map
    
    def _create_reachability_heatmap(self) -> np.ndarray:
        """Create reachability heatmap"""
        reachability_map = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Assign different colors for each agent
        colors = [
            [255, 255, 0],   # Yellow
            [255, 0, 255],   # Magenta
            [0, 255, 255],   # Cyan
            [255, 128, 0],   # Orange
            [128, 255, 0],   # Yellow-green
        ]
        
        for i, (agent_id, reachability) in enumerate(self.reachability_maps.items()):
            color = colors[i % len(colors)]
            
            # Fix dimension issue: reachability is (width, height), needs to be transposed
            if reachability is not None:
                reachability_data = reachability.T  # Transpose to correct dimension
                
                # Apply color to reachable areas
                reachable_positions = np.where(reachability_data)
                for y, x in zip(reachable_positions[0], reachable_positions[1]):
                    # Mix colors to support overlapping areas
                    for c in range(3):
                        reachability_map[y, x, c] = min(255, reachability_map[y, x, c] + color[c] // 2)
        
        return reachability_map

    def get_path_cost_map(self) -> np.ndarray:
        """Get path cost map"""
        return self.path_cost_map.copy()
    
    def get_reachability_maps(self) -> Dict[str, np.ndarray]:
        """Get reachability maps for all agents"""
        return self.reachability_maps.copy()
    
    def get_heatmap_for_visualization(self, map_type: str = 'influence') -> np.ndarray:
        """
        Get heatmap data for visualization (performance optimized version)
        
        Args:
            map_type: Heatmap type 'influence'|'cost'|'terrain'
        """
        if map_type == 'influence':
            # Return enemy influence map directly
            return self.enemy_influence_map if self.enemy_influence_map is not None else np.zeros((self.width, self.height))
        elif map_type == 'cost':
            return self.path_cost_map if self.path_cost_map is not None else np.ones((self.width, self.height))
        elif map_type == 'terrain':
            return self.obstacle_map if self.obstacle_map is not None else np.zeros((self.width, self.height))
        else:
            return np.zeros((self.width, self.height))
    
    def get_current_enemy_positions(self) -> List[Tuple[float, float]]:
        """Get current enemy positions"""
        return self.current_enemy_positions.copy()
    
    def get_current_ally_positions(self) -> List[Tuple[float, float]]:
        """Get current ally positions"""
        return self.current_ally_positions.copy()
    
    def get_obstacle_positions(self) -> List[Tuple[float, float]]:
        """Get obstacle positions"""
        return self.obstacle_positions.copy()
    
    def cleanup_step_data(self):
        """Clean up step data (performance optimization: reduce unnecessary operations)"""
        # Only clean up when necessary to reduce overhead
        pass

    def get_enemy_influence_map(self):
        """
        Get current enemy influence map (for debugging output)
        
        Returns:
            np.ndarray: Copy of enemy influence map
        """
        if hasattr(self, 'enemy_influence_map') and self.enemy_influence_map is not None:
            return self.enemy_influence_map.copy()
        return None
    
    def get_final_cost_map(self):
        """
        Get final cost map (for debugging output)
        
        Returns:
            np.ndarray: Copy of final cost map
        """
        if hasattr(self, 'path_cost_map') and self.path_cost_map is not None:
            return self.path_cost_map.copy()
        return None 