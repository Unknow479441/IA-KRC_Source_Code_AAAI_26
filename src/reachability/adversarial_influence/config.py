import logging

logger = logging.getLogger(__name__)

DEFAULT_ADVERSARIAL_CONFIG = {
    'opponent_learning_rate': 0.0,
    'opponent_update_frequency': 100,
    'experience_buffer_size': 10000,
    'batch_size': 32,
    
    'influence_decay_rate': 0.3,
    'base_influence_strength': 2.0,
    'ally_weakening_factor': 0.0,
    'cost_multiplier': 1.5,
    'alpha': 0.5,  # Angle influence factor as described in paper
    
    'k_step': 0,
    'move_amount': 2,
    
    'heatmap_resolution': (64, 64),
    'update_visualization_frequency': 10
}

SC2_REFERENCE_PARAMETERS = {
    'marine_health': 45.0,
    'marine_attack_range': 5.0,
    'marine_damage': 6.0,
    'marine_movement_speed': 2.25,
    'marine_sight_range': 9.0,
    
    'influence_range': 5.0,
    'influence_decay_rate': 0.3,
    'base_influence_strength': 2.0,
    'threat_multiplier_range': [0.0, 2.0],
    'alpha': 0.5,  # Angle influence factor as described in paper
    
    'weakening_range': 3.0,
    'weakening_decay_rate': 0.5,
    'base_weakening_strength': 0.3,
    'max_weakening_factor': 0.9,
    
    'cost_multiplier': 1.5,
    'min_path_cost': 1.0,
    'max_path_cost': 10.0,
    
    'learning_rate': 0.001,
    'discount_factor': 0.99,
    'epsilon': 0.1,
    
    'k_step': 0,
    'move_amount': 2,
    
    'enable_debug_logging': False,
    'debug_log_frequency': 50,
    'enable_heatmap_visualization': False,
}

OPPONENT_MODEL_CONFIG = {
    'learning_rate': 0.001,
    'discount_factor': 0.99,
    'exploration_rate': 0.1,
    'experience_buffer_size': 10000,
    'update_frequency': 100,
    'batch_size': 32,
    
    'influence_range': 5.0,
    'influence_decay_rate': 0.3,
    'base_influence_strength': 2.0,
    'weakening_range': 3.0,
    'weakening_decay_rate': 0.5,
    'base_weakening_strength': 0.3,
    'cost_multiplier': 1.5,
    'alpha': 0.5,  # Angle influence factor as described in paper
    'k_step': 0,
    'move_amount': 2,
}

GLOBAL_INFLUENCE_CONFIG = {
    'opponent_learning_rate': 0.001,
    'opponent_update_frequency': 100,
    'experience_buffer_size': 10000,
    'batch_size': 32,
    
    'influence_range': 5.0,
    'base_influence_strength': 2.0,
    'influence_decay_rate': 0.3,
    'weakening_range': 3.0,
    'base_weakening_strength': 0.3,
    'weakening_decay_rate': 0.5,
    'cost_multiplier': 1.5,
    'alpha': 0.5,  # Angle influence factor as described in paper
    
    'enable': True,
    'integration_mode': 'mask_only',
    'consider_in_reachability': True,
    'use_k_step_constraint': True,
    'debug_reachability': False,
    'log_influence_stats': False,
    'mask_update_frequency': 1,
    'k_step': 0,
    'move_amount': 2,
    
    'visualization': {
        'enable_heatmap': True,
        'save_gif': True,
        'save_average': True,
        'start_episode': 0,
        'save_interval': 1,
        'average_interval': 10,
        'save_dir': "adversarial_heatmaps",
        'map_size': [32, 32],
        'dpi': 150,
    },
    
    'enable_debug_logging': False,
    'debug_log_frequency': 50,
}

def load_adversarial_config_from_yaml(alg_config_dict, env_config_dict=None):
    alg_adversarial_config = alg_config_dict.get('adversarial_influence', {})
    
    merged_config = GLOBAL_INFLUENCE_CONFIG.copy()
    merged_config.update(alg_adversarial_config)
    
    if env_config_dict:
        if 'move_amount' in env_config_dict:
            merged_config['move_amount'] = env_config_dict['move_amount']

    if 'k_step' not in merged_config:
        merged_config['k_step'] = alg_config_dict.get('k_step', 0)
    
    if 'move_amount' not in merged_config:
        merged_config['move_amount'] = 2

    param_mappings = [
        'integration_mode', 'consider_in_reachability', 
        'use_k_step_constraint', 'debug_reachability', 'log_influence_stats',
        'mask_update_frequency', 'enable_debug_logging', 'debug_log_frequency'
    ]
    
    for param in param_mappings:
        if param in alg_adversarial_config:
            merged_config[param] = alg_adversarial_config[param]
    
    return merged_config

PARAMETER_MAPPING = {
    'k_step': 'self.comm_controller.k_step',
    'move_amount': 'self.env.move_amount'
}

INFLUENCE_PARAMETERS = {
    'geometry_influence_range': [1.0, 10.0],
    'geometry_decay_rate': [0.1, 1.0],
    
    'topology_connection_weight': [0.5, 2.0],
    'topology_choke_penalty': [1.0, 5.0],
    
    'experience_success_bonus': [-0.5, 0.5],
    'experience_failure_penalty': [0.0, 2.0],
}

ENEMY_UNIT_FEATURES = {
    'unit_type': 'RL_marine',
    'position': (float, float),
    'health': float,
    'attack_range': float,
    'damage': float,
    'movement_speed': float,
    'last_seen_position': (float, float),
    'trajectory_history': list,
    'attack_history': list,
    'threat_level': float
}

ALLY_UNIT_FEATURES = {
    'unit_type': 'RL_marine',
    'position': (float, float),
    'health': float,
    'weakening_range': float,
    'weakening_strength': float
} 