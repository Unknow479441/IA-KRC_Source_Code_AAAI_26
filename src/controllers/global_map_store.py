import numpy as np
import threading
import time
import pickle
import logging
from multiprocessing import shared_memory, Lock
import atexit
from typing import Dict

class SharedMemoryPool:
    def __init__(self):
        self._pool = {}
        self._size_cache = {}
        self._lock = threading.RLock()
        self._error_count = {}
        self._max_errors = 5
    
    def get_or_create(self, name: str, size: int, create: bool = False):
        with self._lock:
            try:
                if name in self._error_count and self._error_count[name] >= self._max_errors:
                    return None
                
                if name in self._pool:
                    shm = self._pool[name]
                    try:
                        if len(shm.buf) >= size:
                            return shm
                        else:
                            self._cleanup_single_memory(name)
                    except Exception as e:
                        self._cleanup_single_memory(name)
                        self._increment_error_count(name)
                
                try:
                    shm = shared_memory.SharedMemory(name=name, create=False)
                    if len(shm.buf) >= size:
                        self._pool[name] = shm
                        self._size_cache[name] = len(shm.buf)
                        self._error_count.pop(name, None)
                        return shm
                    else:
                        shm.close()
                except FileNotFoundError:
                    pass
                except Exception as e:
                    self._increment_error_count(name)
                
                if create:
                    try:
                        safety_margin = min(size // 10, 1024 * 1024)
                        actual_size = max(size + safety_margin, 1024 * 1024)
                        
                        shm = shared_memory.SharedMemory(name=name, create=True, size=actual_size)
                        self._pool[name] = shm
                        self._size_cache[name] = actual_size
                        self._error_count.pop(name, None)
                        return shm
                    except FileExistsError:
                        try:
                            shm = shared_memory.SharedMemory(name=name, create=False)
                            if len(shm.buf) >= size:
                                self._pool[name] = shm
                                self._size_cache[name] = len(shm.buf)
                                self._error_count.pop(name, None)
                                return shm
                            else:
                                shm.close()
                                self._increment_error_count(name)
                        except Exception as e:
                            self._increment_error_count(name)
                    except Exception as e:
                        self._increment_error_count(name)
                
                return None
                
            except Exception as e:
                self._increment_error_count(name)
                return None
    
    def _increment_error_count(self, name: str):
        self._error_count[name] = self._error_count.get(name, 0) + 1
    
    def _cleanup_single_memory(self, name: str):
        try:
            if name in self._pool:
                shm = self._pool[name]
                shm.close()
                if hasattr(shm, 'unlink'):
                    try:
                        shm.unlink()
                    except:
                        pass
                del self._pool[name]
            
            self._size_cache.pop(name, None)
        except Exception as e:
            pass
    
    def _cleanup_by_pattern(self, name_pattern: str):
        with self._lock:
            keys_to_remove = [name for name in self._pool.keys() if name_pattern in name]
            for name in keys_to_remove:
                try:
                    self._cleanup_single_memory(name)
                except Exception as e:
                    pass
    
    def cleanup(self):
        with self._lock:
            cleanup_errors = []
            for name, shm in list(self._pool.items()):
                try:
                    shm.close()
                    if hasattr(shm, 'unlink'):
                        try:
                            shm.unlink()
                        except Exception as unlink_e:
                            pass
                except Exception as e:
                    cleanup_errors.append(f"{name}: {e}")
            
            self._pool.clear()
            self._size_cache.clear()
            self._error_count.clear()

    
    def get_status(self):
        with self._lock:
            return {
                'pool_size': len(self._pool),
                'total_memory': sum(self._size_cache.values()),
                'error_counts': dict(self._error_count)
            }

_memory_pool = SharedMemoryPool()

class IncrementalSerializer:
    def __init__(self):
        self._last_state = {}
        self._version = 0
        self._lock = threading.Lock()
        self._corruption_count = 0
        self._max_corruption_threshold = 5
        
    def serialize_diff(self, current_state: dict) -> bytes:
        with self._lock:
            try:
                self._version += 1
                diff_data = {
                    'version': self._version,
                    'changes': {},
                    'full_sync': False,
                    'timestamp': time.time()
                }
                
                need_full_sync = (
                    not self._last_state or 
                    self._version % 50 == 0 or
                    self._corruption_count >= self._max_corruption_threshold
                )
                
                if need_full_sync:
                    diff_data['full_sync'] = True
                    diff_data['changes'] = current_state.copy()
                    self._corruption_count = 0
                else:
                    try:
                        for key, value in current_state.items():
                            if key not in self._last_state:
                                diff_data['changes'][key] = value
                            elif self._last_state[key] != value:
                                if isinstance(value, set) and isinstance(self._last_state[key], set):
                                    if value != self._last_state[key]:
                                        diff_data['changes'][key] = value
                                else:
                                    diff_data['changes'][key] = value
                    except Exception as e:
                        diff_data['full_sync'] = True
                        diff_data['changes'] = current_state.copy()
                        self._corruption_count += 1
                
                self._last_state = current_state.copy()
                
                try:
                    data = pickle.dumps(diff_data, protocol=pickle.HIGHEST_PROTOCOL)
                    return data
                except Exception as e:
                    return pickle.dumps({'version': self._version, 'error': str(e)}, protocol=pickle.HIGHEST_PROTOCOL)
                    
            except Exception as e:
                self._corruption_count += 1
                return pickle.dumps({'version': self._version, 'error': str(e)}, protocol=pickle.HIGHEST_PROTOCOL)
    
    def deserialize_diff(self, data: bytes, current_state: dict) -> dict:
        try:
            diff_data = pickle.loads(data)
            
            if 'error' in diff_data:
                return current_state
            
            if 'version' in diff_data and 'timestamp' in diff_data:
                version = diff_data['version']
                timestamp = diff_data['timestamp']
                if time.time() - timestamp > 30:
                    return current_state
            
            if diff_data.get('full_sync', False):
                changes = diff_data.get('changes', {})
                if self._validate_state_data(changes):
                    return changes.copy()
                else:
                    return current_state
            else:
                result = current_state.copy()
                changes = diff_data.get('changes', {})
                
                if self._validate_changes_data(changes):
                    for key, value in changes.items():
                        result[key] = value
                    return result
                else:
                    return current_state
                    
        except (pickle.PickleError, EOFError):
            self._corruption_count += 1
            return current_state
        except Exception:
            self._corruption_count += 1
            return current_state
    
    def _validate_state_data(self, state_data: dict) -> bool:
        try:
            required_fields = ['obstacle_cells', 'visited_cells', 'cell_attr']
            for field in required_fields:
                if field not in state_data:
                    return False
            
            if not isinstance(state_data.get('obstacle_cells'), set):
                return False
            if not isinstance(state_data.get('visited_cells'), set):
                return False
            if not isinstance(state_data.get('cell_attr'), dict):
                return False
                
            return True
        except Exception:
            return False
    
    def _validate_changes_data(self, changes_data: dict) -> bool:
        try:
            if not isinstance(changes_data, dict):
                return False
            
            for field in ['obstacle_cells', 'visited_cells', 'cell_attr']:
                if field in changes_data:
                    value = changes_data[field]
                    if field in ['obstacle_cells', 'visited_cells'] and not isinstance(value, set):
                        return False
                    elif field == 'cell_attr' and not isinstance(value, dict):
                        return False
            
            return True
        except Exception:
            return False
    
    def reset(self):
        with self._lock:
            self._last_state = {}
            self._version = 0
            self._corruption_count = 0

_incremental_serializer = IncrementalSerializer()

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(name)s] %(levelname)s: %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

DEFAULT_ENV_INDEX = 0  

SHARED_MAP_NAME = "shared_map_data_fixed"
SHARED_POSITIONS_PREFIX = "shared_positions_data_env_fixed"
SHARED_SHAPE_NAME = "shared_map_shape_fixed"
SHARED_GRAPH_STATE_NAME = "shared_graph_state_fixed"

SHARED_GRAPH_NAME = "shared_graph_pickle_data_fixed"
INITIAL_SHARED_GRAPH_SIZE = 5 * 1024 * 1024
MAX_SHARED_GRAPH_SIZE = 2 * 1024 * 1024 * 1024

_shared_memories = {}
_shared_locks = {}

_map_shape = None

_global_map_matrices = {}
_global_agent_positions = {}
_global_graph_manager = None

_global_enemy_positions = {}

EXPLORATION_THRESHOLD = 0.975
EXPERIENCE_BUFFER_LIMIT = 100000
_exploration_threshold_reached = False

_graph_pickle_lock = Lock()

class GlobalDummyEnv:
    def __init__(self, map_shape):
        real_map_matrix = get_map_matrix(0)
        if real_map_matrix is not None:
            self.map_matrix = real_map_matrix.copy()
            map_shape = real_map_matrix.shape
        else:
            self.map_matrix = np.zeros(map_shape, dtype=np.int8)
        self.map_x = map_shape[0]
        self.map_y = map_shape[1]

class EnvManager:
    def __init__(self):
        self.env_instances = {}
        
    def register_env(self, env_index, env_instance):
        self.env_instances[env_index] = env_instance
        
    def get_env(self, env_index):
        return self.env_instances.get(env_index, None)
        
    def get_visible_enemies(self, env_index, agent_positions=None):
        env = self.get_env(env_index)
        if env is None:
            return {}
            
        try:
            if hasattr(env, 'get_visible_enemy_positions'):
                return env.get_visible_enemy_positions()
            elif hasattr(env, 'get_all_enemy_positions'):
                return env.get_all_enemy_positions()
            else:
                return {}
        except Exception as e:
            print(f"[ERROR] 获取敌方位置失败: {e}")
            return {}

_env_manager = EnvManager()

def _cleanup_shared_memory():
    global _memory_pool
    try:
        _memory_pool.cleanup()
    except Exception as e:
        pass
    
    for name, shm in list(_shared_memories.items()):
        try:
            shm.close()
            try:
                if hasattr(shm, 'unlink'):
                    shm.unlink()
            except Exception:
                pass
        except Exception as e:
            pass

atexit.register(_cleanup_shared_memory)

def get_locks_and_memories():
    if "map_lock" not in _shared_locks:
        _shared_locks["map_lock"] = Lock()
    
    if "positions_locks" not in _shared_locks:
        _shared_locks["positions_locks"] = {}
    
    if "shape_lock" not in _shared_locks:
        _shared_locks["shape_lock"] = Lock()
    
    return _shared_locks

def get_shared_memory(name, size=None, create=False):
    global _memory_pool
    
    if size is not None:
        shm = _memory_pool.get_or_create(name, size, create)
        if shm is not None:
            return shm
    
    if name in _shared_memories and not create:
        return _shared_memories[name]
    
    try:
        if not create:
            shm = shared_memory.SharedMemory(name=name, create=False)
            _shared_memories[name] = shm
            return shm
        
        if create and size is not None:
            try:
                shm = shared_memory.SharedMemory(name=name, create=True, size=size)
                _shared_memories[name] = shm
                return shm
            except FileExistsError:
                shm = shared_memory.SharedMemory(name=name, create=False)
                _shared_memories[name] = shm
                return shm
    except Exception as e:
        return None
    
    return None

def store_map_shape(shape):
    global _map_shape
    _map_shape = shape
    
    locks = get_locks_and_memories()
    
    try:
        serialized_data = pickle.dumps(shape, protocol=4)
        
        with locks["shape_lock"]:
            shm = get_shared_memory(SHARED_SHAPE_NAME, size=len(serialized_data) + 8, create=True)
            if shm is None:
                return False
            
            size_bytes = len(serialized_data).to_bytes(8, byteorder='little')
            shm.buf[:8] = size_bytes
            shm.buf[8:8+len(serialized_data)] = serialized_data
        return True
    except Exception as e:
        return False

def get_map_shape():
    global _map_shape
    
    if _map_shape is not None:
        return _map_shape
    
    locks = get_locks_and_memories()
    
    try:
        with locks["shape_lock"]:
            shm = get_shared_memory(SHARED_SHAPE_NAME)
            if shm is None:
                return None
            
            try:
                size_bytes = bytes(shm.buf[:8])
                data_size = int.from_bytes(size_bytes, byteorder='little')
                
                if data_size <= 0 or data_size > len(shm.buf) - 8:
                    return None
                
                serialized_data = bytes(shm.buf[8:8+data_size])
                
                if len(serialized_data) == 0 or all(b == 0 for b in serialized_data):
                    return None
                
                if len(serialized_data) < 2 or serialized_data[0] not in [0x80, 0x5d]:
                    return None
                
                shape = pickle.loads(serialized_data)
                _map_shape = shape
                return shape
            except Exception as e:
                return None
    except Exception as e:
        return None

def get_file_paths(env_index=DEFAULT_ENV_INDEX):
    map_name = f"{SHARED_MAP_NAME}_{env_index}"
    positions_name = f"{SHARED_POSITIONS_PREFIX}_{env_index}"
    return map_name, positions_name, "map_lock", f"positions_lock_{env_index}", "shape_lock"

def update_map_matrix(matrix, env_index=DEFAULT_ENV_INDEX):
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)
    if matrix is None:
        return
    
    store_map_shape(matrix.shape)
    
    map_name, _, _, _, _ = get_file_paths(0)
    locks = get_locks_and_memories()
    
    try:
        if not isinstance(matrix, np.ndarray):
            matrix = np.array(matrix)
        
        size = matrix.nbytes
        
        with locks["map_lock"]:
            shm = get_shared_memory(map_name, size=size, create=True)
            if shm is None:
                return
            
            existing_array = np.ndarray(matrix.shape, dtype=matrix.dtype, buffer=shm.buf)
            
            merged_matrix = existing_array.copy()
            
            obstacle_mask = (matrix == -1)
            merged_matrix[obstacle_mask] = -1
            
            unit_mask = (matrix == 1) | (matrix == 2)
            merged_matrix[unit_mask] = matrix[unit_mask]
            
            empty_mask = (matrix == 0) & (existing_array != -1)
            merged_matrix[empty_mask] = 0
            
            existing_array[:] = merged_matrix[:]

    except Exception as e:
        pass

def get_map_matrix(env_index=DEFAULT_ENV_INDEX):
    map_name, _, _, _, _ = get_file_paths(env_index)
    locks = get_locks_and_memories()
    
    try:
        with locks["map_lock"]:
            shm = get_shared_memory(map_name)
            if shm is None:
                return None
            
            shape = get_map_shape()
            if shape is None:
                return None
            
            shared_array = np.ndarray(shape, dtype=np.int8, buffer=shm.buf)
            return shared_array.copy()
    except Exception as e:
        return None

def update_agent_positions(positions, env_index=DEFAULT_ENV_INDEX):
    if positions is None:
        return
    
    _, positions_name, _, _, _ = get_file_paths(env_index)
    locks = get_locks_and_memories()
    
    if f"positions_lock_{env_index}" not in locks["positions_locks"]:
        locks["positions_locks"][f"positions_lock_{env_index}"] = Lock()
    
    try:
        if not isinstance(positions, dict):
            positions = dict(positions)
        
        serialized_data = pickle.dumps(positions, protocol=4)
        data_size = len(serialized_data)
        
        required_size = data_size + 8
        safe_size = max(required_size * 2, INITIAL_SHARED_GRAPH_SIZE)

        with locks["positions_locks"][f"positions_lock_{env_index}"]:
            shm = get_shared_memory(positions_name, create=False)
            
            if shm is None:
                shm = get_shared_memory(positions_name, size=safe_size, create=True)
                if shm is None:
                    return
            
            elif len(shm.buf) < required_size:
                old_name = positions_name
                new_name = f"{positions_name}_new_{int(time.time())}"
                
                try:
                    new_shm = get_shared_memory(new_name, size=min(safe_size, MAX_SHARED_GRAPH_SIZE), create=True)
                    if new_shm is None:
                        _force_cleanup_shared_memory(old_name)
                        time.sleep(0.1)
                        shm = get_shared_memory(positions_name, size=safe_size, create=True)
                        if shm is None:
                            return
                    else:
                        try:
                            shm.close()
                        except:
                            pass
                        
                        try:
                            if hasattr(shm, 'unlink'):
                                shm.unlink()
                        except:
                            pass
                        
                        shm = get_shared_memory(positions_name, size=safe_size, create=True)
                        if shm is None:
                            shm = new_shm
                        else:
                            try:
                                new_shm.close()
                                if hasattr(new_shm, 'unlink'):
                                    new_shm.unlink()
                            except:
                                pass
                except Exception as rebuild_error:
                    _force_cleanup_shared_memory(old_name)
                    time.sleep(0.2)
                    shm = get_shared_memory(positions_name, size=safe_size, create=True)
                    if shm is None:
                        return
            
            if len(shm.buf) >= required_size:
                size_bytes = data_size.to_bytes(8, byteorder='little')
                shm.buf[:8] = size_bytes
                shm.buf[8:8+data_size] = serialized_data
                
    except Exception as e:
        if "memory" in str(e).lower() or "cannot allocate" in str(e).lower():
            try:
                _force_cleanup_shared_memory(f"{SHARED_POSITIONS_PREFIX}_{env_index}")
            except:
                pass

def _force_cleanup_shared_memory(name_pattern):
    global _shared_memories, _memory_pool
    
    try:
        keys_to_remove = [k for k in _shared_memories.keys() if name_pattern in k]
        for key in keys_to_remove:
            try:
                shm = _shared_memories[key]
                shm.close()
                if hasattr(shm, 'unlink'):
                    shm.unlink()
                del _shared_memories[key]
            except:
                pass
        
        if hasattr(_memory_pool, '_cleanup_by_pattern'):
            _memory_pool._cleanup_by_pattern(name_pattern)
        
        import gc
        gc.collect()
    except Exception as e:
        pass

def get_agent_positions(env_index=DEFAULT_ENV_INDEX):
    _, positions_name, _, _, _ = get_file_paths(env_index)
    locks = get_locks_and_memories()
    
    if f"positions_lock_{env_index}" not in locks["positions_locks"]:
        locks["positions_locks"][f"positions_lock_{env_index}"] = Lock()
    
    try:
        with locks["positions_locks"][f"positions_lock_{env_index}"]:
            shm = get_shared_memory(positions_name)
            if shm is None:
                return None
            
            size_bytes = bytes(shm.buf[:8])
            data_size = int.from_bytes(size_bytes, byteorder='little')
            
            if data_size <= 0 or data_size > len(shm.buf) - 8:
                return None
            
            serialized_data = bytes(shm.buf[8:8+data_size])
            
            if len(serialized_data) == 0 or all(b == 0 for b in serialized_data):
                return None
            
            if len(serialized_data) < 2 or serialized_data[0] not in [0x80, 0x5d]:
                return None
            
            positions = pickle.loads(serialized_data)
            return positions
    except Exception as e:
        return None

def get_all_map_matrices(batch_size_run=None):
    all_matrices = {}
    
    if batch_size_run is not None:
        for i in range(batch_size_run):
            matrix = get_map_matrix(i)
            if matrix is not None:
                all_matrices[i] = matrix
        return all_matrices
    
    for i in range(2):
        matrix = get_map_matrix(i)
        if matrix is not None:
            all_matrices[i] = matrix
    
    return all_matrices

def get_all_agent_positions(batch_size_run=None):
    if batch_size_run is None:
        return {}
    
    all_positions = {}
    
    for env_index in range(batch_size_run):
        positions = get_agent_positions(env_index)
        if positions is not None:
            all_positions[env_index] = positions
    
    return all_positions

_global_graph_manager = None
_graph_lock = threading.Lock()

def init_graph_manager(map_shape=None, grid_resolution=1.0, 
                      enable_adversarial=False, adversarial_config=None, k_step=None, move_amount=None):
    global _global_graph_manager, _exploration_threshold_reached
    
    if _global_graph_manager is not None:
        return _global_graph_manager
    
    with _graph_lock:
        if _global_graph_manager is not None:
            return _global_graph_manager
        
        _exploration_threshold_reached = False
        
        try:
            from reachability import create_reachability_manager
            from reachability.adversarial_influence.config import load_adversarial_config_from_yaml
            
            dummy_env = GlobalDummyEnv(map_shape)

            final_adversarial_config = None
            if enable_adversarial and adversarial_config:
                config_with_k_step = adversarial_config.copy()
                if k_step is not None:
                    config_with_k_step['k_step'] = k_step
                
                adversarial_section = config_with_k_step.get('adversarial_influence', {})
                if adversarial_section.get('enable', False) and enable_adversarial:
                    env_move_amount = move_amount if move_amount is not None else 2
                    
                    config_with_k_step = config_with_k_step.copy()
                    config_with_k_step['adversarial_influence'] = adversarial_section.copy()
                    config_with_k_step['adversarial_influence']['move_amount'] = env_move_amount
                else:
                    enable_adversarial = False

                final_adversarial_config = load_adversarial_config_from_yaml(
                    alg_config_dict=config_with_k_step, 
                    env_config_dict={'move_amount': move_amount if move_amount is not None else 2}
                )

            manager = create_reachability_manager(
                dummy_env, 
                grid_resolution=1.0,
                debug=True,
                include_topology=False,
                enable_adversarial=enable_adversarial,
                adversarial_config=final_adversarial_config
            )
            
            if hasattr(manager, 'experience') and hasattr(manager.experience, 'set_buffer_limit'):
                manager.experience.set_buffer_limit(EXPERIENCE_BUFFER_LIMIT)
            
            _global_graph_manager = manager
            
            existing_state = _load_graph_state_from_shared_memory()
            if existing_state is not None:
                _apply_graph_state_to_manager(manager, existing_state)

            return manager
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None

def get_graph_manager():
    global _global_graph_manager

    if _global_graph_manager is not None:
        return _global_graph_manager
    
    return None

def update_graph_with_sight_range(agent_id=None, pos=None, sight_range=None, positions=None, env_index=DEFAULT_ENV_INDEX):
    global _exploration_threshold_reached, _incremental_serializer
    
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            manager = get_graph_manager()
            if manager is None:
                return
            
            with _graph_lock:
                try:
                    with _graph_pickle_lock:
                        latest_state = _load_graph_state_from_shared_memory()
                        if latest_state is not None:
                            success = _apply_graph_state_to_manager(manager, latest_state)
                            if not success:
                                retry_count += 1
                                if retry_count >= max_retries:
                                    _incremental_serializer.reset()
                                    return
                    
                    current_progress = get_exploration_progress_graph()
                    if current_progress >= EXPLORATION_THRESHOLD:
                        if not _exploration_threshold_reached:
                            _exploration_threshold_reached = True
                        if positions is not None:
                            for agent_id_key, agent_pos in positions.items():
                                if hasattr(manager, 'experience'):
                                    pass
                        return

                    map_matrix = get_map_matrix(0)
                    if map_matrix is None:
                        return
                    
                    visible_enemies = get_visible_enemies(env_index, positions)
                    enemy_positions = []
                    ally_positions = []
                    
                    if visible_enemies:
                        enemy_positions = list(visible_enemies.values())
                    
                    if positions:
                        ally_positions = list(positions.values())
                    
                    visual_observations = {
                        'enemy_positions': enemy_positions,
                        'ally_positions': ally_positions
                    }
                    
                    update_success = False
                    if positions is not None and len(positions) > 0:
                        for agent_id_key, agent_pos in positions.items():
                            try:
                                manager.update_from_observation(
                                    agent_pos=agent_pos, 
                                    map_matrix=map_matrix,
                                    sight_range=sight_range,
                                    visual_observations=visual_observations,
                                    enemy_positions=enemy_positions,
                                    ally_positions=ally_positions,
                                    current_agent_positions=positions,
                                    incremental=True
                                )
                                update_success = True
                            except Exception as e:
                                continue
                    elif pos is not None:
                        try:
                            single_agent_positions = {'agent_0': pos}
                            manager.update_from_observation(
                                agent_pos=pos, 
                                map_matrix=map_matrix,
                                sight_range=sight_range,
                                visual_observations=visual_observations,
                                enemy_positions=enemy_positions,
                                ally_positions=ally_positions,
                                current_agent_positions=single_agent_positions,
                                incremental=True
                            )
                            update_success = True
                        except Exception as e:
                            pass
                    
                    if update_success:
                        with _graph_pickle_lock:
                            success = _store_graph_state_to_shared_memory(manager)
                            if not success:
                                retry_count += 1
                                continue
                    
                    break
                    
                except Exception as inner_e:
                    retry_count += 1
                    if retry_count >= max_retries:
                        return
                    time.sleep(0.01 * retry_count)
                    continue
                
        except Exception as e:
            retry_count += 1
            if retry_count >= max_retries:
                _incremental_serializer.reset()
                return
            time.sleep(0.01 * retry_count)
            continue

def _debug_print_graph_structure():
    manager = get_graph_manager()
    if manager is None:
        return
    
    try:
        with _graph_pickle_lock:
            latest_state = _load_graph_state_from_shared_memory()
            if latest_state is not None:
                _apply_graph_state_to_manager(manager, latest_state)

        pass
    except Exception as e:
        import traceback
        traceback.print_exc()

def get_exploration_progress_graph():
    global _exploration_threshold_reached
    
    manager = get_graph_manager()
    if manager is None:
        return 0.0
    
    try:
        with _graph_pickle_lock:
            latest_state = _load_graph_state_from_shared_memory()
            if latest_state is not None:
                _apply_graph_state_to_manager(manager, latest_state)
        
        visited_cells = len(manager.geometry.visited_cells)
        total_cells = len([cell for cell in manager.geometry.cell_attr.keys() 
                          if cell not in manager.geometry.obstacle_cells])
        progress = visited_cells / max(total_cells, 1)
        
        if progress >= EXPLORATION_THRESHOLD and not _exploration_threshold_reached:
            _exploration_threshold_reached = True

        return progress
    except Exception as e:
        return 0.0

def query_reachability_graph(start_pos, end_pos):
    manager = get_graph_manager()
    if manager is None:
        return False, float('inf')
    
    try:
        reachable, cost = manager.query_reachability(start_pos, end_pos)
        return reachable, cost
    except Exception as e:
        return False, float('inf')

def register_env_instance(env_index, env_instance):
    global _env_manager
    _env_manager.register_env(env_index, env_instance)

def update_enemy_positions(enemy_positions, env_index=DEFAULT_ENV_INDEX):
    global _global_enemy_positions
    
    if not isinstance(enemy_positions, dict):
        return
        
    _global_enemy_positions[env_index] = enemy_positions.copy()

def get_enemy_positions(env_index=DEFAULT_ENV_INDEX):
    global _global_enemy_positions
    return _global_enemy_positions.get(env_index, {})

def get_visible_enemies(env_index=DEFAULT_ENV_INDEX, agent_positions=None):
    global _env_manager
    
    visible_enemies = _env_manager.get_visible_enemies(env_index, agent_positions)
    
    if visible_enemies:
        update_enemy_positions(visible_enemies, env_index)
        return visible_enemies
    else:
        return get_enemy_positions(env_index)

def reset_graph_manager():
    global _global_graph_manager
    
    with _graph_lock:
        _global_graph_manager = None

def is_exploration_threshold_reached():
    global _exploration_threshold_reached
    return _exploration_threshold_reached

def get_experience_buffer_status():
    manager = get_graph_manager()
    if manager is None or not hasattr(manager, 'experience'):
        return {"size": 0, "limit": EXPERIENCE_BUFFER_LIMIT, "usage": 0.0}
    
    try:
        current_size = len(manager.experience.stats)
        usage = current_size / EXPERIENCE_BUFFER_LIMIT if EXPERIENCE_BUFFER_LIMIT > 0 else 0.0
        
        return {
            "size": current_size,
            "limit": EXPERIENCE_BUFFER_LIMIT,
            "usage": usage
        }
    except Exception as e:
        return {"size": 0, "limit": EXPERIENCE_BUFFER_LIMIT, "usage": 0.0}

_global_visualizer = None
_visualizer_config = None


def init_visualizer(config: Dict):
    global _global_visualizer, _visualizer_config
    
    if not config.get('enable_heatmap', False):
        return
    
    _visualizer_config = config

def get_visualizer():
    return _global_visualizer


def update_visualizer_data(episode_id: int, step: int, env_index: int = DEFAULT_ENV_INDEX, group_info: Dict = None):
    global _global_visualizer
    

    if env_index != 0:
        return

    if _global_visualizer is None:
        return
    
    try:
        manager = get_graph_manager()
        if manager is None or not hasattr(manager, 'adversarial_influence'):
            return
        
        influence_map = manager.adversarial_influence
        if influence_map is None:
            if step == 1:
                return
        
        start_episode = _visualizer_config.get('start_episode', 0)
        save_interval = _visualizer_config.get('save_interval', 1)
        
        if episode_id < start_episode:
            return
        
        if save_interval > 1 and (episode_id - start_episode) % save_interval != 0:
            return
        
        cost_map = influence_map.get_final_cost_map()
        if cost_map is None:
            return


        agent_positions_dict = get_agent_positions(env_index)
        ally_positions = []
        if agent_positions_dict:
            ally_positions = list(agent_positions_dict.values())

        enemy_positions_dict = get_visible_enemies(env_index, agent_positions_dict)
        enemy_positions = []
        if enemy_positions_dict:
            enemy_positions = list(enemy_positions_dict.values())

        map_matrix = get_map_matrix(env_index)
        obstacle_positions = []
        if map_matrix is not None:
            obstacle_coords = np.where(map_matrix == -1)
            if len(obstacle_coords[0]) > 0:
                obstacle_positions = list(zip(obstacle_coords[0].tolist(), obstacle_coords[1].tolist()))

        _global_visualizer.add_step_data(
            episode_id=episode_id,
            step=step,
            cost_map=cost_map,
            ally_positions=ally_positions,
            enemy_positions=enemy_positions,
            obstacle_positions=obstacle_positions,
            group_info=group_info
        )
    except Exception as e:
        import traceback
        print(traceback.format_exc())


def finalize_episode_visualization(episode_id: int):
    global _global_visualizer
    
    if _global_visualizer is not None:
        try:
            _global_visualizer.force_episode_end(episode_id)
        except Exception as e:
            pass


def finalize_visualizer():
    global _global_visualizer
    
    if _global_visualizer is not None:
        try:
            _global_visualizer.finalize()
        except Exception as e:
            pass
        finally:
            _global_visualizer = None

def _store_graph_state_to_shared_memory(manager):
    global _incremental_serializer
    try:
        graph_state = {
            'obstacle_cells': set(manager.geometry.obstacle_cells),
            'visited_cells': set(manager.geometry.visited_cells),
            'cell_attr': dict(manager.geometry.cell_attr),
            'exploration_progress': len(manager.geometry.visited_cells) / max(len([cell for cell in manager.geometry.cell_attr.keys() if cell not in manager.geometry.obstacle_cells]), 1),
        }

        data = _incremental_serializer.serialize_diff(graph_state)
        size = len(data)

        required_size = size + 8
        memory_sizes = [
            min(required_size * 2, 100 * 1024 * 1024),
            min(required_size * 3, 500 * 1024 * 1024),
            min(required_size * 4, MAX_SHARED_GRAPH_SIZE)
        ]
        
        shm = None
        for attempt_size in memory_sizes:
            if required_size <= attempt_size:
                shm = get_shared_memory(SHARED_GRAPH_NAME, size=attempt_size, create=True)
                if shm is not None:
                    break
        
        if shm is None:
            return False
        
        if len(shm.buf) < required_size:
            return False
        
        shm.buf[:8] = (0).to_bytes(8, byteorder='little')
        shm.buf[8:8+size] = data
        shm.buf[:8] = size.to_bytes(8, byteorder='little')
        return True
    except Exception as e:
        return False

def _load_graph_state_from_shared_memory():
    global _incremental_serializer
    
    _current_state_cache = getattr(_load_graph_state_from_shared_memory, '_cache', {})
    
    max_retries = 2
    for attempt in range(max_retries):
        try:
            shm = get_shared_memory(SHARED_GRAPH_NAME)
            if shm is None:
                return None
            
            size = int.from_bytes(bytes(shm.buf[:8]), byteorder='little')
            if size == 0:
                time.sleep(0.01)
                continue
                
            if size <= 0 or size > MAX_SHARED_GRAPH_SIZE - 8:
                return None

            size_check = int.from_bytes(bytes(shm.buf[:8]), byteorder='little')
            if size != size_check:
                continue

            data = bytes(shm.buf[8:8+size])
            if len(data) != size:
                continue

            if len(data) == 0 or all(b == 0 for b in data):
                if attempt < max_retries - 1:
                    time.sleep(0.02)
                    continue
                return None

            if len(data) < 2 or data[0] not in [0x80, 0x5d]:
                if attempt < max_retries - 1:
                    time.sleep(0.02)
                    continue
                return None
            try:
                graph_state = _incremental_serializer.deserialize_diff(data, _current_state_cache)
            except (pickle.PickleError, EOFError):
                if attempt < max_retries - 1:
                    continue
                return _current_state_cache if _current_state_cache else None

            if not isinstance(graph_state, dict):
                continue

            _load_graph_state_from_shared_memory._cache = graph_state.copy()
            return graph_state
            
        except (pickle.PickleError, EOFError):
            if attempt < max_retries - 1:
                time.sleep(0.02)
                continue
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(0.02)
                continue
    
    return None

def _apply_graph_state_to_manager(manager, graph_state):
    try:
        if not isinstance(graph_state, dict):
            return False
        
        if 'visited_cells' not in graph_state:
            return False
        
        if 'obstacle_cells' in graph_state and isinstance(graph_state['obstacle_cells'], set):
            manager.geometry.obstacle_cells = set(graph_state['obstacle_cells'])
        
        if 'visited_cells' in graph_state and isinstance(graph_state['visited_cells'], set):
            manager.geometry.visited_cells = set(graph_state['visited_cells'])
        
        if 'cell_attr' in graph_state and isinstance(graph_state['cell_attr'], dict):
            manager.geometry.cell_attr = dict(graph_state['cell_attr'])
        
        try:
            if hasattr(manager.geometry, '_adj'):
                manager.geometry._adj.clear()
            if hasattr(manager.geometry, '_build_grid'):
                manager.geometry._build_grid()
        except Exception as e:
            pass
        
        return True
    except Exception as e:
        return False

def check_graph_manager_health():
    global _global_graph_manager
    
    if _global_graph_manager is None:
        return False
    
    try:
        if not hasattr(_global_graph_manager, 'geometry'):
            return False
        
        geometry = _global_graph_manager.geometry
        if not hasattr(geometry, 'obstacle_cells') or not hasattr(geometry, 'visited_cells'):
            return False
        
        if not isinstance(geometry.obstacle_cells, set) or not isinstance(geometry.visited_cells, set):
            return False
        
        return True
    except Exception as e:
        return False

def recover_graph_manager_if_needed():
    global _global_graph_manager, _incremental_serializer
    
    if not check_graph_manager_health():

        _incremental_serializer.reset()

        try:
            with _graph_pickle_lock:
                shm = get_shared_memory(SHARED_GRAPH_NAME)
                if shm is not None:
                    shm.buf[:8] = (0).to_bytes(8, byteorder='little')
        except Exception as e:
            pass
        

        with _graph_lock:
            _global_graph_manager = None


