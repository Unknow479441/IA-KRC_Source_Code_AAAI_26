from __future__ import annotations

import math
import logging
import heapq
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Iterable, Optional, Set, Any, Union

logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _fmt = logging.Formatter('[%(asctime)s] %(levelname)s %(name)s: %(message)s')
    _handler.setFormatter(_fmt)
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

try:
    from .adversarial_influence import AdversarialInfluenceMap
except ImportError:
    AdversarialInfluenceMap = None

Node = Tuple[float, float]
Edge = Tuple[Node, Node, float]

class BaseLayer:
    def __init__(self, name: str):
        self.name = name
        self._adj: Dict[Node, Dict[Node, float]] = defaultdict(dict)

    def nodes(self) -> Iterable[Node]:
        for u in self._adj:
            yield u
            for v in self._adj[u]:
                yield v

    def edges(self) -> Iterable[Edge]:
        for u, nbrs in self._adj.items():
            for v, w in nbrs.items():
                yield (u, v, w)

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def _add_edge(self, u: Node, v: Node, weight: float):
        self._adj[u][v] = weight

    def _remove_edge(self, u: Node, v: Node):
        if v in self._adj.get(u, {}):
            del self._adj[u][v]
            if not self._adj[u]:
                del self._adj[u]

class GeometryLayer(BaseLayer):
    def __init__(self, bounds: Tuple[float, float, float, float], resolution: float = 0.25):
        super().__init__('geometry')
        self.bounds = bounds
        self.resolution = resolution
        self.obstacle_cells: Set[Node] = set()
        self.cell_attr: Dict[Node, Dict[str, Any]] = {}
        self.entity_cells: Set[Node] = set()
        self.entity_ids: Set[str] = set()
        self.visited_cells: Set[Node] = set()

        xmin, xmax, ymin, ymax = bounds
        
        if resolution == 1.0 and xmin == 0 and ymin == 0:
            self.grid_size = (
                int(xmax - xmin),
                int(ymax - ymin)
            )
        else:
            self.grid_size = (
                int((xmax - xmin) / resolution) + 1,
                int((ymax - ymin) / resolution) + 1
            )

        self._build_grid()

    def add_obstacle(self, rect: Tuple[float, float, float, float]):
        self._accumulate_border_cells(rect)
        self._build_grid()

    def update(self, obstacles: Optional[List[Tuple[float, float, float, float]]] = None, **_):
        if obstacles is not None:
            self.obstacle_cells.clear()
            for rect in obstacles:
                self._accumulate_border_cells(rect)
            self._build_grid()

    def add_entity_cell(self, pos: Node):
        self.entity_cells.add(pos)

    def _accumulate_border_cells(self, rect: Tuple[float, float, float, float]):
        left, right, bottom, top = rect
        step = self.resolution
        
        def align(v):
            return round(round(v / step) * step, 3)

        l = align(left)
        r_ = align(right)
        b = align(bottom)
        t = align(top)
        
        x = l
        while x <= r_ + 1e-6:
            y = b
            while y <= t + 1e-6:
                self.obstacle_cells.add((round(x, 3), round(y, 3)))
                y += step
            x += step

    def _build_grid(self):
        self._adj.clear()
        xmin, xmax, ymin, ymax = self.bounds
        step = self.resolution
        dirs = [(step, 0), (-step, 0), (0, step), (0, -step)]

        valid_nodes: Set[Node] = set()
        x = xmin
        while x <= xmax + 1e-6:
            y = ymin
            while y <= ymax + 1e-6:
                pt = (round(x, 3), round(y, 3))
                if pt not in self.obstacle_cells:
                    valid_nodes.add(pt)
                    if pt not in self.cell_attr:
                        self.cell_attr[pt] = {}
                y += step
            x += step

        for u in valid_nodes:
            for dx, dy in dirs:
                v = (round(u[0] + dx, 3), round(u[1] + dy, 3))
                if v in valid_nodes:
                    self._add_edge(u, v, math.hypot(dx, dy))

    def edges(self) -> Iterable[Edge]:
        return super().edges()

    def mark_cell_attr(self, cell: Node, key: str, value=True):
        if cell not in self.cell_attr:
            self.cell_attr[cell] = {}
        self.cell_attr[cell][key] = value

    def update_from_sight(self, agent_pos: Node, map_matrix=None, pathing_grid=None, sight_range=None):
        if map_matrix is None or sight_range is None:
            return
        if isinstance(agent_pos, (list, tuple)) and len(agent_pos) >= 2:
            agent_x, agent_y = int(round(float(agent_pos[0]))), int(round(float(agent_pos[1])))
        else:
            print(f"[ERROR] Invalid agent_pos: {agent_pos}")
            return
        
        if not (0 <= agent_x < map_matrix.shape[0] and 0 <= agent_y < map_matrix.shape[1]):
            print(f"[ERROR] agent_pos out of map bounds: ({agent_x}, {agent_y}), map size: {map_matrix.shape}")
            return

        new_obstacles_found = set()
        existing_in_sight = set()
        new_visited_found = set()
        
        for i in range(max(0, agent_x - sight_range), min(map_matrix.shape[0], agent_x + sight_range + 1)):
            for j in range(max(0, agent_y - sight_range), min(map_matrix.shape[1], agent_y + sight_range + 1)):
                distance = ((i - agent_x) ** 2 + (j - agent_y) ** 2) ** 0.5
                if distance <= sight_range:
                    if self._has_line_of_sight(agent_x, agent_y, i, j, map_matrix):
                        grid_node = (round(float(i), 3), round(float(j), 3))
                            
                        if map_matrix[i, j] == -1:
                            if grid_node not in self.obstacle_cells:
                                new_obstacles_found.add(grid_node)
                            else:
                                existing_in_sight.add(grid_node)
                        elif map_matrix[i, j] == 0:
                            if (grid_node not in self.obstacle_cells and 
                                grid_node not in self.entity_cells):
                                visited_node = (int(round(float(i))), int(round(float(j))))
                                if visited_node not in self.visited_cells:
                                    new_visited_found.add(visited_node)

        if new_obstacles_found:
            old_count = len(self.obstacle_cells)
            self.obstacle_cells.update(new_obstacles_found)
            new_count = len(self.obstacle_cells)
            self._build_grid()
        
        if new_visited_found:
            self.visited_cells.update(new_visited_found)

    def _has_line_of_sight(self, x1, y1, x2, y2, map_matrix):
        if x1 == x2 and y1 == y2:
            return True
        
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy
        
        current_x, current_y = x1, y1
        
        while True:
            if not (current_x == x1 and current_y == y1) and not (current_x == x2 and current_y == y2):
                if (0 <= current_x < map_matrix.shape[0] and 
                    0 <= current_y < map_matrix.shape[1] and 
                    map_matrix[current_x, current_y] == -1):
                    return False
            
            if current_x == x2 and current_y == y2:
                return True
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                current_x += sx
            if e2 < dx:
                err += dx
                current_y += sy

    def mark_visited(self, cell: Node):
        if cell not in self.obstacle_cells and cell not in self.entity_cells:
            normalized_cell = (int(round(float(cell[0]))), int(round(float(cell[1]))))
            self.visited_cells.add(normalized_cell)

class TopologyLayer(BaseLayer):
    def __init__(self):
        super().__init__('topology')
        self.portals: Dict[Node, Tuple[Node, float]] = {}
        self.unknown_portals: Set[Node] = set()
        self.blocked_edges: Set[Tuple[Node, Node]] = set()

    def add_corridor_edge(self, u: Node, v: Node, weight: float):
        if u == v:
            return
        self._add_edge(u, v, weight)

    def add_portal(self, entrance: Node, exit_: Node, cost: float = 1.0):
        self.portals[entrance] = (exit_, cost)
        self._add_edge(entrance, exit_, cost)
        if hasattr(self, 'geom_ref') and self.geom_ref:
            self.geom_ref.mark_cell_attr(entrance, 'portal', exit_)

    def add_portal_entrance(self, entrance: Node):
        self.unknown_portals.add(entrance)

    def update(self, *_, **__):
        pass

    def block_edge(self, u: Node, v: Node):
        self.blocked_edges.add((u, v))
        if u in self._adj and v in self._adj[u]:
            del self._adj[u][v]
        if hasattr(self, 'geom_ref') and self.geom_ref:
            self.geom_ref.mark_cell_attr(u, 'blocked', True)

class ExperienceLayer(BaseLayer):
    def __init__(self):
        super().__init__('experience')
        self.stats: Dict[Tuple[Node, Node], Dict[str, int]] = defaultdict(lambda: {'succ': 0, 'total': 0})
        self.succ_total = 0
        self.fail_total = 0
        self.buffer_limit = 100000

    def set_buffer_limit(self, limit: int):
        """Set buffer limit for experience storage"""
        self.buffer_limit = limit

    def record_traverse(self, path: List[Node], success: Union[bool, List[bool]] = True):
        """Record path traversal experience"""
        if not path or len(path) < 2:
            return
        
        is_list = isinstance(success, list)
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            edge = tuple(sorted((u, v)))
            s = success[i] if is_list and i < len(success) else success
            
            stat = self.stats[edge]
            stat['total'] += 1
            if s:
                stat['succ'] += 1
                self.succ_total += 1
            else:
                self.fail_total += 1
        
        # Simple FIFO eviction when buffer is full
        if len(self.stats) > self.buffer_limit:
            num_to_remove = len(self.stats) - self.buffer_limit
            for k in list(self.stats.keys())[:num_to_remove]:
                old_stat = self.stats[k]
                self.succ_total -= old_stat['succ']
                self.fail_total -= (old_stat['total'] - old_stat['succ'])
                del self.stats[k]

    def get_action_stats_str(self):
        return f"[ExperienceLayer] Total successful actions: {self.succ_total} | Total failed actions: {self.fail_total}"

    def update(self, *_, **__):
        pass

    def edges(self) -> Iterable[Edge]:
        # Experience layer doesn't contribute graph edges directly
        return []

class ConfidenceLayer(BaseLayer):
    def __init__(self, confidence_threshold: float = 0.7, update_rate_decay: float = 0.95):
        super().__init__('confidence')
        self.confidence_threshold = confidence_threshold
        self.update_rate_decay = update_rate_decay
        self.last_low_conf_updates: int = 0
        self.total_blocked_edges: int = 0
        self.current_low_conf_edges: int = 0
        self.dynamic_update_rate = 1.0  # Start with frequent updates

    def record_updates(self, cnt: int):
        """Record number of low confidence updates"""
        self.last_low_conf_updates = cnt

    def update_current_stats(self, experience_stats: Dict[Tuple[Node, Node], Dict[str, int]]):
        """Update current statistics based on experience layer"""
        low_conf_count = 0
        blocked_count = 0
        
        for stat in experience_stats.values():
            if stat['total'] > 0:
                success_rate = stat['succ'] / stat['total']
                if success_rate < self.confidence_threshold:
                    low_conf_count += 1
                    if success_rate < 0.1:  # Very low success rate
                        blocked_count += 1
        
        self.current_low_conf_edges = low_conf_count
        self.total_blocked_edges = blocked_count

    def update(self, experience_layer: 'ExperienceLayer', *args, **kwargs):
        """Update confidence metrics and adjust update rates"""
        # Update statistics based on experience layer
        self.update_current_stats(experience_layer.stats)
        
        # Adjust update rate based on confidence changes
        if self.current_low_conf_edges > self.last_low_conf_updates:
            # If confidence is decreasing, increase update frequency
            self.dynamic_update_rate = 1.0
        else:
            # Otherwise, gradually decrease update frequency  
            self.dynamic_update_rate *= self.update_rate_decay
            self.dynamic_update_rate = max(self.dynamic_update_rate, 0.1)  # Minimum update rate
        
        self.last_low_conf_updates = self.current_low_conf_edges

    def should_update_layer(self, layer_name: str) -> bool:
        """Determine if a layer should be updated based on confidence"""
        import random
        return random.random() < self.dynamic_update_rate

    def print_layers_status(self):
        """Print status of confidence layer"""
        print(f"[ConfidenceLayer] Low confidence edges: {self.current_low_conf_edges}, "
              f"Blocked edges: {self.total_blocked_edges}, "
              f"Update rate: {self.dynamic_update_rate:.3f}")

    def edges(self) -> Iterable[Edge]:
        # Confidence layer doesn't contribute graph edges directly
        return []

class BaseEnvAdapter:
    def __init__(self, env):
        self.env = env

    def world_bounds(self) -> Tuple[float, float, float, float]:
        raise NotImplementedError

    def static_obstacles(self) -> List[Tuple[float, float, float, float]]:
        return []

    def corridors(self) -> List[Dict]:
        return []

    def portals(self) -> List[Tuple[Node, Node, float]]:
        return []

class MPEAdapter(BaseEnvAdapter):
    def world_bounds(self):
        b = getattr(self.env, 'border_size', 3.5)
        return (-b, b, -b, b)

    def static_obstacles(self):
        obs = []
        for ent in self.env.world.entities:
            if 'landmark' in ent.name and ent.collide:
                sx = getattr(ent, 'size_x', ent.size * 2)
                sy = getattr(ent, 'size_y', ent.size * 2)
                x, y = ent.state.p_pos
                obs.append((x - sx, x + sx, y - sy, y + sy))
        return obs

    def corridors(self):
        return getattr(self.env.world, 'corridors', [])

class SMACAdapter(BaseEnvAdapter):
    def world_bounds(self):
        size_x = getattr(self.env, 'map_x', 32)
        size_y = getattr(self.env, 'map_y', 32)
        return (0, size_x, 0, size_y)
    
    def static_obstacles(self) -> List[Tuple[float, float, float, float]]:
        obstacles = []
        
        if hasattr(self.env, 'map_matrix') and self.env.map_matrix is not None:
            map_matrix = self.env.map_matrix
            for i in range(map_matrix.shape[0]):
                for j in range(map_matrix.shape[1]):
                    if map_matrix[i, j] == -1:
                        obstacles.append((float(i), float(i+1), float(j), float(j+1)))
        
        return obstacles

class GraphManager:
    def __init__(self, adapter: BaseEnvAdapter, grid_resolution: float = 0.25, enable_debug: bool = False,
                 include_topology: bool = False, enable_adversarial: bool = False, adversarial_config: Optional[Dict] = None):
        self.adapter = adapter
        if enable_debug:
            logger.setLevel(logging.DEBUG)
        bounds = adapter.world_bounds()
        self.geometry = GeometryLayer(bounds, resolution=grid_resolution)
        self.topology = TopologyLayer()
        self.experience = ExperienceLayer()
        self.confidence = ConfidenceLayer()
        self.layers: List[BaseLayer] = [self.geometry, self.topology, self.experience, self.confidence]
        self._include_topology = include_topology

        self.enable_adversarial = enable_adversarial
        self.adversarial_influence = None
        if enable_adversarial and AdversarialInfluenceMap is not None:
            xmin, xmax, ymin, ymax = bounds
            
            is_smac_env = isinstance(adapter, SMACAdapter)
            if is_smac_env:
                map_width = int(xmax - xmin)
                map_height = int(ymax - ymin)
            else:
                map_width = int((xmax - xmin) / grid_resolution) + 1
                map_height = int((ymax - ymin) / grid_resolution) + 1
            
            self.adversarial_influence = AdversarialInfluenceMap(
                map_dimensions=(map_width, map_height),
                resolution=grid_resolution,
                config=adversarial_config
            )
            
        elif enable_adversarial:
            logger.warning("Adversarial influence map module import failed, functionality disabled")

        self._gt_obstacle_cells: Set[Node] = self._build_gt_obstacle_cells()
        self._gt_entity_cells: Set[Node] = self._build_gt_entity_cells()
        self._gt_topology_edges: Set[Tuple[Node, Node]] = self._build_gt_topology_edges(grid_resolution)
        self._gt_free_cells: Set[Node] = self._calc_gt_free_cells(grid_resolution)

        self._init_static_layers()

        self.topology.geom_ref = self.geometry

        self._synced_obstacles: Set[Node] = set()
        self._synced_entities: Set[Node] = set()
        self._synced_visited: Set[Node] = set()
        self._synced_edges: Set[Tuple[Node, Node]] = set()
        self._synced_exp: Dict[Tuple[Node, Node], Dict[str, int]] = defaultdict(lambda: {'succ':0,'total':0})

    def print_layers_status(self):
        pass

    def update_from_observation(self, **kwargs):
        agent_pos = kwargs.get('agent_pos')
        if agent_pos is not None:
            x, y = agent_pos
            agent_node = (int(round(x)), int(round(y)))
            
            grid_node = (round(float(x), 3), round(float(y), 3))
            if (grid_node not in self.geometry.obstacle_cells and 
                grid_node not in self.geometry.entity_cells):
                self.geometry.visited_cells.add(agent_node)

        map_matrix = kwargs.get('map_matrix')
        sight_range = kwargs.get('sight_range')
        if agent_pos is not None and map_matrix is not None and sight_range is not None:
            self.geometry.update_from_sight(agent_pos, map_matrix=map_matrix, sight_range=sight_range)
        
        # Handle adversarial influence processing
        adversarial_config = kwargs.get('adversarial_config', {})
        if not adversarial_config.get('enable', True):
            pass
        elif self.enable_adversarial and self.adversarial_influence is not None:
            visual_observations = kwargs.get('visual_observations', {})
            enemy_positions = kwargs.get('enemy_positions', [])
            ally_positions = kwargs.get('ally_positions', [])
            current_agent_positions = kwargs.get('current_agent_positions', {})
            
            if not visual_observations and (enemy_positions or ally_positions):
                visual_observations = {
                    'enemy_positions': enemy_positions,
                    'ally_positions': ally_positions
                }
            
            if not current_agent_positions and agent_pos is not None:
                current_agent_positions = {'agent_0': agent_pos}
            
            if visual_observations or current_agent_positions:
                adversarial_costs, reachability_maps = self.adversarial_influence.process_step_observations(
                    visual_observations, current_agent_positions
                )
                
                self._integrate_adversarial_costs(adversarial_costs)

        # Update topology layer if enabled
        if self._include_topology:
            static_obstacles = self.adapter.static_obstacles()
            if static_obstacles:
                self.geometry.update(obstacles=static_obstacles)
        
        # Update confidence layer first to get dynamic update rates
        self.confidence.update(self.experience)
        
        # Update other layers based on confidence assessment
        for layer in self.layers:
            if layer.name == 'confidence':
                continue  # Already updated
            elif layer.name in ['geometry', 'topology', 'experience']:
                # Use confidence-based update rate for performance optimization
                if self.confidence.should_update_layer(layer.name):
                    layer.update()
            else:
                layer.update()
        
        # Record experience if path information is provided
        if kwargs.get('done', False):
            path = kwargs.get('path')
            success = kwargs.get('path_success', True)
            if path:
                self.record_experience(path, success)

    def _integrate_adversarial_costs(self, adversarial_costs: np.ndarray):
        if adversarial_costs is None:
            return
        
        expected_shape = (self.geometry.grid_size[1], self.geometry.grid_size[0])
        if adversarial_costs.shape != expected_shape:
            logger.warning(f"Adversarial cost map size mismatch: expected {expected_shape}, got {adversarial_costs.shape}")
            return
        
        if not hasattr(self.geometry, 'adversarial_costs'):
            self.geometry.adversarial_costs = np.zeros_like(adversarial_costs)
        
        self.geometry.adversarial_costs = adversarial_costs.copy()

    def get_adversarial_cost(self, pos: Node) -> float:
        if not hasattr(self.geometry, 'adversarial_costs'):
            return 0.0
        
        x, y = pos
        grid_x = int((x - self.geometry.bounds[0]) / self.geometry.resolution)
        grid_y = int((y - self.geometry.bounds[2]) / self.geometry.resolution)
        
        height, width = self.geometry.adversarial_costs.shape
        if 0 <= grid_x < width and 0 <= grid_y < height:
            return float(self.geometry.adversarial_costs[grid_y, grid_x])
        
        return 0.0

    def record_experience(self, path: List[Node], success: Union[bool, List[bool]] = True):
        """Record path traversal experience"""
        self.experience.record_traverse(path, success)

    def query_reachability(self, start: Node, goal: Node, max_cost: float = float('inf'), 
                         consider_adversarial: bool = None) -> Tuple[bool, float, List[Node]]:
        if consider_adversarial is None:
            consider_adversarial = self.enable_adversarial and hasattr(self.geometry, 'adversarial_costs')
        
        graph = self._aggregate_graph()
        
        if consider_adversarial and hasattr(self.geometry, 'adversarial_costs'):
            graph = self._apply_adversarial_costs_to_graph(graph)
        
        if start not in graph:
            return False, float('inf'), []
        if goal not in graph:
            return False, float('inf'), []
        if start == goal:
            return True, 0.0, [start]
        
        def heuristic(u: Node, v: Node) -> float:
            return abs(u[0] - v[0]) + abs(u[1] - v[1])
        
        open_set = [(0.0 + heuristic(start, goal), 0.0, start)]
        closed_set = set()
        came_from = {}
        g_score = {start: 0.0}
        
        while open_set:
            current_f, current_g, current = heapq.heappop(open_set)
            
            if current in closed_set:
                continue
            
            closed_set.add(current)
            
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return True, g_score[goal], path
            
            neighbors = graph.get(current, {})
            for neighbor, weight in neighbors.items():
                if neighbor in closed_set:
                    continue
                    
                tentative_g = current_g + weight
                
                if tentative_g > max_cost:
                    continue
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))
        
        return False, float('inf'), []

    def _apply_adversarial_costs_to_graph(self, graph: Dict[Node, Dict[Node, float]]) -> Dict[Node, Dict[Node, float]]:
        modified_graph = {}
        
        for u, neighbors in graph.items():
            modified_graph[u] = {}
            for v, base_weight in neighbors.items():
                adversarial_cost = self.get_adversarial_cost(v)
                
                final_weight = base_weight + adversarial_cost
                modified_graph[u][v] = final_weight
        
        return modified_graph

    def query_adversarial_reachability(self, start: Node, goal: Node, agent_id: str = None, 
                                     max_cost: float = float('inf')) -> Tuple[bool, float, List[Node]]:
        if not self.enable_adversarial or self.adversarial_influence is None:
            return self.query_reachability(start, goal, max_cost)
        
        is_reachable, cost, grid_path = self.adversarial_influence.query_adversarial_reachability(
            start, goal, agent_id
        )
        
        if not is_reachable:
            return False, float('inf'), []
        
        world_path = []
        for grid_x, grid_y in grid_path:
            world_x = grid_x * self.adversarial_influence.resolution + self.adversarial_influence.resolution/2
            world_y = grid_y * self.adversarial_influence.resolution + self.adversarial_influence.resolution/2
            world_path.append((world_x, world_y))
        
        return is_reachable, cost, world_path

    def _init_static_layers(self):
        is_smac_env = isinstance(self.adapter, SMACAdapter)
        
        if not is_smac_env:
            obstacles = self.adapter.static_obstacles()
            self.geometry.update(obstacles=obstacles)
        
        if self._include_topology:
            corridors = self.adapter.corridors()
            for cor in corridors:
                self.topology.add_corridor_edge(cor['start'], cor['end'], cor['weight'])
            
            portals = self.adapter.portals()
            for entrance, exit_, cost in portals:
                self.topology.add_portal(entrance, exit_, cost)

    def _aggregate_graph(self) -> Dict[Node, Dict[Node, float]]:
        adj: Dict[Node, Dict[Node, float]] = {}
        
        all_nodes = set()
        for layer in self.layers:
            for u, v, w in layer.edges():
                all_nodes.add(u)
                all_nodes.add(v)
        
        for node in all_nodes:
            adj[node] = {}
        
        for layer in self.layers:
            for u, v, w in layer.edges():
                if v not in adj[u] or w < adj[u][v]:
                    adj[u][v] = w
        
        return adj

    @staticmethod
    def _frange(start: float, stop: float, step: float) -> List[float]:
        vals = []
        x = start
        if start <= stop:
            while x <= stop + 1e-6:
                vals.append(round(x, 3))
                x += step
        else:
            while x >= stop - 1e-6:
                vals.append(round(x, 3))
                x -= step
        return vals

    def _build_gt_obstacle_cells(self) -> Set[Node]:
        res = self.geometry.resolution
        obstacle_cells: Set[Node] = set()
        
        is_smac_env = isinstance(self.adapter, SMACAdapter)
        
        if is_smac_env:
            if hasattr(self.adapter.env, 'map_matrix') and self.adapter.env.map_matrix is not None:
                map_matrix = self.adapter.env.map_matrix
                for i in range(map_matrix.shape[0]):
                    for j in range(map_matrix.shape[1]):
                        if map_matrix[i, j] == -1:
                            obstacle_node = (round(float(i), 3), round(float(j), 3))
                            obstacle_cells.add(obstacle_node)
            return obstacle_cells
        else:
            for rect in self.adapter.static_obstacles():
                left, right, bottom, top = rect
                def align(v):
                    return round(round(v / res) * res, 3)
                l = align(left); r_ = align(right); b = align(bottom); t = align(top)

                x = l
                while x <= r_ + 1e-6:
                    obstacle_cells.add((round(x, 3), round(b, 3)))
                    obstacle_cells.add((round(x, 3), round(t, 3)))
                    x += res
                y = b + res
                while y <= t - 1e-6:
                    obstacle_cells.add((round(l, 3), round(y, 3)))
                    obstacle_cells.add((round(r_, 3), round(y, 3)))
                    y += res

            return obstacle_cells

    def _build_gt_entity_cells(self) -> Set[Node]:
        res = self.geometry.resolution
        entity_cells: Set[Node] = set()

        for ent, exit_, _ in self.adapter.portals():
            ent_cell = (round(round(ent[0]/res)*res,3), round(round(ent[1]/res)*res,3))
            entity_cells.add(ent_cell)
            exit_cell = (round(round(exit_[0]/res)*res,3), round(round(exit_[1]/res)*res,3))
            entity_cells.add(exit_cell)
        
        if hasattr(self.adapter, 'env') and hasattr(self.adapter.env, 'world'):
            for landmark in self.adapter.env.world.landmarks:
                if not landmark.collide:
                    hx, hy = landmark.state.p_pos
                    portal_cell = (round(round(hx/res)*res,3), round(round(hy/res)*res,3))
                    entity_cells.add(portal_cell)
            
        return entity_cells

    def _build_gt_topology_edges(self, res: float) -> Set[Tuple[Node, Node]]:
        edges: Set[Tuple[Node, Node]] = set()
        geom_tmp = GeometryLayer(self.geometry.bounds, resolution=res)
        geom_tmp.update(obstacles=self.adapter.static_obstacles())
        for u, v, _ in geom_tmp.edges():
            edges.add((u, v))
        for corr in self.adapter.corridors():
            left, right, bottom, top = corr['left'], corr['right'], corr['bottom'], corr['top']
            orientation = corr.get('orientation', 'horizontal')
            direction = corr.get('direction', 'LR')
            if orientation == 'horizontal':
                y = (bottom + top) / 2
                x_vals = self._frange(left, right, res)
                pts = [(x, y) for x in x_vals]
                seq = pts if direction in ('LR', 'UD') else list(reversed(pts))
            else:
                x = (left + right) / 2
                y_vals = self._frange(bottom, top, res)
                pts = [(x, y) for y in y_vals]
                seq = pts if direction in ('UD', 'LR') else list(reversed(pts))
            for u, v in zip(seq[:-1], seq[1:]):
                edges.add((u, v))
        for ent, exit_, _ in self.adapter.portals():
            edges.add((ent, exit_))
        return edges

    def _calc_gt_free_cells(self, res: float) -> Set[Node]:
        xmin, xmax, ymin, ymax = self.geometry.bounds
        free: Set[Node] = set()
        occupied_cells = self._gt_obstacle_cells | self._gt_entity_cells
        x = xmin
        while x <= xmax + 1e-6:
            y = ymin
            while y <= ymax + 1e-6:
                pt = (round(x,3), round(y,3))
                if pt not in occupied_cells:
                    free.add(pt)
                y += res
            x += res
        return free

    def get_layers_stats(self):
        stats = {}
        stats['geometry_grid'] = f"{self.geometry.grid_size[0]}x{self.geometry.grid_size[1]}"
        stats['geometry_resolution'] = self.geometry.resolution
        
        total_obs = len(self._gt_obstacle_cells)
        total_ent = len(self._gt_entity_cells)
        total_combined = len(self._gt_obstacle_cells | self._gt_entity_cells)
        
        known_obs = len(self.geometry.obstacle_cells)
        known_ent = len(self.geometry.entity_cells)
        known_combined = len(self.geometry.obstacle_cells | self.geometry.entity_cells)
        
        obs_intersect = self.geometry.obstacle_cells & self._gt_obstacle_cells
        precision_obs = len(obs_intersect)/known_obs if known_obs else 0
        coverage_obs = len(obs_intersect)/total_obs if total_obs else 0
        
        ent_intersect = self.geometry.entity_cells & self._gt_entity_cells
        precision_ent = len(ent_intersect)/known_ent if known_ent else 0
        coverage_ent = len(ent_intersect)/total_ent if total_ent else 0
        
        combined_intersect = (self.geometry.obstacle_cells | self.geometry.entity_cells) & (self._gt_obstacle_cells | self._gt_entity_cells)
        precision_combined = len(combined_intersect)/known_combined if known_combined else 0
        coverage_combined = len(combined_intersect)/total_combined if total_combined else 0
        
        stats['geometry_obstacle_precision'] = precision_obs
        stats['geometry_obstacle_coverage'] = coverage_obs
        stats['geometry_obstacle_known'] = known_obs
        stats['geometry_obstacle_total'] = total_obs
        
        stats['geometry_entity_precision'] = precision_ent
        stats['geometry_entity_coverage'] = coverage_ent
        stats['geometry_entity_known'] = known_ent
        stats['geometry_entity_total'] = total_ent
        
        total_free = len(self._gt_free_cells)
        actual_visited_free = self.geometry.visited_cells - self.geometry.obstacle_cells - self.geometry.entity_cells
        known_free = len(actual_visited_free)
        coverage_free = known_free/total_free if total_free else 0
        stats['geometry_free_coverage'] = coverage_free
        stats['geometry_free_known'] = known_free
        stats['geometry_free_total'] = total_free
        edge_cnt = sum(len(v) for v in self.topology._adj.values())
        known_set = set((u, v) for u, v, _ in self.topology.edges())
        overlap = known_set & self._gt_topology_edges
        known_edges = len(overlap)
        prec_top = (known_edges/len(known_set)) if known_set else 0.0
        cov_top = known_edges/len(self._gt_topology_edges) if self._gt_topology_edges else 1.0
        stats['topology_edge_precision'] = prec_top
        stats['topology_edge_coverage'] = cov_top
        stats['topology_edge_known'] = known_edges
        stats['topology_edge_total'] = len(self._gt_topology_edges)
        stats['topology_edge_cnt'] = edge_cnt
        stats['topology_portal_cnt'] = len(self.topology.portals)
        stats['experience_path_cnt'] = len(self.experience.stats)
        stats['experience_succ_total'] = self.experience.succ_total
        stats['experience_fail_total'] = self.experience.fail_total
        total_action = self.experience.succ_total + self.experience.fail_total
        succ_rate = (self.experience.succ_total / total_action) if total_action else 0.0
        stats['experience_succ_rate'] = succ_rate
        stats['confidence_last_low_fix'] = getattr(self.confidence, 'last_low_conf_updates', 0)
        stats['confidence_total_blocked'] = getattr(self.confidence, 'total_blocked_edges', 0)
        stats['confidence_current_low'] = getattr(self.confidence, 'current_low_conf_edges', 0)
        return stats

    def diff(self) -> Dict[str, Any]:
        diff: Dict[str, Any] = {}

        new_obs = self.geometry.obstacle_cells - self._synced_obstacles
        new_ent = self.geometry.entity_cells - self._synced_entities
        new_vis = self.geometry.visited_cells - self._synced_visited
        if new_obs:
            diff['obstacle_cells'] = list(new_obs)
        if new_ent:
            diff['entity_cells'] = list(new_ent)
        if new_vis:
            diff['visited_cells'] = list(new_vis)

        cur_edges = set((u, v) for u, v, _ in self.topology.edges())
        new_edges = cur_edges - self._synced_edges
        if new_edges:
            diff['topology_edges'] = [list(e) for e in new_edges]

        exp_updates: Dict[str, List[int]] = {}
        for (u, v), st in self.experience.stats.items():
            last = self._synced_exp[(u, v)]
            ds = st['succ'] - last['succ']
            dt = st['total'] - last['total']
            if ds or dt:
                key = f"{u[0]},{u[1]}|{v[0]},{v[1]}"
                exp_updates[key] = [ds, dt]
        if exp_updates:
            diff['exp_updates'] = exp_updates

        self._synced_obstacles |= new_obs
        self._synced_entities |= new_ent
        self._synced_visited |= new_vis
        self._synced_edges |= new_edges
        for key, val in exp_updates.items():
            u_str, v_str = key.split('|')
            u = tuple(map(float, u_str.split(',')))
            v = tuple(map(float, v_str.split(',')))
            self._synced_exp[(u, v)]['succ'] += val[0]
            self._synced_exp[(u, v)]['total'] += val[1]

        return diff

    def apply_diff(self, diff: Dict[str, Any]):
        if not diff:
            return

        if 'obstacle_cells' in diff:
            self.geometry.obstacle_cells.update({tuple(n) for n in diff['obstacle_cells']})
        if 'entity_cells' in diff:
            self.geometry.entity_cells.update({tuple(n) for n in diff['entity_cells']})
        if 'visited_cells' in diff:
            self.geometry.visited_cells.update({tuple(n) for n in diff['visited_cells']})

        self._synced_obstacles |= set(map(tuple, diff.get('obstacle_cells', [])))
        self._synced_entities |= set(map(tuple, diff.get('entity_cells', [])))
        self._synced_visited |= set(map(tuple, diff.get('visited_cells', [])))

        for e in diff.get('topology_edges', []):
            u = tuple(e[0]); v = tuple(e[1])
            if self._is_valid_topology_edge(u, v) and v not in self.topology._adj.get(u, {}):
                self.topology.add_corridor_edge(u, v, weight=self.geometry.resolution)
            self._synced_edges.add((u, v))

        for key, (ds, dt) in diff.get('exp_updates', {}).items():
            u_str, v_str = key.split('|')
            u = tuple(map(float, u_str.split(',')))
            v = tuple(map(float, v_str.split(',')))
            st = self.experience.stats[(u, v)]
            st['succ'] += ds
            st['total'] += dt
            self.experience.succ_total += ds
            self.experience.fail_total += max(dt - ds, 0)
            self._synced_exp[(u, v)]['succ'] += ds
            self._synced_exp[(u, v)]['total'] += dt

    def _is_valid_topology_edge(self, u: Node, v: Node) -> bool:
        """Check if topology edge is valid"""
        # Basic validity check: nodes should not be obstacles
        if u in self.geometry.obstacle_cells or v in self.geometry.obstacle_cells:
            return False
        
        # Check if nodes are within bounds
        xmin, xmax, ymin, ymax = self.geometry.bounds
        for node in [u, v]:
            x, y = node
            if not (xmin <= x <= xmax and ymin <= y <= ymax):
                return False
        
        return True

def create_reachability_manager(env, grid_resolution: float = 0.25, debug: bool = False,
                                include_topology: bool = False, enable_adversarial: bool = False,
                                adversarial_config: Optional[Dict] = None) -> GraphManager:
    if env.__class__.__name__.startswith('MultiAgentParticleEnv'):
        adapter = MPEAdapter(env)
    elif env.__class__.__name__.lower().find('starcraft') != -1 or hasattr(env, 'map_x'):
        adapter = SMACAdapter(env)
    else:
        adapter = BaseEnvAdapter(env)
    return GraphManager(adapter, grid_resolution, enable_debug=debug, 
                       include_topology=include_topology, enable_adversarial=enable_adversarial,
                       adversarial_config=adversarial_config)