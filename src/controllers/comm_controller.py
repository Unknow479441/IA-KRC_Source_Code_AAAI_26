from .entity_controller import EntityMAC
import torch as th
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from modules.layers.comm_mixer import AverageMessageEncoder
from .global_map_store import get_map_matrix, get_agent_positions, get_all_map_matrices, get_all_agent_positions, get_graph_manager
import numpy as np
from collections import deque

MAP_MATRIX = None

class CommMAC(EntityMAC):
    def __init__(self, scheme, groups, args):
        assert args.use_msg
        super(CommMAC, self).__init__(scheme, groups, args)
        input_shape = self._get_input_shape(scheme)
        self.message_mixer = AverageMessageEncoder()
        self.message=None

    def decide_header(self, avail_actions, test_mode=False, agent_vis_mask=None):
        if self.args.order_leader:
            neighbor_num = th.sum(th.logical_not(agent_vis_mask), dim=-1)
            neighbor_num=neighbor_num.squeeze(1)
            header = th.zeros_like(neighbor_num).bool()
            alive_agent = th.logical_not(avail_actions[:,0,:,0])
            neighbor_num *= alive_agent
            if not self.args.select_by_prob:
                for _ in range(self.args.header_num):
                    _, ind = neighbor_num.max(1)
                    for i in range(neighbor_num.size(0)):
                        if alive_agent[i,ind[i]]:
                            header[i, ind[i]]=True
                        neighbor_num[i, ind[i]]=0
            else:
                for _ in range(self.args.header_num):
                    for i in range(neighbor_num.size(0)):
                        dis = Categorical(probs=neighbor_num[i].float())
                        ind = dis.sample()
                        if alive_agent[i, ind]:
                            header[i,ind]=True
                        neighbor_num[i,ind]=0
            return header.detach()
        else:
            alive_agent = th.logical_not(avail_actions[:,0,:,0])
            bs, _ = alive_agent.shape
            candidate_num = th.sum(alive_agent, axis=1).unsqueeze(1).repeat(1,self.n_agents)
            rnd = alive_agent*th.rand([bs, self.n_agents], device=self.args.device)
            generation_alpha = self.args.generation_alpha if test_mode else 1.0
            header = (rnd*candidate_num < self.args.header_num * generation_alpha) * alive_agent
            return header.detach()
    
    def _bfs_reachable(self, map_matrix, start_pos, max_distance):
        if map_matrix is None:
            return set()
            
        rows, cols = map_matrix.shape
        visited = set()
        queue = [(start_pos, 0)]
        reachable = set()
        
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        while queue:
            (x, y), dist = queue.pop(0)
            
            if dist > max_distance:
                continue
                
            if (x, y) in visited:
                continue
                
            visited.add((x, y))
            
            if 0 <= x < rows and 0 <= y < cols and map_matrix[x, y] != -1 and map_matrix[x, y] != -9:
                reachable.add((x, y))
                
                for dx, dy in directions:
                    new_x, new_y = x + dx, y + dy
                    if 0 <= new_x < rows and 0 <= new_y < cols and (new_x, new_y) not in visited:
                        queue.append(((new_x, new_y), dist + 1))
        
        return reachable
    
    def _pathfinding_reachable(self, graph_manager, start_pos, max_cost):
        if graph_manager is None:
            return set()
        
        reachable_positions = set()
        start_node = (int(round(start_pos[0])), int(round(start_pos[1])))
        
        try:
            graph_adj = graph_manager._aggregate_graph()
            if start_node not in graph_adj:
                return set()
            
            import heapq
            pq = [(0.0, start_node)]
            visited_costs = {start_node: 0.0}
            
            while pq:
                current_cost, current_node = heapq.heappop(pq)
                
                if current_cost > max_cost:
                    continue
                
                if current_cost > visited_costs.get(current_node, float('inf')):
                    continue
                
                reachable_positions.add(current_node)
                
                for neighbor, edge_weight in graph_adj.get(current_node, {}).items():
                    new_cost = current_cost + edge_weight
                    
                    if new_cost <= max_cost and new_cost < visited_costs.get(neighbor, float('inf')):
                        visited_costs[neighbor] = new_cost
                        heapq.heappush(pq, (new_cost, neighbor))
            
            return reachable_positions
            
        except Exception as e:
            print(f"[ERROR] 路径规划可达性计算失败: {e}")
            return set()
    
    def _balance_groups(self, control_message, header, alive_agent):
        bs = control_message.shape[0]
        balanced_control_message = th.zeros_like(control_message)
        
        for b in range(bs):
            leaders = [i for i in range(self.n_agents) if header[b, i]]
            if not leaders:
                continue
                
            followers = []
            for j in range(self.n_agents):
                if not header[b, j] and alive_agent[b, j]:
                    followers.append(j)
            
            visible_followers = {}
            for leader in leaders:
                visible_followers[leader] = []
                for follower in followers:
                    if control_message[b, 0, leader, follower]:
                        visible_followers[leader].append(follower)
            
            groups = {leader: [] for leader in leaders}
            
            assigned_followers = set()
            
            for follower in followers:
                visible_leaders = [l for l in leaders if control_message[b, 0, l, follower]]
                if len(visible_leaders) == 1:
                    leader = visible_leaders[0]
                    groups[leader].append(follower)
                    assigned_followers.add(follower)
            
            remaining_followers = [f for f in followers if f not in assigned_followers]
            while remaining_followers:
                min_group_size = float('inf')
                min_leader = None
                
                for leader in leaders:
                    available_followers = [f for f in remaining_followers if f in visible_followers[leader]]
                    if available_followers and len(groups[leader]) < min_group_size:
                        min_group_size = len(groups[leader])
                        min_leader = leader
                
                if min_leader is None:
                    break
                    
                for follower in remaining_followers:
                    if follower in visible_followers[min_leader]:
                        groups[min_leader].append(follower)
                        remaining_followers.remove(follower)
                        break
            
            for leader, group_followers in groups.items():
                for follower in group_followers:
                    balanced_control_message[b, 0, leader, follower] = True
        
        return balanced_control_message
    
    def decide_group(self, agent_inputs, avail_actions, test_mode=False):
        bs = avail_actions.shape[0]
        
        if self.args.use_comm_sr:
            original_agent_vis_mask = self.gt_mask[:,:, :self.n_agents, :self.n_agents]
        else:
            entity, obs_mask, entity_mask = agent_inputs
            original_agent_vis_mask = obs_mask[:,:, :self.n_agents, :self.n_agents]
        
        agent_vis_mask = original_agent_vis_mask.clone()

        alive_agent = th.logical_not(avail_actions[:,0,:,0])
        
        alive_counts = th.sum(alive_agent, dim=1)
        
        original_header_num = self.args.header_num
        dynamic_header_nums = th.minimum(
            th.tensor([original_header_num] * bs, device=self.args.device),
            (alive_counts / 2).floor().int()
        )
        
        use_explore = getattr(self.args, 'use_explore_map', True)
        use_pathfinding_reachability = getattr(self.args, 'use_pathfinding_reachability', True)

        all_reachable_counts = {}
        
        def _cont2grid(p, rows, cols):
            x_c, y_c = p
            gx = int(np.clip((x_c + 1.0) / 2.0 * (rows - 1), 0, rows - 1))
            gy = int(np.clip((y_c + 1.0) / 2.0 * (cols - 1), 0, cols - 1))
            return gx, gy

        rows, cols = get_map_matrix().shape if get_map_matrix() is not None else (32, 32)

        for b_idx in range(bs):
            if use_pathfinding_reachability:
                graph_manager = get_graph_manager()
                if graph_manager is None:
                    map_matrix = get_map_matrix() if use_explore else get_map_matrix()
                    use_pathfinding = False
                else:
                    use_pathfinding = True
                    
                    adversarial_config = getattr(self.args, 'adversarial_influence', {})
                    if (adversarial_config.get('enable', False) and 
                        hasattr(graph_manager, 'adversarial_influence') and 
                        graph_manager.adversarial_influence is not None):
                        agent_positions = get_agent_positions(b_idx)
                        if agent_positions:
                            from controllers.global_map_store import get_visible_enemies
                            enemy_positions = get_visible_enemies(b_idx, agent_positions)
                            
                            visual_observations = {
                                'agent_positions': agent_positions,
                                'enemy_positions': list(enemy_positions.values()) if enemy_positions else [],
                                'ally_positions': list(agent_positions.values()) if agent_positions else []
                            }
                            
                            try:
                                adversarial_costs, reachability_maps = graph_manager.adversarial_influence.process_step_observations(
                                    visual_observations, agent_positions
                                )
                                graph_manager._integrate_adversarial_costs(adversarial_costs)
                                
                            except Exception as e:
                                print(f"[ERROR] 对抗性影响图更新失败: {e}")
                                import traceback
                                traceback.print_exc()
            else:
                map_matrix = get_map_matrix() if use_explore else get_map_matrix()
                use_pathfinding = False
                
            agent_positions = get_agent_positions(b_idx)
            
            if agent_positions is not None and len(agent_positions) > 0:
                k_step = getattr(self.args, 'k_step', 0)
                move_amount = getattr(self.args, 'move_amount', 2)
                max_cost = k_step * move_amount

                reachable_map = {}
                reachable_counts = {}
                
                for agent_id, pos in agent_positions.items():
                    if agent_id >= self.n_agents:
                        continue
                    
                    grid_pos = _cont2grid(pos, rows, cols)
                    
                    if use_pathfinding:
                        reachable_positions = self._pathfinding_reachable(graph_manager, grid_pos, max_cost)
                    else:
                        reachable_positions = self._bfs_reachable(map_matrix, grid_pos, max_cost)

                    reachable_map[agent_id] = reachable_positions
                    
                    count = 0
                    reachable_agents = []
                    for other_id, other_pos in agent_positions.items():
                        if other_id != agent_id and other_id < self.n_agents:
                            if _cont2grid(other_pos, rows, cols) in reachable_positions:
                                count += 1
                                reachable_agents.append(other_id)
                    
                    reachable_counts[agent_id] = count

                all_reachable_counts[b_idx] = reachable_counts
                
                for agent_id in agent_positions:
                    if agent_id >= self.n_agents:
                        continue
                        
                    for other_id in agent_positions:
                        if other_id >= self.n_agents:
                            continue
                            
                        if other_id != agent_id and _cont2grid(agent_positions[other_id], rows, cols) in reachable_map.get(agent_id, set()):
                            if alive_agent[b_idx, agent_id] and alive_agent[b_idx, other_id]:
                                agent_vis_mask[b_idx, 0, agent_id, other_id] = 0

        header = th.zeros((bs, self.n_agents), dtype=th.bool, device=self.args.device)
        
        for b_idx in range(bs):
            leaders_selected = 0
            max_leaders = int(dynamic_header_nums[b_idx].item())
            
            if b_idx in all_reachable_counts and all_reachable_counts[b_idx] and len(all_reachable_counts[b_idx]) > 0:
                reachable_counts = all_reachable_counts[b_idx]
                sorted_agents = sorted(reachable_counts.items(), key=lambda x: x[1], reverse=True)
                
                for agent_id, count in sorted_agents:
                    if leaders_selected >= max_leaders:
                        break
                    if agent_id < self.n_agents and alive_agent[b_idx, agent_id]:
                        header[b_idx, agent_id] = True
                        leaders_selected += 1

                continue
            
            b_agent_vis_mask = agent_vis_mask[b_idx:b_idx+1]
            b_avail_actions = avail_actions[b_idx:b_idx+1]
            b_header = th.zeros((1, self.n_agents), dtype=th.bool, device=self.args.device)
            
            if self.args.order_leader:
                neighbor_num = th.sum(th.logical_not(b_agent_vis_mask), dim=-1)
                neighbor_num = neighbor_num.squeeze(1)
                b_alive_agent = th.logical_not(b_avail_actions[:,0,:,0])
                neighbor_num *= b_alive_agent
                
                if not self.args.select_by_prob:
                    for _ in range(min(max_leaders, neighbor_num.size(1))):
                        if leaders_selected >= max_leaders:
                            break
                        
                        _, ind = neighbor_num.max(1)
                        for i in range(neighbor_num.size(0)):
                            if b_alive_agent[i, ind[i]]:
                                b_header[i, ind[i]] = True
                                leaders_selected += 1
                            neighbor_num[i, ind[i]] = 0
                else:
                    for _ in range(min(max_leaders, neighbor_num.size(1))):
                        if leaders_selected >= max_leaders:
                            break
                            
                        for i in range(neighbor_num.size(0)):
                            if th.sum(neighbor_num[i]) > 0 and leaders_selected < max_leaders:
                                dis = Categorical(probs=neighbor_num[i].float())
                                ind = dis.sample()
                                if b_alive_agent[i, ind]:
                                    b_header[i, ind] = True
                                    leaders_selected += 1
                                neighbor_num[i, ind] = 0
                
                header[b_idx] = b_header.squeeze(0)
            else:
                b_alive_agent = th.logical_not(b_avail_actions[:,0,:,0])
                candidate_num = th.sum(b_alive_agent, axis=1).unsqueeze(1).repeat(1,self.n_agents)
                rnd = b_alive_agent*th.rand([1, self.n_agents], device=self.args.device)
                generation_alpha = self.args.generation_alpha if test_mode else 1.0
                
                b_header_candidates = (rnd*candidate_num < max_leaders * generation_alpha) * b_alive_agent
                
                if th.sum(b_header_candidates) > max_leaders:
                    candidates = th.nonzero(b_header_candidates.squeeze(0), as_tuple=False).squeeze(1)
                    perm = th.randperm(len(candidates))
                    selected_candidates = candidates[perm[:max_leaders]]
                    b_header = th.zeros_like(b_header_candidates)
                    b_header[0, selected_candidates] = True
                else:
                    b_header = b_header_candidates
                
                header[b_idx] = b_header.squeeze(0)
            

        control_message = header.unsqueeze(1).unsqueeze(3).repeat(1,1,1,self.n_agents)*th.logical_not(agent_vis_mask)
        control_message *= th.logical_not(header.unsqueeze(1).unsqueeze(2).repeat(1,1,self.n_agents,1))
        
        if getattr(self.args, 'use_balanced_group', True):
            control_message = self._balance_groups(control_message, header, alive_agent)

        try:
            groups = {}
            leaders = set()
            reachable_counts = {}
            
            if bs > 0:
                leader_indices = th.where(header[0])[0].cpu().numpy()
                for leader_id in leader_indices:
                    leaders.add(int(leader_id))
                
                control_msg_0 = control_message[0]
                group_id = 0
                processed_agents = set()
                
                for leader_id in leader_indices:
                    if leader_id in processed_agents:
                        continue
                    
                    followers = []
                    for follower_id in range(self.n_agents):
                        if control_msg_0[0, leader_id, follower_id] > 0:
                            followers.append(int(follower_id))
                            processed_agents.add(follower_id)
                    
                    group_members = [int(leader_id)] + followers
                    groups[group_id] = group_members
                    processed_agents.add(leader_id)
                    group_id += 1
                
                if 0 in all_reachable_counts:
                    reachable_counts = {k: v for k, v in all_reachable_counts[0].items()}
            
            self.decide_group_results = {
                'groups': groups,
                'leaders': leaders,
                'reachable_counts': reachable_counts,
                'alive_agents': set(int(i) for i in th.where(alive_agent[0])[0].cpu().numpy())
            }
            
        except Exception as e:
            print(f"[ERROR] 存储分组结果失败: {e}")
            import traceback
            traceback.print_exc()

        return control_message.detach(), header

    def message_comm(self, agent_inputs, avail_actions, t, train_mode=False, test_mode=False, env_index=0, **kwargs):
        entity, obs_mask, entity_mask = agent_inputs
        if train_mode:
            message_personal, msg_dis, msg_dis_inf = self.agent(agent_inputs, self.hidden_states, ret_inf_msg=True, **kwargs)
        else:
            message_personal, _ = self.agent(agent_inputs, self.hidden_states)
        message_personal = message_personal.squeeze(1)
        lt = t.stop-t.start

        if lt == 1 and t.start % self.args.msg_T==0:
            env_index = kwargs.get('env_index', 0)
            message_matrix, header = self.decide_group(agent_inputs, avail_actions, test_mode=test_mode)
            with th.no_grad():
                message_matrix = message_matrix.squeeze(1)+th.diag_embed(header)
                message_personal_r = message_personal.unsqueeze(1).repeat(1,self.n_agents,1,1)
                message_pass_h = message_matrix.unsqueeze(3)*message_personal_r
                message_header = self.message_mixer(message_pass_h, message_matrix)

                if getattr(self.args, 'use_leader_comm', True):
                    bs, n_agents, msg_dim = message_header.shape
                    for b in range(bs):
                        leader_indices = th.where(header[b])[0]
                        if len(leader_indices) > 1:
                            leader_msgs = message_header[b, leader_indices]
                            leader_msgs_agg = leader_msgs.mean(dim=0, keepdim=True).repeat(len(leader_indices), 1)
                            message_header[b, leader_indices] = leader_msgs_agg

                message_header_r = message_header.unsqueeze(2).repeat(1,1,self.n_agents,1)
                message_pass_a = th.sum(message_matrix.unsqueeze(3)*message_header_r, dim=1)
                if self.args.no_feedback:
                    receive_matrix = header
                else:
                    receive_matrix = th.max(message_matrix,dim=1)[0]
                self.message = receive_matrix.unsqueeze(2) * message_pass_a + th.logical_not(receive_matrix).unsqueeze(2) * message_personal
        elif not self.args.only_use_head_msg:
            self.message = message_personal.detach()
        if train_mode:
            return message_personal, self.message, msg_dis, msg_dis_inf
        else:
            return message_personal, self.message

    def forward(self, ep_batch, t, test_mode=False, fix_msg=None, train_mode=False, env_index=0, **kwargs):
        if t is None:
            t = slice(0, ep_batch["avail_actions"].shape[1])
            int_t = False
        elif type(t) is int:
            t = slice(t, t + 1)
            int_t = True

        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        if train_mode:
            p_msg = ep_batch["self_message"]
            h_msg = ep_batch["head_message"]
            if kwargs.get('imagine', False):
                agent_outs, self.hidden_states, groups = self.agent(agent_inputs, self.hidden_states, msg = h_msg, **kwargs)
            else:
                agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states, msg = h_msg)
            _, _, msg_dis, msg_dis_inf = self.message_comm(agent_inputs, avail_actions, t, train_mode=True, env_index=env_index, **kwargs)
        else:
            if fix_msg is None:
                p_msg, h_msg = self.message_comm(agent_inputs, avail_actions, t, test_mode=test_mode, env_index=env_index, **kwargs)
            else:
                p_msg = h_msg = fix_msg
            if kwargs.get('imagine', False):
                agent_outs, self.hidden_states, groups = self.agent(agent_inputs, self.hidden_states, msg = h_msg, **kwargs)
            else:
                agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states, msg = h_msg)

        outs = (agent_outs.squeeze(1),) if int_t else (agent_outs,)
        if kwargs.get('imagine', False):
            outs += (groups,)
        if fix_msg is None:
            outs += (p_msg, h_msg)
        if train_mode:
            outs += (msg_dis, msg_dis_inf)
        return outs

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False, ret_agent_outs=False, ret_msg=False):
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        
        if isinstance(bs, slice):
            env_indices = list(range(ep_batch.batch_size))
        else:
            env_indices = bs
        
        if len(env_indices) == 1:
            env_index = env_indices[0]
            agent_outputs, self_msg, head_msg = self.forward(ep_batch, t_ep, test_mode=test_mode, fix_msg=None, env_index=env_index)
        else:
            agent_outputs, self_msg, head_msg = self.forward(ep_batch, t_ep, test_mode=test_mode, fix_msg=None)
            
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)

        actions_np = chosen_actions.cpu().numpy() if hasattr(chosen_actions, 'cpu') else chosen_actions
        flat_actions = actions_np.flatten()
        action_hist = np.bincount(flat_actions, minlength=avail_actions[bs].shape[-1])

        if ret_agent_outs:
            return chosen_actions, agent_outputs[bs], self_msg[bs], head_msg[bs]
        if ret_msg:
            return chosen_actions, self_msg[bs], head_msg[bs]

        if hasattr(chosen_actions, 'shape') and len(chosen_actions.shape) == 2 and len(avail_actions[bs].shape) == 3:
            for b in range(avail_actions[bs].shape[0]):
                for a in range(avail_actions[bs].shape[1]):
                    act = chosen_actions[b, a].item()
                    if avail_actions[bs][b, a, act] != 1:
                        raise AssertionError("Picked unavailable action!")

        return chosen_actions

    def _build_inputs(self, batch, t):
        bs = batch.batch_size
        entities = []
        entities.append(batch["entities"][:, t])
        if self.args.entity_last_action:
            ent_acs = th.zeros(bs, t.stop - t.start, self.args.n_entities,
                               self.args.n_actions, device=batch.device,
                               dtype=batch["entities"].dtype)
            if t.start == 0:
                ent_acs[:, 1:, :self.args.n_agents] = (
                    batch["actions_onehot"][:, slice(0, t.stop - 1)])
            else:
                ent_acs[:, :, :self.args.n_agents] = (
                    batch["actions_onehot"][:, slice(t.start - 1, t.stop - 1)])
            entities.append(ent_acs)
        entities = th.cat(entities, dim=3)
        if self.args.gt_mask_avail:
            return (entities, batch["obs_mask"][:, t], batch["entity_mask"][:, t], batch["gt_mask"][:, t])
        if self.args.use_comm_sr:
            self.gt_mask = batch["gt_mask"][:, t]
        return (entities, batch["obs_mask"][:, t], batch["entity_mask"][:, t])
