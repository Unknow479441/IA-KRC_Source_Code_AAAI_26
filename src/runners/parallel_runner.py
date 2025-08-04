from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
from multiprocessing import Pipe, Process
import numpy as np
import torch as th


class ParallelRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run

        self.parent_conns, self.worker_conns = zip(*[Pipe() for _ in range(self.batch_size)])
        env_fn = env_REGISTRY[self.args.env]
        base_seed = self.args.env_args.pop('seed')
        
        sight_range = getattr(self.args, "exploration", {}).get("sight_range", 9)
        exploration_threshold = getattr(self.args, "exploration", {}).get("threshold", 0.95)
        
        env_args = self.args.env_args.copy()
        if 'self_play' not in env_args:
            env_args['self_play'] = getattr(self.args, "self_play", False)
        
        self.ps = [Process(target=env_worker, args=(worker_conn, self.args.entity_scheme,
                                                  CloudpickleWrapper(partial(env_fn, 
                                                                           seed=base_seed + rank,
                                                                           env_index=rank,
                                                                           exploration_threshold=exploration_threshold,
                                                                           **env_args))))
                  for rank, worker_conn in enumerate(self.worker_conns)]
        self.args.env_args['seed'] = base_seed

        for p in self.ps:
            p.daemon = True
            p.start()

        self.parent_conns[0].send(("get_env_info", args))
        self.env_info = self.parent_conns[0].recv()
        self.episode_limit = self.env_info["episode_limit"]

        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        self.log_train_stats_t = -100000
        self.n_agents = self.env_info["n_agents"]
        
        if env_args['self_play']:
            self.n_enemies = self.env_info.get("n_enemies", self.n_agents)
        else:
            self.n_enemies = 0

        self.debug_health_index = False
        
        self.debug_episode_done = False
        
        self.current_episode_id = 0
        
        self._init_adversarial_visualizer()

    def _init_adversarial_visualizer(self):
        try:
            adversarial_config = getattr(self.args, 'adversarial_influence', {})
            if not adversarial_config.get('enable', False):
                return
            
            visualization_config = adversarial_config.get('visualization', {})
            if not visualization_config.get('enable_heatmap', False):
                return
            
            final_config = visualization_config.copy()
            
            env_config = {
                'move_amount': getattr(self.args, 'move_amount', 2),
                'k_step': getattr(self.args, 'k_step', 0)
            }
            
            if final_config.get('enable_heatmap', False):
                from controllers.global_map_store import init_graph_manager, reset_graph_manager
                
                reset_graph_manager()
                
                map_size = final_config.get('map_size', [32, 32])
                init_graph_manager(
                    map_shape=(map_size[0], map_size[1]),
                    enable_adversarial=True,
                    adversarial_config=adversarial_config,
                    k_step=env_config['k_step'],
                    move_amount=env_config['move_amount']
                )
                
                from controllers.global_map_store import init_visualizer
                init_visualizer(final_config)

        
        except Exception as e:
            if hasattr(self.logger, 'error'):
                self.logger.error(f"Failed to initialize adversarial influence map visualizer: {e}")
            else:
                print(f"[ERROR] Failed to initialize adversarial influence map visualizer: {e}")
                if hasattr(self.logger, 'warn'):
                    self.logger.warn(f"Failed to initialize adversarial influence map visualizer: {e}")
                elif hasattr(self.logger, 'warning'):
                    self.logger.warning(f"Failed to initialize adversarial influence map visualizer: {e}")
                else:
                    import logging
                    logging.error(f"Failed to initialize adversarial influence map visualizer: {e}")

    def setup(self, scheme, groups, preprocess, mac, enemy_mac=None):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac
        self.enemy_mac = enemy_mac
        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess

    def get_env_info(self):
        return self.env_info

    def save_replay(self):
        pass

    def close_env(self):
        try:
            from controllers.global_map_store import finalize_visualizer
            finalize_visualizer()
        except Exception as e:
            if hasattr(self.logger, 'error'):
                self.logger.error(f"Failed to finalize visualizer: {e}")
            else:
                print(f"[ERROR] Failed to finalize visualizer: {e}")
                if hasattr(self.logger, 'warn'):
                    self.logger.warn(f"Failed to finalize visualizer: {e}")
                elif hasattr(self.logger, 'warning'):
                    self.logger.warning(f"Failed to finalize visualizer: {e}")
                else:
                    import logging
                    logging.error(f"Failed to finalize visualizer: {e}")
        
        for parent_conn in self.parent_conns:
            parent_conn.send(("close", None))

    def reset(self, **kwargs):
        self.batch = self.new_batch()

        for parent_conn in self.parent_conns:
            parent_conn.send(("reset", kwargs))

        pre_transition_data = {}
        for parent_conn in self.parent_conns:
            data = parent_conn.recv()
            
            self_play_mode = hasattr(self, 'enemy_mac') and self.enemy_mac is not None and hasattr(self.args, 'self_play') and self.args.self_play
            
            if self_play_mode and 'entities' in data:
                for k, v in data.items():
                    if k != 'avail_actions':
                        if k in pre_transition_data:
                            pre_transition_data[k].append(v)
                        else:
                            pre_transition_data[k] = [v]
                
                all_avail_actions = data['avail_actions']
                n_agents = self.args.n_agents
                n_enemies = self.args.n_enemies
                
                ally_avail_actions = all_avail_actions[:n_agents]
                enemy_avail_actions = all_avail_actions[n_agents:n_agents+n_enemies]
                
                if 'avail_actions' in pre_transition_data:
                    pre_transition_data['avail_actions'].append(ally_avail_actions)
                else:
                    pre_transition_data['avail_actions'] = [ally_avail_actions]
                    
                if 'enemy_avail_actions' in pre_transition_data:
                    pre_transition_data['enemy_avail_actions'].append(enemy_avail_actions)
                else:
                    pre_transition_data['enemy_avail_actions'] = [enemy_avail_actions]
            else:
                for k, v in data.items():
                    if k in pre_transition_data:
                        pre_transition_data[k].append(v)
                    else:
                        pre_transition_data[k] = [v]

        self.batch.update(pre_transition_data, ts=0)

        self.t = 0
        self.env_steps_this_run = 0

    def run(self, test_mode=False, test_scen=None, index=None, vid_writer=None, constrain_num=None):
        if test_scen is None:
            test_scen = test_mode
        assert vid_writer is None, "Writing videos not supported for ParallelRunner"
        if self.args.test_unseen:
            constrain_num=self.args.test_map_num if test_mode else self.args.train_map_num
        else:
            constrain_num=None
        if self.args.env == "traffic_junction":
            self.reset(t_env=self.t_env)
        else:
            self.reset(test=test_scen, index=index, constrain_num=constrain_num)

        all_terminated = False
        episode_returns = [0 for _ in range(self.batch_size)]
        episode_lengths = [0 for _ in range(self.batch_size)]
        
        self.mac.init_hidden(batch_size=self.batch_size)
        if hasattr(self, 'enemy_mac') and self.enemy_mac is not None:
            self.enemy_mac.init_hidden(batch_size=self.batch_size)
            
        if test_mode:
            self.mac.eval()
            if hasattr(self, 'enemy_mac') and self.enemy_mac is not None:
                self.enemy_mac.eval()
        else:
            self.mac.train()
            if hasattr(self, 'enemy_mac') and self.enemy_mac is not None:
                self.enemy_mac.train()
                
        terminated = [False for _ in range(self.batch_size)]
        envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
        final_env_infos = []

        while True:
            self_play_mode = hasattr(self, 'enemy_mac') and self.enemy_mac is not None and hasattr(self.args, 'self_play') and self.args.self_play
            
            actions_chosen = {}
            if self.args.mac in ["comm_mac", "comm_mac_sp", "heucomm_mac", "dppcomm_mac", "commformer_mac", "commformer_mac_sp"]:
                actions, p_msg, h_msg = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, bs=envs_not_terminated, test_mode=test_mode, ret_msg=True)
                cpu_actions = actions.to("cpu").numpy()
                cpu_p_msg = p_msg.detach().cpu().numpy()
                cpu_h_msg = h_msg.detach().cpu().numpy()
                actions_chosen.update({
                    "actions": actions.unsqueeze(1),
                    "self_message": th.tensor(cpu_p_msg, dtype=th.float32, device=self.args.device).unsqueeze(1),
                    "head_message": th.tensor(cpu_h_msg, dtype=th.float32, device=self.args.device).unsqueeze(1)
                })
            elif self.args.mac == "rlcomm_mac":
                actions, p_msg, h_msg, head_prob, election_actions = self.mac.select_actions(
                    self.batch, t_ep = self.t, t_env = self.t_env, bs=envs_not_terminated,
                    test_mode=test_mode, ret_msg=True)
                cpu_actions = actions.to("cpu").numpy()
                cpu_p_msg = p_msg.detach().cpu().numpy()
                cpu_h_msg = h_msg.detach().cpu().numpy()
                actions_chosen.update({
                    "actions": actions.unsqueeze(1),
                    "self_message": th.tensor(cpu_p_msg, dtype=th.float32, device=self.args.device).unsqueeze(1),
                    "head_message": th.tensor(cpu_h_msg, dtype=th.float32, device=self.args.device).unsqueeze(1)
                })
                if head_prob is not None:
                    head_chosen = {
                        "head_probs": head_prob[envs_not_terminated],
                        "head_actions": election_actions[envs_not_terminated], 
                    }
                    actions_chosen.update(head_chosen)
            elif self.args.mac == "rlcomm_mac_sp":
                actions = self.mac.select_actions(
                    self.batch, t_ep = self.t, t_env = self.t_env, bs=envs_not_terminated,
                    test_mode=test_mode)
                cpu_actions = actions.to("cpu").numpy()
                actions_chosen.update({"actions": actions.unsqueeze(1)})
                zeros_msg = th.zeros((len(envs_not_terminated), 1, self.n_agents, self.args.msg_dim),
                                      dtype=th.float32, device=self.args.device)
                actions_chosen.update({
                    "self_message": zeros_msg,
                    "head_message": zeros_msg,
                })
            else:
                if self.args.save_entities_and_attn_weights:
                    actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, bs=envs_not_terminated, test_mode=test_mode, ret_attn_weights=True)
                else:
                    actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, bs=envs_not_terminated, test_mode=test_mode)
                cpu_actions = actions.to("cpu").numpy()
                actions_chosen.update({
                    "actions": actions.unsqueeze(1)
                })
            
            if self_play_mode:
                enemy_avail_all = [
                    (self.batch["enemy_avail_actions"][bs, self.t] if "enemy_avail_actions" in self.batch.scheme else self.batch["avail_actions"][bs, self.t])
                    for bs in envs_not_terminated
                ]

                enemy_test_mode = test_mode

                enemy_p_msgs = enemy_h_msgs = None
                enemy_head_prob_tensor = enemy_election_actions_tensor = None

                if self.args.enemy_mac in ["comm_mac", "comm_mac_sp", "heucomm_mac", "dppcomm_mac", "commformer_mac", "commformer_mac_sp"]:
                    ret = self.enemy_mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env,
                                                        bs=envs_not_terminated, test_mode=enemy_test_mode, ret_msg=True)
                    if isinstance(ret, tuple) and len(ret) == 3:
                        enemy_actions, enemy_p_msgs, enemy_h_msgs = ret
                    else:
                        enemy_actions = ret
                elif self.args.enemy_mac in ["rlcomm_mac", "rlcomm_mac_sp"]:
                    ret = self.enemy_mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env,
                                                        bs=envs_not_terminated, test_mode=enemy_test_mode, ret_msg=True)
                    if isinstance(ret, tuple) and len(ret) == 5:
                        enemy_actions, enemy_p_msgs, enemy_h_msgs, enemy_head_prob_tensor, enemy_election_actions_tensor = ret
                    else:
                        enemy_actions = ret if not isinstance(ret, tuple) else ret[0]
                elif self.args.enemy_mac in ["gat_mac_sp"]:
                    enemy_actions = self.enemy_mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env,
                                                                  bs=envs_not_terminated, test_mode=enemy_test_mode)
                else:
                    enemy_actions = self.enemy_mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env,
                                                                  bs=envs_not_terminated, test_mode=enemy_test_mode)

                if enemy_actions.shape[0] != len(envs_not_terminated):
                    enemy_actions = enemy_actions[envs_not_terminated]
                    if enemy_p_msgs is not None:
                        enemy_p_msgs = enemy_p_msgs[envs_not_terminated]
                    if enemy_h_msgs is not None:
                        enemy_h_msgs = enemy_h_msgs[envs_not_terminated]
                    if enemy_head_prob_tensor is not None:
                        enemy_head_prob_tensor = enemy_head_prob_tensor[envs_not_terminated]
                        enemy_election_actions_tensor = enemy_election_actions_tensor[envs_not_terminated]

                enemy_cpu_actions = enemy_actions.cpu().numpy()

                if (enemy_p_msgs is None or enemy_h_msgs is None) and hasattr(self.enemy_mac, "forward"):
                    try:
                        with th.no_grad():
                            fw_outs = self.enemy_mac.forward(self.batch, self.t, test_mode=enemy_test_mode)
                        if len(fw_outs) >= 3:
                            enemy_p_msgs = fw_outs[1]
                            enemy_h_msgs = fw_outs[2]
                    except Exception:
                        zeros_shape = (len(envs_not_terminated), self.n_agents, self.args.msg_dim)
                        enemy_p_msgs = th.zeros(zeros_shape, device=self.args.device)
                        enemy_h_msgs = enemy_p_msgs.clone()

                if enemy_p_msgs is None:
                    zeros_shape = (len(envs_not_terminated), self.n_agents, self.args.msg_dim)
                    enemy_p_msgs = th.zeros(zeros_shape, device=self.args.device)
                    enemy_h_msgs = enemy_p_msgs.clone()

                for env_idx, bs in enumerate(envs_not_terminated):
                    avail = self.batch["avail_actions"][bs, self.t]
                    avail_np = avail.cpu().numpy() if hasattr(avail, "cpu") else np.array(avail)
                    acts    = cpu_actions[env_idx]

                    invalid_mask = avail_np[np.arange(self.n_agents), acts] == 0
                    if invalid_mask.any():
                        replacement = (avail_np == 1).argmax(axis=1)
                        acts[invalid_mask] = replacement[invalid_mask]
                    cpu_actions[env_idx] = acts

                for env_idx, enemy_avail in enumerate(enemy_avail_all):
                    avail_np = enemy_avail.cpu().numpy() if hasattr(enemy_avail, "cpu") else np.array(enemy_avail)
                    acts     = enemy_cpu_actions[env_idx]
                    invalid_mask = avail_np[np.arange(self.n_enemies), acts] == 0
                    if invalid_mask.any():
                        replacement = (avail_np == 1).argmax(axis=1)
                        acts[invalid_mask] = replacement[invalid_mask]
                    enemy_cpu_actions[env_idx] = acts

                enemy_actions_tensor = th.tensor(enemy_cpu_actions, dtype=th.int64, device=self.args.device).unsqueeze(1).unsqueeze(-1)
                actions_chosen.update({"enemy_actions": enemy_actions_tensor})

                if enemy_p_msgs is not None and enemy_h_msgs is not None:
                    actions_chosen.update({
                        "enemy_self_message": enemy_p_msgs.to(self.args.device).unsqueeze(1),
                        "enemy_head_message": enemy_h_msgs.to(self.args.device).unsqueeze(1)
                    })
                if enemy_head_prob_tensor is not None:
                    actions_chosen.update({
                        "enemy_head_probs": enemy_head_prob_tensor.to(self.args.device),
                        "enemy_head_actions": enemy_election_actions_tensor.to(self.args.device)
                    })

            if len(envs_not_terminated) > 0:
                try:
                    self.batch.update(actions_chosen, bs=envs_not_terminated, ts=self.t, mark_filled=False)
                except ValueError as e:
                    import traceback
                    traceback.print_exc()
                    if 'enemy_actions_tensor' in locals() and actions.shape[0] != enemy_actions_tensor.shape[0]:
                        min_envs = min(actions.shape[0], enemy_actions_tensor.shape[0])
                        adjusted_envs = envs_not_terminated[:min_envs]
                        self.batch.update(actions_chosen, bs=adjusted_envs, ts=self.t, mark_filled=False)
                    else:
                        raise

            action_idx = 0
            for idx, parent_conn in enumerate(self.parent_conns):
                if idx in envs_not_terminated and not terminated[idx]:
                    if self_play_mode and enemy_cpu_actions is not None:
                        ally_actions_full = cpu_actions[action_idx].flatten()
                        enemy_actions_full = enemy_cpu_actions[action_idx].flatten()
                        
                        actions_to_send = np.zeros(self.args.n_agents + self.args.n_enemies, dtype=int)
                        actions_to_send[:len(ally_actions_full)] = ally_actions_full
                        actions_to_send[self.args.n_agents:self.args.n_agents + len(enemy_actions_full)] = enemy_actions_full
                        
                        parent_conn.send(("step", actions_to_send.tolist()))
                    else:
                        ally_actions = cpu_actions[action_idx].flatten()
                        actions_to_send = np.zeros(self.args.n_agents, dtype=int)
                        actions_to_send[:len(ally_actions)] = ally_actions
                        parent_conn.send(("step", actions_to_send.tolist()))
                    action_idx += 1

            post_transition_data = {
                "reward": [],
                "terminated": []
            }
            if self.args.entity_scheme:
                pre_transition_data = {
                    "entities": [],
                    "obs_mask": [],
                    "entity_mask": [],
                    "avail_actions": []
                }
                
                if self_play_mode:
                    pre_transition_data["enemy_avail_actions"] = []
            else:
                pre_transition_data = {
                    "state": [],
                    "avail_actions": [],
                    "obs": []
                }

            envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
            all_terminated = all(terminated)
            if all_terminated:
                break

            active_envs = [idx for idx in range(self.batch_size) if not terminated[idx]]
            received_data = {}
            
            if len(active_envs) > 0:
                for idx in active_envs:
                    received_data[idx] = self.parent_conns[idx].recv()
            
            for idx in range(self.batch_size):
                if not terminated[idx]:
                    data = received_data[idx]
                    post_transition_data["reward"].append((data["reward"],))

                    episode_returns[idx] += data["reward"]
                    episode_lengths[idx] += 1
                    if not test_mode:
                        self.env_steps_this_run += 1

                    env_terminated = False
                    if data["terminated"]:
                        final_env_infos.append(data["info"])
                    if data["terminated"] and not data["info"].get("episode_limit", False):
                        env_terminated = True
                    terminated[idx] = env_terminated
                    post_transition_data["terminated"].append((env_terminated,))

                    if self.args.entity_scheme:
                        ent = data["entities"]
                        if hasattr(ent, "cpu"): ent = ent.cpu().numpy()
                        pre_transition_data["entities"].append(ent)
                        obs_mask = data["obs_mask"]
                        if hasattr(obs_mask, "cpu"): obs_mask = obs_mask.cpu().numpy()
                        pre_transition_data["obs_mask"].append(obs_mask)
                        entity_mask = data["entity_mask"]
                        if hasattr(entity_mask, "cpu"): entity_mask = entity_mask.cpu().numpy()
                        pre_transition_data["entity_mask"].append(entity_mask)
                        
                        if self_play_mode:
                            all_avail_actions = data["avail_actions"]
                            n_agents = self.args.n_agents
                            n_enemies = self.args.n_enemies
                            
                            ally_avail_actions = all_avail_actions[:n_agents]
                            enemy_avail_actions = all_avail_actions[n_agents:n_agents+n_enemies]
                            
                            pre_transition_data["avail_actions"].append(ally_avail_actions)
                            pre_transition_data["enemy_avail_actions"].append(enemy_avail_actions)
                        else:
                            avail = data["avail_actions"]
                            if hasattr(avail, "cpu"): avail = avail.cpu().numpy()
                            pre_transition_data["avail_actions"].append(avail)
                        
                        if "gt_mask" in data:
                            if "gt_mask" not in pre_transition_data:
                                pre_transition_data["gt_mask"] = []
                            gt_mask = data["gt_mask"]
                            if hasattr(gt_mask, "cpu"): gt_mask = gt_mask.cpu().numpy()
                            pre_transition_data["gt_mask"].append(gt_mask)
                    else:
                        state = data["state"]
                        if hasattr(state, "cpu"): state = state.cpu().numpy()
                        pre_transition_data["state"].append(state)
                        avail = data["avail_actions"]
                        if hasattr(avail, "cpu"): avail = avail.cpu().numpy()
                        pre_transition_data["avail_actions"].append(avail)
                        obs = data["obs"]
                        if hasattr(obs, "cpu"): obs = obs.cpu().numpy()
                        pre_transition_data["obs"].append(obs)
                else:
                    post_transition_data["reward"].append((0.0,))
                    post_transition_data["terminated"].append((True,))
                    if self.args.entity_scheme:
                        pre_transition_data["entities"].append(np.zeros_like(self.batch["entities"][0, self.t].cpu().numpy()))
                        pre_transition_data["obs_mask"].append(np.ones_like(self.batch["obs_mask"][0, self.t].cpu().numpy()))
                        pre_transition_data["entity_mask"].append(np.ones_like(self.batch["entity_mask"][0, self.t].cpu().numpy()))
                        pre_transition_data["avail_actions"].append(np.zeros_like(self.batch["avail_actions"][0, self.t].cpu().numpy()))
                        if self_play_mode:
                            pre_transition_data["enemy_avail_actions"].append(np.zeros_like(self.batch["enemy_avail_actions"][0, self.t].cpu().numpy()))
                    else:
                        pre_transition_data["state"].append(np.zeros_like(self.batch["state"][0, self.t].cpu().numpy()))
                        pre_transition_data["avail_actions"].append(np.zeros_like(self.batch["avail_actions"][0, self.t].cpu().numpy()))
                        pre_transition_data["obs"].append(np.zeros_like(self.batch["obs"][0, self.t].cpu().numpy()))

            self.batch.update(post_transition_data, bs=range(self.batch_size), ts=self.t, mark_filled=False)

            self.t += 1

            self.batch.update(pre_transition_data, bs=range(self.batch_size), ts=self.t, mark_filled=True)

            self._handle_adversarial_visualization()
            if self_play_mode and self.t == 0:
                self.enemy_mac.init_hidden(batch_size=self.batch_size)

        self._handle_episode_end_visualization()

        if not test_mode:
            self.t_env += self.env_steps_this_run

        cur_stats   = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix  = "test_" if test_mode else ""

        env_stats = None
        if 'sc2' in self.args.env:
            for parent_conn in self.parent_conns:
                parent_conn.send(("get_stats", None))

            env_stats = [parent_conn.recv() for parent_conn in self.parent_conns]
        
        self.debug_episode_done = True

        infos = [cur_stats] + final_env_infos
        cur_stats.update({k: sum(d.get(k, 0) for d in infos) for k in set.union(*[set(d) for d in infos])})
        cur_stats["n_episodes"] = self.batch_size + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = sum(episode_lengths) + cur_stats.get("ep_length", 0)
        if self.args.entity_scheme:
            vis, vis_b10, vis_p10  = self.calc_visibility(self.batch["obs_mask"], self.batch["entity_mask"], entities = self.batch["entities"])
            cur_stats["visibility"] = sum(vis) + cur_stats.get("visibility", 0)
            cur_stats["visibility_b10"] = sum(vis_b10) + cur_stats.get("visibility_b10", 0)
            cur_stats["visibility_p10"] = sum(vis_p10) + cur_stats.get("visibility_p10", 0)
        


        cur_returns.extend(episode_returns)

        n_test_runs = max(1, self.args.test_nepisode // self.batch_size) * self.batch_size
        if test_mode and (len(self.test_returns) == n_test_runs):
            self.rm = self._log(cur_returns, cur_stats, log_prefix, env_stats)
        elif not test_mode and self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix, env_stats)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            if 'sc2' in self.args.env:
                self.logger.log_stat("forced_restarts",
                                     sum(es['restarts'] for es in env_stats),
                                     self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix, env_stats=None):
        rm = np.mean(returns)
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()
        for k, v in stats.items():
            if k == "n_episodes":
                continue
            if k.startswith("battle") or k.startswith("battles_"):
                continue
            self.logger.log_stat(prefix + k + "_mean", v / stats["n_episodes"], self.t_env)
        wins  = stats.get("battles_won",  stats.get("battle_won",  0))
        loses = stats.get("battles_lost", stats.get("battle_lost", 0))
        draws = stats.get("battles_draw", stats.get("battle_draw", 0))
        games = stats.get("battles_game", stats.get("battle_game", wins + loses + draws))
        if games == 0:
            games = wins + loses + draws
        if games > 0:
            self.logger.log_stat(prefix + "win_rate_mean",  wins  / games, self.t_env)
            self.logger.log_stat(prefix + "draw_rate_mean", draws / games, self.t_env)
            self.logger.log_stat(prefix + "loss_rate_mean", loses / games, self.t_env)

        u_wins  = stats.get("unit_battles_won",  stats.get("unit_battle_won",  0))
        u_loses = stats.get("unit_battles_lost", stats.get("unit_battle_lost", 0))
        u_draws = stats.get("unit_battles_draw", stats.get("unit_battle_draw", 0))
        u_games_raw = stats.get("unit_battles_game", stats.get("unit_battle_game", 0))
        u_games_sum = u_wins + u_loses + u_draws
        u_games = max(u_games_raw, u_games_sum)
        if u_games == 0:
            u_games = u_games_sum
        if u_games > 0:
            self.logger.log_stat(prefix + "unit_battle_won_mean",  u_wins  / u_games, self.t_env)
            self.logger.log_stat(prefix + "unit_battle_draw_mean", u_draws / u_games, self.t_env)
            self.logger.log_stat(prefix + "unit_battle_lost_mean", u_loses / u_games, self.t_env)

        if env_stats is not None:
            total_u_wins  = sum(es.get("unit_battles_won", 0)  for es in env_stats)
            total_u_draws = sum(es.get("unit_battles_draw", 0) for es in env_stats)
            total_u_loses = sum(es.get("unit_battles_lost", 0) for es in env_stats)
            total_u_games = sum(es.get("unit_battles_game", 0) for es in env_stats)

            if total_u_games == 0:
                total_u_games = total_u_wins + total_u_draws + total_u_loses

            if total_u_games > 0:
                self.logger.log_stat(prefix + "unit_win_rate_mean",  total_u_wins  / total_u_games, self.t_env)
                self.logger.log_stat(prefix + "unit_draw_rate_mean", total_u_draws / total_u_games, self.t_env)
                self.logger.log_stat(prefix + "unit_loss_rate_mean", total_u_loses / total_u_games, self.t_env)

        stats.clear()
        return rm

    def calc_visibility(self, obs_mask, agent_mask, entities=None):
        health_ind = {"3-8sz_symmetric": 33, "3-8MMM_symmetric": 39, "3-8csz_symmetric": 31, "3m_fixed": 19, "3-8m_symmetric": 43}

        if 'sc2custom' in self.args.env:
            ind = health_ind[self.args.scenario]
            agent_mask = (entities[:,:,:,ind] > 0).float()
        else:
            agent_mask = 1-agent_mask
        obs_mask = 1-obs_mask
        obs_mask = obs_mask.masked_fill((1-agent_mask).bool().unsqueeze(-1),0)
        agent_num = agent_mask[:, :, :self.n_agents].sum(2)
        entity_num = agent_mask.sum(2)
        invalid_frame = th.logical_or((agent_num == 0), (entity_num == 0))
        seen_num=obs_mask.sum(3)
        vis_percent = seen_num[:,:,:self.n_agents].sum(2)/agent_num/entity_num
        vis_percent = vis_percent.masked_fill(invalid_frame, 0.0)
        t_length = th.logical_not(invalid_frame).sum(1)
        t_length_safe = t_length.clone().float()
        t_length_safe[t_length_safe == 0] = 1.0
        visibility = (vis_percent.sum(1)/t_length_safe).detach().cpu().numpy()
        t_length0 = th.logical_not(invalid_frame)[:,:10].sum(1)
        t_length1 = th.logical_not(invalid_frame)[:,10:].sum(1)
        t_length0_safe = t_length0.clone().float(); t_length0_safe[t_length0_safe==0] = 1.0
        t_length1_safe = t_length1.clone().float(); t_length1_safe[t_length1_safe==0] = 1.0
        visibility0 = (vis_percent[:,:10].sum(1)/t_length0_safe).detach().cpu().numpy()
        visibility1 = (vis_percent[:,10:].sum(1)/t_length1_safe).detach().cpu().numpy()
        return visibility, visibility0, visibility1

    def _handle_adversarial_visualization(self):
        try:
            adversarial_config = getattr(self.args, 'adversarial_influence', {})
            if not adversarial_config.get('enable', False):
                return
            
            visualization_config = adversarial_config.get('visualization', {})
            if not visualization_config.get('enable_heatmap', False):
                return
            
            from controllers.global_map_store import get_visualizer, get_graph_manager, get_agent_positions, get_visible_enemies, get_map_matrix
            visualizer = get_visualizer()
            
            if visualizer is None:
                return
            
            manager = get_graph_manager()
            if manager is None:
                return
            if not hasattr(manager, 'adversarial_influence'):
                return
            
            influence_map = manager.adversarial_influence
            if influence_map is None:
                return
            
            cost_map = influence_map.get_final_cost_map()
            if cost_map is None:
                return
            
            episode_id = self.current_episode_id
            step = self.t
            
            agent_positions_dict = get_agent_positions(0)
            ally_positions = []
            if agent_positions_dict:
                ally_positions = list(agent_positions_dict.values())
            
            all_enemy_positions_dict = {}
            all_enemy_positions = []
            
            visible_enemy_positions_dict = {}
            visible_enemy_positions = []
            
            try:
                self.parent_conns[0].send(("get_all_enemy_positions", None))
                all_enemy_data = self.parent_conns[0].recv()
                if all_enemy_data:
                    all_enemy_positions_dict = all_enemy_data
                    all_enemy_positions = list(all_enemy_data.values())
                
                self.parent_conns[0].send(("get_enemy_positions", None))
                visible_enemy_data = self.parent_conns[0].recv()
                if visible_enemy_data:
                    visible_enemy_positions_dict = visible_enemy_data
                    visible_enemy_positions = list(visible_enemy_data.values())
                    
            except Exception as e:
                print(f"  Failed to get enemy positions: {e}")
            
            map_matrix = get_map_matrix(0)
            
            visual_observations = {
                'enemy_positions': visible_enemy_positions,
                'ally_positions': ally_positions
            }
            
            agent_positions_for_reachability = {}
            for i, ally_pos in enumerate(ally_positions):
                agent_positions_for_reachability[f'agent_{i}'] = ally_pos
            
            try:
                if map_matrix is not None and hasattr(influence_map, 'set_obstacle_map'):
                    influence_map.set_obstacle_map(map_matrix)
                
                updated_cost_map, _ = influence_map.process_step_observations(
                    visual_observations, agent_positions_for_reachability
                )
                cost_map = updated_cost_map
            except Exception as e:
                print(f"  Failed to force update influence map: {e}")
            
            obstacle_positions = []
            if map_matrix is not None:
                obstacle_coords = np.where(map_matrix == -1)
                if len(obstacle_coords[0]) > 0:
                    obstacle_positions = list(zip(obstacle_coords[0].tolist(), obstacle_coords[1].tolist()))
            
            group_info = None
            try:
                if hasattr(self.mac, 'decide_group_results'):
                    group_info = getattr(self.mac, 'decide_group_results', None)
            except Exception as e:
                print(f"[ERROR] Failed to get group info: {e}")
                pass
            
            visualizer.add_step_data(
                episode_id=episode_id,
                step=step,
                cost_map=cost_map,
                ally_positions=ally_positions,
                enemy_positions=all_enemy_positions,
                obstacle_positions=obstacle_positions,
                group_info=group_info
            )
            
        except Exception as e:
            pass

    def _handle_episode_end_visualization(self):
        try:
            adversarial_config = getattr(self.args, 'adversarial_influence', {})
            if not adversarial_config.get('enable', False):
                return
            
            visualization_config = adversarial_config.get('visualization', {})
            if not visualization_config.get('enable_heatmap', False):
                return
            
            from controllers.global_map_store import get_visualizer, finalize_episode_visualization
            visualizer = get_visualizer()
            
            if visualizer is not None:
                episode_id = self.current_episode_id
                finalize_episode_visualization(episode_id)
                
                self.current_episode_id += 1
                
        except Exception as e:
            print(f"[WARNING] Failed to handle episode end visualization: {e}")


def env_worker(remote, entity_scheme, env_fn):
    env = env_fn.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            actions = data
            reward, terminated, env_info = env.step(actions)
            send_dict = {
                "reward": reward,
                "terminated": terminated,
                "info": env_info
            }
            
            avail_actions = env.get_avail_actions()
            
            if entity_scheme:
                masks = env.get_masks()
                if len(masks) == 2:
                    obs_mask, entity_mask = masks
                    gt_mask = None
                else:
                    obs_mask, entity_mask, gt_mask = masks
                send_dict["obs_mask"] = obs_mask
                send_dict["entity_mask"] = entity_mask
                if gt_mask is not None:
                    send_dict["gt_mask"] = gt_mask
                    
                entities = env.get_entities()
                send_dict["entities"] = entities
            else:
                send_dict["state"] = env.get_state()
                send_dict["obs"] = env.get_obs()
                
            send_dict["avail_actions"] = avail_actions
                
            remote.send(send_dict)
        elif cmd == "get_debug_info":
            debug_dict = {
                "entities": env.get_entities()
            }
            remote.send(debug_dict)
        elif cmd == "reset":
            env.reset(**data)
            
            self_play_mode = hasattr(env, 'self_play') and env.self_play
            
            if entity_scheme:
                masks = env.get_masks()
                if len(masks) == 2:
                    obs_mask, entity_mask = masks
                    gt_mask = None
                else:
                    obs_mask, entity_mask, gt_mask = masks
                    
                avail_actions = env.get_avail_actions()
                
                send_dict = {
                    "entities": env.get_entities(),
                    "avail_actions": avail_actions,
                    "obs_mask": obs_mask,
                    "entity_mask": entity_mask
                }
                if gt_mask is not None:
                    send_dict["gt_mask"] = gt_mask
                remote.send(send_dict)
            else:
                remote.send({
                    "state": env.get_state(),
                    "avail_actions": env.get_avail_actions(),
                    "obs": env.get_obs()
                })
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_env_info":
            remote.send(env.get_env_info(data))
        elif cmd == "get_stats":
            remote.send(env.get_stats())
        elif cmd == "get_enemy_avail":
            if hasattr(env, "self_play") and env.self_play:
                enemy_avail = []
                for enemy_id in range(env.n_enemies):
                    avail = env.get_avail_enemy_actions(enemy_id)
                    enemy_avail.append(avail)
                remote.send(enemy_avail)
            else:
                remote.send([])
        elif cmd == "get_enemy_positions":
            if hasattr(env, 'get_visible_enemy_positions'):
                enemy_positions = env.get_visible_enemy_positions()
                remote.send(enemy_positions)
            else:
                remote.send({})
        elif cmd == "get_all_enemy_positions":
            if hasattr(env, 'get_all_enemy_positions'):
                all_enemy_positions = env.get_all_enemy_positions()
                remote.send(all_enemy_positions)
            else:
                remote.send({})
        elif cmd == "get_enemy_health":
            try:
                if hasattr(env, 'enemies'):
                    enemy_health = {e_id: (e_unit.pos.x, e_unit.pos.y, e_unit.health) 
                                  for e_id, e_unit in env.enemies.items()}
                    remote.send(enemy_health)
                else:
                    remote.send({})
            except Exception as e:
                remote.send({})
        else:
            raise NotImplementedError


class CloudpickleWrapper():
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

