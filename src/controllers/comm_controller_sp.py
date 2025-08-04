from .entity_controller import EntityMAC
import torch as th
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from modules.layers.comm_mixer import AverageMessageEncoder
from .global_map_store import get_map_matrix, get_agent_positions, get_all_map_matrices, get_all_agent_positions
import numpy as np
from collections import deque

MAP_MATRIX = None

class CommMACSP(EntityMAC):
    def __init__(self, scheme, groups, args):
        assert args.use_msg
        super(CommMACSP, self).__init__(scheme, groups, args)
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
    
    def decide_group(self, agent_inputs, avail_actions, test_mode=False):
        if self.args.use_comm_sr:
            agent_vis_mask = self.gt_mask[:,:, :self.n_agents, :self.n_agents]
        else:
            entity, obs_mask, entity_mask = agent_inputs
            agent_vis_mask = obs_mask[:,:, :self.n_agents, :self.n_agents]

        if getattr(self.args, 'use_euclid_group', False):
            sight_range = getattr(self.args, 'sight_range', 9)
            bs = avail_actions.shape[0]
            for b_idx in range(bs):
                positions = get_agent_positions(b_idx)
                if not positions:
                    continue
                for i, pos_i in positions.items():
                    if i >= self.n_agents:
                        continue
                    for j, pos_j in positions.items():
                        if j >= self.n_agents or i==j:
                            continue
                        dist = ((pos_i[0]-pos_j[0])**2 + (pos_i[1]-pos_j[1])**2) ** 0.5
                        vis = dist <= sight_range
                        agent_vis_mask[b_idx, 0, i, j] = 0 if vis else 1

        header = self.decide_header(avail_actions, test_mode=test_mode, agent_vis_mask=agent_vis_mask)
        bs, n = header.shape
        control_message = header.unsqueeze(1).unsqueeze(3).repeat(1,1,1,n) * th.logical_not(agent_vis_mask)
        control_message *= th.logical_not(header.unsqueeze(1).unsqueeze(2).repeat(1,1,n,1))
        if getattr(self.args, 'random_master', False):
            ind_lst = []
            for i in range(n):
                random_ind = th.randperm(n, device=self.args.device)
                ind_lst.append(F.one_hot(random_ind[th.max(control_message[:,:,random_ind,i], dim=2)[1]], num_classes=n))
            ind = th.cat(ind_lst, dim=1).permute(0,2,1).unsqueeze(1)
        else:
            ind = F.one_hot(th.max(control_message, dim=2)[1], n).permute(0,1,3,2)
        control_message *= ind.bool()

        if getattr(self.args, 'use_balanced_group', False):
            alive_agent = th.logical_not(avail_actions[:,0,:,0])
            control_message = self._balance_groups(control_message, header, alive_agent)

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
        if "enemy_avail_actions" in ep_batch.scheme:
            avail_actions = ep_batch["enemy_avail_actions"][:, t]
        else:
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
        if "enemy_avail_actions" in ep_batch.scheme:
            avail_actions = ep_batch["enemy_avail_actions"][:, t_ep]
            side = 'ENEMY'
        else:
            avail_actions = ep_batch["avail_actions"][:, t_ep]
            side = 'ALLY'
        agent_outputs, *_ = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs, avail_actions, t_env, test_mode=test_mode)
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

    def _balance_groups(self, control_message, header, alive_agent):
        bs = control_message.shape[0]
        balanced_control_message = th.zeros_like(control_message)

        for b in range(bs):
            leaders = [i for i in range(self.n_agents) if header[b, i]]
            if not leaders:
                continue

            followers = [j for j in range(self.n_agents) if (not header[b, j]) and alive_agent[b, j]]

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
                    available = [f for f in remaining_followers if f in visible_followers[leader]]
                    if available and len(groups[leader]) < min_group_size:
                        min_group_size = len(groups[leader])
                        min_leader = leader

                if min_leader is None:
                    break

                for follower in remaining_followers:
                    if follower in visible_followers[min_leader]:
                        groups[min_leader].append(follower)
                        remaining_followers.remove(follower)
                        break

            for leader, flw in groups.items():
                for follower in flw:
                    balanced_control_message[b, 0, leader, follower] = True

        return balanced_control_message
