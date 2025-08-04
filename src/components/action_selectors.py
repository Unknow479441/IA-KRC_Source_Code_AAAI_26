import numpy as np
import torch as th
from torch.autograd import Variable
from torch.distributions import Categorical
from torch.nn.functional import softmax
from .epsilon_schedules import DecayThenFlatSchedule

REGISTRY = {}

class MultinomialActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)
        self.test_greedy = getattr(args, "test_greedy", True)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        masked_policies = agent_inputs.clone()
        masked_policies[avail_actions == 0.0] = 0.0

        self.epsilon = self.schedule.eval(t_env)

        if test_mode and self.test_greedy:
            picked_actions = masked_policies.max(dim=2)[1]
        else:
            picked_actions = Categorical(masked_policies).sample().long()

        return picked_actions

REGISTRY["multinomial"] = MultinomialActionSelector


class EpsilonGreedyActionSelector():

    def __init__(self, args):
        self.args = args
        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time, decay="linear")
        if getattr(self.args, 'two_phase_decay', False):
            self.schedule1 = DecayThenFlatSchedule(args.epsilon1_start, args.epsilon1_finish, args.epsilon1_anneal_time, decay="linear")
        self.epsilon = self.schedule.eval(0)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        self.epsilon = self.schedule.eval(t_env)
        if getattr(self.args, 'two_phase_decay', False):
            self.epsilon = max(self.epsilon, self.schedule1.eval(t_env))

        if test_mode:
            self.epsilon = 0.0

        masked_q_values = agent_inputs.clone()
        masked_q_values[avail_actions == 0.0] = -float("inf")

        random_numbers = th.rand_like(agent_inputs[:,:,0])
        pick_random = (random_numbers < self.epsilon).long()

        random_actions = th.zeros_like(agent_inputs[:,:,0], dtype=th.long)

        for b in range(avail_actions.shape[0]):
            for a in range(avail_actions.shape[1]):
                curr_avail_actions = avail_actions[b, a]

                if curr_avail_actions.sum() > 0:
                    avail_indices = curr_avail_actions.nonzero(as_tuple=False).squeeze(-1)
                    if len(avail_indices) > 0:
                        random_idx = th.randint(0, len(avail_indices), (1,))
                        random_actions[b, a] = avail_indices[random_idx]
                    else:
                        random_actions[b, a] = 0
                else:
                    random_actions[b, a] = 0

        greedy_actions = masked_q_values.max(dim=2)[1]

        picked_actions = pick_random * random_actions + (1 - pick_random) * greedy_actions

        for b in range(avail_actions.shape[0]):
            for a in range(avail_actions.shape[1]):
                action_idx = picked_actions[b, a].item()
                if action_idx >= avail_actions.shape[2] or avail_actions[b, a, action_idx] == 0:
                    avail_indices = avail_actions[b, a].nonzero(as_tuple=False).squeeze(-1)
                    if len(avail_indices) > 0:
                        picked_actions[b, a] = avail_indices[0]
                    else:
                        picked_actions[b, a] = 0
                        
        return picked_actions

REGISTRY["epsilon_greedy"] = EpsilonGreedyActionSelector
