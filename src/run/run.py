import datetime
from functools import partial
from math import ceil
from re import A
import imageio
import os
import os.path as osp
import numpy as np
import pickle
import pprint
import time
import json
import threading
from copy import deepcopy
import torch as th
from numpy.random import RandomState
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath, basename, join, splitext

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from envs import s_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot


def run(_run, _config, _log):
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    exploration_threshold = getattr(args, "exploration", {}).get("threshold", 0.95)
    sight_range = getattr(args, "exploration", {}).get("sight_range", 9)
    _log.info(f"Exploration parameters: threshold={exploration_threshold}, sight_range={sight_range}")

    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f"))
    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(dirname(abspath(__file__)))), "results", args.tb_dirname)
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    logger.setup_sacred(_run)

    if args.evaluate_multi_model:
        model_dir = deepcopy(args.checkpoint_path.split(","))
        sight_ranges = deepcopy(args.env_args["sight_range_kind"])
        args.evaluate=True
        rm_dic={}
        sc2_sr_kind_dict = {0:3, 1:3, 2:9, 3:9}
        sc2_group_kind_dict = {0:True, 1:False, 2:True, 3:False}
        for sr in sight_ranges:
            rmm = []
            for md in model_dir:
                args.checkpoint_path = osp.join(args.checkpoint_prefix, md)
                if "sc2" in args.env:
                    args.env_args["sight_range"] = sc2_sr_kind_dict[sr]
                    args.env_args["divide_group"] = sc2_group_kind_dict[sr]
                else:
                    args.env_args["sight_range_kind"] = sr
                rm = run_sequential(args=args, logger=logger)
                print("model:", md, ", sight range:", sr, ", return mean:", rm)
                rm_dic.update({(md, sr):rm})
                rmm.append(rm)
                print("Exiting Main")

                print("Stopping all threads")
                for t in threading.enumerate():
                    if t.name != "MainThread":
                        print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
                        t.join(timeout=1)
                        print("Thread joined")
            rm_dic.update({('Average', sr):np.mean(rmm)})
        for (md, sr), rm in rm_dic.items():
            print("model:", md, ", sight range:", sr, ", return mean:", rm)
    else:
        run_sequential(args=args, logger=logger)
        print("Exiting Main")

        print("Stopping all threads")
        for t in threading.enumerate():
            if t.name != "MainThread":
                print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
                t.join(timeout=1)
                print("Thread joined")

    print("Exiting script")

    os._exit(os.EX_OK)


def evaluate_sequential(args, runner, logger, load_time_step=0):
    vw = None
    if args.video_path is not None:
        os.makedirs(dirname(args.video_path), exist_ok=True)
        vid_basename_split = splitext(basename(args.video_path))
        if vid_basename_split[1] == '.mp4':
            vid_basename = ''.join(vid_basename_split)
        else:
            vid_basename = ''.join(vid_basename_split) + '.mp4'
        vid_filename = join(dirname(args.video_path), vid_basename)
        vw = imageio.get_writer(vid_filename, format='FFMPEG', mode='I',
                                fps=12, codec='h264', quality=10)

    if args.eval_path is not None:
        os.makedirs(dirname(args.eval_path), exist_ok=True)
        eval_basename_split = splitext(basename(args.eval_path))
        if eval_basename_split[1] == '.json':
            eval_basename = ''.join(eval_basename_split)
        else:
            eval_basename = ''.join(eval_basename_split) + '.json'
        eval_basename = eval_basename[:-5]+"_"+args.unique_token+".json"
        eval_filename = join(dirname(args.eval_path), eval_basename)

    res_dict = {}

    if args.eval_all_scen:
        if 'sc2' in args.env:
            dict_key = 'scenarios'
            n_scen = len(args.env_args['scenario_dict'][dict_key])
        elif 'particle' in args.env:
            assert args.env_args["sr_list"] is not None
            n_scen = len(args.env_args["sr_list"])
        else:
            raise Exception("Environment (%s) does not incorporate multiple scenarios")
    else:
        n_scen = 1
    n_test_batches = max(1, args.test_nepisode // runner.batch_size)
    can_close_env=True
    for i in range(n_scen):
        run_args = {'test_mode': True, 'vid_writer': vw,
                    'test_scen': True}
        if args.eval_all_scen:
            run_args['index'] = i
        for j in range(n_test_batches):
            batch = runner.run(**run_args)
            if args.save_entities_and_attn_weights:
                can_close_env = True                  
                dic = {'entities':batch['entities'].detach().cpu().numpy(),
                        'obs_mask':batch['obs_mask'].detach().cpu().numpy(),
                        'entity_mask':batch['entity_mask'].detach().cpu().numpy(),
                        'attn_weights':runner.mac.agent.attn_weights.detach().cpu().numpy()}
                if args.save_global_attn:
                    dic.update({
                        'attn_w1':runner.mac.mixer.hyper_w_1.attn_weights.detach().cpu().numpy(),
                        'attn_b1':runner.mac.mixer.hyper_b_1.attn_weights.detach().cpu().numpy(),
                        'attn_wf':runner.mac.mixer.hyper_w_final.attn_weights.detach().cpu().numpy(),
                        'attn_v':runner.mac.mixer.V.attn_weights.detach().cpu().numpy(),
                    })
                
                token = osp.split(args.checkpoint_path)[-1]
                save_path = os.path.join(args.local_results_path, "attn_weights", token, str(load_time_step))
                os.makedirs(save_path, exist_ok=True)
                file_name = osp.join(save_path, str(j)+".pkl")
                with open(file_name, "wb") as f:
                    pickle.dump(dic, f)
                logger.console_logger.info("Saving attn_weights to {}".format(file_name))
        rm = runner.rm
        curr_stats = dict((k, v[-1][1]) for k, v in logger.stats.items())
        if args.eval_all_scen:
            if 'particle' in args.env:
                curr_scen = args.env_args["sr_list"][i]
            else:
                curr_scen = args.env_args['scenario_dict'][dict_key][i]
            if 'sc2' in args.env:
                scen_str = "-".join("%i%s" % (count, name[:3]) for count, name in sorted(curr_scen[0], key=lambda x: x[1]))
            elif 'particle' in args.env:
                scen_str = 'sr_kind:'+str(i)
            else:
                scen_str = "".join(curr_scen[0])
            res_dict[scen_str] = curr_stats
        else:
            res_dict.update(curr_stats)

    if vw is not None:
        vw.close()

    if args.eval_path is not None:
        with open(eval_filename, 'w') as f:
            json.dump(res_dict, f)

    if args.save_replay:
        runner.save_replay()
    if can_close_env:
        runner.close_env()
    logger.print_stats_summary()
    return rm 


def run_sequential(args, logger):
    if 'entity_scheme' in args.env_args:
        args.entity_scheme = args.env_args['entity_scheme']
    else:
        args.entity_scheme = False

    if ('sc2custom' in args.env):
        rs = RandomState(0)
        from envs.starcraft2.custom_scenarios import symmetric_armies
        if args.scenario == "3-8m_symmetric" and 'fixed_m' in args.env_args:
            n_fixed = int(args.env_args['fixed_m'])
            args.env_args['scenario_dict'] = symmetric_armies(
                [(('Marine',), (n_fixed, n_fixed))],
                rotate=True, ally_centered=False,
                separation=14, jitter=1, episode_limit=150, map_name="empty_passive",
                rs=rs)
            args.env_args.pop('fixed_m')
            args.train_map_num = [n_fixed]
            args.test_map_num = [n_fixed]
        else:
            args.env_args['scenario_dict'] = s_REGISTRY[args.scenario](rs=rs)
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    if not args.entity_scheme:
        args.state_shape = env_info["state_shape"]
        scheme = {
            "state": {"vshape": env_info["state_shape"]},
            "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
            "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
            "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
            "reward": {"vshape": (1,)},
            "terminated": {"vshape": (1,), "dtype": th.uint8},
        }
        groups = {
            "agents": args.n_agents
        }
        if 'masks' in env_info:
            args.obs_masks, args.state_masks = env_info['masks']
        if 'unit_dim' in env_info:
            args.unit_dim = env_info['unit_dim']
    else:
        args.entity_shape = env_info["entity_shape"]
        args.n_entities = env_info["n_entities"]
        args.gt_mask_avail = env_info.get("gt_mask_avail", False)
        
        if getattr(args, "self_play", False):
            args.n_enemies = env_info.get("n_enemies", args.n_agents)
            
            groups = {
                "agents": args.n_agents,
                "enemy_agents": args.n_enemies,
                "entities": args.n_entities
            }
            
            scheme = {
                "entities": {"vshape": env_info["entity_shape"], "group": "entities"},
                "obs_mask": {"vshape": env_info["n_entities"], "group": "entities", "dtype": th.uint8},
                "entity_mask": {"vshape": env_info["n_entities"], "dtype": th.uint8},
                
                "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
                "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
                
                "enemy_actions": {"vshape": (1,), "group": "enemy_agents", "dtype": th.long},
                "enemy_avail_actions": {"vshape": (env_info["n_actions"],), "group": "enemy_agents", "dtype": th.int},
                
                "reward": {"vshape": (1,)},
                "terminated": {"vshape": (1,), "dtype": th.uint8},
            }
            
            if args.use_msg:
                if args.no_summary:
                    args.msg_dim = args.attn_embed_dim
                scheme["self_message"] = {"vshape": (args.msg_dim,), "group": "agents"}
                scheme["head_message"] = {"vshape": (args.msg_dim,), "group": "agents"}
                scheme["enemy_self_message"] = {"vshape": (args.msg_dim,), "group": "enemy_agents"}
                scheme["enemy_head_message"] = {"vshape": (args.msg_dim,), "group": "enemy_agents"}
            
            if args.gt_mask_avail:
                scheme["gt_mask"] = {"vshape": env_info["n_entities"], "group": "agents", "dtype": th.uint8}
                scheme["enemy_gt_mask"] = {"vshape": env_info["n_entities"], "group": "enemy_agents", "dtype": th.uint8}
            elif args.env == "particle" and args.env_args["scenario_id"] == "resource_collection.py" and args.env_args["comm_sr"] is not None:
                scheme["gt_mask"] = {"vshape": env_info["n_entities"], "group": "entities", "dtype": th.uint8}
        else:
            groups = {
                "agents": args.n_agents,
                "entities": args.n_entities
            }
            
            scheme = {
                "entities": {"vshape": env_info["entity_shape"], "group": "entities"},
                "obs_mask": {"vshape": env_info["n_entities"], "group": "entities", "dtype": th.uint8},
                "entity_mask": {"vshape": env_info["n_entities"], "dtype": th.uint8},
                "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
                "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
                "reward": {"vshape": (1,)},
                "terminated": {"vshape": (1,), "dtype": th.uint8},
            }
            
            if args.gt_mask_avail:
                scheme["gt_mask"] = {"vshape": env_info["n_entities"], "group": "agents", "dtype": th.uint8}
            elif args.env == "particle" and args.env_args["scenario_id"] == "resource_collection.py" and args.env_args["comm_sr"] is not None:
                scheme["gt_mask"] = {"vshape": env_info["n_entities"], "group": "entities", "dtype": th.uint8}
            if args.use_msg:
                if args.no_summary:
                    args.msg_dim=args.attn_embed_dim
                scheme["self_message"] = {"vshape":(args.msg_dim,), "group": "agents"}
                scheme["head_message"] = {"vshape":(args.msg_dim,), "group": "agents"}

    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    if getattr(args, "self_play", False):
        args.n_enemies = env_info.get("n_enemies", args.n_agents)

        if "enemy_agents" not in groups:
            groups["enemy_agents"] = args.n_enemies

        scheme.update({
            "enemy_actions":       {"vshape": (1,), "group": "enemy_agents", "dtype": th.long},
            "enemy_avail_actions": {"vshape": (env_info["n_actions"],), "group": "enemy_agents", "dtype": th.int},
        })

        if args.use_msg:
            if args.no_summary:
                args.msg_dim = args.attn_embed_dim
            scheme.update({
                "enemy_self_message": {"vshape": (args.msg_dim,), "group": "enemy_agents"},
                "enemy_head_message": {"vshape": (args.msg_dim,), "group": "enemy_agents"},
            })

        preprocess["enemy_actions"] = ("enemy_actions_onehot", [OneHot(out_dim=args.n_actions)])

    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)

    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)
    
    enemy_mac = None
    if getattr(args, "self_play", False):
        args.n_enemies = env_info.get("n_enemies", args.n_agents)
        args.enemy_mac = getattr(args, "enemy_mac", "comm_mac_sp")
        
        original_agent = args.agent
        if hasattr(args, "enemy_agent") and args.enemy_agent is not None:
            args.agent = args.enemy_agent
        
        original_learner = args.learner
        if hasattr(args, "enemy_learner") and args.enemy_learner is not None:
            args.learner = args.enemy_learner
            
        try:
            enemy_mac = mac_REGISTRY[args.enemy_mac](buffer.scheme, groups, args)
        except Exception as e:
            logger.console_logger.error(f"Failed to create enemy_mac: {e}")
            enemy_mac = None
        finally:
            args.agent = original_agent
            args.learner = original_learner

    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac, enemy_mac=enemy_mac)

    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)
    enemy_learner = None
    if getattr(args, "self_play", False) and getattr(args, "train_enemy", True):
        try:
            enemy_learner = le_REGISTRY[args.learner](enemy_mac, buffer.scheme, logger, args)
            enemy_learner.tag = "enemy_"
        except Exception as e:
            enemy_learner = None

    if args.use_cuda:
        learner.cuda()
        if enemy_learner is not None:
            enemy_learner.cuda()

    if args.checkpoint_path != "":
        if type(args.load_step) == list:
            assert args.evaluate and not args.evaluate_multi_model
        else:
            args.load_step = [args.load_step]
        for load_step in args.load_step:
            timesteps = []
            timestep_to_load = 0

            if not os.path.isdir(args.checkpoint_path):
                logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
                return

            for name in os.listdir(args.checkpoint_path):
                full_name = os.path.join(args.checkpoint_path, name)
                if os.path.isdir(full_name) and name.isdigit():
                    timesteps.append(int(name))

            if load_step == 0:
                timestep_to_load = max(timesteps)
            else:
                timestep_to_load = min(timesteps, key=lambda x: abs(x - load_step))

            model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

            logger.console_logger.info("Loading model from {}".format(model_path))
            learner.load_models(model_path, evaluate=args.evaluate)
            
            if enemy_learner is not None:
                enemy_load_step = getattr(args, 'enemy_load_step', load_step)
                
                if enemy_load_step != load_step:
                    if enemy_load_step == 0:
                        enemy_timestep_to_load = max(timesteps)
                    else:
                        enemy_timestep_to_load = min(timesteps, key=lambda x: abs(x - enemy_load_step))
                else:
                    enemy_timestep_to_load = timestep_to_load
                
                enemy_model_path = os.path.join(args.checkpoint_path, "enemy_model_{}".format(enemy_timestep_to_load))
                if os.path.exists(enemy_model_path):
                    logger.console_logger.info("Loading enemy model from {} (step: {})".format(enemy_model_path, enemy_timestep_to_load))
                    enemy_learner.load_models(enemy_model_path)
                else:
                    fallback_model_path = os.path.join(args.checkpoint_path, str(enemy_timestep_to_load))
                    if os.path.exists(fallback_model_path):
                        logger.console_logger.info("No dedicated enemy model found, loading from ally model path: {} (step: {})".format(fallback_model_path, enemy_timestep_to_load))
                        enemy_learner.load_models(fallback_model_path)
                    else:
                        logger.console_logger.warning("Neither enemy model nor ally model found for step {}, using ally model step {}".format(enemy_timestep_to_load, timestep_to_load))
                    enemy_learner.load_models(model_path)
                
            runner.t_env = timestep_to_load

            if args.evaluate or args.save_replay:
                rm = evaluate_sequential(args, runner, logger, load_time_step = timestep_to_load)
                args.load_step=0
                return rm

    args.load_step=0

    episode = 0
    last_test_T = 0
    last_log_T = 0
    model_save_time = 0
    insert_buffer_num = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    while runner.t_env <= args.t_max:

        if args.runner == 'total_episode' and buffer.can_sample(runner.batch_size):
            replay_batch=buffer.sample(runner.batch_size)
            episode_batch = runner.run(test_mode=False, replay_batch=replay_batch)
        else:
            episode_batch = runner.run(test_mode=False)

        buffer.insert_episode_batch(episode_batch)
        insert_buffer_num += runner.batch_size
        if args.save_memory and insert_buffer_num % args.save_interval == 0:
            map_name = args.get('scenario', None)
            if map_name is None:
                map_name = args.env_args["map_name"]
            os.makedirs(os.path.join(args.local_results_path, "replay_memory", args.unique_token+"_"+map_name), exist_ok=True)
            save_path = os.path.join(args.local_results_path, "replay_memory", args.unique_token+"_"+map_name, "{:07d}".format(insert_buffer_num)+".pkl")
            with open(save_path, "wb") as f:
                pickle.dump(buffer, f, protocol=4)
            print("Replay saved to", save_path)


        if buffer.can_sample(args.batch_size):
            for _ in range(args.training_iters):
                episode_sample = buffer.sample(args.batch_size)

                max_ep_t = episode_sample.max_t_filled()
                episode_sample = episode_sample[:, :max_ep_t]

                if episode_sample.device != args.device:
                    episode_sample.to(args.device)

                learner.train(episode_sample, runner.t_env, episode)
                
                if enemy_learner is not None:
                    trans_data = episode_sample.data.transition_data

                    backup_dict = {}
                    for key in ["actions", "avail_actions", "self_message", "head_message", "reward"]:
                        if key in trans_data:
                            backup_dict[key] = trans_data[key].clone()
                    enemy_actions      = trans_data.get("enemy_actions", None)
                    enemy_avail        = trans_data.get("enemy_avail_actions", None)
                    enemy_self_message = trans_data.get("enemy_self_message", None)
                    enemy_head_message = trans_data.get("enemy_head_message", None)

                    if enemy_actions is not None:
                        if enemy_actions.shape != trans_data["actions"].shape:
                            bs, t, n_agent, _ = trans_data["actions"].shape
                            bs_e, t_e, n_enemy, _ = enemy_actions.shape
                            assert bs == bs_e and t == t_e, "Batch/time dim mismatch"
                            if n_enemy > n_agent:
                                enemy_actions = enemy_actions[:, :, :n_agent, :]
                            elif n_enemy < n_agent:
                                pad = th.zeros(bs, t, n_agent - n_enemy, 1, dtype=enemy_actions.dtype, device=enemy_actions.device)
                                enemy_actions = th.cat([enemy_actions, pad], dim=2)
                        trans_data["actions"] = enemy_actions.clone()
                    if enemy_avail is not None:
                        if enemy_avail.shape != trans_data["avail_actions"].shape:
                            bs, t, n_agent, a_dim = trans_data["avail_actions"].shape
                            bs_e, t_e, n_enemy, a_dim_e = enemy_avail.shape
                            assert a_dim == a_dim_e and bs == bs_e and t == t_e, "Batch/time/act dim mismatch"
                            if n_enemy > n_agent:
                                enemy_avail = enemy_avail[:, :, :n_agent, :]
                            elif n_enemy < n_agent:
                                pad = th.zeros(bs, t, n_agent - n_enemy, a_dim, dtype=enemy_avail.dtype, device=enemy_avail.device)
                                enemy_avail = th.cat([enemy_avail, pad], dim=2)
                        trans_data["avail_actions"] = enemy_avail.clone()
                    if enemy_self_message is not None and "self_message" in trans_data:
                        trans_data["self_message"] = enemy_self_message.clone().detach()
                    if enemy_head_message is not None and "head_message" in trans_data:
                        trans_data["head_message"] = enemy_head_message.clone().detach()
                    if "reward" in trans_data:
                        r = trans_data["reward"].clone()
                        r_enemy = th.where(r == -1000, r, -r)
                        trans_data["reward"] = r_enemy

                    enemy_learner.train(episode_sample, runner.t_env, episode)

                    for k, v in backup_dict.items():
                        trans_data[k] = v

                    if runner.t_env % args.log_interval == 0 and episode % 50 == 0:
                        logger.console_logger.info("Enemy agent training with same parameters")

        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

            logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()

            last_test_T = runner.t_env
            for _ in range(n_test_runs):
                if args.runner == 'total_episode':
                    replay_batch=buffer.sample(runner.batch_size)
                    runner.run(test_mode=True, replay_batch=replay_batch)
                else:
                    runner.run(test_mode=True)

        if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or
                                model_save_time == 0 or
                                runner.t_env > args.t_max):
            model_save_time = runner.t_env
            save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            learner.save_models(save_path)
            
            if enemy_learner is not None:
                enemy_save_path = os.path.join(args.local_results_path, "models", args.unique_token, "enemy_model_" + str(runner.t_env))
                os.makedirs(enemy_save_path, exist_ok=True)
                logger.console_logger.info("Saving enemy models to {}".format(enemy_save_path))
                enemy_learner.save_models(enemy_save_path)

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    runner.close_env()
    logger.console_logger.info("Finished Training")
    return None


def args_sanity_check(config, _log):

    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]

    return config


def get_custom_envs(args):
    env_id = args.env_map_id if args.env_map_id else 0
    envs = []
    if "sc2" in args.env:
        import yaml
        with open(args.env_args["scenario_dict"], 'r') as f:
            sc_dict = yaml.load(f, Loader=yaml.FullLoader)
            
        exploration_threshold = getattr(args, "exploration", {}).get("threshold", 0.95)
        sight_range = getattr(args, "exploration", {}).get("sight_range", 9)
        
        if args.batch_size_run == 1:
            env = env_REGISTRY[args.env](sc_dict,
                                        env_index=env_id,
                                        sight_range=sight_range,
                                        exploration_threshold=exploration_threshold,
                                        **args.env_args)
            envs.append(env)
        else:
            for i in range(args.batch_size_run):
                env = env_REGISTRY[args.env](sc_dict,
                                          env_index=i,
                                          sight_range=sight_range,
                                          exploration_threshold=exploration_threshold,
                                          **args.env_args)
                envs.append(env)
    return envs
