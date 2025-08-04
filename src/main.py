import numpy as np
import os
import collections.abc
from os.path import dirname, abspath
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
import sys
import torch as th
from utils.logging import get_logger
import yaml
import time

from run import REGISTRY as run_REGISTRY

SETTINGS['CAPTURE_MODE'] = "no"
logger = get_logger()

ex = Experiment("pymarl", save_git_info=False)
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")


@ex.main
def main(_run, _config, _log):
    try:
        my_main(_run, _config, _log)
    except ValueError as e:
        import traceback
        try:
            error_str = str(e)
            if "Unsafe reshape" in error_str:
                import re
                shapes = re.findall(r'torch\.Size\(\[(.*?)\]\)', error_str)
                if len(shapes) >= 2:
                    source_shape = shapes[0].split(', ')
                    target_shape = shapes[1].split(', ')
                    if len(source_shape) > 0 and len(target_shape) > 0:
                        if source_shape[0] != target_shape[0]:
                            pass
        except Exception as debug_error:
            pass
        raise

def my_main(_run, _config, _log):
    config = config_copy(_config)
    np.random.seed(config["seed"])
    th.manual_seed(config["seed"])
    config['env_args']['seed'] = config["seed"]

    if 'self_play' in config:
        config['env_args']['self_play'] = config['self_play']
    else:
        config['env_args']['self_play'] = False

    run_REGISTRY[config['run']](_run, config, _log)

def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)

def _get_config(params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name)), "r", encoding='utf-8') as f:
            try:
                config_dict = yaml.load(f,Loader=yaml.FullLoader)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict


def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


if __name__ == '__main__':
    import os

    from copy import deepcopy
    params = deepcopy(sys.argv)
    if not any(["--env-config" in s for s in params]):
        params.append("--env-config=sc2custom")
    if not any(["--config" in s for s in params]):
        params.append("--config=IA-KRC")

    with open(os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r") as f:
        try:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    env_config = _get_config(params, "--env-config", "envs")
    alg_config = _get_config(params, "--config", "algs")
    
    try:
        with open(os.path.join(os.path.dirname(__file__), "config", "IA-KRC.yaml"), "r", encoding='utf-8') as f:
            czy_config = yaml.load(f, Loader=yaml.FullLoader)
    except Exception:
        czy_config = {}
    
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)
    if czy_config:
        config_dict = recursive_dict_update(config_dict, czy_config)
    
    import os as _os
    _os.environ["USE_EXPLORE_MAP"] = "1" if config_dict.get("use_explore_map", 1) else "0"
    
    ex.add_config(config_dict)

    log_dir = config_dict['log_name']
    for cur_dir in params:
        if "log_name" in cur_dir:
            log_dir = cur_dir.split("=")[1]

    if not config_dict['evaluate']:
        if len(log_dir):
            file_obs_path = os.path.join(results_path, "sacred", log_dir)
        else:
            file_obs_path = os.path.join(results_path, "sacred", config_dict['name'])
        while True:
            try:
                ex.observers.append(FileStorageObserver.create(file_obs_path))
                break
            except FileExistsError:
                time.sleep(1)

    ex.run_commandline(params)


