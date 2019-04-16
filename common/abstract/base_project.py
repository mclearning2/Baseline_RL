import os
import gym
import wandb
import torch
import pickle
import numpy as np
from glob import glob
from typing import Tuple, Union, Callable, Dict
from gym.wrappers import Monitor
from abc import ABC, abstractmethod

from common.envs.core import GymEnv
from torch.distributions import Normal, Categorical
from common.help_function import check_path_or_make_dir, set_random_seed, remove_dir

class BaseProject(ABC):
    ''' Baseline of project that helps to implementation of various experiment.

    What you need to implement
        - init_hyper_params(self)
        - init_env(self, hyper_params, render_on, monitor_func)
        - init_model(self, input_size, output_size, hyper_params)
        - init_agent(self, env, model, optim, device, hyper_params)
    
    '''
    def __init__(self, config):
        
        self.config = config

        self.user_name = config.user_name
        self.project = config.project
        self.run_id = config.run_id
        
        self.seed = config.seed
        
        self.test_mode = config.test_mode
        self.restore = config.restore
        self.render = config.render
        self.record = config.record

        self.report_dir = config.report_dir

        self.video_dir = config.video_dir
        self.params_path = config.params_path
        self.hyperparams_path = config.hyperparams_path

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def init_hyper_params(self) -> dict:
        ''' Implement and return hyperparameters.
            if config.restore == True, It's restored from self.hyperparams_path
        
        Example:
            return { "gamma": 0.9531 }
        '''

    @abstractmethod
    def init_env(
        self, 
        render_on: bool,
        monitor_func: Callable,
        hyper_params: dict,
    ) -> GymEnv:
        ''' Implement and return environment class(GymEnv).

        Args:
            render_on: Whether environment is render or not.
            monitor_func: Recording function. argument is video_callable
                          It will be saved in self.video_dir
            hyper_params: dictionary from self.init_hyper_params()

        Examples:
            return GymEnv(
                env_id = 'CartPole-v1', 
                n_envs = hyper_params['n_workers'],
                render_on = render_on,
                max_episode = 300,
                max_episode_steps = hyper_params['max_episode_steps'],
                monitor_func = monitor_func(lambda x: x % 50 == 0)
            )
        '''        

    @abstractmethod
    def init_model(
        self, 
        input_size: int,
        output_size: int,
        device: str,
        hyper_params: dict
    ) -> Dict[str, Union[torch.nn.Module, torch.optim.Optimizer]]:
        ''' Implement and return model.

        Args:
            input_size: The input size of model
            output_size: The input size of model
            hyper_params: dictionary from self.init_hyper_params()
        
        Examples:
            model = MLP(...)
            optimizer = optim.Adam(...)

            return {'model', model, 'optim', optimizer}
        '''

    @abstractmethod
    def init_agent(
        self, 
        env: GymEnv, 
        model: dict, 
        device: str, 
        hyper_params: dict
    ):
        '''Implement and return agent.

        Examples:
            return A2C(...)
        '''

    def monitor_func(self, video_callable=None, *args, **kargs):
        def func(env):
            if self.record:
                print("[INFO] Video(mp4) will be saved in " + self.video_dir)
                return Monitor(
                    env=env,
                    directory=self.video_dir,
                    video_callable=video_callable,
                    force=True,
                    *args,
                    **kargs
                )
            else:
                remove_dir(self.video_dir)
                return env

        return func

    def _restore_wandb(self):
        ''' wandb로부터 학습에 필요한 model과 hyperparameter를 복원
            복원하는 경로
             - config.model_path
             - config.hyper_params_path
        '''

        if self.run_id:
            run_path = os.path.join(self.user_name, self.project, self.run_id)

            print(f"[INFO] Loaded from {run_path}")
            for path in [self.params_path, self.hyperparams_path]:
                root = os.path.dirname(path)
                name = os.path.basename(path)

                downloaded = wandb.restore(
                    name=name,
                    run_path=run_path,
                    replace=True,
                    root=root)

                print(f"Saved to {path}")

    def _init_hyper_params(self):
        ''' pickle 형식(.pkl)으로 저장했던 hyperparameter를 다시 불러온다.
            config.restore 일 때만 반환하며 아닐 경우 None을 반환.
            만약 파일이 없을 경우 에러를 보이며 반환
        '''
        if self.restore:
            with open(self.hyperparams_path, 'rb') as f:
                unpickler = pickle.Unpickler(f)
                hyper_params = unpickler.load()

                print("[INFO] Loaded hyperparameters from " + self.hyperparams_path)

            return hyper_params
        else:
            print("[INFO] Initialized hyperparameters")
            return self.init_hyper_params()

    def _init_model(self, env, hyper_params):
        model = self.init_model(
                    input_size=env.state_size, 
                    output_size=env.action_size,
                    device=self.device,
                    hyper_params=hyper_params
                )

        if self.restore:
            for name, tensor in model.items():    
                params = torch.load(self.params_path)
                model[name].load_state_dict(params[name])
                
            print("[INFO] Loaded model and optimizer from " + self.params_path)
        else:
            print("[INFO] Initialized model and optimizer")
        
        return model

    def _save_model(self, model):
        check_path_or_make_dir(self.params_path)

        params = dict()
        for name, tensor in model.items():
            params[name] = model[name].state_dict()

        torch.save(params, self.params_path)

        print("[INFO] Saved model and optimizer to " + self.params_path)

    def _save_hyper_params(self, hyper_params):
        check_path_or_make_dir(self.hyperparams_path)

        with open(self.hyperparams_path, 'wb+') as f:
            pickle.dump(hyper_params, f)

        print("[INFO] Saved hyperparameters to " + self.hyperparams_path)

    def _save_wandb(self):
        wandb.save(self.params_path)
        wandb.save(self.hyperparams_path)
        files = glob(os.path.join(self.video_dir, "*.mp4"))
        for mp4_file in files:
            wandb.save(mp4_file)

    def run(self):
        # == Restore from cloud == #
        self._restore_wandb()
        
        # == Hyper parameters == #
        hyper_params = self._init_hyper_params()

        # == Environment == #
        env = self.init_env(hyper_params, self.render, self.monitor_func)
        print(f"[INFO] Initialized environment")

        set_random_seed(env, self.seed)
        print(f"[INFO] Seeds are set by {self.seed}")

        # == Model == #
        model = self._init_model(env, hyper_params)

        # == Agent == #
        agent = self.init_agent(env, model, self.device, hyper_params)
        print("[INFO] Initialized agent")

        
        # == Train or Test == *
        if self.test_mode:
            print("[INFO] Starting test...")
            agent.test()
        else:
            print("[INFO] Starting train...")

            wandb.init(
                project=self.project, 
                config=hyper_params, 
                tensorboard=True,
                dir='report'
            )

            print("Model parameters")
            print("=" * 70)
            print(model)
            print("=" * 70, end="\n\n")

            print("Parsing parameters")
            print("=" * 70)
            for key, value in vars(self.config).items():
                print(key, "=",value)
            print("=" * 70, end="\n\n")

            print("Hyperparameters")
            print("=" * 70)
            for key, value in hyper_params.items():
                print(key, "=",value)
            print("=" * 70, end="\n\n")

            try:
                agent.train()
            except KeyboardInterrupt:
                pass

            # Save 
            self._save_model(model)
            self._save_hyper_params(hyper_params)
            self._save_wandb()
        
        env.close()