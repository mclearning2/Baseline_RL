import os
import wandb
import torch
import pickle
import numpy as np
from glob import glob
from typing import Union, Callable, Dict
from gym.wrappers import Monitor
from abc import ABC, abstractmethod

from common.envs.gym import Gym
from common.help_function import check_path_or_make_dir, set_random_seed, remove_dir

class BaseProject(ABC):
    ''' Baseline of project that helps to implementation of various experiment.

    What you need to implement
        - init_hyper_params(self)
        - init_env(self, hyper_params, monitor_func)
        - init_model(self, input_size, output_size, hyper_params)
        - init_agent(self, env, model, optim, device, hyper_params)
    '''
    def __init__(self, config):
        ''' config from {project_folder}/common/parse.'''

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
    def init_hyper_params(self
    ) -> dict:
        ''' Implement and return hyperparameters dictionary
            if self.restore == True, restored from self.hyperparams_path

        Example:
            return { "gamma": 0.99,
                     "epsilon": 0.1 }
        '''

    @abstractmethod
    def init_env(self, 
        hyper_params: dict,
        monitor_func: Callable,         
    ) -> Gym:
        ''' Implement and return environment class.
            Environment is used from one of {project_folder}/common/envs

        Args:
            monitor_func: Recording function. argument is video_callable
                          It will be saved in self.video_dir
            hyper_params: dictionary from self.init_hyper_params()

        Examples:
            return Atari(env_id = 'Breakout-v4', ...)
        '''        

    @abstractmethod
    def init_model(self, 
        env: Gym,
        device: str,
        hyper_params: dict
    ) -> Dict[str, Union[torch.nn.Module, torch.optim.Optimizer]]:
        ''' Implement and return model.

        Args:
            env: environment from self.init_env()
            device: pytorch cuda or cpu
            hyper_params: dictionary from self.init_hyper_params()
        
        Examples:
            model = MLP(...)
            optimizer = optim.Adam(...)

            return {'model', model, 'optim', optimizer}
        '''

    @abstractmethod
    def init_agent(self, 
        env, 
        model: dict, 
        device: str, 
        hyper_params: dict
    ):
        '''Implement and return agent.

        Args:
            env: environment from self.init_env()
            model: dictionary from self.init_model()
            device: PyTorch cuda or cpu
            hyper_params: dictionary from self.init_hyper_params()

        Examples:
            return A2C(...)
        '''

    def monitor_func(self, video_callable=None, *args, **kargs):
        def _func(env):
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
                return env

        return _func

    def _restore_wandb(self):
        ''' Restore model parameters and hyperparameters from wandb
            and save to
             - self.params_path
             - self.hyperparams_path
        '''

        if self.run_id:
            run_path = os.path.join(self.user_name, self.project, self.run_id)

            print(f"[INFO] Loaded from {run_path} in wandb cloud")
            video_paths = glob(os.path.join(self.video_dir, "*.mp4"))

            for path in [self.params_path, self.hyperparams_path] + list(video_paths):
                
                root = os.path.dirname(path)
                name = os.path.basename(path)
                
                downloaded = wandb.restore(
                    name=name,
                    run_path=run_path,
                    replace=True,
                    root=root)

                print(f"  - Saved to {path}")

    def _init_hyper_params(self):
        ''' pickle 형식(.pkl)으로 저장했던 hyperparameter를 다시 불러온다.
            config.restore 일 때만 반환하며 아닐 경우 None을 반환.
            만약 파일이 없을 경우 에러를 보이며 반환
        '''

        hyper_params = self.init_hyper_params()

        if self.restore:
            try:
                with open(self.hyperparams_path, 'rb') as f:
                    unpickler = pickle.Unpickler(f)
                    hyper_params = unpickler.load()

                    print("[INFO] Loaded hyperparameters from " + self.hyperparams_path)
            except FileNotFoundError:
                print("[INFO] Failed to load hyperparameters. So it is initialized")
        else:
            print("[INFO] Initialized hyperparameters")

        return hyper_params
        
    def _init_model(self, env, hyper_params):
        model = self.init_model(
                    input_size=env.state_size,
                    output_size=env.action_size,
                    device=self.device,
                    hyper_params=hyper_params
                )

        if self.restore:
            params = torch.load(self.params_path)
            for name, tensor in model.items():
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
        # Restore from cloud 
        self._restore_wandb()
        
        # Hyper parameters
        hyper_params = self._init_hyper_params()

        # Environment
        env = self.init_env(hyper_params, self.monitor_func)
        env.render_available = self.render
        print(f"[INFO] Initialized environment")
       
        set_random_seed(env, self.seed)
        print(f"[INFO] Seeds are set by {self.seed}")

        # Model
        model = self._init_model(env, hyper_params)

        # Agent
        agent = self.init_agent(env, model, self.device, hyper_params)
        print("[INFO] Initialized agent")

        # Test
        if self.test_mode:
            print("[INFO] Starting test...")
            agent.test()
        # Train
        else:
            print("[INFO] Starting train...")

            wandb.init(
                project=self.project, 
                config=hyper_params, 
                dir='report'
            )

            try:
                agent.train()
            except KeyboardInterrupt:
                pass

            # Save 
            self._save_model(model)
            self._save_hyper_params(hyper_params)
            self._save_wandb()
        
        env.close()