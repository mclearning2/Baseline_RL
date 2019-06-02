import os
import wandb
import torch
import pickle
import random
import argparse
import numpy as np
from glob import glob
from typing import Union, Callable, Dict
from gym.wrappers import Monitor
from abc import ABC, abstractmethod

from common.envs.gym import Gym
from common.utils import check_path_and_make_dir, remove_dir
from common.utils import restore_wandb, restore_hyper_params

class BaseProject(ABC):
    ''' 이 클래스를 상속해서 다음과 같은 함수들을 구현하면
        알고리즘이 그에 맞게 실행을 하게 된다.

        - init_hyper_params(self)
        - init_env(self, hyper_params, monitor_func)
        - init_model(self, input_size, output_size, hyper_params)
        - init_agent(self, env, model, optim, device, hyper_params)
    '''
    def __init__(self, config: argparse.Namespace):

        self.user_name = config.user_name
        self.project = config.project
        self.run_id = config.run_id
        
        self.seed = config.seed
        
        self.test = config.test
        self.restore = config.restore
        self.render = config.render

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
        env: Gym, 
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
            print("[INFO] Video(mp4) will be saved in " + self.video_dir)
            return Monitor(
                env=env,
                directory=self.video_dir,
                video_callable=video_callable,
                force=True,
                *args,
                **kargs
            )
        
        return _func

    def run(self):
        # Restore from cloud 
        #===================================================================
        if self.run_id:
            restore_wandb(
                user_name=self.user_name,
                project=self.project,
                run_id=self.run_id,
                params_path=self.params_path,
                hyperparams_path=self.hyperparams_path
                )
            print(f"[INFO] Loaded from {self.run_id} in wandb cloud")
        #===================================================================
        
        # Hyper parameters
        #===================================================================
        if self.restore:
            hyper_params = self._init_hyper_params()
            print("[INFO] Initialized hyperparameters")
        else:
            hyper_params = restore_hyper_params(self.hyperparams_path)
            print("[INFO] Loaded hyperparameters from " \
                 + self.hyperparams_path)
        #===================================================================

        # Environment
        #===================================================================
        env = self.init_env(hyper_params, self.monitor_func)
        env.render_available = self.render
        print(f"[INFO] Initialized environment")
        #===================================================================
        
        # Seed
        #===================================================================
        env.seed(self.seed)        
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        print(f"[INFO] Seeds are set by {self.seed}")
        #===================================================================

        # Model
        #===================================================================
        model = self.init_model(
            input_size=env.state_size,
            output_size=env.action_size,
            device=self.device,
            hyper_params=hyper_params
        )
        
        if self.restore:
            restore_model_params(model, self.params_path)
            print("[INFO] Loaded model and optimizer from " \
                  + self.params_path)
        else:
            print("[INFO] Initialized model and optimizer")
        #===================================================================

        # Agent
        #===================================================================
        agent = self.init_agent(env, model, self.device, hyper_params)
        print("[INFO] Initialized agent")
        #===================================================================

        # Test
        if self.test:
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

            print(model)

            try:
                agent.train()
            except KeyboardInterrupt:
                pass

            # Save 
            save_model_params(model, self.params_path)
            print("[INFO] Saved model parameters to " + self.hyperparams_path)
            save_hyper_params(self.hyperparams_path)
            print("[INFO] Saved hyperparameters to " + self.hyperparams_path)
            save_wandb(self.params_path, self.hyperparams_path, self.video_dir)
            print("[INFO] Saved all to cloud(wandb)")
        
        env.close()

        