import os
import wandb
import torch
import random
import argparse
import numpy as np
from glob import glob
from typing import Union, Callable, Dict
from gym.wrappers import Monitor
from abc import ABC, abstractmethod

from common.utils import restore_wandb
from common.utils import restore_hyper_params
from common.utils import restore_model_params

from common.utils import save_hyper_params
from common.utils import save_model_params
from common.utils import save_wandb

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
        self.record = config.record

        self.report_dir = config.report_dir

        self.video_dir = config.video_dir
        self.params_path = config.params_path
        self.hyperparams_path = config.hyperparams_path
        self.tensorboard_path = config.tensorboard_path

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def init_hyper_params(self) -> dict:
        ''' project에서 사용될 하이퍼파라미터들을 딕셔너리로 구현

            만약 self.restore == True일 경우,
            `report/model/{project name}/hyperparams.pkl`에서 복원해서 넣는다.

        Example:
            return { "gamma": 0.99,
                     "epsilon": 0.1 }
        '''

    @abstractmethod
    def init_env(self, 
        hyper_params: dict, 
        render_available: bool,
        monitor_func: Callable):
        ''' `common/envs/ 에 있는 클래스 들 중 하나를 구현 후 반환

        Args:
            hyper_params: self.init_hyper_params()에서 구현한 딕셔너리
            monitor_func: self.monitor_func
            

        Examples:
            return Atari(env_id = 'Breakout-v4', 
                         n_envs = 4,
                         monitor_func = monitor_func(lambda x: x % 50 == 0)
                         )
        '''

    @abstractmethod
    def init_model(self, 
        state_size: Union[list, int],
        action_size: int,
        device: str,
        hyper_params: dict
    ) -> Dict[str, Union[torch.nn.Module, torch.optim.Optimizer]]:
        ''' 모델과 옵티마이저를 구현하고 반환

        Args:
            state_size: model의 입력으로 사용되는 환경 상태 크기
            action_size: model의 출력으로 사용되는 환경 행동 수 
            device: self.device
            hyper_params: self.init_hyper_params()에서 구현한 딕셔너리
        
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
        hyper_params: dict,
        tensorboard_path: str,
    ):
        ''' algorithms/ 안의 에이전트 구현 후 반환

        Args:
            env: self.init_env()에서 구현한 환경
            model: self.init_model()에서 구현한 모델
            device: PyTorch cuda or cpu
            hyper_params: self.init_hyper_params()에서 구현한 하이퍼파라미터
            tensorboard_path: 텐서보드를 저장할 경로

        Examples:
            return A2C(...)
        '''

    def monitor_func(self, video_callable=None, *args, **kargs):
        ''' init_env의 argument에 들어가는 함수.
            init_env 
        '''
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
            hyper_params = restore_hyper_params(self.hyperparams_path)
            print("[INFO] Loaded hyperparameters from " \
                 + self.hyperparams_path)
        else:
            hyper_params = self.init_hyper_params()
            print("[INFO] Initialized hyperparameters")
        #===================================================================

        # Environment
        #===================================================================
        env = self.init_env(hyper_params, self.render, self.monitor_func)
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
            state_size=env.state_size,
            action_size=env.action_size,
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
        agent = self.init_agent(
            env = env, 
            model = model, 
            device = self.device, 
            hyper_params = hyper_params,
            tensorboard_path = self.tensorboard_path)
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

        