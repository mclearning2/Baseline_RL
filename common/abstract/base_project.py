import os
import wandb
import torch
import random
import argparse
import numpy as np
from typing import Union, Callable, Dict
from gym.wrappers import Monitor
from abc import ABC, abstractmethod

from env.atari import Atari
from env.classic import Classic
from common.utils import restore_wandb, save_wandb
from common.utils import restore_hyper_params, save_hyper_params
from common.utils import restore_model_params, save_model_params

class BaseProject(ABC):
    def __init__(self, config: argparse.Namespace):
        self.__config = config

        self.video_dir = os.path.join('report/videos', config.project)
        self.params_path = os.path.join('report/model', config.project,
                                            'model.pt')
        self.hyperparams_path = os.path.join('report/model', config.project,
                                                'hyperparams.pkl')
        self.tensorboard_path = os.path.join('report/tensorboard',
                                                config.project)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() 
                                   else "cpu")

    @abstractmethod
    def init_hyper_params(self) -> dict:
        ''' train/test 할 때 사용될 하이퍼파라미터들을 딕셔너리로 반환

            만약 self.restore == True일 경우, 여기서 반환한 값이 아닌
            config.user_name, config.project, config.run_id를 통해 wandb에서 불러온
            하이퍼파라미터를 사용한다.

        Example:
            return { "gamma": 0.99,
                     "epsilon": 0.1 }
        '''

    @abstractmethod
    def init_env(self, hyper_params: dict) -> Union[Atari, Classic]:
        ''' `envs/ 에 있는 클래스 중 객체 하나를 만들어서 반환.

        Args:
            hyper_params: self.init_hyper_params()에서 구현한 딕셔너리

        Examples:
            return Atari(env_id = 'Breakout-v4',
                         n_envs = 4,
                         monitor_func = self.env_monitor(
                                            lambda iter: iter % 50
                                        )
                         )
        '''

    @abstractmethod
    def init_model(
        self,
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
    def init_agent(
        self,
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

    def is_render(self):
        return self.__config.render

    def is_test(self):
        return self.__config.test

    def is_restore(self):
        return self.__config.restore

    def is_record(self):
        return self.__config.record

    def get_video_dir(self):
        return self.__config.video_dir

    def monitor_func(self, video_callable=None, *args, **kargs):
        ''' init_env의 argument에 들어가는 함수. '''
        def _func(env):
            if self.__config.record:
                print("[INFO] Video(mp4) will be saved in here:")
                print(" > " + self.video_dir)
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
        # ======================================================================
        if self.run_id:
            restore_wandb(
                user_name=self.user_name,
                project=self.project,
                run_id=self.run_id,
                params_path=self.params_path,
                hyperparams_path=self.hyperparams_path
                )
            print(f"[INFO] Loaded from {self.run_id} in wandb cloud")
        # ======================================================================

        # Hyper parameters
        # ======================================================================
        if self.restore:
            hyper_params = restore_hyper_params(self.hyperparams_path)
            print("[INFO] Loaded hyperparameters from " +
                  self.hyperparams_path)
        else:
            hyper_params = self.init_hyper_params()
            print("[INFO] Initialized hyperparameters")
        # ======================================================================

        # Environment
        # ======================================================================
        env = self.init_env(hyper_params, self.render, self.monitor_func)
        print(f"[INFO] Initialized environment")
        # ======================================================================

        # Seed
        # ======================================================================
        env.seed(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        print(f"[INFO] Seeds are set by {self.seed}")
        # ======================================================================

        # Model
        # ======================================================================
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
        # ======================================================================

        # Agent
        # ======================================================================
        agent = self.init_agent(
            env=env,
            model = model, 
            device = self.device, 
            hyper_params = hyper_params,
            tensorboard_path = self.tensorboard_path
            )
        print("[INFO] Initialized agent")
        # ======================================================================

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
            save_hyper_params(hyper_params, self.hyperparams_path)
            print("[INFO] Saved hyperparameters to " + self.hyperparams_path)
            save_wandb(self.params_path, self.hyperparams_path, self.video_dir)
            print("[INFO] Saved all to cloud(wandb)")
        
        env.close()

        