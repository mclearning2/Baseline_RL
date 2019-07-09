import os
import wandb
import torch
import random
import argparse
import numpy as np
from pyfiglet import Figlet
from datetime import datetime
from common.logger import logger
from typing import Union, Callable, Dict, Type
from gym.wrappers import Monitor
from abc import ABC, abstractmethod

from environments.atari import Atari
from environments.gym import Gym
from common.utils import restore_wandb, save_wandb
from common.utils import restore_hyperparams, save_hyperparams
from common.utils import restore_model_params, save_model_params


class BaseProject(ABC):
    '''

    '''

    def __init__(self, config: argparse.Namespace):
        ''' Args
        '''
        self._config = config

        reports_dir = config.reports_dir

        current_time = str(datetime.now().strftime("%y%m%d_%H%M%S"))

        self.video_path = os.path.join(reports_dir, 'videos', config.project)
        self.params_path = os.path.join(reports_dir, 'model', config.project,
                                        'model.pt')
        self.hyperparams_path = os.path.join(reports_dir, 'model',
                                             config.project, 'hyperparams.pkl')
        self.tensorboard_path = os.path.join(reports_dir, 'tensorboard',
                                             config.project, current_time)

        self.device = torch.device("cuda:0" if torch.cuda.is_available()
                                   else "cpu")

    @abstractmethod
    def init_hyperparams(self) -> dict:
        ''' train/test 할 때 사용될 하이퍼파라미터들을 딕셔너리로 반환

        if self._config.restore == True:
            restored hyperparameter will be used
        else:
            hyperparameter here will be used

        Example:
            return { "gamma": 0.99, "epsilon": 0.1 }
        '''

    @abstractmethod
    def init_env(self, hyperparams: dict) -> Union[Atari, Gym]:
        ''' `environments/ 에 있는 클래스 중 객체 하나를 만들어서 반환.

        Args:
            hyperparams: self.init_hyperparams()에서 반환한 딕셔너리

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
        env: Union[Atari, Gym],
        hyperparams: dict
    ) -> Dict[str, Union[Type[torch.nn.Module], Type[torch.optim.Optimizer]]]:
        ''' 모델과 옵티마이저를 구현하고 반환

        Args:
            env: init_env() 함수에서 반환한 환경 클래스
            hyperparams: self.init_hyperparams()에서 구현한 딕셔너리

        Examples:
            model = MLP(...)
            optimizer = optim.Adam(...)

            return {'model', model, 'optim', optimizer}
        '''

    @abstractmethod
    def init_agent(
        self,
        env: Union[Atari, Gym],
        model: dict,
        hyperparams: dict,
    ):
        ''' algorithms/ 안의 에이전트 구현 후 반환

        Args:
            env: self.init_env()에서 구현한 환경
            model: self.init_model()에서 구현한 모델
            device: PyTorch cuda or cpu
            hyperparams: self.init_hyperparams()에서 구현한 하이퍼파라미터
            tensorboard_path: 텐서보드를 저장할 경로

        Examples:
            return A2C(...)
        '''

    @property
    def is_render(self):
        return self._config.render

    @property
    def is_test(self):
        return self._config.test

    @property
    def is_restore(self):
        return True if self._config.restore else False

    @property
    def is_record(self):
        return self._config.record

    def monitor_func(
            self,
            video_callable: Callable,
            force: bool = True,
            *args,
            **kargs):
        ''' init_env를 할 때 record를 하고 싶은 경우 이 monitor_func를 환경에 전달한다.
            video_callable은 gym.wrappers.Monitor의 파라미터이다. 
            그 외의 값들도 전달 가능.
        '''
        def _func(env):
            if self.is_record:
                logger.info("Video(mp4) will be saved in here > " 
                            + self.video_path)
                return Monitor(
                    env=env,
                    directory=self.video_path,
                    video_callable=video_callable,
                    force=True,
                    *args,
                    **kargs
                )
            else:
                return env

        return _func

    def run(self):
        # Restore files from wandb
        # ======================================================================
        if self._config.restore:
            user_name, project, run_id = self._config.restore.split('/')
            restore_wandb(
                user_name=user_name,
                project=project,
                run_id=run_id,
                params_path=self.params_path,
                hyperparams_path=self.hyperparams_path
            )
            logger.info("Loaded from {self.run_id} in wandb cloud")
        # ======================================================================

        # Hyper parameters
        # ======================================================================
        if self._config.restore:
            hyperparams = restore_hyperparams(self.hyperparams_path)
            logger.info("Loaded hyperparameters from " +
                        self.hyperparams_path)
        else:
            hyperparams = self.init_hyperparams()
            logger.info("Initialized hyperparameters")
        # ======================================================================

        # Environment
        # ======================================================================
        env = self.init_env(hyperparams)
        logger.info("Initialized environment")
        # ======================================================================

        # Seed
        # ======================================================================
        env.seed(self._config.seed)
        torch.manual_seed(self._config.seed)
        np.random.seed(self._config.seed)
        random.seed(self._config.seed)
        logger.info(f"Seeds are set by {self._config.seed}")
        # ======================================================================

        # Model
        # ======================================================================
        model = self.init_model(env, hyperparams)

        if self.is_restore:
            restore_model_params(model, self.params_path)
            logger.info(f"Loaded model and optimizer from {self.params_path}")
        else:
            logger.info("Initialized model and optimizer")
        # ======================================================================

        # Agent
        # ======================================================================
        agent = self.init_agent(
            env=env,
            model=model,
            device=self.device,
            hyperparams=hyperparams,
            tensorboard_path=self.tensorboard_path
        )
        logger.info("Initialized agent")
        # ======================================================================

        f = Figlet(font='slant')
        if self.is_test:
            print(f.renderText("T E S T"))
            agent.test()
        # Train
        else:
            print(f.renderText("T R A I N"))

            wandb.init(
                project=self._config.project,
                config=hyperparams,
                dir=self._config.reports_dir
            )

            print(model)

            try:
                agent.train()
            except KeyboardInterrupt:
                pass

            # Save
            save_model_params(model, self.params_path)
            logger.info(f"Saved model parameters to {self.hyperparams_path}")

            save_hyperparams(hyperparams, self.hyperparams_path)
            logger.info(f"Saved hyperparameters to {self.hyperparams_path}")

            save_wandb(self.params_path,
                       self.hyperparams_path, self.video_path)
            logger.info(f"Saved all to cloud(wandb)")

        env.close()
