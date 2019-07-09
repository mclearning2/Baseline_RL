import argparse

def get_config() -> argparse.Namespace:

     parser = argparse.ArgumentParser(description="Pytorch RL algorithms")

     # TODO: experiment로 이름 변환
     parser.add_argument("-p", "--project", type=str, 
                         default=None,
                         help="실행할 실험 이름. report에 저장될 폴더 이름으로도 쓰인다."
                              "pyinquirer로 선택")

     parser.add_argument("-r", "--restore", type=str,
                         default=None,
                         help="None이 아닌 경우 \
                              wandb로부터 모델 파라미터와 하이퍼파라미터를 불러온다. \
                              {user_name}/{project}/{run_id} 와 같이 입력한다. \
                              e.g. mclearning2/A2C_CartPole-v1/fx8nn0dh"
                              )

     parser.add_argument("-s", "--seed", type=int, default=1,
                         help="이전 실험 결과를 그대로 보기 위한 랜덤 시드")

     parser.add_argument("-t", "--test", dest="test", action="store_true",
                         help="테스트 모드 ON")

     parser.add_argument("-d", "--render", dest="render", action="store_true",
                         help="학습/테스트 중 rendering")

     parser.add_argument("-c", "--record", dest="record", action="store_true",
                         help="mp4 파일 형태로 비디오 녹화")

     parser.add_argument("-f", "--projects_dir", default="projects")
     parser.add_argument("-g", "--reports_dir", default="reports")

     parser.set_defaults(test=False)
     parser.set_defaults(render=False)
     parser.set_defaults(record=False)

     config = parser.parse_args()

     return config




