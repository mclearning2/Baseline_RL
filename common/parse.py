import argparse

def get_config() -> argparse.Namespace:

     parser = argparse.ArgumentParser(description="Pytorch RL algorithms")

     parser.add_argument("-u", "--user_name", type=str, 
                         default="mclearning2",
                         help="Wandb(Weights & biases)에서 사용하는 user name. \n"
                              "restore_wandb()에서 user_name 파라미터로 사용하여 "
                              "모델 파라미터나 그 외 파라미터들을 불러온다.")

     # TODO: experiment로 이름 변환
     parser.add_argument("-p", "--project", type=str, 
                         default=None,
                         help="실행할 실험 이름. report에 저장될 폴더 이름으로도 쓰인다."
                              "pyinquirer로 선택")

     parser.add_argument("-i", "--run_id", type=str, 
                         default=None,
                         help="https://app.wandb.ai/에서 해당 "
                              "user_name, project, run_id을 이용해서 불러온다.")

     parser.add_argument("-s", "--seed", type=int, default=1,
                         help="이전 실험 결과를 그대로 보기 위한 랜덤 시드")

     parser.add_argument("-t", "--test", dest="test", action="store_true",
                         help="테스트 모드 ON")
                         
     parser.add_argument("-r", "--restore", dest="restore", action="store_true",
                         help="wandb로부터 모델 파라미터와 하이퍼파라미터를 불러온다."
                              "이 때 user_name, project, run_id 값이 유효해야한다.")

     parser.add_argument("-d", "--render", dest="render", action="store_true",
                         help="학습/테스트 중 rendering")

     parser.add_argument("-c", "--record", dest="record", action="store_true",
                         help="mp4 파일 형태로 비디오 녹화")

     parser.add_argument("-f", "--projects_dir", default="projects")
     parser.add_argument("-g", "--reports_dir", default="records")

     parser.set_defaults(test=False)
     parser.set_defaults(restore=False)
     parser.set_defaults(render=False)
     parser.set_defaults(record=False)

     config = parser.parse_args()

     return config




