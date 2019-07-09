import argparse

def get_config() -> argparse.Namespace:

     parser = argparse.ArgumentParser(description="Pytorch RL algorithms")

     parser.add_argument("-u", "--user_name", type=str, 
                         default="mclearning2",
                         help="")

     # TODO: experiment로 이름 변환
     parser.add_argument("-p", "--project", type=str, 
                         default=None,
                         help="실행할 실험 이름. report에 저장될 폴더 이름으로도 쓰인다."
                              "pyinquirer로 선택")

     parser.add_argument("-i", "--run_id", type=str, 
                         default=None,
                         help="")

     parser.add_argument("-s", "--seed", type=int, default=1,
                         help="이전 실험 결과를 그대로 보기 위한 랜덤 시드")

     parser.add_argument("-t", "--test", dest="test", action="store_true",
                         help="테스트 모드 ON")

     parser.add_argument("-l", "--load", dest="load", action="store_true",
                         help="reports/model에서 해당 project 파라미터 불러온다.")

     parser.add_argument("-d", "--render", dest="render", action="store_true",
                         help="학습/테스트 중 rendering")

     parser.add_argument("-c", "--record", dest="record", action="store_true",
                         help="mp4 파일 형태로 비디오 녹화")

     parser.add_argument("-f", "--projects_dir", default="projects")
     parser.add_argument("-g", "--reports_dir", default="reports")

     parser.set_defaults(test=False)
     parser.set_defaults(load=False)
     parser.set_defaults(render=False)
     parser.set_defaults(record=False)

     config = parser.parse_args()

     return config




