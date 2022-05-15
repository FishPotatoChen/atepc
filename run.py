import torch
from utils.Pytorch_GPUManager import GPUManager
from model.ATEPC import ATEPC
from utils.ATEPC_data_utils import NoAspectError

if __name__ == "__main__":
    index = GPUManager().auto_choice()
    device = torch.device("cuda:" + str(index)
                          if torch.cuda.is_available() else "cpu")
    main = ATEPC(device, "./output/trainedATE.pt", "./output/trainedAPC.pt")
    with open("comment.txt", encoding='utf-8') as f:
        for s in f.readlines():
            string = ''.join(s.strip().split())  # 字符串去空格后合并
            print('-'*80)
            print(string)
            try:
                x, y = main.input_string(string)
            except NoAspectError as e:
                print(e)
                continue
            for i, j in zip(x, y):
                print(i, j)
