from flask import Flask
from flask import Flask, render_template, request, redirect, url_for
from app.form import IndexForm, ResultForm
from utils.ATEPC_data_utils import NoAspectError
import torch
from utils.Pytorch_GPUManager import GPUManager
from model.ATEPC import ATEPC

app = Flask(__name__)
app.secret_key = "123456"

index = GPUManager().auto_choice()
device = torch.device("cuda:" + str(index)
                      if torch.cuda.is_available() else "cpu")
main = ATEPC(device, "./output/trainedATE.pt", "./output/trainedAPC.pt")


@app.route('/', methods=['GET', 'POST'])
def index():
    form = IndexForm()
    if request.method == 'POST':
        if form.submit1.data:
            output = []
            input = form.string.data.split('\r')
            for s in input:
                string = ''.join(s.strip().split())
                output.append('-'*80)
                output.append(string)
                try:
                    x, y = main.input_string(string)
                except NoAspectError as e:
                    output.append(e)
                    continue
                for i, j in zip(x, y):
                    output.append(str(i)+' '+j)
            return render_template('result.html', form=ResultForm(), output=output)
        elif form.submit2.data:
            output = []
            with open("comment.txt", encoding='utf-8') as f:
                for s in f.readlines():
                    string = ''.join(s.strip().split())  # 字符串去空格后合并
                    output.append('-'*80)
                    output.append(string)
                    try:
                        x, y = main.input_string(string)
                    except NoAspectError as e:
                        output.append(e)
                        continue
                    for i, j in zip(x, y):
                        output.append(str(i)+' '+j)
            return render_template('result.html', form=ResultForm(), output=output)
    return render_template('index.html', form=form)
