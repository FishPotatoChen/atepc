import torch
from utils.Pytorch_GPUManager import GPUManager
from model.ATEPC import ATEPC
from utils.ATEPC_data_utils import NoAspectError
from flask import Flask
from flask import Flask, render_template, request, flash
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, TextAreaField

app = Flask(__name__)
app.secret_key = "123456"


class Newform(FlaskForm):
    text1 = StringField(
        '请选择评论读取方式：文本框中输入或者文件读取。')
    text2 = StringField(
        '文件读取请将评论放入comment.txt文件中之后点击按钮`文件`即可；文本框右下角可以拖拽，根据评论多少可以自行放大或者缩小')
    string = TextAreaField( render_kw={"class": "text-control"})
    submit1 = SubmitField('文本框', render_kw={"class": "form-control"})
    submit2 = SubmitField('文件', render_kw={"class": "form-control"})


@app.route('/', methods=['GET', 'POST'])
def index():
    form = Newform()
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
            return render_template('index.html', form=form, output=output)
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

            return render_template('index.html', form=form, output=output)
    return render_template('index.html', form=form)


if __name__ == '__main__':
    index = GPUManager().auto_choice()
    device = torch.device("cuda:" + str(index)
                          if torch.cuda.is_available() else "cpu")
    main = ATEPC(device, "./output/trainedATE.pt", "./output/trainedAPC.pt")
    app.run()
