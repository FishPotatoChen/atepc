from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, TextAreaField

class IndexForm(FlaskForm):
    text1 = StringField(
        '请选择评论读取方式：文本框中输入或者文件读取。')
    text2 = StringField(
        '文件读取请将评论放入comment.txt文件中之后点击按钮`文件`即可；文本框右下角可以拖拽，根据评论多少可以自行放大或者缩小')
    string = TextAreaField(render_kw={"class": "text-control"})
    submit1 = SubmitField('文本框', render_kw={"class": "form-control"})
    submit2 = SubmitField('文件', render_kw={"class": "form-control"})

class ResultForm(FlaskForm):
    submit = SubmitField('返回', render_kw={"class": "form-control"})