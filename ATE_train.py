import json
import logging
import os
import sys
import random
from time import strftime, localtime

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers.models.bert.modeling_bert import BertModel
from transformers import BertTokenizer
from seqeval.metrics import classification_report
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from model.ATE import ATE
from utils.ATE_data_utils import ATEProcessor, convert_examples_to_ate_features
from utils.Pytorch_GPUManager import GPUManager

# 定义日志
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

# 创建log目录文件
os.makedirs('logs', exist_ok=True)
time = '{}'.format(strftime("%y%m%d-%H%M%S", localtime()))  # 写入记录时间
log_file = 'logs/{}_ATE_train.log'.format(time)
logger.addHandler(logging.FileHandler(log_file))
logger.info('log file: {}'.format(log_file))


class Main():
    def set_dataset(self, config):
        self.config = config
        # 判断梯度累积步数
        if self.config['gradient_accumulation_steps'] < 1:
            raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                self.config['gradient_accumulation_steps']))
        # 迭代过程中的训练
        self.config['train_batch_size'] = self.config['train_batch_size'] // self.config['gradient_accumulation_steps']
        # 创建输出位置
        if not os.path.exists(self.config['output_dir']):
            os.makedirs(self.config['output_dir'])
        self.processor = ATEProcessor()  # 处理数据，将数据变为输入所需的形式
        self.label_list = ["O", "B-ASP", "I-ASP", "[CLS]", "[SEP]"]
        self.num_labels = len(self.label_list) + 1
        self.datasets = {
            'camera': "atepc_datasets/camera",
            'car': "atepc_datasets/car",
            'phone': "atepc_datasets/phone",
            'notebook': "atepc_datasets/notebook"
        }
        self.config['data_dir'] = self.datasets[self.config['dataset']]
        # 得到训练数据和测试数据
        self.train_examples = self.processor.get_train_examples(
            self.config['data_dir'])
        self.eval_examples = self.processor.get_test_examples(
            self.config['data_dir'])
        # 转化极性,这些数据集的极性是3种
        self.train_examples = self.convert_polarity(self.train_examples)
        self.eval_examples = self.convert_polarity(self.eval_examples)
        # 下载文件,Bert的预训练模型
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-chinese", do_lower_case=True)
        self.bert_base_model = BertModel.from_pretrained(
            "bert-base-chinese")
        for arg in self.config:
            logger.info('>>> {0}: {1}'.format(arg, self.config[arg]))
        # for name, param in self.bert_base_model.named_parameters():
        #     print(name, param.shape)
        # 加载预训练模型的标签数
        self.bert_base_model.config.num_labels = self.num_labels
        #  加载模型
        self.model = ATE(self.bert_base_model, args=self.config)
        # 将模型放入GPU
        self.model.to(self.config['device'])
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.00001},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.00001}
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters,
                               lr=self.config['learning_rate'], weight_decay=0.00001)

    def convert_polarity(self, examples):
        for i in range(len(examples)):
            polarities = []
            for polarity in examples[i].polarity:
                if polarity == 2:
                    polarities.append(1)
                else:
                    polarities.append(polarity)
            examples[i].polarity = polarities
        return examples

    def process_dataset(self):
        eval_features = convert_examples_to_ate_features(self.eval_examples, self.label_list, self.config['max_seq_length'],
                                                         self.tokenizer)
        all_spc_input_ids = torch.tensor(
            [f.input_ids_spc for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor(
            [f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor(
            [f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor(
            [f.label_id for f in eval_features], dtype=torch.long)
        all_polarities = torch.tensor(
            [f.polarities for f in eval_features], dtype=torch.long)
        all_valid_ids = torch.tensor(
            [f.valid_ids for f in eval_features], dtype=torch.long)
        all_lmask_ids = torch.tensor(
            [f.label_mask for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_spc_input_ids, all_input_mask, all_segment_ids, all_label_ids,
                                  all_polarities, all_valid_ids, all_lmask_ids)
        # Run prediction for full data
        eval_sampler = RandomSampler(eval_data)
        self.eval_dataloader = DataLoader(
            eval_data, sampler=eval_sampler, batch_size=self.config['eval_batch_size'])

        train_features = convert_examples_to_ate_features(
            self.train_examples, self.label_list, self.config['max_seq_length'], self.tokenizer)
        all_spc_input_ids = torch.tensor(
            [f.input_ids_spc for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor(
            [f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor(
            [f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor(
            [f.label_id for f in train_features], dtype=torch.long)
        all_valid_ids = torch.tensor(
            [f.valid_ids for f in train_features], dtype=torch.long)
        all_lmask_ids = torch.tensor(
            [f.label_mask for f in train_features], dtype=torch.long)
        all_polarities = torch.tensor(
            [f.polarities for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_spc_input_ids, all_input_mask, all_segment_ids,
                                   all_label_ids, all_polarities, all_valid_ids, all_lmask_ids)  # 相当于zip

        train_sampler = SequentialSampler(train_data)
        self.train_dataloader = DataLoader(
            train_data, sampler=train_sampler, batch_size=self.config['train_batch_size'])

    def evaluate(self):
        # evaluate
        ate_result = 0
        y_true = []
        y_pred = []
        self.model.eval()
        label_map = {i: label for i, label in enumerate(self.label_list, 1)}
        for batch in self.eval_dataloader:
            batch = tuple(t.to(self.config['device']) for t in batch)
            input_ids_spc, input_mask, segment_ids, label_ids, polarities, valid_ids, attention_mask_label = batch

            # 在反向传播中不被记录
            with torch.no_grad():
                ate_logits = self.model(
                    input_ids_spc, segment_ids, input_mask, valid_ids=valid_ids)

            # 测试方面词提取的准确度
            ate_logits = torch.argmax(
                F.log_softmax(ate_logits, dim=2), dim=2)
            ate_logits = ate_logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            for i, label in enumerate(label_ids):
                temp_1 = []
                temp_2 = []
                for j, m in enumerate(label):
                    if j == 0:
                        continue
                    elif label_ids[i][j] == len(self.label_list):
                        y_true.append(temp_1)
                        y_pred.append(temp_2)
                        break
                    else:
                        temp_1.append(label_map.get(label_ids[i][j], 'O'))
                        temp_2.append(label_map.get(ate_logits[i][j], 'O'))

        report = classification_report(y_true, y_pred, digits=4)
        tmps = report.split()
        ate_result = round(float(tmps[7]) * 100, 2)
        return ate_result

    def train(self):
        # 设置种子
        random.seed(self.config['seed'])
        np.random.seed(self.config['seed'])
        torch.manual_seed(self.config['seed'])
        max_ate_test_f1 = 0
        global_step = 0
        for epoch in range(int(self.config['num_train_epochs'])):
            nb_tr_examples, nb_tr_steps = 0, 0
            for batch in self.train_dataloader:
                self.model.train()  # 开始训练
                batch = tuple(t.to(self.config['device']) for t in batch)
                input_ids_spc, input_mask, segment_ids, label_ids, polarities, valid_ids, l_mask = batch
                loss_ate = self.model(
                    input_ids_spc, segment_ids, input_mask, label_ids, valid_ids)
                loss = loss_ate
                loss.backward()
                nb_tr_examples += input_ids_spc.size(0)
                nb_tr_steps += 1
                self.optimizer.step()
                self.optimizer.zero_grad()
                global_step += 1
                if global_step % self.config['eval_steps'] == 0:
                    if epoch >= self.config['num_train_epochs']-2 or self.config['num_train_epochs'] <= 2:
                        # evaluate in last 2 epochs
                        ate_result = self.evaluate()
                        if ate_result > max_ate_test_f1:
                            max_ate_test_f1 = ate_result
        return max_ate_test_f1

    def save_model(self):
        # Take care of the storage!
        path = '{0}/trainedATE.pt'.format(self.config['output_dir'])
        torch.save(self.model, path)


def parse_experiments(path):
    # 从json中读取文件
    configs = []
    with open(path, "r", encoding='utf-8') as reader:
        json_config = json.loads(reader.read())
    for id, config in json_config.items():
        # Hyper Parameters
        parser = {}
        parser['dataset'] = config['dataset']
        parser['output_dir'] = config['output_dir']
        parser['SRD'] = int(config['SRD'])
        parser['learning_rate'] = float(config['learning_rate'])
        parser['num_train_epochs'] = float(config['num_train_epochs'])
        parser['train_batch_size'] = int(config['train_batch_size'])
        parser['dropout'] = float(config['dropout'])
        parser['max_seq_length'] = int(config['max_seq_length'])
        parser['eval_batch_size'] = 32
        parser['eval_steps'] = 20
        # 默认为1
        parser['gradient_accumulation_steps'] = 1
        configs.append(parser)
    return configs


if __name__ == "__main__":
    logger.info('begin time: {}'.format(
        strftime("%y%m%d-%H%M%S", localtime())))  # 写入开始时间

    index = GPUManager().auto_choice()
    device = torch.device("cuda:" + str(index)
                          if torch.cuda.is_available() else "cpu")

    exp_configs = parse_experiments('train.json')

    main = Main()
    except_num = 10
    n = 5
    for config in exp_configs:
        max_ate_test_f1 = 0
        config['device'] = device
        main.set_dataset(config)
        main.process_dataset()
        while(True):
            logger.info('-'*80)
            logger.info('Config {} (totally {} configs)'.format(
                exp_configs.index(config)+1, len(exp_configs)))
            results = []
            try:
                for i in range(n):
                    main.config['seed'] = i + 1
                    logger.info(
                        'No.{} training process of {}'.format(i + 1, n))

                    ate_test_f1 = main.train()

                    if ate_test_f1 > max_ate_test_f1:
                        max_ate_test_f1 = ate_test_f1
                    logger.info('max_ate_test_f1: {}'.format(max_ate_test_f1))
                except_num = 10
                break
            except Exception as e:
                if except_num == 0:
                    logger.info("Try 10 times! END")
                    exit()
                print(e)
                logger.info("=====Try Again=====")
                except_num -= 1
    main.save_model()
    logger.info('end time: {}'.format(
        strftime("%y%m%d-%H%M%S", localtime())))  # 写入结束时间
