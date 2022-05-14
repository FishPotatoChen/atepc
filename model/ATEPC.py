from itertools import count
import torch
import torch.nn.functional as F

from transformers import BertTokenizer
from torch.utils.data import DataLoader, TensorDataset
from utils.ATEPC_data_utils import INPUTProcessor, convert_examples_to_eval_features, NoAspectError


class ATEPC():
    def __init__(self, device, atepath, apcpath):
        self.device = device
        self.processor = INPUTProcessor()
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-chinese", do_lower_case=True)
        self.atemodel = torch.load(atepath)
        self.apcmodel = torch.load(apcpath)
        self.atemodel.to(self.device)
        self.apcmodel.to(self.device)
        self.label_list = ["O", "B-ASP", "I-ASP", "[CLS]", "[SEP]"]
        self.num_labels = len(self.label_list) + 1

    def input_string(self, string):
        all_polar = []
        all_aspect = []
        string_list = []
        sentence_lenght = 60
        n = int((len(string))/sentence_lenght)
        for i in range(n):
            string_list.append(
                string[sentence_lenght*i:sentence_lenght*i+sentence_lenght])
        string_list.append(string[sentence_lenght*n:])
        for s in string_list:
            ate_examples = self.processor.create_input_sentence_examples_for_ate(
                s)
            eval_features = convert_examples_to_eval_features(
                ate_examples, self.label_list, 80, self.tokenizer)
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
            ate_eval_dataloader = DataLoader(eval_data)

            y_true = []
            y_pred = []
            self.atemodel.eval()
            label_map = {i: label for i,
                         label in enumerate(self.label_list, 1)}
            for batch in ate_eval_dataloader:
                batch = tuple(t.to(self.device) for t in batch)
                input_ids_spc, input_mask, segment_ids, label_ids, polarities, valid_ids, attention_mask_label = batch

                ate_logits = self.atemodel(input_ids_spc, valid_ids=valid_ids)

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
            aspect_tag = y_pred[0]

            try:
                apc_examples, aspect = self.processor.create_input_sentence_examples_for_apc(
                    s, aspect_tag)
            except NoAspectError:
                continue
            all_aspect.extend(aspect)
            eval_features = convert_examples_to_eval_features(
                apc_examples, self.label_list, 80, self.tokenizer)
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
            apc_eval_dataloader = DataLoader(eval_data)

            test_apc_logits_all = None
            self.apcmodel.eval()
            for batch in apc_eval_dataloader:
                batch = tuple(t.to(self.device) for t in batch)
                input_ids_spc, input_mask, segment_ids, label_ids, polarities, valid_ids, attention_mask_label = batch
                apc_logits = self.apcmodel(
                    input_ids_spc, labels=label_ids, valid_ids=valid_ids)

                if test_apc_logits_all is None:
                    test_apc_logits_all = apc_logits
                else:
                    test_apc_logits_all = torch.cat(
                        (test_apc_logits_all, apc_logits), dim=0)
            all_polarities_label = torch.argmax(
                test_apc_logits_all, -1).detach().cpu().numpy()

            for i in all_polarities_label:
                if i == 0:
                    all_polar.append("负面")
                elif i == 1:
                    all_polar.append("正面")
                else:
                    all_polar.append("未识别出情感")
        if all_aspect == []:
            raise NoAspectError()
        return all_aspect, all_polar
