from transformers.models.bert.modeling_bert import BertForTokenClassification
import torch
from torch.nn import CrossEntropyLoss, Dropout


class ATE(BertForTokenClassification):
    def __init__(self, bert_base_model, args):
        super(ATE, self).__init__(config=bert_base_model.config)
        self.bert_for_global_context = bert_base_model
        self.args = args
        self.dropout = Dropout(self.args['dropout'])

    def forward(self, input_ids_spc, token_type_ids=None, attention_mask=None, labels=None, valid_ids=None):
        global_context_out = self.bert_for_global_context(
            input_ids_spc, token_type_ids, attention_mask)['last_hidden_state']

        batch_size, max_len, feat_dim = global_context_out.shape
        global_valid_output = torch.zeros(
            batch_size, max_len, feat_dim, dtype=torch.float32).to(self.args['device'])
        for i in range(batch_size):
            jj = -1
            for j in range(max_len):
                if valid_ids[i][j].item() == 1:
                    jj += 1
                    global_valid_output[i][jj] = global_context_out[i][j]
        global_context_out = self.dropout(
            global_valid_output)  # 全局Bert输出,Bert-base或者Bert-spc
        ate_logits = self.classifier(global_context_out)  # 全连接层进行分类,底层是Linear

        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=0)
            loss_ate = loss_fct(
                ate_logits.view(-1, self.num_labels), labels.view(-1))
            return loss_ate
        else:
            return ate_logits
