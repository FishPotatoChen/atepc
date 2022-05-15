from copy import deepcopy


class NoAspectError(Exception):
    def __init__(self):
        self.value = "未检测到方面词！"

    def __str__(self):
        return self.value


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, sentence_label=None, aspect_label=None, polarity=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.sentence_label = sentence_label
        self.aspect_label = aspect_label
        self.polarity = polarity


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids_spc, input_mask, segment_ids, label_id, polarities=None, valid_ids=None, label_mask=None):
        self.input_ids_spc = input_ids_spc
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.valid_ids = valid_ids
        self.label_mask = label_mask
        self.polarities = polarities


def readfile(filename):
    '''
    read file
    '''
    f = open(filename, encoding='utf8')
    data = []
    sentence = []
    tag = []
    polarity = []
    for line in f:
        if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
            if len(sentence) > 0:
                data.append((sentence, tag, polarity))
                sentence = []
                tag = []
                polarity = []
            continue
        splits = line.split(' ')
        if len(splits) != 3:
            print('warning! detected error line(s) in input file:{}'.format(line))
        sentence.append(splits[0])
        tag.append(splits[-2])
        polarity.append(int(splits[-1][:-1]))
    if len(sentence) > 0:
        data.append((sentence, tag, polarity))
    return data


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        return readfile(input_file)


class INPUTProcessor(DataProcessor):
    def create_input_sentence_examples_for_ate(self, str):
        str_list = list(str)
        length = len(str_list)
        aspect_tag = ['O']*length
        aspect_polarity = [-1]*length
        return self._create_examples([(str_list, aspect_tag, aspect_polarity)], "test")

    def _create_examples(self, lines, set_type):
        '''
        text_a为句子
        text_b为方面词
        '''
        examples = []
        for i, (sentence, tag, polarity) in enumerate(lines):
            aspect = []
            aspect_tag = []
            aspect_polarity = [-1]
            for w, t, p in zip(sentence, tag, polarity):
                if p != -1:
                    aspect.append(w)
                    aspect_tag.append(t)
                    aspect_polarity.append(-1)
            guid = "%s-%s" % (set_type, i)
            text_a = sentence
            text_b = aspect
            polarity.extend(aspect_polarity)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, sentence_label=tag,
                                         aspect_label=aspect_tag, polarity=polarity))  # 将输入变为InputExample的列表
        return examples

    def create_input_sentence_examples_for_apc(self, str, sentence_label):
        # aspect_tag为一个列表，比如：['o','o','B-ASP','I-ASP','o'])
        str_list = list(str)
        length = len(str_list)
        polarity = [-1]*length
        aspect = []
        temp_asp = []
        aspect_label = []
        temp_tag = []
        flag = True
        for i in range(length):
            if flag and 'B' in sentence_label[i]:
                temp_asp.append(str_list[i])
                temp_tag.append(sentence_label[i])
                flag = False
            elif not flag:
                if 'I' in sentence_label[i]:
                    temp_asp.append(str_list[i])
                    temp_tag.append(sentence_label[i])
                else:
                    aspect.append(temp_asp)
                    temp_asp = []
                    aspect_label.append(temp_tag)
                    temp_tag = []
                    flag = True
        # 将方面词放入一个列表
        if temp_asp != []:
            aspect.append(temp_asp)
            aspect_label.append(temp_tag)
        examples = []
        # 无方面词错误
        if aspect == []:
            raise NoAspectError()
        all_aspect = []
        for i, text_b in enumerate(aspect):
            temp_sentence_label = []
            temp_i = i
            for j in sentence_label:
                if 'B' in j:
                    if temp_i == 0:
                        temp_sentence_label.extend(aspect_label[i])
                        break
                    temp_i -= 1
                else:
                    temp_sentence_label.append('O')
            # 屏蔽多余方面词
            temp_sentence_label.extend(
                ['O']*(len(sentence_label)-len(temp_sentence_label)))
            assert len(temp_sentence_label) == len(sentence_label)
            all_aspect.append(text_b)
            examples.append(InputExample(guid="%s-%s" % ("test", i), text_a=str_list,
                                         text_b=text_b, sentence_label=temp_sentence_label, aspect_label=aspect_label[i], polarity=polarity+[-1]*(len(text_b)+1)))
        return examples, all_aspect


def convert_examples_to_eval_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    label_map = {label: i for i, label in enumerate(label_list, 1)}
    features = []
    for example in examples:
        text_spc_tokens = deepcopy(example.text_a)
        aspect_tokens = deepcopy(example.text_b)
        sentence_label = deepcopy(example.sentence_label)
        aspect_label = deepcopy(example.aspect_label)
        polaritiylist = deepcopy(example.polarity)
        tokens = []
        labels = []
        polarities = []
        valid = []
        label_mask = []
        text_spc_tokens.extend(['[SEP]'])
        sentence_label.extend(['[SEP]'])
        text_spc_tokens.extend(aspect_tokens)
        sentence_label.extend(aspect_label)
        enum_tokens = text_spc_tokens
        label_lists = sentence_label
        for i, word in enumerate(enum_tokens):
            token = tokenizer.tokenize(word)  # 使用Bert的预训练模型
            tokens.extend(token)
            for m in range(len(token)):
                if m == 0:
                    # word是有效的
                    labels.append(label_lists[i])
                    polarities.append(polaritiylist[i])
                    valid.append(1)
                    label_mask.append(1)
                else:
                    valid.append(0)
        # 如果句子长度超过80,就截断句子,保持句子有效长度为78,前插入[CLS],后插入[SEP]
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            polarities = polarities[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
            valid = valid[0:(max_seq_length - 2)]
            label_mask = label_mask[0:(max_seq_length - 2)]

        ntokens = []
        segment_ids = []
        label_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        valid.insert(0, 1)
        label_mask.insert(0, 1)
        label_ids.append(label_map["[CLS]"])
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            if len(labels) > i:
                label_ids.append(label_map[labels[i]])
        ntokens.append("[SEP]")
        segment_ids.append(0)
        valid.append(1)
        label_mask.append(1)
        label_ids.append(label_map["[SEP]"])
        input_ids_spc = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids_spc)
        label_mask = [1] * len(label_ids)

        while len(input_ids_spc) < max_seq_length:
            input_ids_spc.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            valid.append(1)
            label_mask.append(0)
        while len(label_ids) < max_seq_length:
            label_ids.append(0)
            label_mask.append(0)
        while len(polarities) < max_seq_length:
            polarities.append(-1)
        assert len(input_ids_spc) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(valid) == max_seq_length
        assert len(label_mask) == max_seq_length

        features.append(
            InputFeatures(input_ids_spc=input_ids_spc,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_ids,
                          polarities=polarities,
                          valid_ids=valid,
                          label_mask=label_mask))  # 变为InputFeatures类的列表之后输出
    return features
