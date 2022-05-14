import os
from tokenize import String
from random import randint


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


class drawProcessor(DataProcessor):
    """Processor for the CoNLL-2003 data set."""
    # 得到训练数据

    def get_train_examples(self, data_dir):
        """See base class."""
        if 'car' in data_dir:
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "car.atepc.train.dat")), "cartrain")
        elif 'phone' in data_dir:
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "phone.atepc.train.dat")), "phonetrain")
        elif 'camera' in data_dir:
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "camera.atepc.train.dat")), "cameratrain")
        elif 'notebook' in data_dir:
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "notebook.atepc.train.dat")), "notebooktrain")

    # 得到测试数据
    def get_test_examples(self, data_dir):
        """See base class."""
        if 'car' in data_dir:
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "car.atepc.test.dat")), "cartest")
        elif 'phone' in data_dir:
            return self._create_examples(self._read_tsv(os.path.join(data_dir, "phone.atepc.test.dat")), "phonetest")
        elif 'camera' in data_dir:
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "camera.atepc.test.dat")), "cameratest")
        elif 'notebook' in data_dir:
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "notebook.atepc.test.dat")), "notebooktest")

    def _create_examples(self, lines, set_type):
        with open(set_type+"length.txt", "w") as f:
            for i, (sentence, tag, polarity) in enumerate(lines):
                f.write(str(len(sentence))+',')