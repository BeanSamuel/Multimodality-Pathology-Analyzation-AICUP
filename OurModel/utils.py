import os
import csv
import logging
import random
import sys
import numpy as np
import requests
from random import shuffle
import pandas as pd
from sklearn.model_selection import train_test_split
import torchaudio
import torch
import math
import librosa

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

from pathlib import Path
from urllib.parse import urlparse

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
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
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


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
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines
        
class PgProcessor(DataProcessor):
    """Processor for the PG data set."""

    def get_train_examples(self, df):
        """See base class."""
        train_text_dataset = self.datatotext(df)
        return self._create_examples(
            train_text_dataset, "train")

    def get_dev_examples(self, df):
        """See base class."""
        dev_text_dataset = self.datatotext(df)
        return self._create_examples(
            dev_text_dataset, "dev")

    def get_test_examples(self, df):
        """See base class."""
        test_text_dataset = self.datatotext(df)
        return self._create_examples(
            test_text_dataset, "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1","2","3","4"]
        
    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    def datatotext(self,df):
        t = []
        sex_table = ['male', 'female']
        for i in range(len(df)):
            s = ""
            uid = df['ID'][i]
            sex = sex_table[(int)(df['Sex'][i])-1]
            age = (int)(df['Age'][i])
            Subject = 'He' if sex == 'male' else 'She'
            Object = 'His' if sex == 'male' else 'Her'
            s += f'{uid} is a {age} years old {sex}.'
            if (int)(df['Narrow pitch range'][i]) == 1:
                s += f'{Subject} feels the range of pitch is narrow '
            if (int)(df['Decreased volume'][i]) == 1:
                s += f'{Object} volume is decreased '
            if (int)(df['Fatigue'][i]) == 1:
                s += f'{Subject} feels fatigue easily when he talks a lot '
            if (int)(df['Dryness'][i]) == 1:
                s += f'{Object} throat often feels dry '
            if (int)(df['Lumping'][i]) == 1:
                s += f'{Subject} has a foreign body sensation or lump in {Object} throat '
            if (int)(df['heartburn'][i]) == 1:
                s += f'{Subject} has a burning sensation in {Object} chest '
            if (int)(df['Choking'][i]) == 1:
                s += f'{Subject} chokes easily '
            if (int)(df['Eye dryness'][i]) == 1:
                s += f'{Object} eyes are dry '
            if (int)(df['PND'][i]) == 1:
                s += f'{Subject} has Post-nasal drip '
            if (int)(df['Diabetes'][i]) == 1:
                s += f'{Subject} has diabetes '
            if (int)(df['Hypertension'][i]) == 1:
                s += f'{Subject} has hypertension '
            if (int)(df['CAD'][i]) == 1:
                s += f'{Subject} has cardiovascular disease '
            if (int)(df['Head and Neck Cancer'][i]) == 1:
                s += f'{Subject} has Head and Neck Cancer '
            if (int)(df['Head injury'][i]) == 1:
                s += f'{Subject} suffered a head injury '
            if (int)(df['CVA'][i]) == 1:
                s += f'{Subject} has haemorrhagic stroke '
            if (int)(df['Smoking'][i]) == 0:
                s += f'{Subject} never smoking '
            elif (int)(df['Smoking'][i]) == 1:
                s += f'{Subject} was smoking past '
            elif (int)(df['Smoking'][i]) == 2:
                s += f'{Subject} is smoking active '
            else:
                s += f'{Subject} is smoking e-cigarette '
            if not pd.isna(df["PPD"][i]):
                s += f'{Subject} smokes {(int)(df["PPD"][i])} pack of cigarette per day '
            if (int)(df['Drinking'][i]) == 0:
                s += f'{Subject} never drinking '
            elif (int)(df['Drinking'][i]) == 1:
                s += f'{Subject} was drinking past '
            else: 
                s += f'{Subject} is drinking active '
            if (int)(df['frequency'][i]) == 0:
                s += f'{Subject} does not drinking '
            elif (int)(df['frequency'][i]) == 1:
                s += f'{Subject} drinking occasionally '
            elif (int)(df['frequency'][i]) == 2:
                s += f'{Subject} drinking weekly '    
            else:
                s += f'{Subject} drinking daily '
            if (int)(df['Onset of dysphonia '][i]) == 1:
                s += f'{Subject} onset of dysphonia sudden '
            elif (int)(df['Onset of dysphonia '][i]) == 2:        
                s += f'{Subject} onset of dysphonia gradually '
            elif (int)(df['Onset of dysphonia '][i]) == 3:    
                s += f'{Subject} onset of dysphonia on and off '
            elif (int)(df['Onset of dysphonia '][i]) == 4:    
                s += f'{Subject} onset of dysphonia since childhood '
            else:
                s += f'{Subject} onset of dysphonia other '
            if (int)(df['Noise at work'][i]) == 1:
                s += f'{Object} working environment is not noisy '
            elif (int)(df['Noise at work'][i]) == 2:       
                s += f'{Object} working environment is a little noisy ' 
            else:
                s += f'{Object} working environment is very noisy '
            if (int)(df['Diurnal pattern'][i]) == 1:
                s += f'{Object} voice worse in the morning '
            elif (int)(df['Diurnal pattern'][i]) == 2:
                s += f'{Object} voice worse in the afternoon '
            elif (int)(df['Diurnal pattern'][i]) == 3:
                s += f'{Object} voice as worse as similar all day '
            else:
                s += f'{Object} voice condition was fluctuating '
            if (int)(df['Occupational vocal demand'][i]) == 1:
                s += f'{Subject} use {Object} vocal occupational always ' 
            elif (int)(df['Occupational vocal demand'][i]) == 2:        
                s += f'{Subject} use {Object} vocal occupational frequently '
            elif (int)(df['Occupational vocal demand'][i]) == 3:
                s += f'{Subject} use {Object} vocal occupational occasional '
            else:
                s += f'{Subject} use {Object} vocal occupational minimal '
            if not (pd.isna(df["Voice handicap index - 10"][i])):
                s += f'{Object} voice handicap index is {(int)(df["Voice handicap index - 10"][i])}.'
            t.append([s.lower(),str(df['Disease category'][i])])
        return t
    
    



def accuracy_7(out, labels):
    return np.sum(np.round(out) == np.round(labels)) / float(len(labels))

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}
    features = []
    for (ex_index, example) in enumerate(examples):

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        # label_id = label_map[example.label]
        label_id = example.label
        label_id = float(label_id)
        if ex_index < 0:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %s)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop(0) #For dialogue context
        else:
            tokens_b.pop()


def accuracy(out, labels):
    return (np.argmax(out, axis=-1)==labels).sum()


# def accuracy(out, labels):
#     outputs = np.argmax(out, axis=1)
#     return np.sum(outputs == labels)


def F1_score(out):
    outputs = np.argmax(out, axis=1)
    return outputs

def pearson(vector1, vector2):
    n = len(vector1)
    #simple sums
    sum1 = sum(float(vector1[i]) for i in range(n))
    sum2 = sum(float(vector2[i]) for i in range(n))
    #sum up the squares
    sum1_pow = sum([pow(v, 2.0) for v in vector1])
    sum2_pow = sum([pow(v, 2.0) for v in vector2])
    #sum up the products
    p_sum = sum([vector1[i]*vector2[i] for i in range(n)])
    #分子num，分母den
    num = p_sum - (sum1*sum2/n)
    den = math.sqrt((sum1_pow-pow(sum1, 2)/n)*(sum2_pow-pow(sum2, 2)/n))
    if den == 0:
        return 0.0
    return num/den

class AudioUtil():
    def open(audio_fail):# -> sig: Tensor, sr: int
        sig, sr = torchaudio.load(audio_fail)
        return (sig, sr)
    
    def rechannel(aud, new_che):
        sig, sr = aud

        if sig.shape[0] == new_che:
            return (sig, sr)
        if new_che == 1:
            sig = sig[:1]
        if new_che == 2:
            sig = torch.cat((sig, sig), 0)

        return (sig, sr)
    
    def resample(aud, new_sr):
        sig, sr = aud
        sig = torchaudio.transforms.Resample(sr, new_sr)(sig)
        return (sig, new_sr)
    
    def pad_trunc(aud, max_len):
        sig, sr = aud

        wav_che, wav_len = sig.shape
        max_len = int(sr * max_len / 1000)

        if wav_len >= max_len:
            sig = sig[:, :max_len]
        else:
            pad_f = random.randint(0, max_len - wav_len)
            pad_b = max_len - wav_len - pad_f
            sig = torch.cat((torch.zeros(wav_che, pad_f), sig, torch.zeros(wav_che, pad_b)), 1)
        
        return (sig, sr)
    
    def time_shift(aud, shi_lim):
        sig, sr = aud
        sig_rot = int(random.random() * shi_lim * sig.shape[1])
        return (sig.roll(sig_rot), sr)
    
    def spectro_gram(aud, n_mels=128, n_fft=1024, hop_len=None):
        sig, sr = aud

        sig = torchaudio.transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)
        sig = torch.tensor(librosa.power_to_db(sig.numpy(), ref=np.max))
        # sig = torchaudio.transforms.AmplitudeToDB(top_db=top_db)(sig)

        return (sig, sr)