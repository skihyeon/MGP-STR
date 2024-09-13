import torch
import numpy as np
import argparse
from transformers import BertTokenizerFast, BertTokenizer, GPT2Tokenizer, PreTrainedTokenizerFast

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TokenLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, opt):
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        self.SPACE = '[s]'
        self.GO = '[GO]'

        self.list_token = [self.GO, self.SPACE]
        self.character = self.list_token + list(opt.character)

        self.dict = {word: i for i, word in enumerate(self.character)}
        self.batch_max_length = opt.batch_max_length + len(self.list_token)
        # self.bpe_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.bpe_tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2")
        # self.wp_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.wp_tokenizer = BertTokenizerFast.from_pretrained("kykim/bert-kor-base")

    def encode(self, text):
        """ convert text-label into text-index.
        """
        batch_text = torch.LongTensor(len(text), self.batch_max_length).fill_(self.dict[self.GO])
        for i, t in enumerate(text):
            txt = [self.GO] + list(t) + [self.SPACE]
            txt = [self.dict[char] for char in txt]
            batch_text[i][:len(txt)] = torch.LongTensor(txt)  # batch_text[:, 0] = [GO] token
        return batch_text.to(device)
    
    def char_encode(self, text):
        """ convert text-label into text-index.
        """
        batch_len = torch.LongTensor(len(text), 2).fill_(self.dict[self.GO])
        batch_text = torch.LongTensor(len(text), self.batch_max_length).fill_(self.dict[self.GO])
        for i, t in enumerate(text):
            length = len(t) 
            batch_len[i][1] = torch.LongTensor([length])  # batch_text[:, 0] = [GO] token
            
            txt = [self.GO] + list(t) + [self.SPACE]
            txt = [self.dict[char] for char in txt]
            batch_text[i][:len(txt)] = torch.LongTensor(txt)
            
        return batch_len.to(device), batch_text.to(device)

    def char_decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
        return texts
    
    def bpe_encode(self, text):
        batch_text = torch.LongTensor(len(text), self.batch_max_length).fill_(self.dict[self.GO])
        for i, t in enumerate(text):
            # token = self.bpe_tokenizer(t,max_length=self.batch_max_length-2, truncation=True)['input_ids']
            token = self.bpe_tokenizer(t)['input_ids']
            txt = [1] + token + [2]
            batch_text[i][:len(txt)] = torch.LongTensor(txt)
            
        return batch_text.to(device)
    
    # def bpe_encode(self, text):
    #     batch_text = torch.LongTensor(len(text), self.batch_max_length).fill_(self.dict[self.GO])
    #     for i, t in enumerate(text):
    #         # KoGPT-2 토크나이저를 사용하여 토큰화
    #         token = self.bpe_tokenizer.encode(t)
    #         txt = [self.dict[self.GO]] + token + [self.dict[self.SPACE]]
    #         batch_text[i][:len(txt)] = torch.LongTensor(txt)
    #     return batch_text.to(device)
    
    
    def bpe_decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            tokenstr = self.bpe_tokenizer.decode(text_index[index,:])
            texts.append(tokenstr)
        return texts

    def wp_encode(self, text):
        wp_target = self.wp_tokenizer(text,padding='max_length',max_length=self.batch_max_length,truncation=True,return_tensors="pt")
        return wp_target["input_ids"].to(device)
          
    def wp_decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            tokenstr = self.wp_tokenizer.decode(text_index[index,:])
            tokenlist = tokenstr.split()
            texts.append(''.join(tokenlist))
        return texts


class CTCLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character):
        # character (str): set of the possible characters.
        dict_character = list(character)

        self.dict = {}
        for i, char in enumerate(dict_character):
            # NOTE: 0 is reserved for 'CTCblank' token required by CTCLoss
            self.dict[char] = i + 1

        self.character = ['[CTCblank]'] + dict_character  # dummy '[CTCblank]' token for CTCLoss (index 0)

    def encode(self, text, batch_max_length=25):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default

        output:
            text: text index for CTCLoss. [batch_size, batch_max_length]
            length: length of each text. [batch_size]
        """
        length = [len(s) for s in text]

        # The index used for padding (=0) would not affect the CTC loss calculation.
        batch_text = torch.LongTensor(len(text), batch_max_length).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text = [self.dict[char] for char in text]
            batch_text[i][:len(text)] = torch.LongTensor(text)
        return (batch_text.to(device), torch.IntTensor(length).to(device))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            t = text_index[index, :]

            char_list = []
            for i in range(l):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):  # removing repeated characters and blank.
                    char_list.append(self.character[t[i]])
            text = ''.join(char_list)

            texts.append(text)
        return texts

class Averager(object):
    """Compute average for torch.Tensor, used for loss average."""

    def __init__(self):
        self.reset()

    def add(self, v):
        count = v.data.numel()
        v = v.data.sum()
        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res


def NED(s1, s2):
    def levenshtein(s1, s2, cost=None):
        if len(s1) < len(s2):
            s1, s2 = s2, s1

        if len(s2) == 0:
            return len(s1)

        if cost is None:
            cost = {}

        def substitution_cost(c1, c2):
            return 0 if c1 == c2 else cost.get((c1, c2), 1)

        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + substitution_cost(c1, c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]
    def decompose(c):
        if not character_is_korean(c):
            return None
        i = ord(c)
        if 0x3131 <= i <= 0x3146 or 0x3147 <= i <= 0x315B:
            return (c, ' ', ' ') if 0x3131 <= i <= 0x3146 else (' ', c, ' ')

        i -= 0xAC00
        cho  = chr((i // 0x24C) + 0x1100)
        jung = chr(((i % 0x24C) // 0x1C) + 0x1161)
        jong = chr(((i % 0x24C) % 0x1C) + 0x11A7) if ((i % 0x24C) % 0x1C) != 0 else ' '
        return (cho, jung, jong)

    def character_is_korean(c):
        i = ord(c)
        return (0xAC00 <= i <= 0xD7A3) or (0x3131<= i <= 0x3146) or (0x3147 <= i <= 0x315B)
    
    def cal_ned(s1, s2):
        s1_korean = re.sub('[^가-힣]', '', s1)
        s2_korean = re.sub('[^가-힣]', '', s2)
        s1_non_korean = re.sub('[가-힣]', '', s1)
        s2_non_korean = re.sub('[가-힣]', '', s2)
        
        if not s1_korean or not s2_korean:
            ned_korean = 0
        else:
            decompose_s1_korean = ''.join(comp for c in s1_korean for comp in decompose(c))
            decompose_s2_korean = ''.join(comp for c in s2_korean for comp in decompose(c))
            max_len_korean = max(len(s1_korean), len(s2_korean))
            ned_korean = (levenshtein(decompose_s1_korean, decompose_s2_korean) / 3) / max_len_korean
        
        if not s1_non_korean or not s2_non_korean:
            ned_non_korean = 0
        else:
            max_len_non_korean = max(len(s1_non_korean), len(s2_non_korean))
            ned_non_korean = levenshtein(s1_non_korean, s2_non_korean) / max_len_non_korean
        
        ned = (ned_korean + ned_non_korean) / 2
        return ned
    
    ned = cal_ned(s1, s2)
    return ned


def get_device(verbose=True):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if verbose:
        print("Device:", device)
    return device


def get_args(is_train=True):
    parser = argparse.ArgumentParser(description='STR')

    # for test
    parser.add_argument('--eval_data', help='path to evaluation dataset')
    parser.add_argument('--benchmark_all_eval', action='store_true', help='evaluate 10 benchmark evaluation datasets')
    parser.add_argument('--calculate_infer_time', action='store_true', help='calculate inference timing')
    parser.add_argument('--flops', action='store_true', help='calculates approx flops (may not work)')

    # for train
    parser.add_argument('--exp_name', help='Where to store logs and models')
    parser.add_argument('--train_data', required=is_train, help='path to training dataset')
    parser.add_argument('--valid_data', required=is_train, help='path to validation dataset')
    parser.add_argument('--manualSeed', type=int, default=1111, help='for random seed setting')
    parser.add_argument('--workers', type=int, help='number of data loading workers. Use -1 to use all cores.', default=4)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--num_iter', type=int, default=300000, help='number of iterations to train for')
    parser.add_argument('--valInterval', type=int, default=2000, help='Interval between each validation')
    parser.add_argument('--saved_model', default='', help="path to model to continue training")
    parser.add_argument('--saved_path', default='./saved_models', help="path to save")
    parser.add_argument('--FT', action='store_true', help='whether to do fine-tuning')
    parser.add_argument('--sgd', action='store_true', help='Whether to use SGD (default is Adadelta)')
    parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is Adadelta)')
    parser.add_argument('--lr', type=float, default=1, help='learning rate, default=1.0 for Adadelta')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
    parser.add_argument('--rho', type=float, default=0.95, help='decay rate rho for Adadelta. default=0.95')
    parser.add_argument('--eps', type=float, default=1e-8, help='eps for Adadelta. default=1e-8')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping value. default=5')
    """ Data processing """
    parser.add_argument('--select_data', type=str, default='MJ-ST',
                        help='select training data (default is MJ-ST, which means MJ and ST used as training data)')
    parser.add_argument('--batch_ratio', type=str, default='0.5-0.5',
                        help='assign ratio for each selected data in the batch')
    parser.add_argument('--total_data_usage_ratio', type=str, default='1.0',
                        help='total data usage ratio, this ratio is multiplied to total number of data.')
    # parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--batch_max_length', type=int, default=50, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=128, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str,
                        default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    parser.add_argument('--data_filtering_off', action='store_true', help='for data_filtering_off mode')
    
    """ Model Architecture """
    parser.add_argument('--Transformer', type=str, required=True, help='Transformer stage. mgp-str|char-str')

    choices = ["mgp_str_base_patch4_3_32_128", "mgp_str_large_patch4_3_32_128", "mgp_str_tiny_patch4_3_32_128", 
                "mgp_str_small_patch4_3_32_128", "char_str_base_patch4_3_32_128", "char_str_large_patch8_1_32_224", "char_str_custom"]
    parser.add_argument('--TransformerModel', default='', help='Which mgp_str transformer model', choices=choices)
    parser.add_argument('--Transformation', type=str, default='', help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, default='',
                        help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, default='', help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, default='', help='Prediction stage. None|CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=3,
                        help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

    # selective augmentation 
    # can choose specific data augmentation
    parser.add_argument('--issel_aug', action='store_true', help='Select augs')
    parser.add_argument('--sel_prob', type=float, default=1., help='Probability of applying augmentation')
    parser.add_argument('--pattern', action='store_true', help='Pattern group')
    parser.add_argument('--warp', action='store_true', help='Warp group')
    parser.add_argument('--geometry', action='store_true', help='Geometry group')
    parser.add_argument('--weather', action='store_true', help='Weather group')
    parser.add_argument('--noise', action='store_true', help='Noise group')
    parser.add_argument('--blur', action='store_true', help='Blur group')
    parser.add_argument('--camera', action='store_true', help='Camera group')
    parser.add_argument('--process', action='store_true', help='Image processing routines')

    # use cosine learning rate decay
    parser.add_argument('--scheduler', action='store_true', help='Use lr scheduler')

    parser.add_argument('--intact_prob', type=float, default=0.5, help='Probability of not applying augmentation')
    parser.add_argument('--isrand_aug', action='store_true', help='Use RandAug')
    parser.add_argument('--augs_num', type=int, default=3, help='Number of data augment groups to apply. 1 to 8.')
    parser.add_argument('--augs_mag', type=int, default=None, help='Magnitude of data augment groups to apply. None if random.')

    # for comparison to other augmentations
    parser.add_argument('--issemantic_aug', action='store_true', help='Use Semantic')
    parser.add_argument('--isrotation_aug', action='store_true', help='Use ')
    parser.add_argument('--isscatter_aug', action='store_true', help='Use ')
    parser.add_argument('--islearning_aug', action='store_true', help='Use ')

    # orig paper uses this for fast benchmarking
    parser.add_argument('--fast_acc', action='store_true', help='Fast average accuracy computation')
   
    parser.add_argument("--local-rank", type=int, default=-1)
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    
    # mask train
    parser.add_argument('--mask_ratio', default=0.5, type=float,
                        help='ratio of the visual tokens/patches need be masked')
    parser.add_argument("--patch_size", type=int, default=4)

    # for eval
    parser.add_argument('--eval_img', action='store_true', help='eval imgs dataset')
    parser.add_argument('--range', default=None, help="start-end for example(800-1000)")
    parser.add_argument('--model_dir', default='') 
    parser.add_argument('--demo_imgs', default='')
    

    parser.add_argument('--wandb', action='store_true')

    args = parser.parse_args()
    return args
