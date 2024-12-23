import validators

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F

from utils import Averager, CTCLabelConverter
from dataset import hierarchical_dataset, AlignCollate, ImgDataset
from models import Model
from utils import get_args, NED

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from tqdm import tqdm
def validation(model, criterion, evaluation_loader, converter, opt):
    """ validation or evaluation """
    char_n_correct = 0

    length_of_data = 0
    infer_time = 0
    valid_loss_avg = Averager()
    for i, (image_tensors, labels, imgs_path) in enumerate(tqdm(evaluation_loader, desc="Validation Progress")):
        batch_size = image_tensors.size(0)
        length_of_data = length_of_data + batch_size
        image = image_tensors.to(device)

        text_for_loss, length_for_loss = converter.encode(labels, batch_max_length=opt.batch_max_length)

        preds = model(image, is_eval=True)
        preds = preds[1]
        preds_size = torch.IntTensor([preds.size(1)] * batch_size)

        cost = criterion(preds.log_softmax(2).permute(1, 0, 2), text_for_loss, preds_size, length_for_loss)
        valid_loss_avg.add(cost)
        # 예측 결과 디코딩
        _, preds_index = preds.max(2)
        preds_str = converter.decode(preds_index.data, preds_size.data)
        # calculate accuracy & confidence score
        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        confidence_score_list = []

        NEDs = []
        for gt,pred,pred_max_prob in zip(labels, preds_str,preds_max_prob):
            pred_strip = pred.strip()
            gt_strip = gt.strip()
            if pred_strip == gt_strip:
                char_n_correct += 1
            try:
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]
            except:
                confidence_score = 0
            ned = NED(pred_strip, gt_strip)
            NEDs.append(ned)
            confidence_score_list.append(confidence_score)

    char_accuracy = char_n_correct/float(length_of_data) * 100
    
    ned_avg = sum(NEDs) / len(NEDs) if NEDs else 0
    return valid_loss_avg.val(), char_accuracy, preds_str, confidence_score_list, labels, infer_time, length_of_data, char_n_correct, ned_avg

def test(opt):
    """ model configuration """
    converter = CTCLabelConverter(opt.character)
    opt.num_class = len(converter.character)
    
    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)

    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)
    model = torch.nn.DataParallel(model).to(device)

    # load model
    print('loading pretrained model from %s' % opt.saved_model)

    if validators.url(opt.saved_model):
        model.load_state_dict(torch.hub.load_state_dict_from_url(opt.saved_model, progress=True, map_location=device))
    else:
        model.load_state_dict(torch.load(opt.saved_model, map_location=device))
    
    """ setup loss """
    criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    """ evaluation """
    model.eval()
    opt.eval = True
    with torch.no_grad():
        AlignCollate_evaluation = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD, opt=opt)
        eval_data, eval_data_log = hierarchical_dataset(root=opt.eval_data, opt=opt)
        eval_data = torch.utils.data.Subset(eval_data, range(int(len(eval_data))))
        evaluation_loader = torch.utils.data.DataLoader(
            eval_data, batch_size=opt.batch_size,
            shuffle=False,
            num_workers=int(opt.workers),
            collate_fn=AlignCollate_evaluation, pin_memory=True)
        
        char_list_accuracy = []
        total_forward_time = 0
        total_evaluation_data_number = 0
        char_total_correct_number = 0

        _, char_accuracy, preds_str, _, labels, infer_time, length_of_data, char_accur_num, ned = validation(model, criterion, evaluation_loader, converter, opt)
        char_list_accuracy.append(f'{char_accuracy:0.3f}')

        with open('valid_result.log', 'a') as log_file:
            for pred, gt in zip(preds_str, labels):
                correct = gt.lower() == pred.lower()
                log_file.write(f"GT: {gt} || Pred: {pred} || Correct: {correct}\n")

        total_forward_time += infer_time
        total_evaluation_data_number += len(eval_data)
        char_total_correct_number += char_accur_num

        print(f'char_Acc {char_accuracy:0.3f}')
        print(f'NED score: {1-ned:0.4f} // ned: {ned: 0.5f}')

if __name__ == '__main__':
    opt = get_args(is_train=False)

    """ vocab / character number configuration """
    # if opt.sensitive:
    #     opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).
    
    import pickle
    if 'pkl' in opt.character:
        with open(opt.character, 'rb') as f:
            extended_char = pickle.load(f)
        extended_char.extend(['±',' ','△','※','☑','☐','⓪','①','②','③','④','⑤','⑥','⑦','⑧','⑨','⑩','⑪','⑫','⑬','⑭','⑮','⑯','⑰','⑱','⑲','⑳','@'])
        opt.character = ''.join(extended_char)
    else:
        import ocr_dict
        opt.character = ocr_dict.all_chars
    
    
    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    opt.saved_model = opt.model_dir
    test(opt)
