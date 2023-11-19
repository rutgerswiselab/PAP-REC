import sys

from tqdm import tqdm
import numpy as np
import logging
import random

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import T5Tokenizer, T5TokenizerFast
from src.tokenization import P5Tokenizer, P5TokenizerFast
import argparse
from evaluate.metrics4rec import *

class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self


from src.all_yelp_templates import all_tasks as task_templates

def main(args):   
    # args = DotDict()
    args.distributed = False
    args.num_workers = 4
    args.gen_max_length = 64
    # args.dataset = 'beauty'
    args.train = args.valid = args.test = args.dataset
    args.batch_size = 16
    # args.model_size = 'small'
    args.backbone = 't5-' + args.model_size # small or base
    args.output = 'snap/' + args.dataset + '-' + args.model_size
    args.tokenizer = 'p5'
    args.max_text_length = 512
    args.do_lower_case = False
    args.dropout = 0.1
    args.losses = 'rating,sequential,explanation,review,traditional'
    '''
    Set seeds
    '''
    args.seed = 0
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    gpu = 0 # Change GPU ID
    args.gpu = gpu
    args.rank = gpu
    print(f'Process Launching at GPU {gpu}')

    torch.cuda.set_device('cuda:{}'.format(gpu))

    from src.pretrain_model import P5Pretraining

    def create_config(args):
        from transformers import T5Config, BartConfig

        if 't5' in args.backbone:
            config_class = T5Config
        else:
            return None

        config = config_class.from_pretrained(args.backbone)
        config.dropout_rate = args.dropout
        config.dropout = args.dropout
        config.attention_dropout = args.dropout
        config.activation_dropout = args.dropout
        config.losses = args.losses

        return config

    def create_tokenizer(args):
        from transformers import T5Tokenizer, T5TokenizerFast
        from src.tokenization import P5Tokenizer, P5TokenizerFast

        if 'p5' in args.tokenizer:
            tokenizer_class = P5Tokenizer

        tokenizer_name = args.backbone
        
        tokenizer = tokenizer_class.from_pretrained(
            tokenizer_name,
            max_length=args.max_text_length,
            do_lower_case=args.do_lower_case,
        )

        print(tokenizer_class, tokenizer_name)
        
        return tokenizer

    def create_model(model_class, config=None):
        print(f'Building Model at GPU {args.gpu}')

        model_name = args.backbone

        model = model_class.from_pretrained(
            model_name,
            config=config
        )
        return model

    from torch.utils.data import DataLoader, Dataset, Sampler
    from src.pretrain_data import get_loader
    from evaluate.utils import rouge_score, bleu_score, unique_sentence_percent, root_mean_square_error, mean_absolute_error, feature_detect, feature_matching_ratio, feature_coverage_ratio, feature_diversity
    


    # Load Model and Tokenizer

    config = create_config(args)

    if args.tokenizer is None:
        args.tokenizer = args.backbone
        
    tokenizer = create_tokenizer(args)

    model_class = P5Pretraining
    model = create_model(model_class, config)

    model = model.cuda()

    if 'p5' in args.tokenizer:
        model.resize_token_embeddings(tokenizer.vocab_size)
        
    model.tokenizer = tokenizer

    args.load = "snap/" + args.dataset + "-" + args.model_size + ".pth"

    # Load Checkpoint
    from src.utils import load_state_dict, LossMeter, set_global_logging_level
    from pprint import pprint

    def load_checkpoint(ckpt_path):
        state_dict = load_state_dict(ckpt_path, 'cpu')
        results = model.load_state_dict(state_dict, strict=False)
        print('Model loaded from ', ckpt_path)
        pprint(results)

    ckpt_path = args.load
    load_checkpoint(ckpt_path)

    # Pre-define Functions & Variables

    def load_dataloader(my_task_templates, mode='train'):
        test_task_list = {task: [prompt_id]}
        if mode == 'train':
            test_sample_numbers = {'rating': 1, 'sequential': (5, 5, 10), 'explanation': 1, 'review': 1, 'traditional': (10, 5)}
        else:
            test_sample_numbers = {'rating': 1, 'sequential': (1, 1, 1), 'explanation': 1, 'review': 1, 'traditional': (1, 1)}
        if 't5' in args.backbone:
            tokenizer = P5Tokenizer.from_pretrained(
                args.backbone, 
                max_length=args.max_text_length, 
                do_lower_case=args.do_lower_case)

        from src.pretrain_data import P5_Yelp_Dataset, P5_Amazon_Dataset
        if args.dataset == 'yelp':
            dataset = P5_Yelp_Dataset(
                my_task_templates,
                test_task_list,
                tokenizer,
                args,
                test_sample_numbers,
                mode=mode,
                split=args.train,
                rating_augment=False
            )
        else:
            dataset = P5_Amazon_Dataset(
                my_task_templates,
                test_task_list,
                tokenizer,
                args,
                test_sample_numbers,
                mode=mode,
                split=args.train,
                rating_augment=False
            )

        if args.distributed:
            sampler = DistributedSampler(dataset)
        else:
            sampler = None

        if mode == 'train':
            zeroshot_test_loader = DataLoader(
                dataset, batch_size=args.batch_size, shuffle=(sampler is None),
                num_workers=args.num_workers, pin_memory=True, sampler=sampler,
                collate_fn=dataset.collate_fn)
        else:
            zeroshot_test_loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                num_workers=args.num_workers, pin_memory=True,
                sampler=sampler,
                shuffle=None if (sampler is not None) else False,
                collate_fn=dataset.collate_fn,
                drop_last=False)
        print(len(zeroshot_test_loader))
        return zeroshot_test_loader

    def revise_batch(batch, prompt_prefix=None, prompt_suffix=None, dynamic_prompt=None):
        if prompt_prefix is not None:
            batch['input_ids'][:, :prefix_words] = prompt_prefix
        if prompt_suffix is not None:
            end_x, end_y = torch.nonzero(batch['input_ids'] == 1, as_tuple=True)
            for k in range(len(batch['input_ids'])):
                batch['input_ids'][k] = batch['input_ids'][k].slice_scatter(prompt_suffix, start=end_y[k] - suffix_head_from_end,
                                                                            end=end_y[k] - suffix_tail_from_end)
                if dynamic_prompt is not None:
                    batch['input_ids'][k] = batch['input_ids'][k].slice_scatter(dynamic_prompt[batch['user_id'][k]], 
                                                                                start=end_y[k] - dynamic_head_from_end,
                                                                                end=end_y[k] - dynamic_index_from_end)
                    # batch['input_ids'][k][end_y[k] - dynamic_index_from_end] = dynamic_prompt[batch['user_id'][k]]
        return batch

    def get_approximate_metric_batch(batch, prompt_prefix=None, prompt_suffix=None, dynamic_prompt=None, topk=5):
        with torch.no_grad():
            ndcg, hit = [], []
            batch = revise_batch(batch, prompt_prefix=prompt_prefix, prompt_suffix=prompt_suffix, dynamic_prompt=dynamic_prompt)
            beam_outputs = model.generate(
                    batch['input_ids'].to('cuda'), 
                    max_length=(batch['target_ids'].shape[-1] + 1), 
                    num_beams=20,
                    no_repeat_ngram_size=0, 
                    num_return_sequences=topk,
                    early_stopping=True
            )
            assert torch.sum(beam_outputs, dim=0)[0] == 0
            target_ids = batch['target_ids'].to(device)
            target_ids[target_ids < 0] = 0
            for j, target_id in enumerate(target_ids):
                beam_output = beam_outputs[j*topk:(j+1)*topk, 1:] # remove first 0
                width = beam_output.shape[-1]
                if target_id.shape[-1] == width or torch.sum(target_id[(width + 1):]) == 0:
                    diff = torch.sum((beam_output != target_id[:width]).int(), dim=-1)
                    rel = (diff == 0).int().cpu()
                    ndcg.append(ndcg_at_k(rel, topk, 1))
                    hit.append(hit_at_k(rel, topk))
                else:
                    ndcg.append(0.0)
                    hit.append(0.0)
        return ndcg, hit

    def exp_approximate_metric_batch(batch, prompt_prefix=None, prompt_suffix=None, dynamic_prompt=None, topk=5):
        with torch.no_grad():
            tokens_predict = []
            tokens_test = []
            batch = revise_batch(batch, prompt_prefix=prompt_prefix, prompt_suffix=prompt_suffix, dynamic_prompt=dynamic_prompt)
            outputs = model.generate(
                batch['input_ids'].to('cuda'), 
                min_length=10
            )
            results = model.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            BLEU4 = [bleu_score([ll.split()], [l.split()], n_gram=4, smooth=False) for l, ll in zip(results, batch['target_text'])]
            ROUGE = [0.1 * rouge_score([ll], [l])['rouge_l/f_score'] for l, ll in zip(results, batch['target_text'])]
        return BLEU4, ROUGE

    def get_approximate_metric(eval_loader, prompt_prefix=None, prompt_suffix=None, dynamic_prompt=None, topk=5, task='traditional'):
        with torch.no_grad():
            batch_eval = exp_approximate_metric_batch if task == 'explanation' else get_approximate_metric_batch
            avg_metric_1, avg_metric_2, cnt = 0.0, 0.0, 0
            for i, batch in tqdm(enumerate(eval_loader)):
                metric_1, metric_2 = batch_eval(batch, prompt_prefix=prompt_prefix, prompt_suffix=prompt_suffix, dynamic_prompt=dynamic_prompt)
                cnt += len(metric_1)
                avg_metric_1 += sum(metric_1)
                avg_metric_2 += sum(metric_2)
        return avg_metric_1 / cnt, avg_metric_2 / cnt

    def seq_prompt_eval(eval_loader, prompt_prefix=None, prompt_suffix=None, dynamic_prompt=None):
        global task
        all_info = []
        for i, batch in tqdm(enumerate(eval_loader)):
            with torch.no_grad():
                batch = revise_batch(batch, prompt_prefix, prompt_suffix, dynamic_prompt)
                results = model.generate_step(batch)
                beam_outputs = model.generate(
                        batch['input_ids'].to('cuda'), 
                        max_length=50, 
                        num_beams=20,
                        no_repeat_ngram_size=0, 
                        num_return_sequences=10,
                        early_stopping=True
                )
                generated_sents = model.tokenizer.batch_decode(beam_outputs, skip_special_tokens=True)
                for j, item in enumerate(zip(results, batch['target_text'], batch['source_text'])):
                    new_info = {}
                    new_info['target_item'] = item[1]
                    new_info['gen_item_list'] = generated_sents[j*10: (j+1)*10]
                    all_info.append(new_info)

        gt = {}
        ui_scores = {}
        for i, info in enumerate(all_info):
            gt[i] = [int(info['target_item'])]
            pred_dict = {}
            for j in range(len(info['gen_item_list'])):
                try:
                    pred_dict[int(info['gen_item_list'][j])] = -(j+1)
                except:
                    pass
            ui_scores[i] = pred_dict
        
        if task == 'traditional':
            print(evaluate_all(ui_scores, gt, 1))
        print(evaluate_all(ui_scores, gt, 5))
        print(evaluate_all(ui_scores, gt, 10))

    def exp_prompt_eval(eval_loader, prompt_prefix=None, prompt_suffix=None, dynamic_prompt=None):
        tokens_predict = []
        tokens_test = []
        for i, batch in tqdm(enumerate(eval_loader)):
            with torch.no_grad():
                batch = revise_batch(batch, prompt_prefix=prompt_prefix, prompt_suffix=prompt_suffix, dynamic_prompt=dynamic_prompt)
                outputs = model.generate(
                    batch['input_ids'].to('cuda'), 
                    min_length=10
                )
                results = model.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                tokens_predict.extend(results) 
                tokens_test.extend(batch['target_text'])
                
        new_tokens_predict = [l.split() for l in tokens_predict]
        new_tokens_test = [ll.split() for ll in tokens_test]
        BLEU1 = bleu_score(new_tokens_test, new_tokens_predict, n_gram=1, smooth=False)
        BLEU4 = bleu_score(new_tokens_test, new_tokens_predict, n_gram=4, smooth=False)
        ROUGE = rouge_score(tokens_test, tokens_predict)
        
        print('BLEU-1 {:7.4f}'.format(BLEU1))
        print('BLEU-4 {:7.4f}'.format(BLEU4))
        for (k, v) in ROUGE.items():
            print('{} {:7.4f}'.format(k, v))

    def dfs_freeze(model):
        for name, child in model.named_children():
            for param in child.parameters():
                param.requires_grad = False
            dfs_freeze(child)


    task = args.task # = 'sequential' # 'explanation' # 'traditional'
    test_func = exp_prompt_eval if task == 'explanation' else seq_prompt_eval
    # prompt_id = args.prompt_id = '2-3' # '3-11' # '5-7'
    if task == 'sequential':
        prompt_id = args.prompt_id = '2-3'
    elif task == 'explanation':
        prompt_id = args.prompt_id = '3-11'
    else:
        prompt_id = args.prompt_id = '5-7'

    from copy import deepcopy
    prefix_words = args.prefix_words = 0
    postfix_words = args.postfix_words = 5

    # adjust dynamic token position
    # dynamic_index_from_begin = args.dynamic_index_from_begin = 1
    dynamic_length = args.dynamic_length # = 1
    if dynamic_length > 0:
        dynamic_index_from_end = args.dynamic_index_from_end = 0
        dynamic_head_from_end = dynamic_index_from_end + dynamic_length
    suffix_head_from_end = args.suffix_head_from_end = dynamic_length + postfix_words
    suffix_tail_from_end = args.suffix_tail_from_end = dynamic_length

    my_task_templates = {task: {prompt_id: deepcopy(task_templates[task][prompt_id])}}
    key_info = 'user_{} : \n {} \n'
    print(task_templates[task][prompt_id])

    test_loader = load_dataloader(my_task_templates, mode='test')
    single_word = [0] * len(test_loader.dataset.tokenizer.get_vocab())
    for v, key in test_loader.dataset.tokenizer.get_vocab().items():
        if not v.startswith('▁'):
            single_word[key] = -1e32
    single_word = torch.tensor(single_word)
    user_cnt = len(test_loader.dataset.user_id2name) + 1  # user_id index starting from 1
    single_word_index = (single_word > -1).nonzero(as_tuple=True)[0]
    ## dynamic prompt init
    dynamic_prompt = single_word_index[torch.randint(0, len(single_word_index), (user_cnt, dynamic_length))] if dynamic_length > 0 else None
    device = torch.device('cuda:{}'.format(gpu))
    single_word = single_word.to(device)
    single_word.shape

    # # Test Manual Prompt 

    dfs_freeze(model)
    model.eval()
    with torch.no_grad():
        # exp_prompt_eval(test_loader)
        test_func(test_loader)

    # # Auto Prompt Init

    ## prompt random init
    prompt_prefix = single_word_index[torch.randint(0, len(single_word_index), (prefix_words, ))].to(device) if prefix_words > 0 else None
    prompt_suffix = single_word_index[torch.randint(0, len(single_word_index), (postfix_words + dynamic_length, ))].to(device) if postfix_words > 0 else None
    decoder_prefix = model.tokenizer.decode(prompt_prefix, skip_special_tokens=True).strip() if prompt_prefix is not None else ''
    decoder_suffix = model.tokenizer.decode(prompt_suffix, skip_special_tokens=True).strip() if prompt_suffix is not None else ''
    my_task_templates = {task: {prompt_id: deepcopy(task_templates[task][prompt_id])}}
    my_task_templates[task][prompt_id]['source'] = ' '.join([decoder_prefix, key_info, decoder_suffix]).strip()
    print(my_task_templates)
    train_loader = load_dataloader(my_task_templates, mode='train')
    eval_loader = load_dataloader(my_task_templates, mode='val')
    test_loader = load_dataloader(my_task_templates, mode='test')
    # need to revise if prompt suffix is not the end
    prompt_suffix = prompt_suffix[-suffix_head_from_end:-suffix_tail_from_end] if dynamic_length > 0 else prompt_suffix
    print(prompt_prefix, prompt_suffix)

    best_eval_loss = 0.0
    with torch.no_grad():
        metric_1, metric_2 = get_approximate_metric(eval_loader, task=task)
        if task == 'explanation':
            print('Approximated BLEU-4: %.6lf, ROUGE-L: %.6lf' % (metric_1, metric_2 * 10))
        else:
            print('Approximated NDCG@5: %.6lf, Hit@5: %.6lf' % (metric_1, metric_2))
        best_metric = metric_1 + metric_2

    # Hyper-parameters and logging

    args.epochs = 100
    topk = args.topk = 5

    import logging
    import copy
    from datetime import datetime
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    args.log_file = 'logging/' #  './logging/' + ' '.join([args.model_size, args.dataset, task, 'prelen=%d suflen=%d dynlen=%d' % (prefix_words, postfix_words, dynamic_length)])
    args.verbose = logging.INFO
    print(args.log_file)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=args.log_file, level=args.verbose, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(vars(args))


    # Prompt Generatiuon

    used_prompt = []
    round_for_dynamic = False
    epoch = 0
    batch_eval = exp_approximate_metric_batch if task == 'explanation' else get_approximate_metric_batch
    last_valid_user = -1

    while epoch < args.epochs:
        avg_loss = 0.0
        prefix_grad, suffix_grad, dynamic_grad = [], [], [None for i in range(user_cnt)]
        avg_grad = None
        for param in model.parameters():
            param.grad = None
        pbar = tqdm(enumerate(train_loader))
        for i, batch in pbar:
            batch = revise_batch(batch, prompt_prefix=prompt_prefix, prompt_suffix=prompt_suffix, dynamic_prompt=dynamic_prompt)
            end_x, end_y = torch.nonzero(batch['input_ids'] == 1, as_tuple=True)
            results = model.train_step(batch)
            loss = -results['loss']
            loss.backward()
            avg_loss += results['loss'].item()
            pbar.set_postfix({'training loss': avg_loss / (i + 1)})
            if not round_for_dynamic:
                if postfix_words > 0:
                    grad_slice = []
                    for k in range(len(batch['input_ids'])):
                        grad_slice.append(model.encoder.inputs_embeds.grad.detach().clone()
                                        [k, (end_y[k]-suffix_head_from_end):(end_y[k]-suffix_tail_from_end), :])
                    suffix_grad.append(torch.mean(torch.stack(grad_slice), dim=0))
                if prefix_words > 0:
                    prefix_grad.append(torch.mean(model.encoder.inputs_embeds.grad.detach().clone()[:, :prefix_words, :], dim=0))
            else:
                for k in range(len(batch['input_ids'])):
                    grad = model.encoder.inputs_embeds.grad.detach().clone()[k, (end_y[k] - dynamic_head_from_end):(end_y[k]-dynamic_index_from_end), :].cpu()
                    user_id = batch['user_id'][k]
                    if dynamic_grad[user_id] is None:
                        dynamic_grad[user_id] = [grad.detach().clone()]
                    else:
                        dynamic_grad[user_id].append(grad.detach().clone())
                    last_valid_user_id = user_id
        with torch.no_grad():
            if not round_for_dynamic:
                failed_position = []
                if prefix_words > 0:
                    avg_grad = torch.mean(torch.stack(prefix_grad), dim=0)
                    prefix_score = torch.mm(avg_grad, model.encoder.embed_tokens.weight.t()) + single_word
                if postfix_words > 0:
                    avg_grad = torch.mean(torch.stack(suffix_grad), dim=0)
                    suffix_score = torch.mm(avg_grad, model.encoder.embed_tokens.weight.t()) + single_word
                round_for_dynamic = (dynamic_length > 0)
                while len(failed_position) < postfix_words + prefix_words:      
                    token_to_flip = random.randrange(postfix_words + prefix_words)
                    while token_to_flip in failed_position:
                        token_to_flip = random.randrange(postfix_words + prefix_words)
                    failed_position.append(token_to_flip)
                    score = prefix_score[token_to_flip] if token_to_flip < prefix_words else suffix_score[token_to_flip - prefix_words]
                    _, vocab = score.topk(topk, dim=-1, largest=True, sorted=True)
                    best_cur_metric = None
                    for token in vocab:
                        if token_to_flip < prefix_words:
                            cur_prefix = prompt_prefix.detach().clone()
                            cur_prefix[token_to_flip] = token
                            metric_1, metric_2 = get_approximate_metric(eval_loader, prompt_prefix=cur_prefix, prompt_suffix=prompt_suffix, dynamic_prompt=dynamic_prompt, task=task)
                        else:
                            cur_suffix = prompt_suffix.detach().clone()
                            cur_suffix[token_to_flip - prefix_words] = token
                            metric_1, metric_2 = get_approximate_metric(eval_loader, prompt_prefix=prompt_prefix, prompt_suffix=cur_suffix, dynamic_prompt=dynamic_prompt, task=task)
                        cur_metric = metric_1 + metric_2
                        if best_cur_metric is None or best_cur_metric < cur_metric:
                            best_metric_1, best_metric_2 = metric_1, metric_2
                            best_cur_metric = cur_metric
                            replace_token = token
                    print(best_cur_metric, best_metric)
                    if best_cur_metric > best_metric:
                        failed_position = []
                        if token_to_flip < prefix_words:
                            prompt_prefix[token_to_flip] = replace_token
                            decoder_result = model.tokenizer.decode(prompt_prefix, skip_special_tokens=True).strip()
                        else:
                            prompt_suffix[token_to_flip - prefix_words] = replace_token
                            decoder_result = model.tokenizer.decode(prompt_suffix, skip_special_tokens=True).strip()
                        used_prompt.append((epoch, 
                                            prompt_prefix.detach().clone() if prompt_prefix is not None else None, 
                                            prompt_suffix.detach().clone() if prompt_suffix is not None else None,
                                            dynamic_prompt.detach().clone() if dynamic_prompt is not None else None))
                        best_metric = best_cur_metric
                        if task == 'explanation':
                            logging.info('Eval BLEU-4: %.6lf, Eval ROUGE-L: %.6lf, Prompt suffix: %s', best_metric_1, best_metric_2, decoder_result)
                        else:
                            logging.info('Eval NDCG@5: %.6lf, Eval Hit@5: %.6lf, Prompt suffix: %s', best_metric_1, best_metric_2, decoder_result)
                        
                        print('Epoch %d Testing: ' % epoch)
                        test_func(test_loader, prompt_prefix=prompt_prefix, prompt_suffix=prompt_suffix, dynamic_prompt=dynamic_prompt)
                        break
                if not round_for_dynamic and len(failed_position) >= postfix_words + prefix_words:
                    break
            else:
                token_to_flip = random.randrange(dynamic_length)
                user_grad = torch.stack([torch.ones_like(dynamic_grad[last_valid_user_id][0])[token_to_flip] if t is None 
                                        else torch.mean(torch.stack(t), dim=0)[token_to_flip] for t in dynamic_grad[1:]]).to(device) # user_id index starting from 1
                score = torch.mm(user_grad, model.encoder.embed_tokens.weight.t()) + single_word
                _, vocab = score.topk(topk, dim=-1, largest=True, sorted=True)
                baseline_score = [0] * user_cnt
                for eval_batch in eval_loader:
                    metric_1, metric_2 = batch_eval(eval_batch, prompt_prefix=prompt_prefix, prompt_suffix=prompt_suffix, dynamic_prompt=dynamic_prompt)
                    for item in zip(eval_batch['user_id'], metric_1, metric_2):
                        baseline_score[item[0]] += item[1] + item[2]
                cur_dynamic = dynamic_prompt.detach().clone()
                for cand_index in range(topk):
                    update_score = [0] * user_cnt
                    cur_dynamic[1:, token_to_flip] = vocab[:, cand_index]
                    for _, eval_batch in tqdm(enumerate(eval_loader)):
                        metric_1, metric_2 = batch_eval(eval_batch, prompt_prefix=prompt_prefix, prompt_suffix=prompt_suffix, dynamic_prompt=cur_dynamic)
                        for item in zip(eval_batch['user_id'], metric_1, metric_2):
                            update_score[item[0]] += item[1] + item[2]
                    for user_id in range(1, user_cnt):  # user_id index starting from 1
                        if update_score[user_id] > baseline_score[user_id]:
                            dynamic_prompt[user_id] = vocab[user_id - 1][cand_index]
                            baseline_score[user_id] = update_score[user_id]
                used_prompt.append((epoch, 
                                    prompt_prefix.detach().clone() if prompt_prefix is not None else None, 
                                    prompt_suffix.detach().clone() if prompt_suffix is not None else None,
                                    dynamic_prompt.detach().clone() if dynamic_prompt is not None else None))
                print('Epoch %d Testing: ' % epoch)
                test_func(test_loader, prompt_prefix=prompt_prefix, prompt_suffix=prompt_suffix, dynamic_prompt=dynamic_prompt)
                metric_1, metric_2 = get_approximate_metric(eval_loader, prompt_prefix=prompt_prefix, prompt_suffix=prompt_suffix, dynamic_prompt=dynamic_prompt, task=task)
                best_metric = metric_1 + metric_2
                round_for_dynamic = False
        avg_loss /= len(train_loader)
        logging.info('Epoch %d, Training Loss: %.6lf', epoch, avg_loss)
        avg_loss = 0.0
        epoch += 1

    for epoch, prefix, suffix, dynamic in used_suffix[::-1]:
        decoder_prefix = model.tokenizer.decode(prefix, skip_special_tokens=True).strip() if prefix is not None else ''
        decoder_suffix = model.tokenizer.decode(suffix, skip_special_tokens=True).strip() if suffix is not None else ''
        logging.info('Epoch: %d, Prompt prefix: %s, Prompt suffix: %s, Personalized: %s', 
                    epoch, decoder_prefix, decoder_suffix, '\t'.join(str(x.item()) for x in dynamic))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='sequential')
    parser.add_argument('--dataset', type=str, default='beauty')
    parser.add_argument('--model_size', type=str, default='small')
    parser.add_argument('--dynamic_length', type=int, default=0)
    args = parser.parse_known_args()[0]
    main(args)

