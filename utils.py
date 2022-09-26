import torch
import torch.nn.functional as F
from tqdm import tqdm
import json
import math
import sys
import openai
import time
import os
import re
import string

import json
import torch
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from transformers import (
    AutoConfig,
    AutoTokenizer,
    default_data_collator,
)

from t0.data_collator import DataCollatorForMultipleChoice
from t0.model import ModelBase

def detokenizer(string):
    # ari custom
    string = string.replace("`` ", '"')
    string = string.replace(" ''", '"')
    string = string.replace("` ", '"')
    string = string.replace(" ' ", '" ')
    # contractions
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    # number separators
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    # punctuation
    string = string.replace(" :", ":")
    string = string.replace(" ;", ";")
    string = string.replace(" .", ".")
    string = string.replace(" !", "!")
    string = string.replace(" ?", "?")
    string = string.replace(" ,", ",")
    # double brackets
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    # string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    # string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    # miscellaneous
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")
    # ari custom
    string = string.replace(" n't ", "n't ")
    string = string.replace(" 'd ", "'d ")
    string = string.replace(" 'm ", "'m ")
    string = string.replace(" 're ", "'re ")
    string = string.replace(" 've ", "'ve ")
    return string


def get_key(source, target):
    return '{}'.format(json.dumps({'source':source, 'target':target}))


def gpt3(prompt, max_len, model_name, temp=0, num_log_probs=100, echo=False, n=None):
    print('calling API')
    # call GPT-3 API until result is provided and then return it
    response = None
    received = False
    while not received:
        try:
            response = openai.Completion.create(engine=model_name, 
                                                prompt=prompt,
                                                max_tokens=max_len,
                                                temperature=temp,
                                                logprobs=num_log_probs,
                                                echo=echo,
                                                stop='\n',
                                                n=n)
            received = True
        except:
            error = sys.exc_info()[0]
            if error == openai.error.InvalidRequestError: 
                # something is wrong: e.g. prompt too long
                print(f"InvalidRequestError\nPrompt passed in:\n\n{prompt}\n\n")
                assert False
            print("API error:", error)
            time.sleep(1)
    return response

def cross_entropy_list_gpt3(inputs, targets, model_name, batch=None,cache=None, calculate = False):
    '''
    get a list of -log P(target|inp) for
    the inputs and targets in inputs, targets
    using gpt3
    '''
    assert(len(inputs) == len(targets))
    
    ### This block at the top handles caching/batching
    ## basically, first log all computations not in the cache
    ## if calculate is False, return dummy values (just
    ## logging computations to do later)
    ## if calculate is True, do all computations that are not done
    ## then return results for this batch
    ###############################
    ## if we are caching results (LAZY EVALUATION)
    # this is useful for efficient batching. First, add all needed
    # calculations to the batch with calculate = False
    # then run with calculate=True to work through all cached calculations
    if cache is not None:
        # log calculations we have not done yet
        for inp,targ in zip(inputs, targets):
            if get_key(inp, targ) not in cache:
                cache[get_key(inp, targ)] = {'source': inp, 'target':targ,'result':None}
        
        # if not calculating, return dummy values
        if not calculate:
            return [1.]*len(inputs), [1.]*len(inputs), None
        
        # if caching and calculating, we calculate for all examples
        # that have been cached but not calculated
        cache_todo = [(v['source'], v['target']) for v in cache.values() if v['result'] is None]
        
        ## if there are calculations to do, do them
        if len(cache_todo) > 0:
            sources_todo = list(zip(*cache_todo))[0]
            targets_todo = list(zip(*cache_todo))[1]
            
            ce_list, t_len_list, result_list = cross_entropy_list_gpt3(sources_todo, targets_todo,  model_name, cache=None, batch=batch)
            for source, target, ce,t_len, result in zip(sources_todo,targets_todo, ce_list, t_len_list, result_list):
                cache[get_key(source, target)]['ce'] = ce
                cache[get_key(source, target)]['result'] = result
                cache[get_key(source, target)]['t_len'] = t_len
        ## return results for thie example
        output = ([cache[get_key(inp, targ)]['ce'] for inp,targ in zip(inputs, targets)],
                  [cache[get_key(inp, targ)]['t_len'] for inp,targ in zip(inputs, targets)],
                  [cache[get_key(inp, targ)]['result'] for inp,targ in zip(inputs, targets)])
        return output
    ###############################           
    
    
    ### batching ####
    if batch is not None:
        result = {'choices':[]}
        ce_list = []
        len_list = []
        while len(inputs) > 0:
            ce_out, len_out, result_out = cross_entropy_list_gpt3(inputs[:batch], targets[:batch], model_name, cache=None, batch=None)
            inputs, targets = inputs[batch:], targets[batch:]
            
            ce_list = ce_list + ce_out
            len_list = len_list + len_out
            result['choices'] = result['choices'] + result_out
            
            return ce_list, len_list, result['choices']  
    #########
    
    
    #####
    ## calculating cross-entropy
    #####
    data = [inp + targ for inp, targ in zip(inputs, targets)]    
    result = gpt3(data, 0, model_name, echo=True, num_log_probs=1)
    
    #with open(out_file, 'a') as out:
    #    out.write(f'{json.dumps(result)}\n')
    ce_list = []
    t_lens = []
    for inp, out in zip(inputs, result['choices']):
        # get the beginning of the target from the response (based on tokenization)
        i = 0
        while out['logprobs']['text_offset'][i] < len(inp):
            i += 1
        t_lens.append(len(out['logprobs']['text_offset']) - i)
        # sum of log probs over the target tokens
        ce = -sum(out['logprobs']["token_logprobs"][i:])
        ce_list.append(ce)
    return ce_list, t_lens, result['choices'] 


def cross_entropy_list(sources, targets, model, cache = None, batch=False, calculate=True):
    '''
    Gets a list of CE values, where the ith item is a list of cross-entropies
    for targets[i] with sources[i] as contexts
    targets and sources are lists of lists of tokens (integers)
    model is a language model
    batch is the batch size to break things up into, batch=False means don't
    break things up into batches, do them all in one go.
    
    CACHING:
    
    cache is a dictionary for single source/target pairs
      accessed by cache[get_key(source,target)]
      it has fields source, target, result
    
    calculate decides whether to immediates calculate for batch of input
      sources/targets or just log them as todo in the cache. To efficiently 
      batch, we can first log many todo calculations by calling cross_entropy_list
      multiple times with calculate=False and the same input cache
      Then finally calling it with calculate=True which will then catch up on all
      todo calculations, caching them together efficiently
    
    '''
    
    ###############################
    # This block handles caching of results (LAZY EVALUATION)
    # this is useful for efficient batching. First, add all todo
    # calculations to the cache with calculate = False (won't do them yet)
    # then run with calculate=True to work through all cached calculations
    # in efficient batches
    if cache is not None:

        # log calculations we have not done yet
        for source,target in zip(sources, targets):
            if get_key(source, target) not in cache:
                cache[get_key(source, target)] = {'source': source, 'target':target,'result':None}
        
        # if not calculating, return dummy values
        if not calculate:
            return [1.]*len(sources)
        
        # if caching and calculating, we calculate for all examples
        # that have been cached but not calculated
        cache_todo = [(v['source'], v['target']) for v in cache.values() if v['result'] is None]
        
        ## if there are calculations to do, do them
        if len(cache_todo) > 0:
            sources_todo = list(zip(*cache_todo))[0]
            targets_todo = list(zip(*cache_todo))[1]
            
            cache_results = cross_entropy_list(sources_todo, targets_todo, model, cache=None, batch=batch)
            for source, target, result in zip(sources_todo,targets_todo, cache_results):
                cache[get_key(source, target)]['result'] = result
    
        ## return results for thie example
        results = [cache[get_key(source, target)]['result'] for source,target in zip(sources, targets)]
        return results
    ###############################        
        
        
        
        
    
    
    
    assert(len(sources ) == len(targets))
    n_seqs = len(sources)
    
    torch.cuda.empty_cache()
    # device = model.transformer.wte.weight.device
    # print("device",device)

    # if batching, break it up into smaller pieces
    if batch:
        ce_list = []
        
        n_batches = math.ceil(len(sources) / batch)
        
        list_fun = (lambda v: tqdm(list(v))) if cache is not None else list
        
        for i in tqdm(list(range(n_batches))):
            ce_list += cross_entropy_list(sources[i*batch:(i+1)*batch], targets[i*batch:(i+1)*batch], model, batch=False)
            #sources, targets = sources[batch:], targets[batch:]
        return ce_list 

    # initialize input tensors
    max_len = max([len(s + t) for s,t in zip(sources, targets)])
    input_ids = torch.zeros((n_seqs, max_len)).long() 
    #-100 is the padding token, which is ignored by F.cross_entropy below
    labels = -100*torch.ones((n_seqs, max_len)).long()
    
    # for each source, target pair, set values in the input tensors
    for i, (source, target) in enumerate(zip(sources,targets)):
        s = torch.tensor(source).long()
        t = torch.tensor(target).long()
        input_ids[i,:len(s)] = s
        input_ids[i,len(s):len(s) + len(t)] = t
        # ignore all predictions except in the target span
        labels[i,len(s):len(s) + len(t)] = t
    
    # get logits from the model
    with torch.no_grad():
        input_ids = input_ids.to('cuda')
        # input_ids = input_ids.to(device)
        logits = model(input_ids).logits.cpu()[:,:-1].contiguous()
    
    # get cross-entropies given the logits
    logit_shape = logits.shape
    logits = logits.view(-1, logit_shape[-1])
    ce_list = F.cross_entropy(logits.float(), labels[:,1:].contiguous().view(-1), reduction='none')
    ce_list = ce_list.view(n_seqs, max_len -1).sum(dim=1).squeeze().tolist()
    
    # if one element (i.e. len(sources) == 1), nest it into a list. Otherwise, give full list
    # this just handles an idiosyncracy of the .tolist() function
    try:
        len(ce_list)
    except:
        ce_list = [ce_list]
    
    return ce_list





def inference_autobatch( model, encoder, example, batch = 1, prelog = False, cache = None):
    '''
    
    if prelog is true, then we're just logging calculations to do in one big batch calculate
    (used for caching)
    
    
    '''
    
    ## if we are just prelogging cross entropy calculations to do later,
    ## we will set caclulate=False for cross_entropy_list and it will output
    ## a dummy value for now and just log calculations to do. Then the output
    ## of inference_autobatch will not be correct, calling it in this case is 
    ## just to log calculations to do in big batches
    if prelog and (cache is not None):
        calculate = False 
    else:
        calculate = True
    
    
    #####
    ## input data handling
    #####
    # i.e. if we're using GPT-3 through the OpenAI API
    if type(model) == str:
        max_len = 2048  
        gpt3 = True
    else:
        max_len = 1024
        gpt3 = False

    options = []
    for opt_raw in example['options']:
        if gpt3:
            options.append(opt_raw)
        else:
            # first, encode the option 
            opt = { key: encoder.encode(opt_raw[key], add_special_tokens=False) for key in opt_raw.keys() }

            ## trim the option to the max length for gpt2
            opt['premise'] = opt['premise'][-(max_len - len(opt['hypothesis'])):]
            assert(len(opt['premise'] + opt['hypothesis']) <= max_len)

            # then add the encoded, trimmed option
            options.append( opt )

    #####
    ## cross-entropy calculation
    #####
    if gpt3:
        ## get conditional CEs
        cond_ce, cond_t_lens, _ = cross_entropy_list_gpt3([opt['premise'] for opt in options], 
                                                          [opt['hypothesis'] for opt in options],
                                                          model,
                                                        cache=cache,calculate = calculate, batch=batch)
        
        ## get domain conditional CEs
        # domain_cond_ce, domain_cond_t_lens, _ = cross_entropy_list_gpt3([opt['uncond_premise'] for opt in options],
        #                                 [opt['uncond_hypothesis'] for opt in options],
        #                                 model,
        #                                 cache=cache,calculate = calculate, batch=batch)

        ## get unconditional CEs
        # uncond_ce, uncond_t_lens, _ = cross_entropy_list_gpt3([':' for opt in options],
        #                                 [opt['uncond_hypothesis'] for opt in options],
        #                                 model,
        #                                 cache=cache,calculate = calculate, batch=batch)
    else:
        ## get conditional CEs
        cond_ce = cross_entropy_list([opt['premise'] for opt in options], 
                                    [opt['hypothesis'] for opt in options],
                                    model, cache=cache, batch=batch, calculate = calculate)

        
        ## get domain conditional CEs
        # domain_cond_ce  = cross_entropy_list([opt['uncond_premise'] for opt in options],
        #                                 [opt['uncond_hypothesis'] for opt in options],
        #                                 model, cache=cache, batch=batch, calculate = calculate)
        
        # ## get unconditional CEs
        # uncond_ce = cross_entropy_list([[25] for opt in options],
        #                                [opt['uncond_hypothesis'] for opt in options],
        #                                model, cache=cache, batch=batch, calculate = calculate)

    ## get average CE by token
    if gpt3:
        avg_cond_ce = [ce/l for ce, l in zip(cond_ce, cond_t_lens)]
    else:
        
        avg_cond_ce = [ce / len(opt['hypothesis']) for ce, opt in zip(cond_ce, options)]
       
    
    #####
    ## prediction
    #####
    # calculate dcpmi
    # dcpmi = [ce_0 - ce_1 for ce_0,ce_1 in zip(domain_cond_ce, cond_ce)]
    # pmi = [ce_0 - ce_1 for ce_0,ce_1 in zip(uncond_ce, cond_ce)]

    
    ## make predictions based on different scores
    lm_pred = cond_ce.index(min(cond_ce))
    lm_avg_pred = avg_cond_ce.index(min(avg_cond_ce))
    # lm_domain_cond_pred = domain_cond_ce.index(min(domain_cond_ce))
    # dcpmi_pred = dcpmi.index(max(dcpmi))
    # pmi_pred = pmi.index(max(pmi))
    pred = {
                 'lm': lm_pred,
                 'tok_mean': lm_avg_pred,
                #  'dcpmi' : dcpmi_pred,
                #  'pmi': pmi_pred,
                #  'domain_cond': lm_domain_cond_pred,
           }
    return pred

        
        



def fwd(model, encoder, examples, batch, cache = None, prompt_name=None):
    '''
    This is designed for gpt2-style language models
    
    Inputs: (any you don't know)
        model - a HuggingFace Transformers gpt-2 model
        encoder - a HuggingFace Transformers tokenizer
        examples = [ex1, ex2, ...]
            where ex = [opt1, opt2, ...] (multiple choice options)
            where opt = (premise, hypothesis) 
        
        batch: is the max allowed batch size (set to 1 for no batching)
    '''
    
    if type(model) != str:
        # print the first example to make sure the format is ok
        print('='*50)
        print('MAKE SURE TOKENIZATION AND FORMATTING LOOKS OK')
        print('\nprint example 0 of {}:'.format(len(examples)))
        ex = examples[0]
        options = ex['options']
        opt = options[0]
        print('CONDITIONAL:')
        print(encoder.encode(opt['hypothesis']))
        print(encoder.decode(encoder.encode(opt['premise'])) + '<BREAK>' + encoder.decode(encoder.encode(opt['hypothesis'])))
        # print('UNCONDITIONAL:')
        # print(encoder.decode(encoder.encode(opt['uncond_premise'])) + '<BREAK>' + encoder.decode(encoder.encode(opt['uncond_hypothesis'])))
        print('='*50)
    else:
        # print the first example to make sure the format is ok
        print('='*50)
        print('MAKE SURE TOKENIZATION AND FORMATTING LOOKS OK')
        print('\nprint example 0 of {}:'.format(len(examples)))
        ex = examples[0]
        options = ex['options']
        opt = options[0]
        print('CONDITIONAL:')
        print(opt['premise'] + '<BREAK>' + opt['hypothesis'])
        # print('UNCONDITIONAL:')
        # print(opt['uncond_premise'] + '<BREAK>' + opt['uncond_hypothesis'])
        print('='*50)

    predictions_list = []
    

    ## in this loop, prelog is set to true so we are just logging cross_entropy_list calculations
    ## but not doing them yet
    if cache is not None:
        print('logging examples')
        for example in tqdm( examples):
            _ = inference_autobatch(model, encoder, example, prelog=True, cache = cache, batch=batch )

    ## in this loop, we actually do the calculations from above in efficient batches, storing results 
    ## in the cache and calculating actual predictions
    print('actually calculating')
    for example in tqdm(examples):
        pred = inference_autobatch(model, encoder, example, prelog=False, cache = cache, batch=batch )
        predictions_list.append(pred)

        
    labels = [ex['label'] for ex in examples]
    # get predictions into list by scoring key
    predictions_dict = {key:list(map(lambda v: v[key], predictions_list)) for key in predictions_list[0].keys()}

    # calculate accuracies
    results = {key: sum(list(map(lambda v: v[0] in v[1], zip(predictions_dict[key] , labels) )))/len(labels) for key in predictions_dict.keys()}

    print(predictions_list[0].keys())

    if 'negation' in prompt_name.lower():
        results_50 = {key: sum(list(map(lambda v: v[0] in v[1], zip(predictions_dict[key][50:100] , labels[50:100]) )))/len(labels[50:100]) for key in predictions_dict.keys()}
    else: 
        results_50 = {key: sum(list(map(lambda v: v[0] in v[1], zip(predictions_dict[key][:50] , labels[:50]) )))/len(labels[:50]) for key in predictions_dict.keys()}

    # save labels for later
    predictions_dict['labels'] = labels
    return results, results_50, predictions_dict, labels

def convert_example_T0(example, tokenizer, eos_token, max_input_length, max_output_length, n_prefix):
    input_ = example['premise']
    # target_ = example['hypothesis']
    

    if eos_token == False: 
        source = tokenizer.batch_encode_plus([str(input_)], max_length=max_input_length, 
                                                        padding='max_length', truncation=True, return_tensors="pt", add_special_tokens=False)
    else: 
        source = tokenizer.batch_encode_plus([str(input_)], max_length=max_input_length, 
                                                        padding='max_length', truncation=True, return_tensors="pt", add_special_tokens=True)
    # targets = tokenizer.batch_encode_plus([str(target_)], max_length=max_output_length, 
    #                                                 padding='max_length', truncation=True, return_tensors="pt")
    # task_tokens = ["<TASK{}>".format(str(i).zfill(2)) for i in range(n_prefix)]
    # task_token_ids = tokenizer(" ".join(task_tokens), return_tensors="pt", add_special_tokens=False)["input_ids"]
    # assert task_token_ids.shape[-1]==n_prefix
    # n_train = source["input_ids"].shape[0]
    # new_input_ids=torch.cat([task_token_ids.repeat(n_train, 1),source["input_ids"]], 1)
    # source["input_ids"] = new_input_ids
    # source["attention_mask"] = torch.cat([torch.ones((n_train, n_prefix), dtype=torch.long), source["attention_mask"]], 1)
    source_ids = source["input_ids"].squeeze()
    # target_ids = targets["input_ids"].squeeze()

    src_mask    = source["attention_mask"].squeeze()
    # target_mask = targets["attention_mask"].squeeze()

    return source_ids, None , src_mask, None
def Dataset_for_T0(batch_item, tokenizer, eos_token, max_input_length, max_output_length, n_prefix):
    source_ids_batch=[]
    target_ids_batch=[]
    src_mask_batch=[]
    target_mask_batch=[]
    for index in range(len(batch_item['premise'])):
        example = {}
        example['premise']=batch_item['premise'][index]
        # example['hypothesis']=batch_item['hypothesis'][index]
        source_ids, target_ids, src_mask, target_mask = convert_example_T0(example, tokenizer, eos_token, max_input_length, max_output_length, n_prefix)
        
        source_ids_batch.append(source_ids)
        # target_ids_batch.append(target_ids)
        src_mask_batch.append(src_mask)
        # target_mask_batch.append(target_mask)
    #print(source_ids_batch, target_ids_batch, src_mask_batch, target_mask_batch)
    return torch.stack(source_ids_batch), torch.stack(src_mask_batch)
def ids_to_clean_text(tokenizer, generated_ids):
    gen_text = tokenizer.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    return lmap(str.strip, gen_text)

def lmap(f, x):
    """list(map(f, x))"""
    return list(map(f, x))
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    
    def rid_of_specials(text):
        text = text.replace("<extra_id_0>", "")
        text = text.replace("<extra_id_1>", "")
        return text
    
    return rid_of_specials(white_space_fix(remove_articles(remove_punc(lower(s)))))

def accuracy_match_score_normalize(prediction, ground_truth):
    if normalize_answer(prediction)== '' or normalize_answer(ground_truth)== '':
        return 0
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))

def accuracy_match_list_score(prediction, answers):
    accuracy_score = 0
    
    accuracy_correct = False
    for answer in answers:
        accuracy = accuracy_match_score_normalize(prediction, answer)
        if accuracy == 1:
            accuracy_correct = True
    if accuracy_correct:
        return 1
    return 0 

def evaluation_T0(model, encoder, examples, batch, args, cache, prompt_name):
    total_cnt = 0
    accuracy_correct_num = 0
    accuracy_correct_50 = 0

    if type(model) != str:
        # print the first example to make sure the format is ok
        print('='*50)
        print('MAKE SURE TOKENIZATION AND FORMATTING LOOKS OK')
        print('\nprint example 0 of {}:'.format(len(examples)))
        ex = examples[0]
        options = ex['options']
        opt = options[0]
        print('CONDITIONAL:')
        print(encoder.encode(opt['hypothesis']))
        print(encoder.decode(encoder.encode(opt['premise'])) + '<BREAK>' + encoder.decode(encoder.encode(opt['hypothesis'])))
        # print('UNCONDITIONAL:')
        # print(encoder.decode(encoder.encode(opt['uncond_premise'])) + '<BREAK>' + encoder.decode(encoder.encode(opt['uncond_hypothesis'])))
        print('='*50)
    else:
        # print the first example to make sure the format is ok
        print('='*50)
        print('MAKE SURE TOKENIZATION AND FORMATTING LOOKS OK')
        print('\nprint example 0 of {}:'.format(len(examples)))
        ex = examples[0]
        options = ex['options']
        opt = options[0]
        print('CONDITIONAL:')
        print(opt['premise'] + '<BREAK>' + opt['hypothesis'])
        # print('UNCONDITIONAL:')
        # print(opt['uncond_premise'] + '<BREAK>' + opt['uncond_hypothesis'])
        print('='*50)
    predictions_dict = {}
    predictions_list = []
    
    for index in range(0, len(examples), batch):
        next_index = min(len(examples), index + batch)
        batch_item_dict = examples[index:next_index]
        batch_item = {}
        batch_item_options = []
        batch_item_labels = []
        batch_item_option_list=[[]]*len(batch_item_dict[0]['option_list'])
        for item in batch_item_dict:
            label_list = []
            batch_item_options.append(item['options'][0]['premise'])
            for label_idx in item['label']:
                label_list.append(item['options'][label_idx]['hypothesis'])
            batch_item_labels.append(label_list)
            for option_cand_index in range(len(item['option_list'])):
                batch_item_option_list[option_cand_index] = batch_item_option_list[option_cand_index]  + [item['option_list'][option_cand_index]]
        batch_item['premise'] = batch_item_options
        # batch_item['hypothesis'] =batch_item_labels
        batch_item['labels'] = batch_item_labels
        batch_item['option_list'] = batch_item_option_list
        '''
        batch_item: dictionary of batch
        premise: 
        hypothesis:
        options:
        '''
        prob_list = []
        accuracy_list = []
        source_ids, src_mask = Dataset_for_T0(batch_item, encoder, args.eos_token, args.max_input_length, args.max_output_length, args.n_prefix)
        if batch_item['option_list'] is not None:
            option_list = batch_item['option_list'] 
        else:
            option_list = -1
        with torch.no_grad():
            for index in range(len(batch_item['option_list'])):
                option_ = encoder.batch_encode_plus(option_list[index], max_length=args.max_output_length,
                                                padding=True, truncation=True, return_tensors="pt")
                
                lm_labels = option_["input_ids"].expand(len(batch_item['option_list'][0]), -1)
                lm_labels[lm_labels[:, :] == encoder.pad_token_id] = -100
                #print(source_ids, target_ids, src_mask, target_mask , option_list, option_, lm_labels)
                # print(source_ids, src_mask, lm_labels,option_["attention_mask"], option_list)
                outputs = model(
                    input_ids=source_ids.cuda(),
                    attention_mask=src_mask.cuda(),
                    labels=lm_labels.cuda(),
                    decoder_attention_mask=option_["attention_mask"].cuda()
                )
                print("outputs", outputs[0])
                logits = option_["attention_mask"].cuda().unsqueeze(-1) * torch.log_softmax(outputs.logits, dim=-1)
                lm_labels=lm_labels.cuda().unsqueeze(-1)
                seq_token_log_prob=torch.zeros(lm_labels.shape)
                #print(seq_token_log_prob.shape, logits.shape, lm_labels.shape)
                for i in range(lm_labels.shape[0]):
                    for j in range(lm_labels.shape[1]):
                        seq_token_log_prob[i][j][0] = logits[i][j][lm_labels[i][j][0]]
                seq_log_prob = seq_token_log_prob.squeeze(dim=-1).sum(dim=-1)
                prob_list.append(seq_log_prob)
            concat = torch.cat(prob_list).view(-1,len(source_ids))
            #print(concat)
            predictions = concat.argmax(dim=0)
            dec = [option_list[i.item()][elem_num] for elem_num, i in enumerate(predictions)]
            #print(dec)
            #print(predictions)
            texts = [encoder.decode(ids) for ids in source_ids]
            # targets = ids_to_clean_text(encoder, target_ids) 
            for i in range(len(source_ids)):
                total_cnt+=1
                print(batch_item['labels'][i])
                ground_truth =batch_item['labels'][i]
                print("ground_truth",ground_truth)
                predicted = dec[i]
                print("prediction:",total_cnt,predicted)
                accuracy = accuracy_match_list_score(predicted, ground_truth)
                if accuracy == 1:
                    accuracy_correct_num+=1
                    if 'negation' in prompt_name.lower():
                        if total_cnt>50 and total_cnt<=100:
                            accuracy_correct_50+=1
                    else: 
                        if total_cnt<=50:
                            accuracy_correct_50+=1
                
                print("ground_truth", ground_truth)

                print("acc",accuracy_correct_num)
                predictions_list.append(predicted)
                accuracy_list.append(accuracy)
    
    predictions_dict['lm'] = predictions_list
    predictions_dict['accuracy'] = accuracy_list
    accs=  accuracy_correct_num/total_cnt  
    accs_50 = accuracy_correct_50/50 
    return accs, accs_50, predictions_dict


def score(model, model_name, encoder, examples, stem, split, batch, prompt_name, icl=0):
    hist_path = f'{stem}{model_name}-{split}.hist'
    model_name_s = model_name.split('/')
    MYDIR = f'{stem}{model_name}'
    # MYDIR = stem
    CHECK_FOLDER = os.path.isdir(MYDIR)
    if not CHECK_FOLDER:
        os.makedirs(MYDIR)
        print("created folder : ", MYDIR)
    #if not os.path.exists(hist_path):
    #    cache = {}
    #    with open(hist_path, 'w') as f:
    #        f.write(json.dumps(cache))
    #else:
    #    MB = os.path.getsize(hist_path)/1000000
    #    print('='*50)
    #    print('Loading existing cache, size {} MB'.format(MB))
    #    print('='*50)
    cache={}
        
    #with open(hist_path, 'r') as f:
    #    cache = json.loads(f.read())
        
    accs, accs_50, preds, labels = fwd(model, encoder, examples, batch, cache, prompt_name)
    
    #print('='*50)
    #print('saving cache to {}'.format(hist_path))
    #print('='*50)
    #with open(hist_path, 'w') as f:
    #    f.write(json.dumps(cache))

    print("accs", accs, accs_50)
    # save scores
    results_path = f'{stem}{model_name}/{split}.accs'
    with open(results_path,'w') as out:
        out.write(json.dumps(accs))
        out.write(json.dumps(accs_50))

    # save predicted labels
    #preds_path = f'{stem}/{model_name}/{split}.preds'
    #with open(preds_path, 'w') as out:
    #    out.write(json.dumps(preds))
    #    out.write(json.dumps(labels))

    return accs, accs_50

def score_T0(model, model_name, encoder, examples, stem, split, batch, args):
    hist_path = f'{stem}{model_name}-{split}.hist'
    MYDIR = stem+"/"+model_name
    CHECK_FOLDER = os.path.isdir(MYDIR)
    if not CHECK_FOLDER:
        os.makedirs(MYDIR)
        print("created folder : ", MYDIR)    
    if not os.path.exists(hist_path):
        cache = {}
        with open(hist_path, 'w') as f:
            f.write(json.dumps(cache))
    else:
        MB = os.path.getsize(hist_path)/1000000
        print('='*50)
        print('Loading existing cache, size {} MB'.format(MB))
        print('='*50)
        
    with open(hist_path, 'r') as f:
        cache = json.loads(f.read())
    accs, accs_50, preds = evaluation_T0(model, encoder, examples, batch, args, cache, args.prompt_name)
    print('='*50)
    print('saving cache to {}'.format(hist_path))
    print('='*50)
    with open(hist_path, 'w') as f:
        f.write(json.dumps(cache))

    # save scores
    results_path = f'{stem}/{model_name}/{split}.accs'
    with open(results_path,'w') as out:
        out.write(json.dumps(accs))
        out.write(json.dumps(accs_50))

    # save predicted labels
    preds_path = f'{stem}/{model_name}/{split}.preds'
    with open(preds_path, 'w') as out:
        out.write(json.dumps(preds))

    return accs, accs_50