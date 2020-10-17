import pickle
import nltk
import re
import numpy as np
import pandas as pd
from sklearn import preprocessing

url_replacer = '<url>'
num_regex = re.compile('^[+-]?[0-9]+\.?[0-9]*$')
ref_scores_dtype = 'int32'
MAX_SENTLEN = 50
MAX_SENTNUM = 100


def replace_url(text):
    replaced_text = re.sub('(http[s]?://)?((www)\.)?([a-zA-Z0-9]+)\.{1}((com)(\.(cn))?|(org))', url_replacer, text)
    return replaced_text


def tokenize(string):
    tokens = nltk.word_tokenize(string)
    for index, token in enumerate(tokens):
        if token == '@' and (index+1) < len(tokens):
            tokens[index+1] = '@' + re.sub('[0-9]+.*', '', tokens[index+1])
            tokens.pop(index)
    return tokens


def shorten_sentence(sent, max_sentlen):
    new_tokens = []
    sent = sent.strip()
    tokens = nltk.word_tokenize(sent)
    if len(tokens) > max_sentlen:
        split_keywords = ['because', 'but', 'so', 'You', 'He', 'She', 'We', 'It', 'They', 'Your', 'His', 'Her']
        k_indexes = [i for i, key in enumerate(tokens) if key in split_keywords]
        processed_tokens = []
        if not k_indexes:
            num = len(tokens) / max_sentlen
            num = int(round(num))
            k_indexes = [(i+1)*max_sentlen for i in range(num)]

        processed_tokens.append(tokens[0:k_indexes[0]])
        len_k = len(k_indexes)
        for j in range(len_k-1):
            processed_tokens.append(tokens[k_indexes[j]:k_indexes[j+1]])
        processed_tokens.append(tokens[k_indexes[-1]:])

        for token in processed_tokens:
            if len(token) > max_sentlen:
                num = len(token) / max_sentlen
                num = int(np.ceil(num))
                s_indexes = [(i+1)*max_sentlen for i in range(num)]

                len_s = len(s_indexes)
                new_tokens.append(token[0:s_indexes[0]])
                for j in range(len_s-1):
                    new_tokens.append(token[s_indexes[j]:s_indexes[j+1]])
                new_tokens.append(token[s_indexes[-1]:])

            else:
                new_tokens.append(token)
    else:
        return [tokens]

    return new_tokens


def tokenize_to_sentences(text, max_sentlength, create_vocab_flag=False):
    sents = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\!|\?)\s', text)
    processed_sents = []
    for sent in sents:
        if re.search(r'(?<=\.{1}|\!|\?|\,)(@?[A-Z]+[a-zA-Z]*[0-9]*)', sent):
            s = re.split(r'(?=.{2,})(?<=\.{1}|\!|\?|\,)(@?[A-Z]+[a-zA-Z]*[0-9]*)', sent)
            ss = " ".join(s)
            ssL = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\!|\?)\s', ss)

            processed_sents.extend(ssL)
        else:
            processed_sents.append(sent)

    if create_vocab_flag:
        sent_tokens = [tokenize(sent) for sent in processed_sents]
        tokens = [w for sent in sent_tokens for w in sent]
        return tokens

    sent_tokens = []
    for sent in processed_sents:
        shorten_sents_tokens = shorten_sentence(sent, max_sentlength)
        sent_tokens.extend(shorten_sents_tokens)
    return sent_tokens


def text_tokenizer(text, replace_url_flag=True, tokenize_sent_flag=True, create_vocab_flag=False):
    text = replace_url(text)
    text = text.replace(u'"', u'')
    if "..." in text:
        text = re.sub(r'\.{3,}(\s+\.{3,})*', '...', text)
    if "??" in text:
        text = re.sub(r'\?{2,}(\s+\?{2,})*', '?', text)
    if "!!" in text:
        text = re.sub(r'\!{2,}(\s+\!{2,})*', '!', text)

    tokens = tokenize(text)
    if tokenize_sent_flag:
        text = " ".join(tokens)
        sent_tokens = tokenize_to_sentences(text, MAX_SENTLEN, create_vocab_flag)
        return sent_tokens
    else:
        raise NotImplementedError


def is_number(token):
    return bool(num_regex.match(token))


def read_word_vocab(read_configs):
    vocab_size = read_configs['vocab_size']
    file_path = read_configs['train_path']
    word_vocab_count = {}

    with open(file_path, encoding='latin-1') as input_file:
        next(input_file)
        for index, line in enumerate(input_file):
            tokens = line.strip().split('\t')
            content = tokens[2].strip()
            content = text_tokenizer(content, True, True, True)
            content = [w.lower() for w in content]
            for word in content:
                try:
                    word_vocab_count[word] += 1
                except KeyError:
                    word_vocab_count[word] = 1

        import operator
        sorted_word_freqs = sorted(word_vocab_count.items(), key=operator.itemgetter(1), reverse=True)
        if vocab_size <= 0:
            vocab_size = 0
            for word, freq in sorted_word_freqs:
                if freq > 1:
                    vocab_size += 1

        word_vocab = {'<pad>': 0, '<unk>': 1, '<num>': 2}
        vcb_len = len(word_vocab)
        index = vcb_len
        for word, _ in sorted_word_freqs[:vocab_size - vcb_len]:
            word_vocab[word] = index
            index += 1
        return word_vocab


def read_essay_sets_word(essay_file, vocab, target_prompt, ret_target_prompt=False, ret_src_prompts=True):
    out_data = {
        'essay_ids': [],
        'words': [],
        'data_y': [],
        'prompt_ids': [],
        'max_sentnum': -1,
        'max_sentlen': -1
    }
    with open(essay_file, encoding='latin-1') as input_file:
        next(input_file)
        for essay in input_file:
            tokens = essay.strip().split('\t')
            essay_id = int(tokens[0])
            essay_set = int(tokens[1])
            content = tokens[2]
            score = int(tokens[6])
            if (essay_set == target_prompt and ret_target_prompt == True) or (essay_set != target_prompt and ret_src_prompts == True):
                out_data['data_y'].append(score)
                sent_tokens = text_tokenizer(content, replace_url_flag=True, tokenize_sent_flag=True)
                sent_tokens = [[w.lower() for w in s] for s in sent_tokens]

                sent_indices = []
                indices = []
                for sent in sent_tokens:
                    length = len(sent)
                    if length > 0:
                        if out_data['max_sentlen'] < length:
                            out_data['max_sentlen'] = length
                        for word in sent:
                            if is_number(word):
                                indices.append(vocab['<num>'])
                            elif word in vocab:
                                indices.append(vocab[word])
                            else:
                                indices.append(vocab['<unk>'])
                        sent_indices.append(indices)
                        indices = []
                out_data['words'].append(sent_indices)
                out_data['prompt_ids'].append(essay_set)
                out_data['essay_ids'].append(essay_id)
                if out_data['max_sentnum'] < len(sent_indices):
                    out_data['max_sentnum'] = len(sent_indices)
    assert(len(out_data['words']) == len(out_data['data_y']))
    print(' word_x size: {}'.format(len(out_data['words'])))
    return out_data


def read_essays_words(read_configs, word_vocab, target_prompt):
    train_data_src = read_essay_sets_word(read_configs['train_path'], word_vocab, target_prompt,
                                          ret_target_prompt=False, ret_src_prompts=True)
    train_data_tgt = read_essay_sets_word(read_configs['train_path'], word_vocab, target_prompt,
                                          ret_target_prompt=True, ret_src_prompts=False)
    dev_data_src = read_essay_sets_word(read_configs['dev_path'], word_vocab, target_prompt,
                                        ret_target_prompt=False, ret_src_prompts=True)
    dev_data_tgt = read_essay_sets_word(read_configs['dev_path'], word_vocab, target_prompt,
                                        ret_target_prompt=True, ret_src_prompts=False)

    return train_data_src, train_data_tgt, dev_data_src, dev_data_tgt
