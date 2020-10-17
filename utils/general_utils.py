import numpy as np


def get_min_max_scores():
    return {
        1: (2, 12),
        2: (1, 6),
        3: (0, 3),
        4: (0, 3),
        5: (0, 4),
        6: (0, 4),
        7: (0, 30),
        8: (0, 60)
    }


def get_scaled_down_scores(scores, prompts):
    min_max_scores = get_min_max_scores()
    score_prompts = zip(scores, prompts)
    scaled_score_list = []
    for score, prompt in score_prompts:
        min_val = min_max_scores[prompt][0]
        max_val = min_max_scores[prompt][1]
        scaled_score = (score - min_val) / (max_val - min_val)
        scaled_score_list.append([scaled_score])
    assert len(scaled_score_list) == len(scores)
    return scaled_score_list


def rescale_tointscore(scaled_scores, set_ids):
    '''
    rescale scaled scores range[0,1] to original integer scores based on  their set_ids
    :param scaled_scores: list of scaled scores range [0,1] of essays
    :param set_ids: list of corresponding set IDs of essays, integer from 1 to 8
    '''
    if isinstance(set_ids, int):
        prompt_id = set_ids
        set_ids = np.ones(scaled_scores.shape[0],) * prompt_id
    assert scaled_scores.shape[0] == len(set_ids)
    int_scores = np.zeros((scaled_scores.shape[0], 1))
    for k, i in enumerate(set_ids):
        assert i in range(1, 9)
        # TODO
        if i == 1:
            minscore = 2
            maxscore = 12
        elif i == 2:
            minscore = 1
            maxscore = 6
        elif i in [3, 4]:
            minscore = 0
            maxscore = 3
        elif i in [5, 6]:
            minscore = 0
            maxscore = 4
        elif i == 7:
            minscore = 0
            maxscore = 30
        elif i == 8:
            minscore = 0
            maxscore = 60
        else:
            print ("Set ID error")

        int_scores[k] = scaled_scores[k]*(maxscore-minscore) + minscore
    return np.around(int_scores).astype(int)


def pad_hierarchical_text_sequences(index_sequences, max_sentnum, max_sentlen):
    X = np.empty([len(index_sequences), max_sentnum, max_sentlen], dtype=np.int32)

    for i in range(len(index_sequences)):
        sequence_ids = index_sequences[i]
        num = len(sequence_ids)

        for j in range(num):
            word_ids = sequence_ids[j]
            length = len(word_ids)
            for k in range(length):
                wid = word_ids[k]
                X[i, j, k] = wid
            X[i, j, length:] = 0

        X[i, num:, :] = 0
    return X


def load_word_embedding_dict(embedding_path):
    print("Loading GloVe ...")
    embedd_dim = -1
    embedd_dict = dict()
    with open(embedding_path, 'r') as file:
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = line.split()
            if embedd_dim < 0:
                embedd_dim = len(tokens) - 1
            else:
                assert (embedd_dim + 1 == len(tokens))
            embedd = np.empty([1, embedd_dim])
            embedd[:] = tokens[1:]
            embedd_dict[tokens[0]] = embedd
    return embedd_dict, embedd_dim, True


def build_embedd_table(word_alphabet, embedd_dict, embedd_dim, caseless):
    scale = np.sqrt(3.0 / embedd_dim)
    embedd_table = np.empty([len(word_alphabet), embedd_dim])
    embedd_table[0, :] = np.zeros([1, embedd_dim])
    oov_num = 0
    for word in word_alphabet:
        ww = word.lower() if caseless else word
        if ww in embedd_dict:
            embedd = embedd_dict[ww]
        else:
            embedd = np.random.uniform(-scale, scale, [1, embedd_dim])
            oov_num += 1
        embedd_table[word_alphabet[word], :] = embedd
    oov_ratio = float(oov_num)/(len(word_alphabet)-1)
    print("OOV number =%s, OOV ratio = %f" % (oov_num, oov_ratio))
    return embedd_table
