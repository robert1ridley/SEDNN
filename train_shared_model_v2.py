import os
import math
import argparse
import random
import numpy as np
import tensorflow as tf
from configs.configs import Configs
from models.shared_model_v2 import SharedModelV2
from evaluators.shared_model_evaluator_v2 import SharedModelEvaluatorV2
from utils.read_data import read_word_vocab, read_essays_words
from utils.general_utils import get_scaled_down_scores, pad_hierarchical_text_sequences, load_word_embedding_dict, \
    build_embedd_table


def batch_generator(data, batch_size):
    """Generate batches of data.

    Given a list of numpy data, it iterates over the list and returns batches of the same size
    """
    all_examples_indices = len(data[0])
    while True:
        mini_batch_indices = np.random.choice(all_examples_indices, size=batch_size, replace=False)
        tbr = [k[mini_batch_indices] for k in data]
        yield tbr


@tf.function
def full_train_step(X_train_src_batch, Y_train_src_batch, X_both, label_both,
                    shared_model, score_loss_fn, disc_loss_fn, optimizer):

    with tf.GradientTape() as tape:
        y_score_pred = shared_model.predict_score(X_train_src_batch, training=True)
        score_loss = get_loss(disc_loss_fn, score_loss_fn, d_logits=None, domain=None, s_logits=y_score_pred,
                              s_labels=Y_train_src_batch)
    score_grad = tape.gradient(score_loss, shared_model.predict_score.trainable_variables)

    with tf.GradientTape(persistent=True) as tape:
        y_domain_pred = shared_model.predict_domain(X_both, training=True)
        dc_loss = get_loss(disc_loss_fn, score_loss_fn, d_logits=y_domain_pred, domain=label_both,
                           s_logits=None, s_labels=None)
        y_score_pred = shared_model.predict_score(X_train_src_batch, training=True)
        temp_score_loss = get_loss(disc_loss_fn, score_loss_fn, d_logits=None, domain=None, s_logits=y_score_pred,
                              s_labels=Y_train_src_batch)
        fe_loss = temp_score_loss + dc_loss
    fe_grad = tape.gradient(fe_loss, shared_model.feature_extractor.trainable_variables)
    dc_grad = tape.gradient(dc_loss, shared_model.discriminator.trainable_variables)
    del tape

    optimizer.apply_gradients(zip(score_grad, shared_model.predict_score.trainable_variables))

    optimizer.apply_gradients(zip(dc_grad, shared_model.discriminator.trainable_variables))

    optimizer.apply_gradients(zip(fe_grad, shared_model.feature_extractor.trainable_variables))

    return fe_loss, score_loss, dc_loss


def get_loss(disc_loss_fn, score_loss_fn, d_logits=None, domain=None, s_logits=None, s_labels=None):
    if s_logits is None:
        return disc_loss_fn(domain, d_logits)
    elif d_logits is None:
        return score_loss_fn(s_labels, s_logits)


def main():
    parser = argparse.ArgumentParser(description="Shared Model")
    parser.add_argument('--test_prompt_id', type=int, default=1, help='prompt id of test essay set')
    parser.add_argument('--seed', type=int, default=12, help='set random seed')
    args = parser.parse_args()
    test_prompt_id = args.test_prompt_id
    seed = args.seed

    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    print("Test prompt id is {} of type {}".format(test_prompt_id, type(test_prompt_id)))
    print("Seed: {}".format(seed))

    configs = Configs()

    data_path = configs.DATA_PATH
    train_path = data_path + '/train.tsv'
    dev_path = data_path + '/dev.tsv'
    pretrained_embedding = configs.PRETRAINED_EMBEDDING
    embedding_path = configs.EMBEDDING_PATH
    embedding_dim = configs.EMBEDDING_DIM
    vocab_size = configs.VOCAB_SIZE
    epochs = configs.EPOCHS
    batch_size = configs.BATCH_SIZE

    read_configs = {
        'train_path': train_path,
        'dev_path': dev_path,
        'vocab_size': vocab_size
    }

    word_vocab = read_word_vocab(read_configs)
    print('vocab complete')
    train_data_src, train_data_tgt, dev_data_src, dev_data_tgt = \
        read_essays_words(read_configs, word_vocab, test_prompt_id)

    if pretrained_embedding:
        embedd_dict, embedd_dim, _ = load_word_embedding_dict(embedding_path)
        embedd_matrix = build_embedd_table(word_vocab, embedd_dict, embedd_dim, caseless=True)
        embed_table = [embedd_matrix]
    else:
        embed_table = None

    max_sentlen = max(train_data_src['max_sentlen'], train_data_tgt['max_sentlen'],
                      dev_data_src['max_sentlen'], dev_data_tgt['max_sentlen'])
    max_sentnum = max(train_data_src['max_sentnum'], train_data_tgt['max_sentnum'],
                      dev_data_src['max_sentnum'], dev_data_tgt['max_sentnum'])
    print('max sent length: {}'.format(max_sentlen))
    print('max sent num: {}'.format(max_sentnum))
    train_data_src['y_scaled'] = get_scaled_down_scores(train_data_src['data_y'], train_data_src['prompt_ids'])
    train_data_tgt['y_scaled'] = get_scaled_down_scores(train_data_tgt['data_y'], train_data_tgt['prompt_ids'])
    dev_data_src['y_scaled'] = get_scaled_down_scores(dev_data_src['data_y'], dev_data_src['prompt_ids'])
    dev_data_tgt['y_scaled'] = get_scaled_down_scores(dev_data_tgt['data_y'], dev_data_tgt['prompt_ids'])

    X_train_src = pad_hierarchical_text_sequences(train_data_src['words'], max_sentnum, max_sentlen)
    X_train_tgt = pad_hierarchical_text_sequences(train_data_tgt['words'], max_sentnum, max_sentlen)
    X_dev_src = pad_hierarchical_text_sequences(dev_data_src['words'], max_sentnum, max_sentlen)
    X_dev_tgt = pad_hierarchical_text_sequences(dev_data_tgt['words'], max_sentnum, max_sentlen)

    X_train_src = X_train_src.reshape((X_train_src.shape[0], X_train_src.shape[1] * X_train_src.shape[2]))
    X_train_tgt = X_train_tgt.reshape((X_train_tgt.shape[0], X_train_tgt.shape[1] * X_train_tgt.shape[2]))
    X_dev_src = X_dev_src.reshape((X_dev_src.shape[0], X_dev_src.shape[1] * X_dev_src.shape[2]))
    X_dev_tgt = X_dev_tgt.reshape((X_dev_tgt.shape[0], X_dev_tgt.shape[1] * X_dev_tgt.shape[2]))

    Y_train_src = np.array(train_data_src['y_scaled'])
    Y_train_tgt = np.array(train_data_tgt['y_scaled'])
    Y_dev_src = np.array(dev_data_src['y_scaled'])
    Y_dev_tgt = np.array(dev_data_tgt['y_scaled'])

    train_src_batches = batch_generator(
        [X_train_src, Y_train_src], batch_size)
    train_tgt_batches = batch_generator(
        [X_train_tgt, Y_train_tgt], batch_size)

    disc_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    score_loss_fn = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    shared_model = SharedModelV2(len(word_vocab), max_sentnum, max_sentlen, embedding_dim, embed_table)

    steps = (X_train_src.shape[0] // batch_size) * epochs

    evaluator = SharedModelEvaluatorV2(test_prompt_id, X_dev_src, X_train_tgt, X_dev_tgt, dev_data_src['prompt_ids'],
                                       train_data_tgt['prompt_ids'], dev_data_tgt['prompt_ids'], Y_dev_src, Y_train_tgt,
                                       Y_dev_tgt)

    evaluator.evaluate(shared_model, 0, print_info=True)

    for step in range(steps):
        src_label = tf.zeros((batch_size, 1))
        tgt_label = tf.ones((batch_size, 1))

        X_train_src_batch, Y_train_src_batch = next(train_src_batches)
        X_train_tgt_batch, Y_train_tgt_batch = next(train_tgt_batches)

        X_both = tf.concat([X_train_src_batch, X_train_tgt_batch], axis=0)
        label_both = tf.concat([src_label, tgt_label], axis=0)

        fe_loss, score_loss, disc_loss = full_train_step(X_train_src_batch, Y_train_src_batch, X_both, label_both,
                                                shared_model, score_loss_fn, disc_loss_fn, optimizer)

        current_step = step + 1
        if current_step % (steps//epochs) == 0:
            print(
                "fe loss (for one batch) at step %d: %.4f"
                % (current_step, float(fe_loss))
            )
            print(
                "score loss (for one batch) at step %d: %.4f"
                % (current_step, float(score_loss))
            )
            print(
                "disc loss (for one batch) at step %d: %.4f"
                % (current_step, float(disc_loss))
            )
            print('steps', steps)
            print('step', current_step)
            print('epochs', epochs)
            print('batch_size', batch_size)
            print('Evaluating epoch', current_step/(steps//epochs))
            if step == 0:
                evaluator.evaluate(shared_model, 0)
            else:
                evaluator.evaluate(shared_model, current_step/(steps//epochs))

    evaluator.print_final_info()


if __name__ == '__main__':
    main()
