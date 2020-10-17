from metrics.metrics import *
from utils.general_utils import rescale_tointscore


class SharedModelEvaluator():

    def __init__(self, test_prompt_id, X_dev_src, X_train_tgt, X_dev_tgt, X_dev_src_prompt_ids, X_train_tgt_prompt_ids,
                 X_dev_tgt_prompt_ids, Y_dev_src, Y_train_tgt, Y_dev_tgt):
        self.test_prompt_id = test_prompt_id
        self.X_dev_src, self.X_train_tgt, self.X_dev_tgt = X_dev_src, X_train_tgt, X_dev_tgt
        self.X_dev_src_prompt_ids, self.X_train_tgt_prompt_ids, self.X_dev_tgt_prompt_ids = \
            X_dev_src_prompt_ids, X_train_tgt_prompt_ids, X_dev_tgt_prompt_ids
        self.Y_dev_src, self.Y_train_tgt, self.Y_dev_tgt = Y_dev_src, Y_train_tgt, Y_dev_tgt
        self.Y_dev_src_org = Y_dev_src.flatten() * 100
        self.Y_train_tgt_org = rescale_tointscore(Y_train_tgt, self.X_train_tgt_prompt_ids)
        self.Y_dev_tgt_org = rescale_tointscore(Y_dev_tgt, self.X_dev_tgt_prompt_ids)
        self.best_src_dev = [-1, -1, -1, -1]
        self.best_tgt_train = [-1, -1, -1, -1]
        self.best_tgt_dev = [-1, -1, -1, -1]

    def calc_correl(self, dev_src_pred, train_tgt_pred, dev_tgt_pred):
        # self.dev_src_pr = pearson(self.Y_dev_src_org, dev_src_pred)
        # self.train_tgt_pr = pearson(self.Y_train_tgt_org, train_tgt_pred)
        # self.dev_tgt_pr = pearson(self.Y_dev_tgt_org, dev_tgt_pred)
        self.dev_src_pr = 0
        self.train_tgt_pr = 0
        self.dev_tgt_pr = 0

        self.dev_src_spr = spearman(self.Y_dev_src_org, dev_src_pred)
        self.train_tgt_spr = spearman(self.Y_train_tgt_org, train_tgt_pred)
        self.dev_tgt_spr = spearman(self.Y_dev_tgt_org, dev_tgt_pred)

    def calc_kappa(self, dev_src_pred, train_tgt_pred, dev_tgt_pred, weight='quadratic'):
        self.dev_src_qwk = kappa(self.Y_dev_src_org, dev_src_pred, weight)
        self.train_tgt_qwk = kappa(self.Y_train_tgt_org, train_tgt_pred, weight)
        self.dev_tgt_qwk = kappa(self.Y_dev_tgt_org, dev_tgt_pred, weight)

    def calc_rmse(self, dev_src_pred, train_tgt_pred, dev_tgt_pred):
        self.dev_src_rmse = root_mean_square_error(self.Y_dev_src_org, dev_src_pred)
        self.train_tgt_rmse = root_mean_square_error(self.Y_train_tgt_org, train_tgt_pred)
        self.dev_tgt_rmse = root_mean_square_error(self.Y_dev_tgt_org, dev_tgt_pred)

    def evaluate(self, model, epoch, print_info=True):
        dev_src_latent = model.feature_generator.predict(self.X_dev_src)
        train_tgt_latent = model.feature_generator.predict(self.X_train_tgt)
        dev_tgt_latent = model.feature_generator.predict(self.X_dev_tgt)

        dev_src_pred = model.scorer.predict(dev_src_latent, batch_size=32).squeeze()
        train_tgt_pred = model.scorer.predict(train_tgt_latent, batch_size=32).squeeze()
        dev_tgt_pred = model.scorer.predict(dev_tgt_latent, batch_size=32).squeeze()

        dev_src_pred_int = dev_src_pred.flatten() * 100
        train_tgt_pred_int = rescale_tointscore(train_tgt_pred, self.X_train_tgt_prompt_ids)
        dev_tgt_pred_int = rescale_tointscore(dev_tgt_pred, self.X_dev_tgt_prompt_ids)

        self.calc_correl(dev_src_pred_int, train_tgt_pred_int, dev_tgt_pred_int)
        self.calc_kappa(dev_src_pred_int, train_tgt_pred_int, dev_tgt_pred_int)
        self.calc_rmse(dev_src_pred_int, train_tgt_pred_int, dev_tgt_pred_int)

        if self.dev_src_qwk > self.best_src_dev[0]:
            self.best_src_dev = [self.dev_src_qwk, self.dev_src_pr, self.dev_src_spr, self.dev_src_rmse]
            self.best_tgt_train = [self.train_tgt_qwk, self.train_tgt_pr, self.train_tgt_spr, self.train_tgt_rmse]
            self.best_tgt_dev = [self.dev_tgt_qwk, self.dev_tgt_pr, self.dev_tgt_spr, self.dev_tgt_rmse]
            self.best_dev_epoch = epoch
        if print_info:
            self.print_info()

    def print_info(self):
        print(
            '[DEV SRC]   QWK:  %.3f, PRS: %.3f, SPR: %.3f, RMSE: %.3f, (Best @ %i: {{%.3f}}, %.3f, %.3f, %.3f)' % (
                self.dev_src_qwk, self.dev_src_pr, self.dev_src_spr, self.dev_src_rmse, self.best_dev_epoch,
                self.best_src_dev[0], self.best_src_dev[1], self.best_src_dev[2], self.best_src_dev[3]))
        print(
            '[TRAIN TGT]  QWK:  %.3f, PRS: %.3f, SPR: %.3f, RMSE: %.3f (Best @ %i: {{%.3f}}, %.3f, %.3f, %.3f)' % (
                self.train_tgt_qwk, self.train_tgt_pr, self.train_tgt_spr, self.train_tgt_rmse, self.best_dev_epoch,
                self.best_tgt_train[0], self.best_tgt_train[1], self.best_tgt_train[2], self.best_tgt_train[3]))
        print(
            '[DEV TGT]  QWK:  %.3f, PRS: %.3f, SPR: %.3f, RMSE: %.3f (Best @ %i: {{%.3f}}, %.3f, %.3f, %.3f)' % (
                self.dev_tgt_qwk, self.dev_tgt_pr, self.dev_tgt_spr, self.dev_tgt_rmse, self.best_dev_epoch,
                self.best_tgt_dev[0], self.best_tgt_dev[1], self.best_tgt_dev[2], self.best_tgt_dev[3]))

        print(
            '--------------------------------------------------------------------------------------------------------------------------')

    def print_final_info(self):
        print(
            '--------------------------------------------------------------------------------------------------------------------------')
        print('Best @ Epoch %i:' % self.best_dev_epoch)
        print('  [DEV SRC]  QWK: %.3f,  PRS: %.3f, SPR: %.3f, RMSE: %.3f' % (
            self.best_src_dev[0], self.best_src_dev[1], self.best_src_dev[2], self.best_src_dev[3]))
        print('  [TRAIN TGT] QWK: %.3f,  PRS: %.3f, SPR: %.3f, RMSE: %.3f' % (
            self.best_tgt_train[0], self.best_tgt_train[1], self.best_tgt_train[2], self.best_tgt_train[3]))
        print('  [DEV TGT] QWK: %.3f,  PRS: %.3f, SPR: %.3f, RMSE: %.3f' % (
            self.best_tgt_dev[0], self.best_tgt_dev[1], self.best_tgt_dev[2], self.best_tgt_dev[3]))



