from x import XExecution, registry
from torch.utils.data import DataLoader
from x.common.util import get_numpy_word_embed, logging_format
import torch.optim as Optim
import torch.nn as nn
import os
import torch
import pdb

@registry.register_execution(name="QuesEstExecution")
class QuesEstExecution(XExecution):

    def __init__(self, _C):
        super(QuesEstExecution,self).__init__(_C)

        self.logger.info("Building model ......")

        if _C.MODEL.use_glove:
            pretrained_word_embed = get_numpy_word_embed(None, os.path.join(_C.DATA.parent_path,
                                                                            _C.MODEL.pretrained_word_path))
            _model_init = registry.get_model(_C.MODEL.name)
            self.model = _model_init(_C.MODEL, pretrained_word_embed)
        else:
            _model_init = registry.get_model(_C.MODEL.name)
            self.model = _model_init(_C.MODEL)

        self.logger_format = logging_format(["loss"], show_iter=True)

    def train(self, dataset, eval_dataset = None):

        self.model.to(self._device)

        if self._C.EXECUTION.n_gpu > 1:
            self.model = nn.DataParallel(self.model)

        # Define loss
        loss_fn = nn.CrossEntropyLoss(reduction="mean",ignore_index = 1).cuda()

        optimizer = Optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                               lr = self._C.OPTIM.lr_base,
                               betas = self._C.OPTIM.opt_betas,
                               eps = self._C.OPTIM.opt_eps)

        if self._C.EXECUTION.resume:
            path = self._C.EXECUTION.loaded_checkpoint_path
            model_dict, optimizer_dict = self._checkpointing.loaded_checkpoint(path)

            self.logger.info("Loading ckpt from {}".format(path))
            self.model.load_state_dict(model_dict)
            optimizer.load_state_dict(optimizer_dict)

        lr_scheduler = self._lr_scheduler(lr_base = self._C.OPTIM.lr_base,
                                          optimizer = optimizer,
                                          method = self._C.OPTIM.lr_scheduler,
                                          step = self._C.OPTIM.step,
                                          step2 = self._C.OPTIM.step2,
                                          total_step = self._C.OPTIM.max_epoch)

        checkpointing = self._checkpointing(model = self.model,
                                            optimizer = optimizer,
                                            checkpoint_dirpath = self._C.EXECUTION.save_path.format(version=self._C.EXECUTION.version))

        data_loader = DataLoader(dataset = self.dataset,
                                 batch_size = self._C.DATA.batch_size,
                                 shuffle = self._C.DATA.shuffle,
                                 num_workers = self._C.DATA.num_workers,
                                 pin_memory = False,
                                 drop_last = True)

        lr_scheduler.step()
        num_step_per_epoch = len(dataset) // self._C.DATA.batch_size
        sum_loss = 0
        for epoch in range(self._C.OPTIM.max_epoch):
            for step, item in enumerate(data_loader):
                self.model.train()
                optimizer.zero_grad()
                question = item["question"].to(self._device)
                reason_in = item["reason_in"].to(self._device)
                reason_gt = item["reason_gt"].to(self._device)

                reason_pre = self.model(question, reason_in)
                reason_gt = reason_gt.view(-1)
                reason_pre = reason_pre.view(self._C.DATA.batch_size*self._C.DATA.max_seq_len,-1)

                loss = loss_fn(reason_pre, reason_gt)

                num_iter = epoch * num_step_per_epoch +  step
                sum_loss += loss

                if num_iter % self._C.EXECUTION.output_steps == 0:
                    print_loss = sum_loss / self._C.EXECUTION.output_steps
                    self.logger.info(self.logger_format % (epoch, num_iter, print_loss))
                    sum_loss = 0

                loss.backward()
                optimizer.step()
            checkpointing.step()
            lr_scheduler.step()

            if self._C.EXECUTION.is_valid:
                self.logger.info("Fire evaluation !!!")
                self.eval(self.eval_dataset, epoch)

    def eval(self, dataset, epoch=None, state_dict_path=None, valid=False):
        if state_dict_path is not None:
            model_dict, _ = self._checkpointing.load_checkpoint(check_point_path = state_dict_path)
            self.model.load_state_dict(model_dict)
        self.model.eval()

        dataloader = DataLoader(dataset = dataset,
                                batch_size= self._C.DATA.valid_batch_size,
                                shuffle=False,
                                num_workers=self._C.DATA.num_workers,
                                pin_memory=False,
                                drop_last=False)

        log_eval_format = logging_format(["accuracy"],False)
        cur_num = 0
        for _, item in enumerate(dataloader):
            question = item["question"].to(self._device)
            decoder_input = torch.tensor([2]).expand(self._C.DATA.valid_batch_size,1).to(self._device) # 2 for <S> token
            reason_gt = item["reason_gt"].to(self._device)
            gt_list = list(reason_gt[0].cpu().numpy())
            max_len = gt_list.index(3)
            decoded_words_id = []

            for i in range(self._C.DATA.max_seq_len):
                decoder_output = self.model(question, decoder_input)
                topv,topi = decoder_output.data.topk(1)  # topi shape (batch_size, 1, 1)
                cur_max_index = topi.squeeze(0)[-1]
                if cur_max_index.item() == 3:
                    decoded_words_id.append(3)
                    break
                else:
                    decoded_words_id.append(cur_max_index.item())
                decoder_input = torch.tensor(decoded_words_id).unsqueeze(0).to(self._device)
            if decoded_words_id == gt_list[:max_len+1]:
                cur_num += 1

            if self._C.EXECUTION.show_valid_words:
                decoder_words = self.dataset.vocabulary.to_word(decoded_words_id)
                gt_words = self.dataset.vocabulary.to_word(reason_gt)
                print("decoder output: ",decoder_words)
                print("gt output: ",gt_words)

        accuracy = cur_num / len(dataloader)
        self.logger.info(log_eval_format % (epoch,accuracy))

    def register_all(self):
        try:
            from model_zoo.reason_flow_net.quer_estimator.query_dataset import QueyDataset
            import model_zoo.reason_flow_net.quer_estimator.query_estimator
        except:
            raise ImportError
