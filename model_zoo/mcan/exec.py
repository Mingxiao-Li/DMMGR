from x import *
from x.common.util import get_numpy_word_embed, logging_format
from x.common.metrics import get_recall_at_k
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as Optim
import os, torch

torch.set_printoptions(profile="full")


@registry.register_execution(name="MCANExecution")
class MCANExecution(XExecution):
    def __init__(self, _C):
        super(MCANExecution, self).__init__(_C)

        self.logger.info("Building model ......")
        #if _C.MODEL.use_glove:
        #    pretrained_word_embed = get_numpy_word_embed(
        #        None, os.path.join(_C.DATA.parent_path, _C.MODEL.pretrained_word_path)
        #    )
        #    _model_init = registry.get_model(_C.MODEL.name)
        #    self.model = _model_init(_C.MODEL, pretrained_word_embed)
        #else:
        _model_init = registry.get_model(_C.MODEL.name)
        self.model = _model_init(_C.MODEL)

        self.logger_format = logging_format(["loss"], show_iter=True)
        self.logger.info(_C)

    def train(self, dataset, eval_dataset=None):

        # self.model = self.model.double()
        self.model.train()
        self.model.to(self._device)
        # Define the multi-gpu training if need
        if self._C.EXECUTION.n_gpu > 1:
            self.model = nn.DataParallel(self.model)

        # Define loss
        loss_fn = nn.CrossEntropyLoss(reduction="mean").cuda()

        optimizer = Optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self._C.OPTIM.lr_base,
            betas=self._C.OPTIM.opt_betas,
            eps=self._C.OPTIM.opt_eps,
        )

        if self._C.EXECUTION.resume:
            path = self._C.EXECUTION.loaded_checkpoint_path
            model_dict, optimizer_dict = self.checkpointing.loaded_checkpoint(path)

            self.logger.info("Loading ckpt from {}".format(path))
            self.model.load_state_dict(model_dict)
            optimizer.load_state_dict(optimizer_dict)

        lr_scheduler = self._lr_scheduler(
            lr_base=self._C.OPTIM.lr_base,
            optimizer=optimizer,
            method=self._C.OPTIM.lr_scheduler,
            step=self._C.OPTIM.step,
            step2=self._C.OPTIM.step2,
            total_step=self._C.OPTIM.max_epoch,
        )

        checkpointing = self._checkpointing(
            model=self.model,
            optimizer=optimizer,
            checkpoint_dirpath=self._C.EXECUTION.save_path.format(
                version=self._C.EXECUTION.version
            ),
        )

        data_loader = DataLoader(
            dataset=self.dataset,
            batch_size=self._C.DATA.batch_size,
            shuffle=self._C.DATA.shuffle,
            num_workers=self._C.DATA.num_workers,
            pin_memory=False,
            drop_last=True,
        )

        lr_scheduler.step()
        num_setp_per_epoch = len(dataset) // self._C.DATA.batch_size
        sum_loss = 0
        for epoch in range(self._C.OPTIM.max_epoch):
            for step, item in enumerate(data_loader):
                self.model.train()
                optimizer.zero_grad()
                question = item["question"].to(self._device)
                answer = item["answer"].to(self._device)
                question_mask = item["question_mask"].to(self._device)
                feature = item["feature"].to(self._device)
                img_loc = item["img_loc"].to(self._device)
                if "facts" in item.keys():
                    kb = item["facts"].to(self._device)
                    kb_mask = item["facts_mask"].to(self._device)
                else:
                    kb = None
                    kb_mask = None
                pred_answer = self.model(question, kb, question_mask, kb_mask, feature)
                loss = loss_fn(pred_answer, answer)

                num_iter = epoch * num_setp_per_epoch + step
                sum_loss += loss
                if num_iter % self._C.EXECUTION.outpu_steps == 0:
                    print_loss = sum_loss / self._C.EXECUTION.outpu_steps
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
            model_dict, _ = self._checkpointing.load_checkpoint(
                check_pointpath=state_dict_path
            )
            self.model.load_state_dict(model_dict)
        self.model.eval()

        data_loader = DataLoader(
            dataset=dataset,
            batch_size=self._C.DATA.valid_batch_size,
            shuffle=False,
            num_workers=self._C.DATA.num_workers,
            pin_memory=False,
            drop_last=False,
        )

        r_1 = 0
        r_3 = 0
        log_eval_format = logging_format(["recall_1", "recall_3"], False)
        for _, item in enumerate(data_loader):
            question = item["question"].to(self._device)
            answer = item["answer"].to(self._device)
            question_mask = item["question_mask"].to(self._device)
            feature = item["feature"].to(self._device)
            img_loc = item["img_loc"].to(self._device)
            if "facts" in item.keys():
                kb = item["facts"].to(self._device)
                kb_mask = item["facts_mask"].to(self._device)
            else:
                kb = None
                kb_mask = None

            pred_answer = self.model(question, kb, question_mask, kb_mask, feature)
            r_1_batch = get_recall_at_k(pred_answer, answer, 1)
            r_3_batch = get_recall_at_k(pred_answer, answer, 3)
            r_1 += r_1_batch
            r_3 += r_3_batch

        r_1 = r_1 / len(dataset)
        r_3 = r_3 / len(dataset)
        self.logger.info(log_eval_format % (epoch, r_1, r_3))

    def register_all(self):
        try:
            from model_zoo.mcan.dataset import MCANDataset
            from model_zoo.mcan.mcan import MACNet
        except:
            raise ImportError
