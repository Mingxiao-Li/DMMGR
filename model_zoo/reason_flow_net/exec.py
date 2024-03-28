from x import *
from x.common.util import get_numpy_word_embed, logging_format
from x.common.metrics import get_recall_at_k
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as Optim
import os
import torch

@registry.register_execution(name="RFExecution")
class RFExecution(XExecution):

    def __init__(self, _C):
        super(RFExecution, self).__init__(_C)

        self.logger.info("Building model ......")
        _model_init = registry.get_model(_C.MODEL.name)
        if _C.MODEL.use_glove:
            pretrained_word_embed = get_numpy_word_embed(None, os.path.join(_C.DATA.parent_path,
                                                                            _C.MODEL.pretrained_word_path))
            self.model = _model_init(_C.MODEL, pretrained_word_embed)
        else:
            self.model = _model_init(_C.MODEL)

        self.logger_format = logging_format(["loss"], show_iter=True)

    def train(self, dataset, eval_dataset = None):

        self.model.to(self._device)

        if self._C.EXECUTION.n_gpu > 1:
            self.model = nn.DataParallel(self.model)

        #Define loss
        loss_fn = nn.CrossEntropyLoss(reduction="mean").cuda()

        optimizer = Optim.Adam(filter(lambda  p: p.requires_grad, self.model.parameters()),
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
                                            checkpoint_dirpath = self._C.EXECUTION.save_path.format(
                                                                 version=self._C.EXECUTION.version))

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
                kg_e1 = item["kg_e_1"].to(self._device)
                kg_e2 = item["kg_e_2"].to(self._devoce)
                kg_r = item["kg_r"].to(self._device)
                r_nodes = item["r_nodes"].to(self._device)
                r_connects = item["r_connects"].to(self._device)
                r_nodes_type = item["r_type"].to(self._device)
                img_feat = item["img_feat"].to(self._device)
                img_connection = item["img_connections"].to(self._device)
                img_loc = item["img_loc"].to(self._device)
                answer = item["answer"].to(self._device)

                pred_answer = self.model(kg_e1,kg_e2,kg_r,r_nodes,r_connects,r_nodes_type,
                                         img_feat,img_connection,img_loc)
                loss = loss_fn(pred_answer, answer)

                num_iter = epoch * num_step_per_epoch + step
                sum_loss += loss
                if num_iter % self._C.EXECUTION.output_steps == 0:
                    print_loss = sum_loss / self._C.EXECUTION.output_steps
                    self.logger.info(self.logger_format % (epoch, num_iter,print_loss))
                    sum_loss = 0
                loss.backward()
                optimizer.step()
            checkpointing.step()
            lr_scheduler.step()

            if self._C.EXECUTION.is_valid:
                self.logger.info("Fire evaluation !!!")
                self.eval(self.eval_dataset, epoch)

    def eval(self,dataset, epoch=None, state_dict_path = None,valid = False):
        if state_dict_path is not None:
            model_dict, _ = self._checkpointing.load_checkpoint(check_pointpath = state_dict_path)
            self.model.load_state_dict(model_dict)
        self.model.eval()

        data_loader = DataLoader(dataset=dataset,
                                 batch_size=self._C.DATA.valid_batch_size,
                                 shuffle=False,
                                 num_workers=self._C.DATA.num_workers,
                                 pin_memory=False,
                                 drop_last=False)

        accuracy = 0
        log_eval_format = logging_format(["accuracy"], False)
        for _, item in enumerate(data_loader):
            kg_e1 = item["kg_e_1"].to(self._device)
            kg_e2 = item["kg_e_2"].to(self._devoce)
            kg_r = item["kg_r"].to(self._device)
            r_nodes = item["r_nodes"].to(self._device)
            r_connects = item["r_connects"].to(self._device)
            r_nodes_type = item["r_type"].to(self._device)
            img_feat = item["img_feat"].to(self._device)
            img_connection = item["img_connections"].to(self._device)
            img_loc = item["img_loc"].to(self._device)
            answer = item["answer"].to(self._device)

            pre_answer = self.model(kg_e1,kg_e2,kg_r,r_nodes,r_connects,r_nodes_type,
                                    img_feat,img_connection,img_loc)
            accuracy_batch = get_recall_at_k(["accuracy"],answer,1)
            accuracy += accuracy_batch
        accuracy = accuracy / len(dataset)
        self.logger.info(log_eval_format % (epoch, accuracy))

    def register_all(self):
        try:
            from model_zoo.reason_flow_net.rfnet import Net
            from model_zoo.reason_flow_net.dataset import RFDataset
        except:
            raise ImportError
