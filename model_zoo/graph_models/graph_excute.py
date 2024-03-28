from x.core.registry import registry
from x.core.execution import XExecution
from x.common.util import logging_format
from torch.utils.data import DataLoader
from x.common.metrics import get_recall_at_k
from model_zoo.graph_models.graph_model_zoo.net_utils import FocalLoss_Multilabel
import torch.nn as nn
import torch.optim as Optim
import os
import torch
import json
import pdb
import time


@registry.register_execution(name="GExecution")
class CGRMExecution(XExecution):
    def __init__(self, config):
        super(CGRMExecution, self).__init__(config)
        self.config = config
        self.logger.info("Build model ......")
        _model_init = registry.get_model(name=config.MODEL.name)
        self.model = _model_init(config.MODEL)
        self.logger.info(self.model)
        self.logger_format = logging_format(
            ["pred_loss"], show_iter=True
        )
        self.logger.info(config)

    def train(self, dataset, eval_dataset=None):
        self.model.to(self._device)

        if self.config.EXECUTION.n_gpu > 1:
            self.model = nn.DataParallel(self.model)

        loss_fn = nn.CrossEntropyLoss(reduction="mean").cuda()
        #loss_q_type = nn.CrossEntropyLoss(reduction="mean").cuda()
        # loss_fn = FocalLoss_Multilabel(self.config.MODEL.num_answer,use_alpha=True)

        optimizer = Optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config.OPTIM.lr_base,
            betas=self.config.OPTIM.opt_betas,
            eps=self.config.OPTIM.opt_eps,
        )

        if self.config.EXECUTION.resume:
            path = self.config.EXECUTION.loaded_checkpoint_path
            model_dict, optimizer_dict = self._checkpointing.loaded_checkpoint(path)

            self.logger.info("Loading ckpt from {}".format(path))
            self.model.load_state_dict(model_dict)
            optimizer.load_state_dict(optimizer_dict)

        lr_scheduler = self._lr_scheduler(
            lr_base=self.config.OPTIM.lr_base,
            optimizer=optimizer,
            method=self.config.OPTIM.lr_scheduler,
            step=self.config.OPTIM.step,
            step2=self.config.OPTIM.step2,
            total_step=self.config.OPTIM.max_epoch,
        )

        checkpointing = self._checkpointing(
            model=self.model,
            optimizer=optimizer,
            checkpoint_dirpath=self.config.EXECUTION.save_path.format(
                version=self.config.EXECUTION.version
            ),
        )

        data_loader = DataLoader(
            dataset=self.dataset,
            batch_size=self.config.DATA.batch_size,
            shuffle=self.config.DATA.shuffle,
            num_workers=self.config.DATA.num_workers,
            pin_memory=False,
            drop_last=True,
        )

        lr_scheduler.step()
        num_step_per_epoch = len(self.dataset) // self.config.DATA.batch_size
        sum_loss = 0

        for epoch in range(self.config.OPTIM.max_epoch):

            for step, item in enumerate(data_loader):
                self.model.train()
                optimizer.zero_grad()
                for k, v in item.items():
                    item[k] = v.to(self._device)
                pred_answer = self.model(item)
                answer = item["answer"]  # .unsqueeze(1)

                # qtype = item["q_type"]
                # answer = torch.zeros(answer.shape[0], pred_answer.shape[1]).to(self._device).scatter_(
                #    1, answer, 1
                # )  uncomment if use bce loss
                loss_pred = loss_fn(pred_answer, answer)
                #loss_qtype = 0  # loss_q_type(q_type_pred, qtype)

                loss = loss_pred
                num_iter = epoch * num_step_per_epoch + step
                sum_loss += loss
                #sum_loss_pred += loss_pred
                #sum_loss_q_type += 10 * loss_qtype

                if num_iter % self.config.EXECUTION.output_steps == 0:
                    print_loss = sum_loss / self.config.EXECUTION.output_steps
                    #print_loss_pred = sum_loss_pred / self.config.EXECUTION.output_steps
                    #print_loss_qtype = (
                    #    sum_loss_q_type / self.config.EXECUTION.output_steps
                    #)

                    self.logger.info(
                        self.logger_format
                        % (
                            epoch,
                            num_iter,
                            print_loss,
                        )
                    )
                    sum_loss = 0
                    #sum_loss_pred = 0
                    #sum_loss_q_type = 0
                loss.backward()
                optimizer.step()

            checkpointing.step()
            lr_scheduler.step()

            if self.config.EXECUTION.is_valid and epoch >= 0:
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
            batch_size=self.config.DATA.valid_batch_size,
            shuffle=False,
            num_workers=self.config.DATA.num_workers,
            pin_memory=False,
            drop_last=False,
        )

        accuracy = 0
        log_eval_formate = logging_format(["accuracy"], False)
        for _, item in enumerate(data_loader):
            for k, v in item.items():
                item[k] = v.to(self._device)

            pred_answer, _ = self.model(item)
            answer = item["answer"]
            accuracy_batch = get_recall_at_k(pred_answer, answer, 1)
            accuracy += accuracy_batch

        accuracy_ave = accuracy / len(dataset)
        self.logger.info(log_eval_formate % (epoch, accuracy_ave))

    def analysis(self, config):

        if config.dataset == "train":
            dataset = self.dataset
        elif config.dataset == "valid":
            dataset = self.eval_dataset
        output_path = config.output_path
        state_dict_path = config.state_dict_path

        assert state_dict_path is not None
        assert output_path is not None

        components = torch.load(state_dict_path)
        model_dict = components["model"]

        self.model.load_state_dict(model_dict)
        self.model.to(self._device)
        self.model.eval()

        data_loader = DataLoader(
            dataset=dataset,
            batch_size=self.config.DATA.valid_batch_size,
            shuffle=False,
            num_workers=self.config.DATA.num_workers,
            pin_memory=False,
            drop_last=False,
        )
        results = {"results": []}
        a = 0
        t = 0
        answer_id2word = dataset.candidate_answers
        for q_id, item in enumerate(data_loader):
            question = item["question"].to(self._device)
            question_mask = item["question_mask"].to(self._device)
            answer = item["answer"].to(self._device)
            img_nodes_feature = item["img_nodes_feature"].to(self._device)
            img_edges_feature = item["img_edges_feature"].to(self._device)
            img_node1_ids_list = item["img_node1_ids_list"].to(self._device)
            img_node2_ids_list = item["img_node2_ids_list"].to(self._device)
            kg_entity_tensor = item["kg_entity_tensor"].to(self._device)
            kg_edge_tensor = item["kg_edge_tensor"].to(self._device)
            kg_node1_ids_list = item["kg_node1_ids_list"].to(self._device)
            kg_node2_ids_list = item["kg_node2_ids_list"].to(self._device)
            question_id = item["question_id"].to(self._device)

            pred_answer = self.model(
                question=question,
                question_mask=question_mask,
                img_nodes=img_nodes_feature,
                img_edges=img_edges_feature,
                img_node1_ids_list=img_node1_ids_list,
                img_node2_ids_list=img_node2_ids_list,
                kg_nodes=kg_entity_tensor,
                kg_edges=kg_edge_tensor,
                kg_node1_ids_list=kg_node1_ids_list,
                kg_node2_ids_list=kg_node2_ids_list,
            )

            batch_size, _ = question.shape

            for b in range(batch_size):
                gt = answer[b].item()
                pred = pred_answer[b]
                _, indices = torch.sort(pred, descending=True)
                pred_index = indices[0].item()
                t += 1
                if pred_index == gt:
                    a += 1
                    is_correct = 1
                else:
                    is_correct = 0

                results["results"].append(
                    {
                        "questinon_id": question_id[b].item(),
                        "predict_answer_index": pred_index,
                        "predict_answer": answer_id2word[pred_index],
                        "ground_truth_index": gt,
                        "ground_truth_answer": answer_id2word[gt],
                        "is_correct": is_correct,
                    }
                )
        results["accuracy"] = a / t
        with open(output_path, "w") as f:
            json.dump(results, f)
        print("DONE !!!!")

    def run(self, run_mode):
        if run_mode == "train":
            if self._C.EXECUTION.is_valid:
                self.train(self.dataset, self.eval_dataset)
            else:
                self.train(self.dataset, None)

        elif run_mode == "val":
            self.eval(self.dataset, valid=True)

        elif run_mode == "test":
            assert (
                self._C.DATA.split == "test"
            ), "Data split is {}. Split has to be 'test' during testing !!".format(
                self._C.DATA.split
            )
            self.eval(self.dataset)
        elif run_mode == "analysis":
            self.analysis(self.config.ANALYSIS)

    def register_all(self):
        from model_zoo.graph_models import models
        from model_zoo.graph_models.graph_dataset import GraphDataset
