from model_zoo.reason_flow_net.quer_estimator.dataset import QueyDataset
from x import registry, XCfgs
from torch.utils.data import DataLoader


if __name__ == "__main__":
    config_path = "../experiment_configs/quer_estimator_rnn.yaml"
    _C = XCfgs(config_path)
    _C.proc()

    config = _C.get_config()
    dataset = registry.get_dataset(name = "QueryDataset")
    dataset = dataset(config.DATA)
    dataloader = DataLoader(dataset = dataset,
                            batch_size = config.DATA.batch_size,
                            num_workers = 0,
                            pin_memory = False,
                            shuffle = False,
                            drop_last=False)
    for i,batch in enumerate(dataloader):
        #print(batch)
        #print(batch[0].shape)
        #print(batch[1].shape)
        k=i