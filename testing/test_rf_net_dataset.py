from model_zoo.reason_flow_net.dataset import RFDataset
from x import registry, XCfgs
from torch.utils.data import DataLoader


if __name__ == "__main__":
    config_path = "../experiment_configs/rf_gt.yaml"
    _C = XCfgs(config_path)
    _C.proc()

    config = _C.get_config()
    dataset = registry.get_dataset(name = "RFDataset")
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
      #  kg_entity = batch["kg_entity"]
      #  kg_connect = batch["kg_connect"]
      #  kg_r = batch["kg_r"]
      #  r_nodes = batch["r_nodes"]
      #  r_connects = batch["r_connects"]
      #  r_type = batch["r_type"]
      #  img_feat = batch["img_feat"]
      #  img_connection = batch["img_connections"]

        for key, value in batch.items():
            print(key)
            print(value.shape)
            print(value)
        break
        print("-"*10)