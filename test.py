import torch
import torch.nn as nn
import torchvision
import torch.optim
import numpy as np

import os
import argparse
import utils
from tqdm import tqdm

from networks import graph_network
from graphmethods import build_adjacency_matrices

import time

def test(config):
    enhan_net = graph_network.graph_net(config.block_size).cuda()
    utils.load_checkpoint(enhan_net, os.path.join(config.checkpoint_path, config.net_name, 'model_best_1.pth'))
    
    print("gpu_id:", config.cudaid)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config.cudaid
    device_ids = [i for i in range(torch.cuda.device_count())]

    if torch.cuda.device_count() > 1:
        enhan_net = nn.DataParallel(enhan_net, device_ids=device_ids)
        
    print(os.path.join(config.ori_images_path, config.dataset_name))
    test_dataset = utils.test_loader(config.ori_images_path)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False,
                                               num_workers=config.num_workers, drop_last=False, pin_memory=True)
    test_adj = build_adjacency_matrices(test_loader, config.block_size)

    result_dir = os.path.join(config.result_path, config.net_name, config.dataset_name)

    enhan_net.eval()


    with torch.no_grad():
        for i, (img_ori, filenames) in enumerate(tqdm(test_loader)):
            test_adj_batch = test_adj[i]
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
            img_ori = img_ori.cuda()
            # depth_image = d_net(img_ori, False)
            from thop import profile
            flops, params = profile(enhan_net, inputs=(img_ori, test_adj_batch))

            enhan_image = enhan_net(img_ori, test_adj_batch)

            for j in range(len(enhan_image)):
                torchvision.utils.save_image(enhan_image[j], os.path.join(result_dir, os.path.basename(filenames[j])))
            # torchvision.utils.save_image(enhan_image, os.path.join(result_dir, filenames[0]))
            # print(filenames[0], "is done!")

if __name__ == '__main__':
    """
        param orig_images_path:original underwater images
    """

    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    parser.add_argument('--block_size', type=int, default="16")

    # Input Parameters
    parser.add_argument('--net_name', type=str, default="net_C")
    # parser.add_argument('--d_net_name', type=str, default="d_net")
    parser.add_argument('--dataset_name', type=str, default="UIEB")
    # parser.add_argument('--ori_images_path', type=str, default="E:/PHD/UIEB/base/UIEB/test/ref/input/")
    parser.add_argument('--ori_images_path', type=str, default="/Datasets/UIEB/test/input")

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--checkpoint_path', type=str, default="./trained_model/")
    parser.add_argument('--result_path', type=str, default="./result/")
    parser.add_argument('--cudaid', type=str, default="0", help="choose cuda device id 0-7).")

    config = parser.parse_args()

    if not os.path.exists(os.path.join(config.result_path, config.net_name)):
        os.mkdir(os.path.join(config.result_path, config.net_name))

    if not os.path.exists(os.path.join(config.result_path, config.net_name, config.dataset_name)):
        os.mkdir(os.path.join(config.result_path, config.net_name, config.dataset_name))
    
    start_time = time.time()
    test(config)
    print("final_time:"+str(time.time()-start_time))
