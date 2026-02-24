import pytorch_lightning as pl
from model.dti import DTIModel
import torch
import os
import argparse
from dataset.dataset import preparedataset
from utils import get_config

def parse_args():
    parser = argparse.ArgumentParser(description='Drug-Target Interaction Prediction (Test Mode)')
    
    # 基本配置参数 (需要与训练时保持一致，用于加载数据和配置)
    parser.add_argument('--base_config', type=str, default='./config/config.yaml',
                        help='Path to base configuration file')
    parser.add_argument('--data', type=str, required=True,
                        help='Dataset name (e.g., biosnap, drugbank, bindingdb etc.)')
    parser.add_argument('--num_worker', type=int, default=0, help='number of workers for dataloader')
    parser.add_argument('--type', type=str, default='random', choices=['random', 'cold', 'cluster'], 
                        help='Training type: random, cold, cluster (must match training)')
    
    # 测试特有参数
    parser.add_argument('--model_path', type=str, required=True, 
                        help='Path to the .ckpt file (best_model_path)')
    parser.add_argument('--gpu', type=str, default='0', help='Specify GPUs to use, e.g. 0')
    parser.add_argument('--batch_size', type=int, default=None, 
                        help='Overwrite test batch size (optional)')
    parser.add_argument('--outname', type=str, default='test_result', 
                            help='Output name (optional for test)')
    return parser.parse_args()

def main():
    # 1. 解析参数
    args = parse_args()
    
    # 2. 检查模型路径是否存在
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {args.model_path}")

    # 3. 加载配置
    # 注意：这里加载配置主要是为了获取数据路径和模型结构参数
    cfg = get_config(args.base_config, args.data)
    
    # 如果命令行指定了 batch_size，覆盖配置文件 (测试时显存占用少，通常可以开大 batch_size)
    if args.batch_size is not None:
        cfg.Global.Batch_Size = args.batch_size

    # 设置种子 (保持复现性)
    pl.seed_everything(cfg.Global.Seed, workers=True)

    print(f"Loading data for {args.data} ({args.type})...")
    
    # 4. 准备数据
    # 我们只需要 test_loader，前面的 train/val 用 _ 占位，确保测试过程只使用测试集
    # 注意：必须使用与训练时相同的参数调用 preparedataset，以确保数据处理逻辑一致
    _, _, test_loader, _, _ = preparedataset(
        cfg.Global.Batch_Size, 
        args.type, 
        args.data, 
        args.num_worker,
        cfg.Drug.Max_Nodes, 
        cfg.Protein.Max_Length
    )

    print(f"Loading model from {args.model_path}...")

    # 5. 加载模型
    # load_from_checkpoint 会自动加载权重
    # 我们传入 cfg 是因为 DTIModel 的 __init__ 需要它
    model = DTIModel.load_from_checkpoint(args.model_path, cfg=cfg)
    
    # 将模型设置为评估模式 (虽然 Trainer.test 会自动处理，但显式设置是个好习惯)
    model.eval()

    # 6. 初始化 Trainer 进行测试
    # 测试通常只需要单卡，不需要 DDPStrategy，除非数据量极其巨大
    # 解析 gpus 参数，转换为 int 或 list
    gpu = [int(g) for g in args.gpu.split(',')]
    
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=gpu,
        logger=False, # 测试模式通常不需要 TensorBoard logger，可以直接打印结果
        precision=16  # 保持与训练一致的精度，或者使用 32
    )

    print("Starting testing...")
    
    # 7. 运行测试
    # 这将自动调用 DTIModel 中的 test_step 和 test_epoch_end (如果有定义)
    results = trainer.test(model, dataloaders=test_loader)

    print("\nTest Finished.")
    print("Results:", results)

if __name__ == '__main__':
    main()