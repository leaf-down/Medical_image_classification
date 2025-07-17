import os
import sys
import json
import argparse
import logging

import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from CNN_Mamba import VSSM as medmamba  # import model


# --------------------- utils for DDP ----------------------
def is_main_process():
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0


def setup_for_distributed(is_master):
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


# --------------------- Logger Setup ----------------------
def setup_logger(log_path):
    logging.basicConfig(
        filename=log_path,
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


# --------------------- Main Training ----------------------
def main():
    parser = argparse.ArgumentParser(description="Train CNN_Mamba model with optional distributed training support.")
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs (default: 1)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size per process (default: 32)')
    parser.add_argument('--local_rank', type=int, default=0, help='[Distributed] Local GPU id passed by torchrun')
    parser.add_argument('--data-path', type=str, default='/app/RetinalOCT_Dataset',
                        help='Root path to training and validation dataset folders')
    parser.add_argument('--save-path', type=str, default='/app/models/cnn_ssd_.pth',
                        help='File path to save the trained model')
    parser.add_argument('--resume', type=str, default='', help='Path to checkpoint to resume from (default: none)')
    parser.add_argument('--log-path', type=str, default='train.log', help='Path to save training log file')

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    if is_main_process():
        setup_logger(args.log_path)
        logging.info("Starting training script")

    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)

    distributed = False
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        distributed = True
        torch.distributed.init_process_group(backend='nccl')
        setup_for_distributed(is_main_process())

    if is_main_process():
        logging.info(f"Using device: {device}")

    data_transform = {
        "train": transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    }

    train_dataset = datasets.ImageFolder(root=os.path.join(args.data_path, "train"),
                                         transform=data_transform["train"])
    val_dataset = datasets.ImageFolder(root=os.path.join(args.data_path, "val"),
                                       transform=data_transform["val"])

    if is_main_process():
        flower_list = train_dataset.class_to_idx
        cla_dict = dict((val, key) for key, val in flower_list.items())
        with open('class_indices.json', 'w') as json_file:
            json.dump(cla_dict, json_file, indent=4)

    train_sampler = DistributedSampler(train_dataset) if distributed else None
    shuffle = train_sampler is None

    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])
    if is_main_process():
        logging.info(f'Using {nw} dataloader workers per process')

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=shuffle,
                                               num_workers=nw,
                                               sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=nw)

    if is_main_process():
        logging.info(f"Training images: {len(train_dataset)} | Validation images: {len(val_dataset)}")

    net = medmamba(num_classes=8).to(device)
    if distributed:
        net = DDP(net, device_ids=[args.local_rank])

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    start_epoch = 0
    best_acc = 0.0

    if args.resume and os.path.isfile(args.resume):
        map_location = {'cuda:%d' % 0: 'cuda:%d' % args.local_rank}
        checkpoint = torch.load(args.resume, map_location=map_location)
        net.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint.get('best_acc', 0.0)
        if is_main_process():
            logging.info(f"Resumed from checkpoint {args.resume}, starting at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        if distributed:
            train_sampler.set_epoch(epoch)

        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout) if is_main_process() else train_loader

        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if is_main_process():
                train_bar.set_description(f"train epoch[{epoch + 1}/{args.epochs}] loss:{loss:.3f}")

        net.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_loader, file=sys.stdout) if is_main_process() else val_loader
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / len(val_dataset)
        if is_main_process():
            logging.info(
                f'[epoch {epoch + 1}] train_loss: {running_loss / len(train_loader):.3f}  val_accuracy: {val_accurate:.3f}')

            if val_accurate > best_acc:
                best_acc = val_accurate
                save_dict = {
                    'epoch': epoch,
                    'model': net.module.state_dict() if distributed else net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_acc': best_acc
                }
                torch.save(save_dict, args.save_path)
                logging.info(f"New best model saved at epoch {epoch + 1} with val_accuracy: {val_accurate:.3f}")

    if is_main_process():
        logging.info('Finished Training')


if __name__ == '__main__':
    main()
