import codecs
import time
from logging import DEBUG, StreamHandler, getLogger

import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torcheval.metrics import MulticlassAccuracy
from torcheval.metrics.toolkit import sync_and_compute
from torchvision import datasets, models, transforms


def test(
    ddp_model: DistributedDataParallel,
    device_id: int,
    test_loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
) -> float:
    count = 0
    ddp_model.eval()
    test_loss = 0.0
    test_correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data_device, target_device = data.to(device_id), target.to(device_id)
            output = ddp_model(data_device)
            loss = criterion(output, target_device)

            # lossの計算
            test_loss += loss.item()
            _, preds = torch.max(output, 1)
            test_correct += torch.sum(preds == target_device)
            count += 1

    # lossの平均値
    test_loss = test_loss / count
    test_correct = float(test_correct) / count

    return test_loss, test_correct


def train(
    ddp_model: DistributedDataParallel,
    rank: int,
    device_id: int,
    train_loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    optimizer: optim.SGD,
    metric: MulticlassAccuracy,
) -> float:
    count = 0
    train_loss = 0.0
    train_correct = 0.0

    ddp_model.train()
    for _batch_idx, (data, label) in enumerate(train_loader):
        count += len(label)
        data_device, label_device = data.to(device_id), label.to(device_id)
        optimizer.zero_grad()
        output = ddp_model(data_device)
        loss = criterion(output, label_device)
        loss.backward()
        optimizer.step()

        # lossの計算
        train_loss += loss.item()
        _, preds = torch.max(output, 1)
        train_correct += torch.sum(preds == label_device)

        metric.update(output, label_device)
        local_compute_result = metric.compute()
        global_compute_result = sync_and_compute(metric)
        if rank == 0:
            logger.info("Accuracy: %s", local_compute_result)
    metric.reset()

    # lossの平均値
    train_loss = train_loss / count
    train_correct = float(train_correct) / count

    return train_loss, train_correct


def learning(
    ddp_model: DistributedDataParallel,
    rank: int,
    device_id: int,
    train_loader: DataLoader,
    test_loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    optimizer: optim.SGD,
    epochs: int,
) -> list:
    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []

    # 複数GPUからのaccを集計する
    metric = MulticlassAccuracy(device=device_id)

    for epoch in range(1, epochs + 1, 1):
        train_loss, train_acc = train(ddp_model, rank, device_id, train_loader, criterion, optimizer, metric)
        test_loss, test_acc = test(ddp_model, device_id, test_loader, criterion)
        # エポック毎の表示
        logger.info(
            "epoch : %d, train_loss : %f, train_acc : %f, test_loss : %f, test_acc : %f,",
            epoch,
            train_loss,
            train_acc,
            test_loss,
            test_acc,
        )
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)

    return train_loss, train_acc, test_loss, test_acc


def main() -> None:
    # 時間計測用
    start = time.time()

    # 変数もろもろ
    # batch_sizeの認識がずれてて datasetの総数//batch_sizeしたものが実際のbatch sizeになってる 50000//100=500的な感じ
    batch_size = 250
    epoch = 200

    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []

    criterion = nn.CrossEntropyLoss()

    # マルチプロセス初期化
    dist.init_process_group("nccl")
    # この部分はグローバルランクの取得
    # 順にrankが割り振られる masterは0
    rank = dist.get_rank()
    # world_sizeはネットワーク全体のGPUの数
    world_size = dist.get_world_size()
    logger.info("This node is rank: %d.", rank)

    # 余りを計算することでcuda0~cuda7のように認識させることができる
    # DDPの場合GPu毎にCPUを割り当てられるのでこういう計算になる
    # K8sの場合はコンテナに認識させる量を絞っているのであまり気にしなくてよい
    device_id = rank % torch.cuda.device_count()

    # torchから提供されるモデルをそのまま使う
    # "weight=None"で事前学習をなしにできる
    model = models.resnet50(weights=None)
    # 最後の全結合層の出力を10にすることでCIFAR10に対応させる
    model.fc = nn.Linear(model.fc.in_features, 10)

    model = model.to(device_id)
    ddp_model = DistributedDataParallel(model, device_ids=[device_id])

    # CIFAR10関連
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
        ],
    )
    train_dataset = datasets.CIFAR10("./pv/data", train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10("./pv/data", train=False, transform=transform)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
    )
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=test_sampler is None,
        sampler=test_sampler,
    )

    # オプティマイザ
    # 比較的精度が出やすい感じのチューンになっている
    # ref. https://qiita.com/TrashBoxx/items/2d441e46643f73c0ca19#3-1cifar-10%E3%82%92%E5%AD%A6%E7%BF%92%E3%81%99%E3%82%8B%E3%82%AF%E3%83%A9%E3%82%B9
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)

    # トレーニング
    train_loss, train_acc, test_loss, test_acc = learning(
        ddp_model=ddp_model,
        rank=rank,
        device_id=device_id,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=epoch,
    )

    # プロセス解放
    dist.destroy_process_group()

    # Masterだけでやること

    checkpoint_dir = "./pv/model.pth"
    if int(rank) == 0:
        print(
            "Process time: ",
            time.time() - start,
            file=codecs.open("./pv/time.txt", "w", "utf-8"),
        )
        torch.save(ddp_model.state_dict(), checkpoint_dir)

        # lossのグラフ描画
        rate = plt.figure()
        plt.plot(range(len(train_loss)), train_loss, c="b", label="train loss")
        plt.plot(range(len(test_loss)), test_loss, c="r", label="test loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.grid()
        plt.show()
        rate.savefig("./pv/rate.png")

        # accのグラフ描画
        acc = plt.figure()
        plt.plot(range(len(train_acc)), train_acc, c="b", label="train acc")
        plt.plot(range(len(test_acc)), test_acc, c="r", label="test acc")
        plt.xlabel("epoch")
        plt.ylabel("acc")
        plt.legend()
        plt.grid()
        plt.show()
        acc.savefig("./pv/acc.png")


if __name__ == "__main__":
    # デバッグ用
    logger = getLogger(__name__)
    handler = StreamHandler()
    handler.setLevel(DEBUG)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)
    logger.propagate = False
    main()
