# ddp-resnet50-cifar10

- ResNet50をCIFAR10で分散学習するためのコード

## 使い方

- Pythonの環境には[UV](https://docs.astral.sh/uv/)を推奨しておきます

- venvやcondaなどその他の環境を使っても 必要なパッケージさえインストールできれば問題ないです

- エディタ&実行環境はVSCodeを想定しています それ以外のエディタは知りません

1. リポジトリのクローン

    ```bash
    git clone https://github.com/rits-menglab/ddp-resnet50-cifar10.git
    ```

1. UVでの仮想環境作成

    ```bash
    uv sync --dev
    ```

1. 以下のコマンドを実行(GPUが1台のマシンに2台搭載されている場合)

    ```bash
    torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29400 __init__.py
    ```
