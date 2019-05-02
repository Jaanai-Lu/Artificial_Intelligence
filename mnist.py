{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "mnist.py",
      "version": "0.3.2",
      "provenance": [],
      "collapsed_sections": [],
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Jaanai-Lu/Deep-learning/blob/master/mnist.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "metadata": {
        "id": "G2B5C06uQ3v3",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "import torch\n",
        "import torch.nn as nn\n",
        "import torch.nn.functional as F\n",
        "import torch.optim as optim\n",
        "from torchvision import datasets, transforms"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "NYZH9bASRA87",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        },
        "outputId": "80bec235-66fb-43d2-c34a-30b02a3a3b6d"
      },
      "cell_type": "code",
      "source": [
        "torch.__version__"
      ],
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "'1.0.1.post2'"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 3
        }
      ]
    },
    {
      "metadata": {
        "id": "AB8LTOLMUI5q",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "BATCH_SIZE = 512\n",
        "EPOCHS = 20\n",
        "DEVICE = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "18E9gtERU3vp",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 255
        },
        "outputId": "0b336b3b-f898-4d8b-f41c-dd3382576884"
      },
      "cell_type": "code",
      "source": [
        "train_loader = torch.utils.data.DataLoader(\n",
        "        datasets.MNIST('data', train = True, download = True,\n",
        "                      transform=transforms.Compose([\n",
        "                          transforms.ToTensor(),\n",
        "                          transforms.Normalize((0.1307,), (0.3081,))\n",
        "                      ])),\n",
        "        batch_size=BATCH_SIZE, shuffle=True)"
      ],
      "execution_count": 5,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "  0%|          | 0/9912422 [00:00<?, ?it/s]"
          ],
          "name": "stderr"
        },
        {
          "output_type": "stream",
          "text": [
            "Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to data/MNIST/raw/train-images-idx3-ubyte.gz\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "stream",
          "text": [
            "9920512it [00:00, 26030023.41it/s]                            \n"
          ],
          "name": "stderr"
        },
        {
          "output_type": "stream",
          "text": [
            "Extracting data/MNIST/raw/train-images-idx3-ubyte.gz\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "stream",
          "text": [
            "32768it [00:00, 454170.99it/s]\n",
            "  1%|          | 16384/1648877 [00:00<00:11, 146491.86it/s]"
          ],
          "name": "stderr"
        },
        {
          "output_type": "stream",
          "text": [
            "Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz to data/MNIST/raw/train-labels-idx1-ubyte.gz\n",
            "Extracting data/MNIST/raw/train-labels-idx1-ubyte.gz\n",
            "Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz to data/MNIST/raw/t10k-images-idx3-ubyte.gz\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "stream",
          "text": [
            "1654784it [00:00, 7244562.04it/s]                            \n",
            "8192it [00:00, 176456.25it/s]\n"
          ],
          "name": "stderr"
        },
        {
          "output_type": "stream",
          "text": [
            "Extracting data/MNIST/raw/t10k-images-idx3-ubyte.gz\n",
            "Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to data/MNIST/raw/t10k-labels-idx1-ubyte.gz\n",
            "Extracting data/MNIST/raw/t10k-labels-idx1-ubyte.gz\n",
            "Processing...\n",
            "Done!\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "metadata": {
        "id": "Vo-qllugOysP",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "test_loader = torch.utils.data.DataLoader(\n",
        "       datasets.MNIST('data', train=False, transform=transforms.Compose([\n",
        "                          transforms.ToTensor(),\n",
        "                          transforms.Normalize((0.1307,), (0.3081,))\n",
        "                      ])),\n",
        "        batch_size=BATCH_SIZE, shuffle=True)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "XySiElmkTEE7",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "class ConvNet(nn.Module):\n",
        "    def __init__(self):\n",
        "        super().__init__()\n",
        "        # 1，28*28\n",
        "        self.conv1 = nn.Conv2d(1, 10, 5) # 10, 24*24, submodule: Conv2d\n",
        "        self.conv2 = nn.Conv2d(10, 20, 3) # 128, 10*10\n",
        "        self.fc1 = nn.Linear(20*10*10, 500)\n",
        "        self.fc2 = nn.Linear(500, 10)\n",
        "    def forward(self, x):\n",
        "        in_size = x.size(0)\n",
        "        out = self.conv1(x) # 24\n",
        "        out = F.relu(out)\n",
        "        out = F.max_pool2d(out, 2, 2) # 12\n",
        "        out = self.conv2(out) # 10\n",
        "        out = F.relu(out)\n",
        "        out = out.view(in_size, -1)\n",
        "        out = self.fc1(out)\n",
        "        out = F.relu(out)\n",
        "        out = self.fc2(out)\n",
        "        out = F.log_softmax(out,dim=1)\n",
        "        return out\n",
        "        "
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "7YqzhfhHe_bk",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "model = ConvNet().to(DEVICE)\n",
        "optimizer = optim.Adam(model.parameters())\n"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "tNQAO3qMfjjK",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "def train(model, device, train_loader, optimizer, epoch):\n",
        "    model.train()\n",
        "    for batch_idx, (data, target) in enumerate(train_loader):\n",
        "        data, target = data.to(device), target.to(device)\n",
        "        optimizer.zero_grad()\n",
        "        output = model(data)\n",
        "        loss = F.nll_loss(output, target)\n",
        "        loss.backward()\n",
        "        optimizer.step()\n",
        "        if(batch_idx+1)%30 == 0:\n",
        "            print('Train Epoch: {} [{}/{} ({:.0f}%)]\\tLoss: {:.6f}'.format(\n",
        "                epoch, batch_idx * len(data), len(train_loader.dataset),\n",
        "                100. * batch_idx / len(train_loader), loss.item()))\n",
        "      "
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "n5Xs_1wGi63V",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "def test(model, device, test_loader):\n",
        "    model.eval()\n",
        "    test_loss = 0\n",
        "    correct = 0\n",
        "    with torch.no_grad():\n",
        "        for data, target in test_loader:\n",
        "            data, target = data.to(device), target.to(device)\n",
        "            output = model(data)\n",
        "            test_loss += F.nll_loss(output, target, reduction='sum').item() \n",
        "            # 将一批的损失相加\n",
        "            pred = output.max(1, keepdim=True)[1] # 找到概率最大的下标\n",
        "            correct += pred.eq(target.view_as(pred)).sum().item()\n",
        "            \n",
        "    test_loss /= len(test_loader.dataset)\n",
        "    print('\\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\\n'.format(\n",
        "        test_loss, correct, len(test_loader.dataset),\n",
        "        100. * correct / len(test_loader.dataset)))\n",
        "    \n",
        "            "
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "C3g2BKn0oih7",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 2057
        },
        "outputId": "d56c385d-0e3a-457b-a947-83bacef5f6d9"
      },
      "cell_type": "code",
      "source": [
        "for epoch in range(1, EPOCHS + 1):\n",
        "    train(model, DEVICE, train_loader, optimizer, epoch)\n",
        "    test(model, DEVICE, test_loader)\n",
        "    "
      ],
      "execution_count": 15,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Train Epoch: 1 [14848/60000 (25%)]\tLoss: 0.292614\n",
            "Train Epoch: 1 [30208/60000 (50%)]\tLoss: 0.134914\n",
            "Train Epoch: 1 [45568/60000 (75%)]\tLoss: 0.115569\n",
            "\n",
            "Test set: Average loss: 0.0939, Accuracy: 9719/10000 (97%)\n",
            "\n",
            "Train Epoch: 2 [14848/60000 (25%)]\tLoss: 0.080601\n",
            "Train Epoch: 2 [30208/60000 (50%)]\tLoss: 0.088773\n",
            "Train Epoch: 2 [45568/60000 (75%)]\tLoss: 0.079390\n",
            "\n",
            "Test set: Average loss: 0.0499, Accuracy: 9843/10000 (98%)\n",
            "\n",
            "Train Epoch: 3 [14848/60000 (25%)]\tLoss: 0.061033\n",
            "Train Epoch: 3 [30208/60000 (50%)]\tLoss: 0.050075\n",
            "Train Epoch: 3 [45568/60000 (75%)]\tLoss: 0.057016\n",
            "\n",
            "Test set: Average loss: 0.0486, Accuracy: 9855/10000 (99%)\n",
            "\n",
            "Train Epoch: 4 [14848/60000 (25%)]\tLoss: 0.040749\n",
            "Train Epoch: 4 [30208/60000 (50%)]\tLoss: 0.026281\n",
            "Train Epoch: 4 [45568/60000 (75%)]\tLoss: 0.036193\n",
            "\n",
            "Test set: Average loss: 0.0429, Accuracy: 9862/10000 (99%)\n",
            "\n",
            "Train Epoch: 5 [14848/60000 (25%)]\tLoss: 0.035125\n",
            "Train Epoch: 5 [30208/60000 (50%)]\tLoss: 0.021073\n",
            "Train Epoch: 5 [45568/60000 (75%)]\tLoss: 0.033027\n",
            "\n",
            "Test set: Average loss: 0.0285, Accuracy: 9910/10000 (99%)\n",
            "\n",
            "Train Epoch: 6 [14848/60000 (25%)]\tLoss: 0.019707\n",
            "Train Epoch: 6 [30208/60000 (50%)]\tLoss: 0.031947\n",
            "Train Epoch: 6 [45568/60000 (75%)]\tLoss: 0.008413\n",
            "\n",
            "Test set: Average loss: 0.0301, Accuracy: 9908/10000 (99%)\n",
            "\n",
            "Train Epoch: 7 [14848/60000 (25%)]\tLoss: 0.014864\n",
            "Train Epoch: 7 [30208/60000 (50%)]\tLoss: 0.007799\n",
            "Train Epoch: 7 [45568/60000 (75%)]\tLoss: 0.023022\n",
            "\n",
            "Test set: Average loss: 0.0301, Accuracy: 9902/10000 (99%)\n",
            "\n",
            "Train Epoch: 8 [14848/60000 (25%)]\tLoss: 0.007452\n",
            "Train Epoch: 8 [30208/60000 (50%)]\tLoss: 0.010756\n",
            "Train Epoch: 8 [45568/60000 (75%)]\tLoss: 0.021713\n",
            "\n",
            "Test set: Average loss: 0.0294, Accuracy: 9896/10000 (99%)\n",
            "\n",
            "Train Epoch: 9 [14848/60000 (25%)]\tLoss: 0.002210\n",
            "Train Epoch: 9 [30208/60000 (50%)]\tLoss: 0.006894\n",
            "Train Epoch: 9 [45568/60000 (75%)]\tLoss: 0.007443\n",
            "\n",
            "Test set: Average loss: 0.0322, Accuracy: 9893/10000 (99%)\n",
            "\n",
            "Train Epoch: 10 [14848/60000 (25%)]\tLoss: 0.004085\n",
            "Train Epoch: 10 [30208/60000 (50%)]\tLoss: 0.008896\n",
            "Train Epoch: 10 [45568/60000 (75%)]\tLoss: 0.003086\n",
            "\n",
            "Test set: Average loss: 0.0303, Accuracy: 9912/10000 (99%)\n",
            "\n",
            "Train Epoch: 11 [14848/60000 (25%)]\tLoss: 0.011350\n",
            "Train Epoch: 11 [30208/60000 (50%)]\tLoss: 0.007481\n",
            "Train Epoch: 11 [45568/60000 (75%)]\tLoss: 0.008421\n",
            "\n",
            "Test set: Average loss: 0.0329, Accuracy: 9901/10000 (99%)\n",
            "\n",
            "Train Epoch: 12 [14848/60000 (25%)]\tLoss: 0.008141\n",
            "Train Epoch: 12 [30208/60000 (50%)]\tLoss: 0.005311\n",
            "Train Epoch: 12 [45568/60000 (75%)]\tLoss: 0.003874\n",
            "\n",
            "Test set: Average loss: 0.0377, Accuracy: 9889/10000 (99%)\n",
            "\n",
            "Train Epoch: 13 [14848/60000 (25%)]\tLoss: 0.001006\n",
            "Train Epoch: 13 [30208/60000 (50%)]\tLoss: 0.005097\n",
            "Train Epoch: 13 [45568/60000 (75%)]\tLoss: 0.005085\n",
            "\n",
            "Test set: Average loss: 0.0392, Accuracy: 9905/10000 (99%)\n",
            "\n",
            "Train Epoch: 14 [14848/60000 (25%)]\tLoss: 0.003974\n",
            "Train Epoch: 14 [30208/60000 (50%)]\tLoss: 0.002187\n",
            "Train Epoch: 14 [45568/60000 (75%)]\tLoss: 0.000493\n",
            "\n",
            "Test set: Average loss: 0.0360, Accuracy: 9894/10000 (99%)\n",
            "\n",
            "Train Epoch: 15 [14848/60000 (25%)]\tLoss: 0.001239\n",
            "Train Epoch: 15 [30208/60000 (50%)]\tLoss: 0.007346\n",
            "Train Epoch: 15 [45568/60000 (75%)]\tLoss: 0.005702\n",
            "\n",
            "Test set: Average loss: 0.0403, Accuracy: 9893/10000 (99%)\n",
            "\n",
            "Train Epoch: 16 [14848/60000 (25%)]\tLoss: 0.002461\n",
            "Train Epoch: 16 [30208/60000 (50%)]\tLoss: 0.001638\n",
            "Train Epoch: 16 [45568/60000 (75%)]\tLoss: 0.000544\n",
            "\n",
            "Test set: Average loss: 0.0354, Accuracy: 9902/10000 (99%)\n",
            "\n",
            "Train Epoch: 17 [14848/60000 (25%)]\tLoss: 0.000772\n",
            "Train Epoch: 17 [30208/60000 (50%)]\tLoss: 0.003961\n",
            "Train Epoch: 17 [45568/60000 (75%)]\tLoss: 0.001129\n",
            "\n",
            "Test set: Average loss: 0.0320, Accuracy: 9914/10000 (99%)\n",
            "\n",
            "Train Epoch: 18 [14848/60000 (25%)]\tLoss: 0.004667\n",
            "Train Epoch: 18 [30208/60000 (50%)]\tLoss: 0.001105\n",
            "Train Epoch: 18 [45568/60000 (75%)]\tLoss: 0.000699\n",
            "\n",
            "Test set: Average loss: 0.0328, Accuracy: 9903/10000 (99%)\n",
            "\n",
            "Train Epoch: 19 [14848/60000 (25%)]\tLoss: 0.002526\n",
            "Train Epoch: 19 [30208/60000 (50%)]\tLoss: 0.000280\n",
            "Train Epoch: 19 [45568/60000 (75%)]\tLoss: 0.000657\n",
            "\n",
            "Test set: Average loss: 0.0415, Accuracy: 9888/10000 (99%)\n",
            "\n",
            "Train Epoch: 20 [14848/60000 (25%)]\tLoss: 0.002222\n",
            "Train Epoch: 20 [30208/60000 (50%)]\tLoss: 0.000813\n",
            "Train Epoch: 20 [45568/60000 (75%)]\tLoss: 0.000499\n",
            "\n",
            "Test set: Average loss: 0.0356, Accuracy: 9907/10000 (99%)\n",
            "\n"
          ],
          "name": "stdout"
        }
      ]
    }
  ]
}