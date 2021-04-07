from __future__ import print_function

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import hyptorch.nn as hypnn
from examples.fewshot.data.color_mnist_dataset import ColorMNIST


class Net_vanilla(nn.Module):
    def __init__(self, args):
        super(Net_vanilla, self).__init__()
        # self.conv1 = nn.Conv2d(3, 20, 5, 1)
        # self.conv2 = nn.Conv2d(20, 50, 5, 1)
        # self.fc1 = nn.Linear(4 * 4 * 50, 500)
        # self.fc2 = nn.Linear(500, args.dim)

        self.conv1 = nn.Conv2d(3, 20, 5, 1)
        self.bn1 = nn.BatchNorm2d(20)
        self.conv2 = nn.Conv2d(20, 50, 3, 1)
        self.bn2 = nn.BatchNorm2d(50)
        self.conv3 = nn.Conv2d(50, 50, 3, 1)
        self.bn3 = nn.BatchNorm2d(50)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, args.dim)
        self.fc_out = nn.Linear(args.dim, 120)

        # self.tp = hypnn.ToPoincare(
        #     c=args.c, train_x=args.train_x, train_c=args.train_c, ball_dim=args.dim
        # )
        # self.mlr = hypnn.HyperbolicMLR(ball_dim=args.dim, n_classes=120, c=args.c)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        # x = x.detach()
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.fc_out(x)
        # x = F.relu(self.conv1(x))
        # x = F.max_pool2d(x, 2, 2)
        # x = F.relu(self.conv2(x))
        # x = F.max_pool2d(x, 2, 2)
        # x = x.view(-1, 4 * 4 * 50)
        # x = F.relu(self.fc1(x))
        # x = self.fc2(x)
        # x = self.tp(x)
        # return F.log_softmax(x, dim=-1)
        return F.log_softmax(x, dim=-1)

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        # self.conv1 = nn.Conv2d(3, 20, 5, 1)
        # self.conv2 = nn.Conv2d(20, 50, 5, 1)
        # self.fc1 = nn.Linear(4 * 4 * 50, 500)
        # self.fc2 = nn.Linear(500, args.dim)

        self.conv1 = nn.Conv2d(3, 20, 5, 1)
        self.bn1 = nn.BatchNorm2d(20)
        self.conv2 = nn.Conv2d(20, 50, 3, 1)
        self.bn2 = nn.BatchNorm2d(50)
        self.conv3 = nn.Conv2d(50, 50, 3, 1)
        self.bn3 = nn.BatchNorm2d(50)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, args.dim)

        self.tp = hypnn.ToPoincare(
            c=args.c, train_x=args.train_x, train_c=args.train_c, ball_dim=args.dim
        )
        self.mlr = hypnn.HyperbolicMLR(ball_dim=args.dim, n_classes=120, c=self.tp.c)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        # x = x.detach()
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # x = F.relu(self.conv1(x))
        # x = F.max_pool2d(x, 2, 2)
        # x = F.relu(self.conv2(x))
        # x = F.max_pool2d(x, 2, 2)
        # x = x.view(-1, 4 * 4 * 50)
        # x = F.relu(self.fc1(x))
        # x = self.fc2(x)
        x = self.tp(x)
        return F.log_softmax(self.mlr(x, c=self.tp.c), dim=-1)


class Netold(nn.Module):
    def __init__(self, args):
        super(Netold, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, args.dim)
        self.tp = hypnn.ToPoincare(
            c=args.c, train_x=args.train_x, train_c=args.train_c, ball_dim=args.dim
        )
        self.mlr = hypnn.HyperbolicMLR(ball_dim=args.dim, n_classes=120, c=args.c)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.tp(x)
        return F.log_softmax(self.mlr(x, c=self.tp.c), dim=-1)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    # for batch_idx, (data, target) in enumerate(train_loader):
    #     data, target = data.to(device), target.to(device)

    for batch_idx, data in enumerate(train_loader):
        images = data["image"].to(device)  # between 0 and 1
        targets = data["target"].to(device)
        image_org = data["image_org"]
        parent_hierarchy = data["parent_hierarchy"]
        parent_idxs = data["parent_idx"].to(device)
        data, target = images.to(device), targets.to(device)
        # optimizer.zero_grad()
        # output, features_euclidean, features_hyperbolic = model(data)

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data in test_loader:
            images = data["image"].to(device)  # between 0 and 1
            targets = data["target"].to(device)
            image_org = data["image_org"]
            parent_hierarchy = data["parent_hierarchy"]
            parent_idxs = data["parent_idx"].to(device)
            data, target = images.to(device), targets.to(device)
            # data, target = data.to(device), target.to(device)
            # data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        metavar="LR",
        help="learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )

    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )

    parser.add_argument(
        "--c", type=float, default=1.0, help="Curvature of the Poincare ball"
    )
    parser.add_argument(
        "--dim", type=int, default=2, help="Dimension of the Poincare ball"
    )
    parser.add_argument(
        "--train_x",
        action="store_true",
        default=False,
        help="train the exponential map origin",
    )
    parser.add_argument(
        "--train_c",
        action="store_true",
        default=True,
        help="train the Poincare ball curvature",
    )

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    color_mnist_train = ColorMNIST("datasets", split="train",
                                     subsample_num=1)
    train_loader = torch.utils.data.DataLoader(
        color_mnist_train,
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs
    )
    color_mnist_test = ColorMNIST("datasets", split="test",
                                     subsample_num=1)
    test_loader = torch.utils.data.DataLoader(
        color_mnist_test,
        batch_size=args.test_batch_size,
        shuffle=True,
        **kwargs
    )
    model = Net(args).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == "__main__":
    main()
