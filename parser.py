import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="SphereFace Implementation")

    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--bs', type=int, default=64, help="batch size")
    parser.add_argument('--epoch', type=int, default=10, help="epoch count")
    parser.add_argument('--lr', type=float, default=1e-1, help="learning rate")
    parser.add_argument('--val_interval', type=int, default=1, help="validation interval")
    parser.add_argument('--train_file', type=str, default='./data/pairsDevTrain.txt')
    parser.add_argument('--test_file', type=str, default='./data/pairsDevTest.txt')