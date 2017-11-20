# LeNet-5

This implements a slightly modified LeNet-5 and achieves an accuracy of ~99% on the MNIST dataset.

## Setup

Install all dependencies using the following command

```
$ pip install -r requirements.txt
```

## Usage

Start the `visdom` server for visualization

```
$ python -m visdom.server
```

Start the training procedure

```
$ python run.py
```

See epoch train loss live graph at `[http://localhost:8097](http://localhost:8097)`.
