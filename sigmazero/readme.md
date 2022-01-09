# Sigma Zero

SigmaZero is an implementation of the AlphaZero algorithm.

## Selfplay

A selfplay process should write replays to standard out, line by line. It should take one argument, a file path to the model file to use for self play. This file should be watched for changes and loaded regularly.

## Training

A training process should read replays from standard in, line by line. It should take one argument, a file path to the model file to use for training. This file should be loaded once and then written to regularly while training. When the model has been updated, the process should write a new line to standard out to indicate this.

## Build

```bash
cd ..
meson build
cd build
ninja training
ninja selfplay
ninja sigmazero
```

## Training

To train, you have to run one training process and one or more selfplay processes. The training program writes the most recent model to the path specified by the first argument, and reads replays from stdin. Each selfplay process takes a path to a model as an argument, which they use to play selfplay games whose results are written to stdout.

Run one selfplayer and one trainer:

```bash
cd ../build
model=model.pt
selfplay $model | ./training $model
```

Run multiple selfplayers and one trainer:

```bash
cd ../build
model=model.pt
./training $model <(./selfplay $model) <(./selfplay $model) <(./selfplay $model)
```

Train using multiple computers in the Olympen lab for selfplay:

Olympen example:

```bash
cd ../scripts
./olympen.sh 01 01 02 02 03 03
```

This will create a directory `~/sigma_{datetime}` to which the latest model, checkpoints and logs will be written. The current olympen computer will be used as trainer and selfplay processes will be started on all hosts specified as arguments (NM is olympen1-1NM). The tjack repo is assumed to be at `~/tjack` but its location can be set with the `SIGMA_REPO` variable. The training session directory can be set with `SIGMA_DIR`.

## Playing

The engine uses the UCI protocol. It takes a path to the model to play with as argument.

Compile the engine:

```bash
cd ../build
model=model.pt
./sigmazero $model
```

It is easiest to use the engine with an UCI compatible interface such as [Cute Chess](https://cutechess.com/), but you can also use it manually. To search the starting position, do:

```bash
uci
position startpos
go infinite
```

This will search for the best move. Write `stop` to stop the search and get the best move. The full UCI documentation is mirrored [here](http://wbec-ridderkerk.nl/html/UCIProtocol.html).
