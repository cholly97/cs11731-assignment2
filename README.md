# Povoit-MT

### Constructing vocab files
```
python3 vocab_main.py --train-src <src file> --train-tar <tar file> --size 50000 --freq-cutoff 2 --vocab-file vocab.bin
```
python3 vocab_main.py --train-src data/train.en-az.az.txt --size 50000 --freq-cutoff 2 --vocab-src vocab/en-az.az.bin

### train one language pair
```
python3 train_main.py --train-src <train src> --train-tar <train tar> --dev-src <dev-src> --dev-tar <dev-tar> --batch-size 64 --valid-iter 500 --log-every 2000 --save-path <save path> --vocab-src <vocab src> --embed-size 256 --hidden-size 256 --dropout 0.8 --patience 8 --max-num-trial 3 --max-epoch 30 --lr-decay 0.02
```

### 
python3 train_main.py --train-src data/train.en-az.az.txt --train-tar data/train.en-az.en.txt --dev-src data/dev.en-az.az.txt --dev-tar data/dev.en-az.en.txt --batch-size 64 --valid-iter 100 --log-every 100 --vocab-src vocab/en-az.az.bin --vocab-tar vocab/en-az.en.bin --embed-size 256 --hidden-size 256 --dropout 1.0 --patience 6 --max-num-trial 3  --max-epoch 25 --lr 0.0005 --lr-decay 0.05 --encoder-bidir 1 --encoder-layer 1