DATA=/media/turing/D741ADF8271B9526/DATA
OUTPUT=/media/turing/D741ADF8271B9526/OUTPUT
python train_rbm.py \
    --DATASET=$DATA/BostonHousePriceDataset/BostonHousePriceDataset.txt \
    --MODEL=$OUTPUT/RBM_MODEL \
    --EPOCH=1000 \
    --LEARNING_RATE=0.00001