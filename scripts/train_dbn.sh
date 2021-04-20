DATA=/media/turing/D741ADF8271B9526/DATA
OUTPUT=/media/turing/D741ADF8271B9526/OUTPUT
python train_dbn.py \
    --MODEL=$OUTPUT/DBN_MODEL \
    --DATASET=$DATA/BostonHousePriceDataset/BostonHousePriceDataset.txt \
    --EPOCH=1000 \
    --LEARNING_RATE=0.00001 \
    --BATCH_SIZE=30 \
    --DISPLAY_STEP=100