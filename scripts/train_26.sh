/usr/bin/python3 ../src/train.py \
--GPU='0' \
--trainable='1' \
--batch_size=64 \
--server=26 \
--method_of_GAN='BEGAN' \
--epoch=20 \
--lr_boundaries='10' \
--lr_values='0.0001,0.00001'
