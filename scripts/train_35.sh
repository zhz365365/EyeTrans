/usr/bin/python3 ../src/train.py \
--GPU='2' \
--trainable='1' \
--batch_size=128 \
--server=35 \
--method_of_GAN='DCGAN' \
--epoch=20 \
--lr_boundaries='10' \
--lr_values='0.0001,0.00001'