/usr/bin/python3 ../src/train.py \
--GPU='1' \
--trainable='1' \
--batch_size=128 \
--server=29 \
--method_of_GAN='WGAN-gp' \
--epoch=20 \
--lr_boundaries='10' \
--lr_values='0.0001,0.00001'