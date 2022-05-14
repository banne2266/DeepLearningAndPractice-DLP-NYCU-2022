BATCHSIZE = 24

all:
	python train_fixed_prior.py --cuda --batch_size $(BATCHSIZE) --num_workers 1 --epoch_size 1 \
	--kl_anneal_cyclical --niter 300 \
	--tfr_start_decay_epoch 100 --tfr_decay_step 0.003 --tfr_lower_bound 0.1

test:
	python test.py --cuda --batch_size $(BATCHSIZE) --num_workers 1