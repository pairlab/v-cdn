CUDA_VISIBLE_DEVICES=0 \
	python train_kp.py \
	--env Ball \
	--stage kp \
	--nf_hidden_kp 16 \
	--n_ball 5 \
	--n_kp 5 \
	--inv_std 10 \
	--batch_size 32 \
	--lr 1e-3 \
	--gen_data 0 \
	--num_workers 10 \
	--kp_epoch -1 \
	--kp_iter -1 \
	--dy_epoch -1 \
	--dy_iter -1