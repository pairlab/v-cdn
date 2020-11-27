CUDA_VISIBLE_DEVICES=0 \
	python eval_kp.py \
	--env Ball \
	--stage kp \
	--nf_hidden_kp 16 \
	--n_kp 5 \
	--inv_std 10 \
	--eval_kp_epoch 2 \
	--eval_kp_iter 10000 \
	--eval_set demo \
	--store_demo 1 \
	--store_result 0 \
	--store_st_idx 0 \
	--store_ed_idx 10 \
	--eval_st_idx 0 \
	--eval_ed_idx 500 \
