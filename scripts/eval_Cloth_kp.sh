CUDA_VISIBLE_DEVICES=0 \
	python eval_kp.py \
	--env Cloth \
	--stage kp \
	--nf_hidden_kp 32 \
	--n_kp 10 \
	--inv_std 10 \
	--eval_kp_epoch 14 \
	--eval_kp_iter 15000 \
	--eval_set valid \
	--store_demo 0 \
	--store_result 1 \
	--store_st_idx 0 \
	--store_ed_idx 200 \
	--eval_st_idx 0 \
	--eval_ed_idx 300 \
