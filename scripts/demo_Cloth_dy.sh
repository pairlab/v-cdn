CUDA_VISIBLE_DEVICES=0 \
	python eval_dy.py \
	--env Cloth \
	--stage dy \
	--baseline 0 \
	--gauss_std 5e-2 \
	--lam_kp 1e1 \
	--en_model cnn \
	--dy_model gnn \
	--nf_hidden_kp 32 \
	--nf_hidden_dy 16 \
	--n_kp 10 \
	--inv_std 10 \
	--min_res 20 \
	--n_identify 20 \
	--n_his 5 \
	--n_roll 5 \
	--node_attr_dim 0 \
	--edge_attr_dim 1 \
	--edge_type_num 2 \
	--edge_st_idx 1 \
	--edge_share 0 \
	--eval_set demo \
	--eval_st_idx 20 \
	--eval_ed_idx 60 \
	--identify_st_idx 0 \
	--identify_ed_idx 20 \
	--eval_kp_epoch 14 \
	--eval_kp_iter 15000 \
	--eval_dy_epoch 7 \
	--eval_dy_iter 0 \
	--store_demo 1 \
	--vis_edge 1 \