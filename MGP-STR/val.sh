CUDA_VISIBLE_DEVICES=0 python test_final.py --eval_data ./data/val/namuwiki2 --Transformer char-str  \
--TransformerModel=char_str_base_patch4_3_32_128 \
--model_dir ./model_files/char_gray_base_max50_extend_dict-Seed456/best_accuracy.pth --batch_size 100 --PAD \
--character ./korean_dict_noCn.pkl --batch_max_length 50 --workers 12 --input_channel 1