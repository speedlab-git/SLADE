#!/bin/bash
python -m vlm_eval.run_evaluation_of \
--eval_flickr30 \
--attack apgd --eps 4 --steps 100 --mask_out context \
--num_samples 500 \
--shots 0 \
--batch_size 1 \
--results_file res9B \
--model open_flamingo \
--out_base_path /c/CodesSpring24/SLADE/cocoadv \
--vision_encoder_path ViT-L-14 \
--checkpoint_path /c/CodesSpring24/SLADE/flamingocheckpoint.pt \
--lm_path anas-awadalla/mpt-7b \
--lm_tokenizer_path anas-awadalla/mpt-7b \
--precision float16 \
--cross_attn_every_n_layers 4 \
--flickr_image_dir_path /c/CodesSpring24/Data/flickr30k_images/flickr30k_images \
--flickr_karpathy_json_path /c/CodesSpring24/SLADE/eval_benchmark/flickr30k/dataset_flickr30k.json \
--flickr_annotations_json_path /c/CodesSpring24/SLADE/eval_benchmark/flickr30k/dataset_flickr30k_coco_style.json \
--vision_encoder_pretrained 

