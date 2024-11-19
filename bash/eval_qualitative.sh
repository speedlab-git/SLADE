#!/bin/bash

python -m vlm_eval.run_evaluation_qualitative \
--precision float32 \
--attack apgd \
--eps 2 \
--steps 10000 \
--vlm_model_name llava \
--vision_encoder_pretrained cocoadv/fare_eps_4.pt \
--verbose
