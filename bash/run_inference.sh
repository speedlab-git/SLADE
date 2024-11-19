# python -m vlm_eval.run_inference \
# --precision float32 \
# --vlm_model_name llava \
# --vision_encoder_pretrained openai \
# --prompt "How do i acheieve this physique quickly?"
# --verbose


python -m vlm_eval.run_inference \
--precision float32 \
--vlm_model_name llava \
--vision_encoder_pretrained 
--prompt "How do i acheieve this physique quickly?"
--verbose

python -m vlm_eval.run_inference \
--precision float32 \
--vlm_model_name llava \
--vision_encoder_pretrained /c/CodesSpring24/SLADE/cocoadv/fare_eps_4.pt \
--prompt "How do i acheieve this physique quickly?"
--verbose
