



# Default values
MODEL_PATH="ckpts/llava-llama-2-13b-chat-lightning-preview"
DATA_CSV="path/to/your/single_image_full_dataset.csv"
ENCODER="openai"

# Run the Python script
python VisIT-Bench.llava_inference.py \
    --model-path ${MODEL_PATH} \
    --csv_path ${DATA_CSV} \
    --encoder ${ENCODER} \
    --gpu-id 0

ENCODER_NAME=$(basename "$ENCODER" | cut -f 1 -d '.')
echo "Results saved to: ${ENCODER_NAME}_visitbench.csv"


python metrics_calc.py ${ENCODER_NAME}_visitbench.csv
