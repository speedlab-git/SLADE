#!/bin/bash

evalDir=/c/CodesSpring24/SLADE/scienceqa_eval
evalModel='LLAVA'
# evalModel='openFlamingo'

modelLocation=${1}
if [ -z "${modelLocation}" ]
then
      echo "\$modelLocation is empty Using robust model from here: "
      modelLocation=SLADE4.pt
      modelAlias=slade
else
      echo "\$modelLocation is NOT empty"
      modelAlias=${modelLocation}
fi

outputFile="${evalModel}_${modelAlias}"
echo "Will save to the following json: "
echo $outputFile

python -m llava.eval.model_vqa_science \
    --model-path liuhaotian/llava-v1.5-7b \
    --eval-model ${evalModel} \
    --pretrained_rob_path ${modelLocation} \
    --question-file "${evalDir}/llava_test_CQM-A.json" \
    --image-folder ScienceQA/data/test/test \
    --answers-file ${evalDir}/answers/${outputFile}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1
# https://drive.google.com/drive/folders/16kuhXdM-MOhYcFIyRj91WvnDnjnF-xHw ScienceQA image dataset
python llava/eval/eval_science_qa.py \
    --base-dir ${evalDir} \
    --result-file ${evalDir}/answers/${outputFile}.jsonl \
    --output-file ${evalDir}/answers/${outputFile}_output.jsonl \
    --output-result ${evalDir}/answers/${outputFile}_result.json
