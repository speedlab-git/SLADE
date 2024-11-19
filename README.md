
<h1 align="center">SLADE: Shielding against Dual Exploits in Large Vision-Language Models</h1>






<!-- ![system architecture](./utils/arch.png) -->


## Contents

- [Env Installation](#installation)
- [Datasets](#dataset)
- [Adversarial Training SLADE](#adversarial-training)
- [Adversarially fine-tuned Encoders](#models)
- [Evaluation](#evaluation)
- [Acknowledgement](#acknowledgement)



## Installation



1. We recommend you to use [Anaconda](https://www.anaconda.com/products/distribution) to maintain installed packages and the environment. We use **Python 3.11** for our training and evaluation. Install required packages using the following commands:

```
conda create -n SLADE python=3.11 -y
conda activate SLADE
pip install -r requirements.txt
```

## Dataset

### Adversarial training dataset

We perform adversarial pre-training of CLIP on the ImageNet dataset. You can download the ImageNet dataset from [this link](https://www.image-net.org/download.php) or use the following command:


```
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
```

After downloading the ImageNet dataset, extract the training and validation data using the provided script in `bash` folder:

```
./bash/imagenet/extract_ILSVRC.sh
```

### Evaluation dataset

<p align="justify">To assess the robustness and effectiveness of our fine-tuned CLIP vision encoder, we employ a variety of datasets designed for specific tasks. For Visual Question Answering (VQA), we use the OKVQA and VizWiz datasets, which offer rigorous benchmarks for evaluating the model's capability to comprehend and respond to questions based on visual information. For image captioning, we utilize the COCO and Flickr30k datasets, known for their extensive annotations and diverse image collections. Below is a table with download links for each dataset used in our studies:</p>

| Dataset Name | Download Link                                                                         |
| ------------ | ------------------------------------------------------------------------------------- |
| OKVQA        | [Download OKVQA](https://okvqa.allenai.org/download.html)                             |
| COCO         | [Download COCO](https://cocodataset.org/#download)                                    |
| Flickr30k    | [Download Flickr30k](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset) |
| VizWiz       | [Download VizWiz](https://vizwiz.org/tasks-and-datasets/)                             |

<!-- https://huggingface.co/datasets/openflamingo/eval_benchmark/tree/main -->

## Adversarial training

In this repository, we provide scripts for running adversarial training with our proposed method, `SLADE`. We have provided bash scripts for easier execution of this training method.

### 1. SLADE<sup>4</sup>

Execute the bash script(you can specify the training parameters inside). Make sure you are in the `SLADE` folder

```
./bash/training/SLADE_train.sh
```


## Load Robust encoder SLADE

To use these models, you can load them using the provided code. For example, to load the SLADE<sup>4</sup> model, you can use the following code snippet:

```
import torch
import open_clip
model, _, image_processor = open_clip.create_model_and_transforms(
            'ViT-L-14', pretrained='openai', device='gpu'
        )

checkpoint = torch.load('SLADE4.pt path', map_location=torch.device('gpu'))
model.vision_encoder.load_state_dict(checkpoint)
```


### **Note:**

- Set `--imagenet_root` with the path of your downloaded ImageNet dataset. Set `eps 2` in the bash script to obtain SLADE<sup>2</sup> model
- If you are facing any issues with the GPU running out of memory, please reduce the `batch size`


## Models

| Model Name           | Type   | Source                                                 | Download Link                                                                                               |
| -------------------- | ------ | ------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------- |
| CLIP                 | Clean  | [OpenAI](https://arxiv.org/pdf/2103.00020)                   | [Load CLIP model](https://huggingface.co/docs/transformers/en/model_doc/clip#transformers.CLIPModel)        |
| SLADE<sup>4</sup> | Robust | Our Method                                                   | [Download SLADE<sup>4</sup>](https://huggingface.co/mdzhossain/SLADE/)        |
| SLADE<sup>2</sup> | Robust | Our Method                                                   | [Download SLADE<sup>2</sup>](https://huggingface.co/mdzhossain/SLADE/)        |
| FARE<sup>4</sup>     | Robust | [Schlarmann et al. (2024)](https://arxiv.org/pdf/2402.12336) | [Download FARE<sup>4</sup>](https://huggingface.co/collections/chs20/robust-clip-65d913e552eca001fdc41978)  |
| FARE<sup>2</sup>     | Robust | [Schlarmann et al. (2024)](https://arxiv.org/pdf/2402.12336) | [Download FARE<sup>2</sup>](https://huggingface.co/collections/chs20/robust-clip-65d913e552eca001fdc41978)  |
| TeCoA<sup>4</sup>    | Robust | [Mao et al. (2023)](https://arxiv.org/abs/2212.07016)        | [Download TeCoA<sup>4</sup>](https://huggingface.co/collections/chs20/robust-clip-65d913e552eca001fdc41978) |
| TeCoA<sup>2</sup>    | Robust | [Mao et al. (2023)](https://arxiv.org/abs/2212.07016)        | [Download TeCoA<sup>2</sup>](https://huggingface.co/collections/chs20/robust-clip-65d913e552eca001fdc41978) |



## Evaluation

### Zero-shot Classification on AutoAttack

Acquire the classification dataset by visiting the Huggingface CLIP_benchmark repository at [Huggingface CLIP_benchmark](https://huggingface.co/clip-benchmark). Configure the models for evaluation in `CLIP_benchmark/benchmark/models.txt` and specify the datasets in `CLIP_benchmark/benchmark/datasets.txt`. Then execute

```

cd CLIP_benchmark
./bash/run_benchmark_adv.sh

```




### Down-stream tasks evaluation (Untargeted Attacks)

Before proceeding with Down-stream tasks evaluations, download validation annotations set from [Huggingface openflamingo repository](https://huggingface.co/datasets/openflamingo/eval_benchmark/tree/main)

### Captioning Tasks

- OpenFlamingo

  To evaluate the OpenFlamingo 9B model, first download the model from [here](https://huggingface.co/openflamingo/OpenFlamingo-9B-vitl-mpt7b/tree/main). Then, supply the downloaded annotation set and flamingo checkpoint paths in `/bash/of_eval_9B_coco.sh` . Set the `--vision_encoder_pretrained` parameter to `openai` or provide the path to a fine-tuned CLIP model checkpoint (e.g., SLADE). Finally, run the evaluation script.

```

./bash/of_eval_coco.sh

```

```
./bash/of_eval_Flickr.sh

```

- LLaVA

  The LLaVA1.5-7B model checkpoint will be automatically downloaded from repository. For LLaVA-13B evluation on downstream tasks, please update the model path with `LLaVA-13B'. Then execute the following command:

```

./bash/LLaVA_eval_coco.sh

```

### Visual Question Answering Tasks

- For VQA, provide the path of OKVQA dataset in the script and then execute the following commands:

For LLaVA run

```

./bash/LLaVA_eval_okvqa.sh

```

For Flamingo run

```

./bash/of_eval_okvqa.sh

```

## Targeted attacks

To perform targeted attacks with the LLaVA model on the COCO or Flickr30k dataset, please run these steps:

```
./bash/eval_targeted.sh
```

**Note**: Default target strings can be updated in `run_evaluation.py`

For targeted attacks on custom images, update `vlm_eval/run_evaluation_qualitative.py` with your images and captions, then execute:

```
python -m vlm_eval.run_evaluation_qualitative --precision float32 --attack apgd --eps 2 --steps 10000 --vlm_model_name LLaVA --vision_encoder_pretrained openai --verbose
```

**Note**:
To increase the strength of the attack, modify the `--attack` parameter with higher steps in the bash script. A higher attack step size results in a stronger attack.




## Jailbreak Attacks 

### VisualAdv 

To run VisualAdv, follow these steps:

1. Change directory to VisualAdv:
    ```
    cd VisualAdv
    ```

2. Set up the environment for LLaVA-LLaMA-2 by following the instructions in the original repository: [LLaVA Repository](https://github.com/haotian-liu/LLaVA)

3. After setting up the environment and obtaining the model, save it to the path:
    ```
    ./ckpts/llava_llama_2_13b_chat_freeze
    ```

4. The adversarial image is already provided in the bash prompt with the adversarial prompts.

5. Run the following command to execute the inference:
    ```
    ./bash/llava_inference_robustPrev.sh
    ```


### ImgJP attack

1. Change directory to ImgJP:
    ```
    cd ImgJP
    ```
2. Set up the environment using the provided YAML file:
    ```
    conda env create -f environment.yml
    conda activate jbattack
    ```

3. For inference, we use LLAVA modules setup in VisualAdv attack. Go to the VisualAdv folder and run:
    ```
    ./bash/llava_inference_ImgJP.sh
    ```


# COT Evaluation


To evaluate the model on the ScienceQA dataset, you can use the provided bash scripts. 

Download the ScienceQA image dataset, please use the following link: [ScienceQA Image Dataset](https://drive.google.com/drive/folders/16kuhXdM-MOhYcFIyRj91WvnDnjnF-xHw)

Follow the instructions below to run the evaluations:


```
./bash/eval_pope.sh openai 

./bash/eval_pope.sh   # change the script with vision encoder

```
# Instruction Following Benchmark
 Download the LLaVA-13B model from the `llava-llama-2-13b-chat-lightning-preview`, you can use the following link:

[Download LLaVA Model](https://huggingface.co/liuhaotian/llava-llama-2-13b-chat-lightning-preview)


```
./bash/eval_visitBench.sh
```

## Acknowledgements

We extend our gratitude to the developers and contributors of the following repositories for their invaluable resources and tools that have significantly aided in the development and evaluation of our project:

- [OpenFlamingo](https://github.com/mlfoundations/open_flamingo)
- [LLaVA](https://github.com/haotian-liu/LLaVA)
- [CLIP Benchmark](https://github.com/LAION-AI/CLIP_benchmark)
- [AutoAttack](https://github.com/fra31/auto-attack)
- [RobustVLM](https://github.com/chs20/RobustVLM)
