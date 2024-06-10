# Harry Potter LLM

This repository is for training small and medium sized GPTs. However, my main focus was on creating a model to generate Harry Potter like text. I followed the excellent video by [Andrej Karpathy Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY), and his [nanoGPT](https://github.com/karpathy/nanoGPT) repository.


## install
```
pip install .
```

## quick start

### Prepare Data
Put the text data you want to train on inside `./data/{dataset_name}`. Then generate `train.bin` and `test.bin` by:
```
python prepare.py --input_file_path "./data/{dataset_name}/{text_file_name}.txt"
```

### Training
To start training, run:
```
python train --dataset "./data/{dataset_name}"
```

To start the model from a GPT2 checkpoint, run:
```
python train --init_from "GPT2" --dataset "./data/{dataset_name}"
```

You can also run `python train.py --help` to see the list of tunable hyper parameters.


### Generate sample
To generate sample text from a trained model:
```
python sample.py --init_from "{dataset_name}.ckpt.pt"
```

