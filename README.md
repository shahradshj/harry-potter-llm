# Harry Potter LLM

This repository is for training small and medium sized GPTs. However, my main focus was on creating a model to generate Harry Potter like text. I followed the excellent video by [Andrej Karpathy Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY), and his [nanoGPT](https://github.com/karpathy/nanoGPT) repository to create this repo.


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

### eamples

Sample generated from a model trained from scratch on my laptop with a RTX 4060 for about 10 minutes using default parameters starting with "Harry, Ron, and Hermione":
```
Harry, Ron, and Hermione plungeded into the trapdoor, on the other side of the damp grass.
It was the tiny black-looking moving and a few feet were dusty-lamps. He had a tumbling silver. He screwed up a little tablecloth, and said, "Oewter" -- his batty face quivered.
He pointed his wand at the ground.
"C'mon," he said, giving Ron the Ravenclaw and Harry Harry put on and Hermione. "He hasn't been to kill. But he was a bit of something. A reallyhaired, light came from the ceiling. He turned around and felt himself behind the dark.
"What's going on?" he said to the other two. "He gave me a nervous old socks. I want to put it to take Sorting!"
"He's coming away," said Harry. "He was just trying to keep him to kill, not for me -- and he's not in your family."
"He's all right," said Ron promptly. "There are both too far, haven't you, I supposed to be surprised it."
"What about?" said Harry, his eyes narrowed. "It's a matter of three of the last time I tried to drop!"
"It's the way -- look --"
"It's the better," said Harry, pointing at the end of the end of the classroom. "You're not going to get back. He'll be really down."
"I see him," said Ron, without looking at the Ravenclaw table. "Sly. Don't you've got it, my Nimbus Two Thousand."
When the end of his Great Hall, Snape was a very fast asleep.
"Oh," he said. "You're in the Fat Lady, Harry Potter...."
Harry turned around, watching, then said, who looked mad. "He's going to Hogsmeade, see you again. There are going to get into a hand in there!" He glanced around.
"Well -- no -- you want to go and get back!"
"Yeah, you!" said Harry. "Come on, runing under the Shrieking Shack and get back. It was a very hard outside, but he was too weak. He pulled off his leg.
"What about?" said Ron. "I've got to look too!"
"What are you planning to go on, Harry?"
```

Sample generated from a model initialized from smallest GPT2 model on my laptop with a RTX 4060 for about 15 minutes starting with "Snow was falling on Hogwarts":
```
Snow was falling on Hogwarts grounds, but Harry wasn't looking at Ron, who was frowning slightly.
"Well, that doesn't make any sense," he said, very slowly.
"It doesn't make any sense," said Hermione, still frowning slightly. "Yes, but there we are, outside in the grounds, we were going upstairs to the library."
"I made a move," said Harry. He looked around. Then he said, "Who's there?"
"Not you," said Hermione, still frowning. "I heard you had that stupid train accident. I heard you'd want to get out of that one as well."

"Oh, I forgot," said Harry, frowning slightly. "I don't know . . ."
"What are you talking about?"
Harry looked down at his robes. He had never seen any of them on his own, and there was nothing from them. It was something very flammable. They were all lying on piles of wood beside a large, carpeted table.
"Harry! Harry! Harry!" Hermione yelled as the two of them hugged each other.
"I'm sorry, Hermione!" said Ron, still staring wildly at the pair of them. "I'm supposed to be helping you. I promised you that you'd be OK! This wasn't what we were going to do. You'd be too late. Ron and Hermione had to hurry --"
"What?" said Harry.
"See, it's OK," said Hermione, tears welling in her eyes. "Just -- come on -- just stay here, Harry. We'll be able to get to you in a minute, then."

"I'm going to have to sit down, and I need to get this thing out of here," Harry told her.
"Harry, you know where I'm going -- I need to --"
Harry was shaking uncontrollably. He had a nasty feeling that he was going to be in the hospital wing for weeks.
"I'm going to be in shock," he said. "It's going to make you feel worse than you ever did."
"I'm going to get you in there --"
```
