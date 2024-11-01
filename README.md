## PLDR-LLM: Large Language Model From Power Law Decoder Representations

This repository is the implementation of the Large Language Model From Power Law Decoder Representations (PLDR-LLM) detailed in the research article: [PLDR-LLM: Large Language Model From Power Law Decoder Representations](https://arxiv.org/abs/2410.16703).

Large Language Model From Power Law Decoder Representations is a deductive-inductive LLM model that utilizes the decoder layers that were first developed for [Power Law Graph Transformer (PLGT)](https://arxiv.org/abs/2107.02039). 

The deductive outputs of PLDR-LLM are generated by the Power Law Graph Attention (PLGA) mechanism at each decoder layer. PLDR-LLM takes advantage of the metric tensor, the potential tensor and energy-curvature tensor to monitor and assess the model response or can be regularized to modify the model behaviour using a DAG loss.

The output and training procedure of PLDR-LLM is similar to LLMs that utilize decoders with Scaled Dot Product Attention (SDPA). The inductive output is same as the transductive output of an SDPA-LLM. At inference time, it is straightforward to replace an SDPA-LLM model with PLDR-LLM model.

#### How to reach pretrained PLDR-LLMs:

- The PLDR-LLMs that were pretrained and studied in the [research paper](https://arxiv.org/abs/2410.16703) can be found at [huggingface.co/fromthesky](https://huggingface.co/fromthesky).
- A fork of the LM Evaluation Harness Suite with PLDR-LLM support is available at [lm-evaluation-harness-with-PLDR-LLM](https://github.com/burcgokden/lm-evaluation-harness-with-PLDR-LLM).

#### Key Features:

- A flexible interface to build, customize and train PLDR-LLMs with deep layers of decoders by specifying hyperparameters such as number of layers, number of attention heads, embedding dimension, vocabulary size and more through a dictionary of hyperparameters.
- Generate continuation text from input text through the following random sampling techniques: temperature, top-k or top-p (nucleus) sampling.
- Train interface that allows keeping track of global loss/accuracy, running loss/accuracy for training and validation datasets as well as DAG loss values for each deductive outputs in a single epoch.
- Run scripts to train PLDR-LLM on single or multiple-gpus.
- Data preparation optimized for pretraining PLDR-LLMs. Implementation is optimized for the Refined-Web dataset used in the research paper.

#### Limitations:
The PLDR-LLM is implemented with vanilla features for now. The distribution of a single multi-billion parameter PLDR-LLM on multiple gpus is not supported and the model is not optimized for fast inference.
- The maximum model size will be limited by the parameter size of a model that can fit a single gpu memory, which is typically in the sub-billion parameter range.
- Inference is slow, and for deeper models, generating text takes longer.

#### Data Preparation:

Below is an example for initializing a pldr_data_prep object for preprocessing first 8 million samples of RefinedWeb dataset. The dataset samples are tokenized, concatenated and chunked into contiguous sequences of 1024 token length with 32 batch size. Tokenizer is a sentencepiece tokenizer model that is wrapped in the format of the [Sentencepiece Tokenizer Wrapper for PLDR-LLM](https://github.com/burcgokden/Sentencepiece-Tokenizer-Wrapper-for-PLDR-LLM). batch_agg_count is a multiplier to concatenate the tokenized samples together in a large sequence to reduce number of padded samples. Data source is 'hf' for huggingface and 'tf' for tensorflow.

```python
import pldr_data_prep

inp_obj = pldr_data_prep.pldr_data_prep(
                 src_lang='en',
                 BUFFER_SIZE=20000,
                 BATCH_SIZE = 32,
                 data_source='hf',
                 dataset_file='tiiuae/falcon-refinedweb',
                 dataset_name='falcon-refinedweb',
                 split_names=None,
                 load_dataset=True,
                 load_from_train=True,
                 split_style='index', 
                 train_intvl= [0, 8000000], 
                 val_offset= 64000,
                 test_offset=None,
                 model_name="/path/to/tokenizer/model",
                 shuffle_set=False,
                 shuffle_files=False,
                 MAX_LENGTH=1024,
                 as_supervised=False,
                 batch_agg_count=100,
                 with_prefetch=True,    
                )
```

#### PLDR-LLM Hyperparameter Definition:

PLDR-LLM hyperparameter dictionary is composed of model architecture parameters and training parameters. The Feedforward Network (FFN) size dff at the end of each decoder layer is scaled by 2/3 since SwiGLU FFN has three weights. The input_vocab_size is obtained from data preparation step by calling the get_vocab_size() method for the tokenizer. The difference between v500 and v900 type of models is the size of linear layer in the SwiGLU residual network units. For v500, it is equal to $d_{k}$=64 and for v900 it is equal to A_dff_depth=112. 

- For PLDR-LLM v500:

```python
num_layers=5
num_heads=14
dk=64

epochs=1
batch_count=125000
tot_steps=batch_count*epochs

pldr_llm_v500_hparams= {
          "num_layers":num_layers,
          "d_model": num_heads*dk,  
          "num_heads": num_heads,
          "dropout_rate":0.0,
          "dff": int(np.floor(num_heads*dk*4*2/3)), 
          "att_dropout_rate_in":0.0,
          "att_dropout_rate_eij":0.0,                                                    
          "Adropout_rate":0.0,
          "A_dff":170,
          "num_reslayerA":8,
          "num_denseA":2,
          "input_vocab_size": inp_obj.tokenizers_src.get_vocab_size().numpy(), 
          "epochs":epochs,
          "save_model_path": "my_pldr_llm_model",       
          "warmup_steps": 8000, 
          "lr_total_steps": tot_steps,
          "learning_rate": 1e-3,
          "lr_alpha": 0.1,
          "adamw_decay": 0.1,
          "activation": tf.nn.silu, 
          }
```

- For PLDR-LLM v900:

```python
num_layers=3
num_heads=17
dk=64

epochs=1
batch_count=125000
tot_steps=batch_count*epochs

pldr_llm_v900_hparams= {
          "num_layers":num_layers,
          "d_model": num_heads*dk,  
          "num_heads": num_heads,
          "dropout_rate":0.0,
          "dff": int(np.floor(num_heads*dk*4*2/3)), 
          "att_dropout_rate_in":0.0,
          "att_dropout_rate_eij":0.0,                                                    
          "Adropout_rate":0.0,
          "A_dff":300,
          "A_dff_depth":112,
          "num_reslayerA":8,
          "num_denseA":2,
          "input_vocab_size": inp_obj.tokenizers_src.get_vocab_size().numpy(), 
          "epochs":epochs,
          "save_model_path": "my_pldr_llm_model",       
          "warmup_steps": 8000, 
          "lr_total_steps": tot_steps,
          "learning_rate": 1e-3,
          "lr_alpha": 0.1,
          "adamw_decay": 0.1,
          "activation": tf.nn.silu, 
          }

```

#### PLDR-LLM Model Initialization and Training for Single GPU

- **Model Initialization:**

Model is initialized for training with randomly initialized weights by setting load_ckpt to None.

```python
import pldr_run_model_v500 as pldr_run_model

e2e_obj=pldr_run_model.pldr_model_e2e(tokenizer_obj_src = inp_obj.tokenizers_src,
                               checkpoint_path = "path/to/checkpoint/saves",
                               hpdict=pldr_llm_v500_hparams,
                               load_ckpt=None
                                     )
```


- **Model Training: Single-GPU**

For training the train_batches and val_batches attributes of the inp_obj instance are used as training and validation datasets. 

```python
batch_size=32
batch_count=125000
verbose_freq = 2000 
val_verbose_freq=12000
train_batches=inp_obj.train_batches.take(batch_count)
chkpt_batches= [int(batch_count*2/8), int(batch_count*3/8), 
                int(batch_count*4/8), int(batch_count*5/8), int(batch_count*6/8)]
chkpt_epochs=None

train_loss, train_accuracy, val_loss, val_accuracy=
                        e2e_obj.train_model(pt_training=True,
                                            pt_trainable=True,
                                            train_batches=train_batches,
                                            val_batches=inp_obj.val_batches,
                                            chkpt_epochs=chkpt_epochs,
                                            chkpt_batches=chkpt_batches,
                                            verbose_freq=verbose_freq,
                                            val_verbose_freq=val_verbose_freq
                                                                      )
```

- **Model Training: Multi-GPU**

For multi-GPU, modify the model and training hyperparameters and path locations for model training and tokenizer within the script *_pretrain_run.py. 

To run on a machine with 2 GPUs:
```
python3 '/path/to/run_script' 2 'localhost:2' 'mpi' 2>&1 | tee '/path/to/log/file'
```

#### PLDR-LLM Model Evaluation

After a model is trained, the latest train checkpoint found at the checkpoint_path can be loaded as follows:
```python
e2e_obje=pldr_run_model.pldr_model_e2e(tokenizer_obj_src = inp_obj.tokenizers_src,
                                       checkpoint_path = "/path/to/checkpoint/folder",
                                       hpdict=pldr_llm_v500_hparams,
                                       load_ckpt="train"
                                      )
```

A custom checkpoint can be loaded as follows:
```python
e2e_obje=pldr_run_model.pldr_model_e2e(tokenizer_obj_src = inp_obj.tokenizers_src,
                                       checkpoint_path = "/path/to/checkpoint/folder",
                                       hpdict=pldr_llm_v500_hparams,
                                       load_ckpt="/path/to/checkpoint/file"
                                      )
```

Above models assume that tokenizer is already loaded using pldr_data_prep module. The tokenizer can be loaded without loading and preprocessing dataset by setting load_dataset to False.

To generate continuation text for an input sentence, generate_text method can be used.

```python
sentence="Write a letter requesting that people use language models responsibly."
translated_text, translated_tokens, att_weights, eval_length = e2e_obje.generate_text(sentence,
                                                                           temperature=1.0,
                                                                           top_k=1,
                                                                           top_p=1.0,
                                                                           max_length=100, 
                                                                           save_att=None)
e2e_obje.print_generated_text(sentence, translated_text, eval_length)
```
The temperature, top_k and top_p sampling can be stacked, but most common is to use one type of sampling. (temperature=1, top_k=0, top_p=1) setting disables all sampling approaches. For greedy sampling top_k=1 can be used (with top_p=1, temperature=1). A nucleus sampling only example at top_p=0.6 would be as follows:
```python
sentence="Write a letter requesting that people use language models responsibly."
translated_text, translated_tokens, att_weights, eval_length = e2e_obje.generate_text(sentence,
                                                                           temperature=1.0,
                                                                           top_k=0,
                                                                           top_p=0.6,
                                                                           max_length=100, 
                                                                           save_att=None)
e2e_obje.print_generated_text(sentence, translated_text, eval_length)
```
#### Deductive Outputs

Below are the deductive outputs used for monitoring and regularizing the PLDR-LLM. For more on deductive outputs, please see the papers for [PLDR-LLM](https://arxiv.org/abs/2410.16703) and [Power Law Graph Transformer](https://arxiv.org/abs/2107.02039).

Metric Tensor **A<sub>LM</sub>** (# decoder layers, # attention heads, $d_k$, $d_k$):
```python
tf.convert_to_tensor([w[1] for w in att_weights[0]])
```

Potential Tensor **P** (# decoder layers, # attention heads, $d_k$, $d_k$):
```python
tf.convert_to_tensor([tf.math.pow(w[1],w[2]) for w in att_weights[0]])
```

Energy-Curvature Tensor **G<sub>LM</sub>**  (# decoder layers, # attention heads, $d_k$, $d_k$):
```python
tf.convert_to_tensor([w[5] for w in att_weights[0]])
```

#### Additional Notes:
- In [the PLDR-LLM research article](https://arxiv.org/abs/2410.16703), PLDRv5-1 model utilizes GPU memory extensively and was trained with the default BFC memory allocation setting on GPU.
- For horovod, Open-MPI version 5.0.3 was used.

#### Citation:

Please cite this work as:
```bibtex
@misc{gokden2024pldrllm,
      title={PLDR-LLM: Large Language Model from Power Law Decoder Representations}, 
      author={Burc Gokden},
      year={2024},
      eprint={2410.16703},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.16703}, 
}
```