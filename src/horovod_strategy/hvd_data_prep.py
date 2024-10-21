'''
Data preparation module for running RefinedWeb dataset on multipl ranks for
pretraining Large Language Model from Power Law Decoder Representations (PLDR-LLM).
'''

import tensorflow as tf
import tensorflow_text as text
import datasets as hfds

MAX_LENGTH=512

def load_tokenizer(model_name):
    '''
    Method to load tokenizer from saved model file.
    Returns tokenizer and vocabulary size.
    '''

    print(f"LOADING TOKENIZER AT {model_name}")
    tokenizers = tf.saved_model.load(model_name)
    print("TOKENIZER LOADED")
    src_lang='en'

    print("THE TOKENIZER ITEMS AVAILABLE ARE:")
    print([item for item in dir(tokenizers) if not item.startswith('_')])
    tokenizers_src=getattr(tokenizers, src_lang, None)

    return tokenizers_src, tokenizers_src.get_vocab_size().numpy()


def load_falcon_refinedweb(tok_model_name, split_interval, rank_cnt, rank_index, 
                       BATCH_SIZE=32, MAX_LENGTH=512, AGG_BATCHES=100, 
                       shuffle_files=False, shuffle_set=False, cache_dir=None):
    '''
    Load falcon-refinedweb dataset.
    tok_model_name: path to tokenizer model
    split_interval: the interval as tuple of start and end indices from dataset to load.
    rank_cnt: total  number of ranks
    rank_index: index of current rank
    BATCH_SIZE: Batch size for processed dataset.
    MAX_LENGTH: Context length
    AGG_BATCHES: the multiplier for batch_size to densify batches by concatenating
    shuffle_files: If True, shuffle dataset files while loading
    shuffle_set: If True, shuffle the dataset while loading
    
    '''

    tokenizer_src, vocab_size=load_tokenizer(tok_model_name)
    
    def tokenize_fun(src):
        '''Use tokenizer model to tokenize input.'''
        src = tokenizer_src.tokenize(src)
        src = src.to_tensor()
        return src
    
    def concat_chunk_batches(X_batched):
        '''concat and chunk all tokens in batch'''
        X_batched=X_batched.merge_dims(0, -1)
        X_batched=tf.convert_to_tensor(X_batched)
        end_seq=int(tf.shape(X_batched)[0])
        row_indices=tf.range(0, end_seq, MAX_LENGTH)
        X_batched=tf.RaggedTensor.from_row_starts(X_batched, row_starts=row_indices)
        X_batched=X_batched.to_tensor()
        return X_batched
    
    print("Loading from falcon-refinedweb dataset")
    split_train='train'
    start_ind, end_ind=split_interval
    dataset_file="tiiuae/falcon-refinedweb"

    examples = hfds.load_dataset(dataset_file, split=[f'{split_train}[{start_ind}:{end_ind}]'], cache_dir=cache_dir, num_proc=8)
    examples[0]=examples[0].shard(num_shards=rank_cnt, index=rank_index)
    examples[0]=examples[0].to_tf_dataset(columns='content', shuffle=shuffle_files, prefetch=False)
    if shuffle_set:
        examples[0]=examples[0].shuffle(buffer_size=20000)
    train_batches = examples[0].batch(1)
    train_batches = train_batches.map(tokenize_fun, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    train_batches=train_batches.unbatch()
    print("Batching by concatenating and chunking samples.")
    train_batches=train_batches.ragged_batch(batch_size=int(BATCH_SIZE*AGG_BATCHES), drop_remainder=False)
    train_batches = train_batches.map(concat_chunk_batches, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    train_batches = train_batches.unbatch()
    train_batches=train_batches.padded_batch(BATCH_SIZE, padded_shapes=None, drop_remainder=True)
    train_batches=train_batches.map(shift_samples, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)

    return train_batches, vocab_size, tokenizer_src


def shift_samples(inp):
    tar_inp = inp[:, :-1]
    tar_real = inp[:, 1:]
    combined_mask= create_masks(tar_inp)
    return tar_inp, tar_real, combined_mask

@tf.function
def create_masks(inp):
    '''
    Create masks for decoder layer for pldr model.
    '''

    def create_padding_mask(seq):
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
        return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

    def create_look_ahead_mask(size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask

    look_ahead_mask = create_look_ahead_mask(tf.shape(inp)[1])
    dec_target_padding_mask = create_padding_mask(inp)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return combined_mask
