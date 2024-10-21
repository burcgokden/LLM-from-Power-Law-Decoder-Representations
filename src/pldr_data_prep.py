'''
Data preparation module for Large Language Model from Power Law Decoder Representations. (PLDR-LLM)
Loads a dataset ready for pretraining. Presets for falcon-refinedweb is available.
'''


import logging

import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow_text as text
import datasets as hfds

logging.getLogger('tensorflow').setLevel(logging.ERROR)


class pldr_data_prep:
    '''
    Prepares dataset for Large Language Model from Power Law Decoder Representations (PLDR-LLM) for pretraining
    Optimized for loading falcon-refinedweb dataset
    '''
    def __init__(self,
                 src_lang="en",
                 BUFFER_SIZE=20000,
                 BATCH_SIZE = 32,
                 data_source = "hf",
                 dataset_file="tiiuae/falcon-refinedweb",
                 dataset_name="falcon-refinedweb",
                 split_names=None,
                 load_dataset=True,
                 load_from_train=False, 
                 split_style="index",
                 train_intvl=None, 
                 val_offset=None,
                 test_offset=None, 
                 model_name = "/path/to/tokenizer/model",
                 shuffle_set=True, 
                 shuffle_files=True,
                 MAX_LENGTH=None,
                 as_supervised=False,
                 batch_agg_count=None, 
                 with_prefetch=True
                 ):
        '''
        Args:
            src_lang: source language abbreviation as string for tokenizer model
            BUFFER_SIZE: Buffer size for shuffling
            BATCH_SIZE: Batch size for dataset
            dataset_file: path to huggingface/tensorflow dataset
            dataset_name: used to apply dataset specific mappings for pretraining, such as falcon-refinedweb
            load_dataset: if True load the dataset
            load_from_train: load from train split only
            split_style: 'percent' or 'index' based splitting of dataset
            train_intvl: input is a tuple integer (start,end), None loads all data. 
            val_offset: input is single integer as offset, None skips validation dataset
            test_offset: input is single integer as offset, None skips test dataset.
            model_name: file path for tokenizer model
            shuffle_set: If True, shuffle the dataset while loading
            shuffle_files: shuffle dataset files while loading
            MAX_LENGTH: maximum number of tokens in each sentence.
            as_supervised: if true, tensorflow dataset will be a tuple of (input, label)
            batch_agg_count: the multiplier for batch_size to densify batches by concatenating
            with_prefetch: if True, use prefetch as last data pipeline step

        Returns batched, tokenized train, validation and/or test datasets ready for pretraining. 
        Tokenizer methods are accessible through instance of this class object
        '''

        self.BUFFER_SIZE=BUFFER_SIZE
        self.BATCH_SIZE=BATCH_SIZE
        self.MAX_LENGTH = MAX_LENGTH if MAX_LENGTH is not None else 512
        self.AGG_BATCHES=batch_agg_count if batch_agg_count is not None else 100
        self.with_prefetch=with_prefetch
        self.model_name = model_name
        self.src_lang = src_lang
        self.as_supervised=as_supervised
        self.tokenizers_src, self.tokenizers = self.load_tokenizer()

        #load dataset
        if split_names is not None:
            split_train, split_val, split_test=split_names
        else:
            split_train='train'
            split_val='validation'
            split_test='test'

        if load_dataset:
            print("LOADING DATASET")
            if load_from_train:
                #Load dataset from train data
                if train_intvl:
                    start_ind, end_ind=train_intvl
                    
                    assert (val_offset is None) or val_offset > 0, "WARNING: validation offset should be positive"
                    assert (test_offset is None) or (test_offset > 0 and test_offset > end_ind+val_offset), \
                                                                    "WARNING: test offset is overlapping validation data"
                    #load only percentage of train data
                    if  not val_offset and not test_offset:
                        if split_style=='percent':
                            if data_source=='tf':
                                examples, metadata = tfds.load(dataset_file,
                                                            split=[f'{split_train}[{start_ind}%:{end_ind}%]'],
                                                            with_info=True, as_supervised=self.as_supervised, shuffle_files=shuffle_files)
                                self.metadata = metadata
                            elif data_source=='hf':
                                examples = hfds.load_dataset(dataset_file,
                                                            split=[f'{split_train}[{start_ind}%:{end_ind}%]'])
                            else:
                                print("Unrecognized data source, choose between tf or hf")
                        elif split_style=='index':
                            if data_source=='tf':
                                examples, metadata = tfds.load(dataset_file,
                                                            split=[f'{split_train}[{start_ind}:{end_ind}]'],
                                                            with_info=True, as_supervised=self.as_supervised, shuffle_files=shuffle_files)
                                self.metadata = metadata
                            elif data_source=='hf':
                                examples = hfds.load_dataset(dataset_file,
                                                            split=[f'{split_train}[{start_ind}:{end_ind}]'])
                            else:
                                print("Unrecognized data source, choose between tf or hf")
                        
                        #convert dataset format to tensorflow after loading dataset for hf
                        if data_source=='hf':
                            print(f"Dataset Info for {dataset_name}:")
                            print(f"Train: {examples[0]}")
                            if dataset_name in ['falcon-refinedweb']:
                                examples[0]=examples[0].to_tf_dataset(columns='content', shuffle=shuffle_files)
                            else:
                                examples[0]=examples[0].to_tf_dataset(shuffle=shuffle_files)
                        self.train_examples = examples[0]
                        self.val_examples = None
                        self.test_examples=None
                    #load only percentage of train data and rest for validation set
                    elif val_offset and not test_offset:
                        if split_style=='percent':
                            if data_source=='tf':
                                examples, metadata = tfds.load(dataset_file,
                                                            split=[f'{split_train}[{start_ind}%:{end_ind}%]', 
                                                                f'{split_train}[{end_ind}%:{end_ind+val_offset}%]'],
                                                            with_info=True, as_supervised=self.as_supervised, shuffle_files=shuffle_files)
                                self.metadata = metadata
                            elif data_source=='hf':
                                examples = hfds.load_dataset(dataset_file,
                                                            split=[f'{split_train}[{start_ind}%:{end_ind}%]', 
                                                                f'{split_train}[{end_ind}%:{end_ind+val_offset}%]'])
                            else:
                                print("Unrecognized data source, choose between tf or hf")
                        elif split_style=='index':
                            if data_source=='tf':
                                examples, metadata = tfds.load(dataset_file,
                                                            split=[f'{split_train}[{start_ind}:{end_ind}]', 
                                                                f'{split_train}[{end_ind}:{end_ind+val_offset}]'],
                                                            with_info=True, as_supervised=self.as_supervised, shuffle_files=shuffle_files)
                                self.metadata = metadata
                            elif data_source=='hf':
                                examples = hfds.load_dataset(dataset_file,
                                                            split=[f'{split_train}[{start_ind}:{end_ind}]', 
                                                                f'{split_train}[{end_ind}:{end_ind+val_offset}]'])
                            else:
                                print("Unrecognized data source, choose between tf or hf")
                        # convert dataset format to tensorflow after loading dataset
                        if data_source=='hf':
                            print(f"Dataset Info for {dataset_name}:")
                            print(f"Train: {examples[0]}")
                            print(f"Validation: {examples[1]}")
                            if dataset_name in ['falcon-refinedweb']:
                                examples[0]=examples[0].to_tf_dataset(columns='content', shuffle=shuffle_files)
                                examples[1]=examples[1].to_tf_dataset(columns='content', shuffle=shuffle_files)
                            else:
                                examples[0]=examples[0].to_tf_dataset(shuffle=shuffle_files)
                                examples[1]=examples[1].to_tf_dataset(shuffle=shuffle_files)
                        self.train_examples = examples[0]
                        self.val_examples = examples[1]
                        self.test_examples=None
                    elif test_offset:
                        if split_style=='percent':
                            if data_source=='tf':
                                examples, metadata = tfds.load(dataset_file,
                                                            split=[f'{split_train}[{start_ind}%:{end_ind}%]', 
                                                                f'{split_train}[{end_ind}%:{end_ind+val_offset}%]',
                                                                f'{split_train}[{end_ind+val_offset}%:{end_ind+val_offset+test_offset}%]'],
                                                            with_info=True, as_supervised=self.as_supervised, shuffle_files=shuffle_files)
                                self.metadata = metadata
                            elif data_source=='hf':
                                examples = hfds.load_dataset(dataset_file,
                                                            split=[f'{split_train}[{start_ind}%:{end_ind}%]', 
                                                                f'{split_train}[{end_ind}%:{end_ind+val_offset}%]',
                                                                f'{split_train}[{end_ind+val_offset}%:{end_ind+val_offset+test_offset}%]'])
                            else:
                                print("Unrecognized data source, choose between tf or hf")
                        elif split_style=='index':
                            if data_source=='tf':
                                examples, metadata = tfds.load(dataset_file,
                                                            split=[f'{split_train}[{start_ind}:{end_ind}]', 
                                                                f'{split_train}[{end_ind}:{end_ind+val_offset}]',
                                                                f'{split_train}[{end_ind+val_offset}:{end_ind+val_offset+test_offset}]'],
                                                            with_info=True, as_supervised=self.as_supervised, shuffle_files=shuffle_files)
                                self.metadata = metadata
                            elif data_source=='hf':
                                examples = hfds.load_dataset(dataset_file,
                                                            split=[f'{split_train}[{start_ind}:{end_ind}]', 
                                                                f'{split_train}[{end_ind}:{end_ind+val_offset}]',
                                                                f'{split_train}[{end_ind+val_offset}:{end_ind+val_offset+test_offset}]'])
                            else:
                                print("Unrecognized data source, choose between tf or hf")
                        if data_source=='hf':
                            print(f"Dataset Info for {dataset_name}:")
                            print(f"Train: {examples[0]}")
                            print(f"Validation: {examples[1]}")
                            print(f"Test: {examples[2]}")
                            if dataset_name in ['falcon-refinedweb']:
                                examples[0]=examples[0].to_tf_dataset(columns='content', shuffle=shuffle_files)
                                examples[1]=examples[1].to_tf_dataset(columns='content', shuffle=shuffle_files)
                                examples[2]=examples[2].to_tf_dataset(columns='content', shuffle=shuffle_files)
                            else:
                                examples[0]=examples[0].to_tf_dataset(shuffle=shuffle_files)
                                examples[1]=examples[1].to_tf_dataset(shuffle=shuffle_files)
                                examples[2]=examples[2].to_tf_dataset(shuffle=shuffle_files)
                        self.train_examples = examples[0]
                        self.val_examples = examples[1]
                        self.test_examples=examples[2]
                    
                else:
                    #load all split at once
                    if data_source=='tf':
                        examples, metadata = tfds.load(dataset_file,
                                                        split=[f'{split_train}'],
                                                        with_info=True, as_supervised=self.as_supervised, shuffle_files=shuffle_files)
                        self.metadata = metadata
                    elif data_source=='hf':
                        examples = hfds.load_dataset(dataset_file,
                                                        split=[f'{split_train}'])
                    else:
                        print("Unrecognized data source, choose between tf or hf")
                    # convert dataset format to tensorflow after loading dataset for hf
                    if data_source=='hf':
                        print(f"Dataset Info for {dataset_name}:")
                        print(f"Train: {examples[0]}")
                        if dataset_name in ['falcon-refinedweb']:
                            examples[0]=examples[0].to_tf_dataset(columns='content', shuffle=shuffle_files)
                        else:
                            examples[0]=examples[0].to_tf_dataset(shuffle=shuffle_files)
                    self.train_examples = examples[0]
                    self.val_examples = None
                    self.test_examples = None                    
            else:
                #Load dataset for train, validation and test
                if train_intvl:
                    start_ind, end_ind=train_intvl
                    if split_style=='percent':
                        if data_source=='tf':
                            examples, metadata = tfds.load(dataset_file,
                                                        split=[f'{split_train}[{start_ind}%:{end_ind}%]', split_val, split_test],
                                                        with_info=True, as_supervised=self.as_supervised, shuffle_files=shuffle_files)
                            self.metadata = metadata 
                        elif data_source=='hf':
                            examples = hfds.load_dataset(dataset_file,
                                                        split=[f'{split_train}[{start_ind}%:{end_ind}%]', split_val, split_test])
                        else:
                            print("Unrecognized data source, choose between tf or hf")
                    elif split_style=='index':
                        if data_source=='tf':
                            examples, metadata = tfds.load(dataset_file,
                                                        split=[f'{split_train}[{start_ind}:{end_ind}]', split_val, split_test],
                                                        with_info=True, as_supervised=self.as_supervised, shuffle_files=shuffle_files)
                            self.metadata = metadata 
                        elif data_source=='hf':
                            examples = hfds.load_dataset(dataset_file,
                                                        split=[f'{split_train}[{start_ind}:{end_ind}]', split_val, split_test])
                        else:
                            print("Unrecognized data source, choose between tf or hf")
                else:
                    if data_source=='tf':
                        examples, metadata = tfds.load(dataset_file,
                                                        split=[split_train, split_val, split_test],
                                                        with_info=True, as_supervised=self.as_supervised, shuffle_files=shuffle_files)
                        self.metadata = metadata  
                    elif data_source=='hf':
                        examples = hfds.load_dataset(dataset_file,
                                                        split=[split_train, split_val, split_test])
                    else:
                        print("Unrecognized data source, choose between tf or hf")
                # convert dataset format to tensorflow after loading dataset
                if data_source=='hf':
                    print(f"Dataset Info for {dataset_name}:")
                    print(f"Train: {examples[0]}")
                    print(f"Validation: {examples[1]}")
                    print(f"Test: {examples[2]}")                    
                    if dataset_name in ['falcon-refinedweb']:
                        examples[0]=examples[0].to_tf_dataset(columns='content', shuffle=shuffle_files)
                        examples[1]=examples[1].to_tf_dataset(columns='content', shuffle=shuffle_files)
                        examples[2]=examples[2].to_tf_dataset(columns='content', shuffle=shuffle_files)
                    else:
                        examples[0]=examples[0].to_tf_dataset(shuffle=shuffle_files)
                        examples[1]=examples[1].to_tf_dataset(shuffle=shuffle_files)
                        examples[2]=examples[2].to_tf_dataset(shuffle=shuffle_files)
                self.train_examples = examples[0]
                self.val_examples = examples[1]
                self.test_examples=examples[2]
        else:
            print("SKIPPED LOADING DATASET")

        print(f"BEGINNING PREPROCESSING EXAMPLES FOR {dataset_name}")
        if dataset_name in ['openwebtext']:
            self.train_examples=self.train_examples.map(lambda X: X["text"])
            if self.val_examples:
                self.val_examples=self.val_examples.map(lambda X: X["text"])
            if self.test_examples:
                self.test_examples=self.test_examples.map(lambda X: X["text"])
            self.tokenize_fun=self.tokenize_src
            self.detokenize_fun=self.detokenize_src
        elif dataset_name in ['falcon-refinedweb']:
            self.tokenize_fun=self.tokenize_src
            self.detokenize_fun=self.detokenize_src
        else:
            print(f"{dataset_name} preprocessing module not found.")
            return None
        print(f"DONE PREPROCESSING EXAMPLES FOR {dataset_name}")           

        #print some info about tokenizer model
        load_tokenizer_model= self.tokenizers_src
        if load_tokenizer_model:
            print("SOURCE AND TARGET TOKENIZERS INFO")
            print(f"Methods for source lang: {self.src_lang}")
            print([item for item in dir(self.tokenizers_src) if not item.startswith('_')])
        else:
            print("PLEASE PROVIDE TOKENIZERS CORRECTLY")

        print("CREATING BATCHED DATASETS FOR TRAINING AND VALIDATION")
        self.train_batches=self.make_dense_padded_batches(self.train_examples, shuffle_set=shuffle_set)
        if self.val_examples:
            self.val_batches=self.make_dense_padded_batches(self.val_examples, shuffle_set=False)
        if self.test_examples:
            self.test_batches=self.make_dense_padded_batches(self.test_examples, shuffle_set=False)

            print("BATCHED DATASETS ARE CREATED")

    def tokenize_src(self, src):
        '''
        Use tokenizer model to tokenize input.
        '''
        src = self.tokenizers_src.tokenize(src)
        # Convert from ragged to dense, padding with zeros.
        src = src.to_tensor()
        return src

    def detokenize_src(self, src):
        '''
        Use tokenizer model to detokenize input.
        '''
        src = self.tokenizers_src.detokenize(src)
        return src

    def load_tokenizer(self):
        '''
        Method to load tokenizer from saved model file.
        Returns tokenizer and tokenizers object.
        '''
        print(f"LOADING TOKENIZER AT {self.model_name}")
        tokenizers = tf.saved_model.load(self.model_name)
        print("THE TOKENIZER ITEMS AVAILABLE ARE:")
        print([item for item in dir(tokenizers) if not item.startswith('_')])
        tokenizers_src=getattr(tokenizers, self.src_lang, None)
        return tokenizers_src, tokenizers

    @tf.function
    def concat_chunk_batches(self, X_batched):
        #concat all tokens in batch by flattening input
        #chunk the flattened tokens into rows with length MAX_LENGTH.
        X_batched=X_batched.merge_dims(0, -1)
        X_batched=tf.convert_to_tensor(X_batched)
        end_seq=int(tf.shape(X_batched)[0])
        row_indices=tf.range(0, end_seq, self.MAX_LENGTH)
        X_batched=tf.RaggedTensor.from_row_starts(X_batched, row_starts=row_indices)
        X_batched=X_batched.to_tensor()
        return X_batched

    def make_dense_padded_batches(self, ds, shuffle_set=True):
        ''' To be used with pretraining train and val dataset'''
        ds_batched = ds.batch(1)
        ds_batched = ds_batched.map(self.tokenize_fun, num_parallel_calls=tf.data.AUTOTUNE)
        ds_batched=ds_batched.unbatch()
        if shuffle_set:
            ds_batched=ds_batched.shuffle(self.BUFFER_SIZE)
        
        #Make ragged batches out of tokens for further processing
        print("Batching by concatenating and chunking samples.")
        ds_batched=ds_batched.ragged_batch(batch_size=int(self.BATCH_SIZE*self.AGG_BATCHES), drop_remainder=False)
        ds_batched = ds_batched.map(self.concat_chunk_batches, num_parallel_calls=tf.data.AUTOTUNE)
        ds_batched = ds_batched.unbatch()
        ds_batched=ds_batched.padded_batch(self.BATCH_SIZE, padded_shapes=None, drop_remainder=True)

        if self.with_prefetch:
            ds_batched = ds_batched.prefetch(tf.data.AUTOTUNE)

        return ds_batched
