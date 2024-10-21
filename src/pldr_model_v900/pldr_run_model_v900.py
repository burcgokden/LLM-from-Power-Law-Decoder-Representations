'''
LLM from Power Law Decoder Representations v900 (PLDR-LLM v900) Single GPU Train and Evaluation Module
'''

import os
import time

import numpy as np
import tensorflow as tf

import pldr_model_v900 as pldr_model
import common as cm


class pldr_model_e2e:
    '''
    Trains and evaluates Large Language Model from Power Law Graph Decoder Representations
    '''

    def __init__(self, tokenizer_obj_src, hpdict=None,
                 checkpoint_path = "./pldr_checkpoints/", load_ckpt=None):

        if hpdict:
            self.hpdict = hpdict
        else:
            print("USING DEFAULT HYPERPARAMETERS")
            self.hpdict={"num_layers": 4,
                         "d_model": int(15*64),
                         "num_heads": 15,
                         "dropout_rate": 0.0,
                         "dff": int(np.floor(15*64*4*2/3)),
                         "att_dropout_rate_in": 0.0,
                         "att_dropout_rate_eij": 0.0,
                         "Adropout_rate":0.0,
                         "num_reslayerA":8,
                         "num_denseA":2,
                         "A_dff":300,
                         "A_dff_depth":112,
                         "input_vocab_size": tokenizer_obj_src.get_vocab_size(),
                         "epochs":1,
                         "save_model_path": "default_pldr_model",
                         "warmup_steps": 2000, 
                         "lr_total_steps": 250000,
                         "learning_rate": 1e-3,
                         "lr_alpha":0.1,
                         "adamw_decay":0.1,
                         "activation":tf.nn.silu,
            }
        print(f"hyperparameters are {self.hpdict}")

        self.tokenizers_src=tokenizer_obj_src


        self.loss_function=masked_loss_function()

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
        self.val_loss = tf.keras.metrics.Mean(name='val_loss')
        self.val_accuracy = tf.keras.metrics.Mean(name='val_accuracy')

        self.pldr_model = pldr_model.PLDR_Model(
                            num_layers=self.hpdict["num_layers"],
                            d_model=self.hpdict["d_model"],
                            num_heads=self.hpdict["num_heads"],
                            dff=self.hpdict["dff"],
                            input_vocab_size=self.hpdict["input_vocab_size"],
                            rate=self.hpdict["dropout_rate"],
                            att_dropout_rate_in=self.hpdict["att_dropout_rate_in"],
                            att_dropout_rate_eij=self.hpdict["att_dropout_rate_eij"],
                            Adropout_rate=self.hpdict["Adropout_rate"],
                            A_dff=self.hpdict["A_dff"],
                            A_dff_depth=self.hpdict["A_dff_depth"],
                            num_reslayerA=self.hpdict["num_reslayerA"],
                            num_denseA=self.hpdict["num_denseA"],
                            activation=hpdict["activation"]
                        )
        
        self.checkpoint_path = checkpoint_path
        self.train_ckpt_path=os.path.join(self.checkpoint_path, "train", self.hpdict["save_model_path"])

        if not os.path.isdir(self.train_ckpt_path):
            print(f"Creating train ckpt dir: {self.train_ckpt_path}")
            os.makedirs(self.train_ckpt_path)
            cm.pklsave(os.path.join(self.train_ckpt_path, self.hpdict["save_model_path"] + "_hparams.pkl"), self.hpdict)

        train_ckpt = tf.train.Checkpoint(pldr_model=self.pldr_model)

        self.train_ckpt_manager = tf.train.CheckpointManager(train_ckpt,
                                                             directory=self.train_ckpt_path,
                                                             checkpoint_name="train_"+self.hpdict["save_model_path"],
                                                             max_to_keep=10)

        # if a checkpoint exists, restore the latest checkpoint.
        if load_ckpt=="train":
            if self.train_ckpt_manager.latest_checkpoint:
                train_ckpt.restore(self.train_ckpt_manager.latest_checkpoint)
                print('Latest train checkpoint restored!!')
        elif load_ckpt is None:
            print('Checkpoint restoration is skipped')
        else:
            print("Attempting to restore the checkpoint path specified...")
            train_ckpt.restore(load_ckpt)
            print("Checkpoint restored.")
    
        print("Using cosine learning schedule with warm start.")
        self.learning_rate=WarmUpCosineDecaySchedule(learning_rate=self.hpdict["learning_rate"],
                                                        total_steps=self.hpdict["lr_total_steps"],
                                                        warmup_steps=self.hpdict["warmup_steps"],
                                                        alpha=self.hpdict["lr_alpha"]
                                                        )

        print("Using AdamW optimizer.")
        self.optimizer=tf.keras.optimizers.AdamW(learning_rate=self.learning_rate, weight_decay=hpdict["adamw_decay"],
                                                    beta_1=0.9, beta_2=0.95, clipvalue=1.0, epsilon=1e-5)


    def create_masks(self, inp):
        '''
        inp: tensor of shape [batch_size, seq_len]
        Create masks for decoder layer for pldr model.
        Used in the attention block in the decoder.
        It is used to pad and mask future tokens in the input received by the decoder.
        '''
        look_ahead_mask = self.create_look_ahead_mask(tf.shape(inp)[1])
        dec_target_padding_mask = self.create_padding_mask(inp)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        return combined_mask

    @staticmethod
    def create_padding_mask(seq):
        '''
        inp: tensor of shape [batch_size, seq_len]
        Create a mask for padding in the input for decoder.
        '''
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

        return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

    @staticmethod
    def create_look_ahead_mask(size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask
    

    train_step_signature = [
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    ]

    @tf.function(input_signature=train_step_signature)
    def train_step(self, inp):
        '''
        Train step for single token for PLDR-LLM with pretrain data input.
        inp: tensor of shape [batch_size, seq_len]
        '''

        #input is shifted by one to compare with predictions
        tar_inp = inp[:, :-1]
        tar_real = inp[:, 1:]

        combined_mask= self.create_masks(tar_inp)

        with tf.GradientTape() as tape:
            predictions, _, _ = self.pldr_model([tar_inp, combined_mask], 
                                                training=self.pt_training, trainable=self.pt_trainable)
            loss = self.loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, self.pldr_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.pldr_model.trainable_variables))

        self.train_loss.update_state(loss)
        self.train_accuracy.update_state(self.accuracy_function(tar_real, predictions))
        self.train_loss_epoch.update_state(loss)
        self.train_accuracy_epoch.update_state(self.accuracy_function(tar_real, predictions))
        self.train_loss_one_epoch.update_state(loss)
        self.train_accuracy_one_epoch.update_state(self.accuracy_function(tar_real, predictions))


    def train_model(self, pt_training, pt_trainable, 
                       train_batches, val_batches=None, 
                       chkpt_batches=None, chkpt_epochs=None,
                       verbose_freq=2000, val_verbose_freq=None):
        '''
        Method for training model for single and multiple epochs with batch indexed checkpointing.
        pt_training: enable drop out during training.
        pt_trainable: set model weights as trainable.
        train_batches: batched dataset for pretraining.
        val_batches: batched dataset for validation.
        chkpt_batches: A list of training steps at which checkpoints will be saved within a single epoch.
        chkpt_epochs: A list of training epochs at the end of which checkpoints will be saved.
        verbose_freq: how often train loss and metrics is saved and printed within e single epoch.
        val_verbose_freq: how often validation loss and metrics is saved and printed within e single epoch.
        '''

        self.pt_training=pt_training
        self.pt_trainable=pt_trainable
        self.train_loss_epoch = tf.keras.metrics.Mean(name='train_loss_epoch')
        self.train_accuracy_epoch = tf.keras.metrics.Mean(name='train_accuracy_epoch')
        self.train_loss_one_epoch = tf.keras.metrics.Mean(name='train_loss_one_epoch')
        self.train_accuracy_one_epoch = tf.keras.metrics.Mean(name='train_accuracy_one_epoch')

        EPOCHS=self.hpdict["epochs"]
        print(f"Train checkpoints are at epochs: {chkpt_epochs}")
        print(f"Batch Train checkpoints are at batches: {chkpt_batches}")

        #initialize lists to collect loss and accuracy data per epoch
        train_loss_lst, train_acc_lst, val_loss_lst, val_acc_lst=[],[],[],[]
        trn_loss_epoch_lst, trn_acc_epoch_lst=[], []
        trn_loss_one_epoch_lst, trn_acc_one_epoch_lst=[], []

        steps_count=0
        self.train_loss_epoch.reset_states()
        self.train_accuracy_epoch.reset_states()
        for epoch in range(EPOCHS):
            start = time.time()

            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            self.train_loss_one_epoch.reset_states()
            self.train_accuracy_one_epoch.reset_states()

            for (batch, inp) in enumerate(train_batches):

                steps_count+=1
                self.train_step(inp)

                if (batch+1) % verbose_freq == 0 or batch==0:
                    time_so_far=time.time()-start
                    loss_batch=self.train_loss.result().numpy()
                    acc_batch=self.train_accuracy.result().numpy()
                    loss_epoch=self.train_loss_epoch.result().numpy()
                    acc_epoch=self.train_accuracy_epoch.result().numpy()
                    loss_one_epoch=self.train_loss_one_epoch.result().numpy()
                    acc_one_epoch=self.train_accuracy_one_epoch.result().numpy()
                    print(f"{time_so_far:.2f}s Epoch {epoch + 1} Batch {batch+1} Loss(G) {loss_epoch:.4f} Acc(G) {acc_epoch:.4f} "+
                          f"Loss(R) {loss_batch:.4f} Acc(R) {acc_batch:.4f} Loss(E) {loss_one_epoch:.4f} Acc(E) {acc_one_epoch:.4f} "+
                          f"LR {self.optimizer.learning_rate.numpy():.4e}")
                        
                    train_loss_lst.append(loss_batch)
                    train_acc_lst.append(acc_batch)
                    trn_loss_epoch_lst.append(loss_epoch)
                    trn_acc_epoch_lst.append(acc_epoch)
                    self.train_loss.reset_states()
                    self.train_accuracy.reset_states()
                
                if val_verbose_freq is not None:
                    if (batch+1) % val_verbose_freq == 0:
                        time_so_far=time.time()-start
                        self.validate_model(val_batches, epoch)
                        val_lossv=self.val_loss.result().numpy()
                        val_accv=self.val_accuracy.result().numpy()
                        val_loss_lst.append(val_lossv)
                        val_acc_lst.append(val_accv)
                        print(f"{time_so_far:.2f}s Epoch {epoch + 1} Batch {batch+1} Val Loss {val_lossv:.4f} Val Accuracy {val_accv:.4f}")
                    
                if chkpt_batches is not None:
                    if (batch+1) in chkpt_batches:
                        ckpt_save_path = self.train_ckpt_manager.save()
                        print(f'Saving train checkpoint for batch {batch+1} in epoch {epoch+1} at {ckpt_save_path}')
            
            if chkpt_epochs is not None:
                if (epoch + 1) in chkpt_epochs:
                    ckpt_save_path = self.train_ckpt_manager.save()
                    print(f'Saving train checkpoint for epoch {epoch + 1} at {ckpt_save_path}')

            print(f"Epoch {epoch + 1} Loss(G) {self.train_loss_epoch.result().numpy():.4f} Accuracy(G) {self.train_accuracy_epoch.result().numpy():.4f} "+
                  f"Loss(E) {self.train_loss_one_epoch.result().numpy():.4f} Accuracy(E) {self.train_accuracy_one_epoch.result().numpy():.4f}")
            if (batch+1)%verbose_freq != 0:
                print(f"End of epoch batch count is {batch+1}. Appending end of epoch loss/accuracy")
                trn_loss_epoch_lst.append(self.train_loss_epoch.result().numpy())
                trn_acc_epoch_lst.append(self.train_accuracy_epoch.result().numpy())
                trn_loss_one_epoch_lst.append(self.train_loss_one_epoch.result().numpy())
                trn_acc_one_epoch_lst.append(self.train_accuracy_one_epoch.result().numpy())

            if val_batches is not None:
                self.validate_model(val_batches, epoch)
                val_lossv=self.val_loss.result().numpy()
                val_accv=self.val_accuracy.result().numpy()
                val_loss_lst.append(val_lossv)
                val_acc_lst.append(val_accv)

            print(f"Total number of steps elapsed: {steps_count}")
            print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')

        #save loss and accuracy data for train and validation runs
        cm.pklsave(self.train_ckpt_path+'/train_loss.pkl', train_loss_lst)
        cm.pklsave(self.train_ckpt_path+'/val_loss.pkl', val_loss_lst)
        cm.pklsave(self.train_ckpt_path+'/train_accuracies.pkl', train_acc_lst)
        cm.pklsave(self.train_ckpt_path+'/val_accuracies.pkl', val_acc_lst)
        cm.pklsave(self.train_ckpt_path+'/train_loss_epoch.pkl', trn_loss_epoch_lst)
        cm.pklsave(self.train_ckpt_path+'/train_loss_one_epoch.pkl', trn_loss_one_epoch_lst)
        cm.pklsave(self.train_ckpt_path+'/train_accuracies_epoch.pkl', trn_acc_epoch_lst)
        cm.pklsave(self.train_ckpt_path+'/train_accuracies_one_epoch.pkl', trn_acc_one_epoch_lst)

        final_ckpt_save_path = self.train_ckpt_manager.save()
        print(f'Saving final train checkpoint for epoch {EPOCHS} at {final_ckpt_save_path}')

        return train_loss_lst, train_acc_lst, val_loss_lst, val_acc_lst

    validate_step_signature = [
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    ]

    @tf.function(input_signature=validate_step_signature)
    def validate_step(self, inp):
        
        tar_inp = inp[:, :-1]
        tar_real = inp[:, 1:]

        combined_mask = self.create_masks(tar_inp)

        predictions, _, _ = self.pldr_model([tar_inp, combined_mask], 
                                            training=False, trainable=False)
        loss = self.loss_function(tar_real, predictions)

        self.val_loss(loss)
        self.val_accuracy(self.accuracy_function(tar_real, predictions))
        
        return loss


    def validate_model(self, val_batches, epoch):
        '''
        This runs the model on val dataset and returns loss and accuracy during training.
        Args:
            val_batches: the validation batches same size as train batches
            epoch: current epoch
        Returns:
            loss: the loss averaged over all batches
            accuracy: the accuracy averaged over all batches
        Validate method for pretrain model
        '''

        self.val_loss.reset_states()
        self.val_accuracy.reset_states()

        for (batch, inp) in enumerate(val_batches):
            self.validate_step(inp)

        print(f'Epoch {epoch + 1} Val Loss {self.val_loss.result():.4f} Val Accuracy {self.val_accuracy.result():.4f}')

        return self.val_loss.result(), self.val_accuracy.result()


    # top-k and top-p implementations are from https://github.com/openai/gpt-2/blob/master/src/sample.py

    @staticmethod
    def top_k_logits(logits, k):
        '''Top-k sampling'''
        if k == 0:
            # no truncation
            return logits

        def _top_k():
            values, _ = tf.nn.top_k(logits, k=k)
            min_values = values[:, -1, tf.newaxis]
            return tf.where(
                            logits < min_values,
                            tf.ones_like(logits, dtype=logits.dtype) * -1e10,
                            logits,
                           )
        return tf.cond(
                        tf.equal(k, 0),
                        lambda: logits,
                        lambda: _top_k(),
                       )


    @staticmethod
    def top_p_logits(logits, p):
        """Nucleus sampling"""
        batch, _ = logits.shape.as_list()
        sorted_logits = tf.sort(logits, direction='DESCENDING', axis=-1)
        cumulative_probs = tf.cumsum(tf.nn.softmax(sorted_logits, axis=-1), axis=-1)
        indices = tf.stack([
                            tf.range(0, batch),
                            # number of indices to include
                            tf.maximum(tf.reduce_sum(tf.cast(cumulative_probs <= p, tf.int32), axis=-1) - 1, 0),
                            ], axis=-1)
        min_values = tf.gather_nd(sorted_logits, indices)
        return tf.where(
            logits < min_values,
            tf.ones_like(logits) * -1e10,
            logits,
        )
    
    def generate_text(self, sentence, temperature=1.0, top_k=0, top_p=1.0, max_length=50, save_att=None):
        '''
        Args:
            sentence: source sentence as input string.
            temperature: parameter to determine how deterministic the output is between (0,1]. 
                         Less deterministic on logits if temperature==1.
            top_k: value to select from top k largest logits, select all if k==0.
            top_p: cumulative probability threshold to select from logits for nucleus sampling. Select all if p == 1.
            max_length: maximum number of iterations to run.
            save_att: path location to save attention weights.
        Returns:
            Predicted text, tokens and attention weights.
        '''
        assert 0.0 < temperature <=1.0, "set temperature between (0, 1]."
        assert 0.0 < top_p <=1.0, "set nucleus sampling probability between (0, 1], p=1 to skip top_p sampling."
        assert top_k >= 0, "set top_k above 0 or 0 to skip top_k sampling."
        temperature = tf.cast(temperature, dtype=tf.float32)

        sentence = tf.convert_to_tensor([sentence])
        sentence = self.tokenizers_src.tokenize(sentence).to_tensor()

        pldr_input = sentence
        end = self.tokenizers_src.tokenize([''])[0]
        output = tf.convert_to_tensor([pldr_input[0][:-1]])
        output = tf.expand_dims(output, 0)
        att_weights=None

        for i in range(max_length):
            combined_mask = self.create_masks(output[0])

            predictions, _, att_weights = self.pldr_model([output[0], combined_mask], training=False)

            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)
            predictions=tf.squeeze(predictions, axis=[1]) #shrink to (batch_size, vocab_size)
            
            #temperature, top_k and nucleus sampling are stackable if needed.
            #scale logits for temperature sampling
            if temperature < 1:
                predictions = predictions/temperature

            #top_p sampling
            if top_p < 1:
                predictions=self.top_p_logits(logits=predictions, p=top_p)

            #top_k sampling
            if top_k > 0:
                predictions=self.top_k_logits(logits=predictions, k=top_k)
            

            predicted_id=tf.random.categorical(logits=predictions, num_samples=1) #(batch_size, 1)
            predicted_id=tf.expand_dims(predicted_id, axis=1) #expand to (batch_size, 1, 1)

            output = tf.concat([output, predicted_id], axis=-1)

            if predicted_id == end:
                break

        text = self.tokenizers_src.detokenize(output[0])[0]

        tokens = self.tokenizers_src.lookup(output[0])[0]

        if save_att is not None:
            print("saving attention weights")
            cm.pklsave(save_att, att_weights)

        return text, tokens, att_weights, max_length


    @staticmethod
    def print_generated_text(sentence, full_output, max_eval_length):
        '''
        sentence: Input to the PLDR-LLM as string
        full_output: sentence+completion as output from PLDR-LLM as tensor string
        max_eval_length: Number of maximum tokens for generation
        '''
        print(f"Max Eval Length: {max_eval_length}")
        print(f'{"Input":15s}: {sentence}')
        print(f'{"Prediction":15s}: {full_output.numpy().decode("utf-8")}')

    @staticmethod
    def accuracy_function(real, pred):
        accuracies = tf.equal(real, tf.argmax(pred, axis=2))

        mask = tf.math.logical_not(tf.math.equal(real, 0))
        accuracies = tf.math.logical_and(mask, accuracies)

        accuracies = tf.cast(accuracies, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)

    
class WarmUpCosineDecaySchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, learning_rate, total_steps, warmup_steps, alpha=0, **kwargs):
        """
        learning_rate: maximum learning rate.
        total_steps; Total number of steps to run.
        warmup_steps:number of steps for linear warm up to maximum learning rate.
        alpha: multiplier for minimum learning rate after decay
        """
        super().__init__(**kwargs)

        assert warmup_steps <= total_steps, "warmup_step is exceeding total steps"
        assert alpha <= 1.0, "select alpha value between 0 and 1.0"

        self.learning_rate=tf.cast(learning_rate, dtype=tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, dtype=tf.float32)
        self.total_steps = tf.cast(total_steps, dtype=tf.float32)
        self.alpha = tf.cast(alpha, dtype=tf.float32)
   
    def __call__(self, step, **kwargs):

        step = tf.cast(step, dtype=tf.float32)
        pi=tf.constant(np.pi)
        
        #keep learning rate at minimum if total step mismatch actual total steps for an epoch.
        step=tf.minimum(step, self.total_steps)

        warmup_rise = (self.learning_rate/self.warmup_steps)*step

        decay_step=step-self.warmup_steps
        decay_rate = self.total_steps-self.warmup_steps
        cosine_decay = self.learning_rate * ((1-self.alpha)*0.5* (1+tf.math.cos(pi*(decay_step/decay_rate)))+self.alpha)

        return tf.where(step <= self.warmup_steps, warmup_rise, cosine_decay)

    def get_config(self):
        config=super().get_config()
        config.update({
            "learning_rate": self.learning_rate,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "alpha": self.alpha
        })
        return config

class masked_loss_function(tf.keras.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        mask = tf.math.logical_not(tf.math.equal(y_true,0))
        loss_ = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')(y_true, y_pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_*=mask
        loss=tf.reduce_sum(loss_)/tf.reduce_sum(mask)
        return loss

    def get_config(self):
        config = super().get_config()
        return config
