'''
Script for pretraining a Large Language Model from Power Law Decoder Representations v500 (PLDR-LLM v500) with
DAG Regularizer on multiple ranks with Refined Web dataset.
This script used the examples found in https://github.com/horovod/horovod/tree/master/examples as a starting point.
'''

import os
import sys
import time
import numpy as np
import tensorflow as tf
import random

import horovod
import horovod.tensorflow as hvd

import pldr_model_v500 as hvdpldrm
import hvd_data_prep as hvdp
import common as cm

from datetime import datetime
from pytz import timezone

def main():

    #HOROVOD: Initialize and assign GPU for each rank
    print("initializing horovod")
    hvd.init()

    print("Setting random seeds")
    random.seed(1234)
    np.random.seed(1234)
    tf.random.set_seed(1234)

    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    print("assigning gpus to rank")
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
    print("Done assigning gpus")

    ########## DEFINE THE PATH LOCATIONS FOR DATASET, TOKENIZER, AND CHECKPOINT #############
    save_model_path = "pldr_llm_v500_sample_model"
    checkpoint_path = os.path.abspath("./chkpt-saved-models")   
    tokenizer_model = os.path.abspath("./tokenizer-models/refinedweb-tokenizer-pldr-llm-paper/refinedweb-sentencepiece-unigram-cased-en-32k-tokenizer-pldr-llm-paper")
    cache_dir = None

    ############## DEFINE THE DATA INTERVAL, BATCH_SIZE and MAX_LENGTH FOR DATA LOADING ######################################

    VAL_BATCH_COUNT_ALL_RANKS=4000
    BATCH_COUNT_ALL_RANKS= int(80000000/16)-VAL_BATCH_COUNT_ALL_RANKS 
    learning_rate=1e-3 
    warmup_steps=2000
    BATCH_COUNT_PER_RANK=250000
    BATCH_SIZE=16
    MAX_LENGTH=1024

    ####### SET THE MODEL ARGUMENTS TO INITIALIZE THE MODEL ############
    epochs=1
    tot_steps=BATCH_COUNT_PER_RANK*epochs 
    alpha=0.1
    num_layers=5
    num_heads=14
    dffv=int(np.floor(num_heads*64*4*2/3))
    A_dffv=int(np.floor(256*2/3))
    activationv=tf.nn.silu
    lambda_dagA=0.001
    lambda_dagApG=0.005

    ######### SET THE TRAINING OBSERVATION AND CHECKPOINT INTERVALS FOR TRAIN PREPARATION #########################################
    EPOCHS=epochs
    verbose_freq= 2000
    val_verbose_freq=int(6*verbose_freq)
    chkpt_batches=[warmup_steps, int(warmup_steps+10000), 31250, 62500, 93750,
                   125000, 175000]
    chkpt_epochs=None

    enable_val=True #False

    ###############################################################################################################################

    if hvd.rank()==0:
        print(f"Checkpoint and batch related parameters:")
        print(f"Warm-up LR steps: {warmup_steps}")
        print(f"Total LR steps: {tot_steps}")
        print(f"Batch Count Per Rank: {BATCH_COUNT_PER_RANK}")
        print(f"learning rate: {learning_rate}")
        print(f"alpha for learning rate: {alpha}")
        print(f"lambda_dagA/lambda_dagApG for learning rate: {lambda_dagA}/{lambda_dagApG}")
        print(f"Number of epochs: {EPOCHS}")
        print(f"Batch Size per Rank; {BATCH_SIZE}")
        print(f"Max Context Length: {MAX_LENGTH}")
        print(f"Checkpoints at Batch Counts: {chkpt_batches}")
        print(f"Checkpoints at Epochs: {chkpt_epochs}")
        print(f"Reporting at every {verbose_freq} batches.")
        if enable_val:
            print(f"Reporting val loss at every {val_verbose_freq} batches.")

    ########################### DATASET PROCESSING ################################################
    pst_time=timezone('US/Pacific')
    if hvd.rank()==0:
        print(f"Start time: {datetime.now(pst_time)}")
        start=time.time()

    ############################ TRAIN DATASET PREPARATION ########################################
    print(f"Getting TRAIN Inputs for {hvd.rank()}")
    shuffle_flag=True if epochs > 1 else False
    print(f"Shuffle status for train dataset is {shuffle_flag}")

    dataset_per_rank, vocab_size, tokenizer_src = hvdp.load_falcon_refinedweb(tok_model_name=tokenizer_model,
                                                    split_interval=[0, int(BATCH_COUNT_ALL_RANKS*BATCH_SIZE)],
                                                    rank_cnt=hvd.size(), rank_index=hvd.rank(),
                                                    BATCH_SIZE=BATCH_SIZE, MAX_LENGTH=MAX_LENGTH, AGG_BATCHES=100,
                                                    shuffle_files=shuffle_flag, shuffle_set=shuffle_flag, cache_dir=cache_dir)


    #### Adjust tf dataset autotune buffer options
    options = tf.data.Options()
    options.autotune.ram_budget = int(1e10)
    dataset_per_rank = dataset_per_rank.with_options(options)
    dataset_per_rank=dataset_per_rank.take(BATCH_COUNT_PER_RANK)
    dataset_per_rank=dataset_per_rank.prefetch(tf.data.AUTOTUNE)
    ###################################### END OF TRAIN DATASET PREPARATION ##################################################

    ###################################### VAL DATASET PREPARATION ########################################
    if enable_val:
        print(f"Getting VALIDATION Inputs for {hvd.rank()}")
        val_split_interval=[int(BATCH_COUNT_ALL_RANKS*BATCH_SIZE), int((BATCH_COUNT_ALL_RANKS+VAL_BATCH_COUNT_ALL_RANKS)*BATCH_SIZE)]

        val_dataset_per_rank, _, _ = hvdp.load_falcon_refinedweb(tok_model_name=tokenizer_model,
                                                          split_interval=val_split_interval,
                                                          rank_cnt=hvd.size(), rank_index=hvd.rank(),
                                                          BATCH_SIZE=BATCH_SIZE, MAX_LENGTH=MAX_LENGTH, AGG_BATCHES=100,
                                                          shuffle_files=False, shuffle_set=False, cache_dir=cache_dir)



        options = tf.data.Options()
        options.autotune.ram_budget = int(1e10)
        val_dataset_per_rank = val_dataset_per_rank.with_options(options)
        val_dataset_per_rank=val_dataset_per_rank.prefetch(tf.data.AUTOTUNE)
    else:
        print("Skipped VALIDATION Input generation.")
    ###################################### END OF VAL DATASET PREPARATION ##################################################
    
    if hvd.rank()==0:
        print(f"Dataset processing time: {(time.time()-start):.2f}s for Rank {hvd.rank()}")
        print(f"BATCH COUNT PER RANK is: {BATCH_COUNT_PER_RANK}")
    print(f"Rank {hvd.rank()} finished data loading")

    ############################ END DATASET PROCESSING #########################################

    ######################### MODEL INITIALIZATION ##############################################
    hpdict = {
            "num_layers": num_layers,
            "d_model": num_heads*64, 
            "num_heads": num_heads, 
            "dropout_rate":0.0,
            "dff": dffv,
            "att_dropout_rate_in":0.0,
            "att_dropout_rate_eij":0.0,                                                     
            "Adropout_rate":0.0,
            "A_dff":A_dffv,
            "num_reslayerA":8,
            "num_denseA":2,
            "input_vocab_size": vocab_size,
            "epochs":epochs, 
            "save_model_path": save_model_path,
            "warmup_steps": warmup_steps,
            "lr_total_steps": tot_steps,
            "learning_rate": learning_rate,
            "lr_alpha": alpha,
            "adamw_decay": 0.1,
            "activation": activationv,
            }
    
    pldr_model = hvdpldrm.PLDR_Model( 
                    num_layers=hpdict["num_layers"],
                    d_model=hpdict["d_model"],
                    num_heads=hpdict["num_heads"],
                    dff=hpdict["dff"],
                    input_vocab_size=hpdict["input_vocab_size"],
                    rate=hpdict["dropout_rate"],
                    att_dropout_rate_in=hpdict["att_dropout_rate_in"],
                    att_dropout_rate_eij=hpdict["att_dropout_rate_eij"],
                    Adropout_rate=hpdict["Adropout_rate"],
                    A_dff=hpdict["A_dff"],
                    num_reslayerA=hpdict["num_reslayerA"],
                    num_denseA=hpdict["num_denseA"],
                    activation=hpdict["activation"]
                )
    print(f"Rank {hvd.rank()} initialized model")


    def loss_function(y_true, y_pred):
        mask = tf.math.logical_not(tf.math.equal(y_true,0))
        loss_ = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')(y_true, y_pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_*=mask
        loss=tf.reduce_sum(loss_)/tf.reduce_sum(mask)
        return loss
    
    def dag_regularizer(att_weights):
        '''
        input is attention weights which are metric tensor, attention, energy-curvature tensor and associated
        learned values. shape is [[layer_num, batch_size, num_head, seq_len or dk, seq_len or dk]]
        input is a list of list of tensors.
        att_weights[0][1] is metric tensor.
        att_weights[0][2] is learned power coeffecients for the metric tensor
        att_weights[0][5] is energy_curvature tensor
        att_weights[0][6] is attention per head

        returns dag regularizer values for metric tensor, potential tensor and energy-curvature tensor 
        '''
        #convert relevant values to tensor
        adjm=tf.convert_to_tensor([t[1] for t in att_weights[0]]) #shape: [layer_num, batch_size, num_heads, dk, dk]
        adjp=tf.convert_to_tensor([tf.math.pow(t[1], t[2]) for t in att_weights[0]])
        adjg=tf.convert_to_tensor([t[5] for t in att_weights[0]])

        #the sum of diagonals need to be equal to dk=dmodel//num_heads
        dval=tf.cast(tf.shape(adjm)[-1], dtype=tf.float32)

        #average only on batch_size version
        adjm_dloss=tf.math.abs(tf.math.log(tf.linalg.trace(tf.linalg.expm(tf.multiply(adjm, adjm)))/dval)) #shape: [layer_num, batch_size, num_head]
        adjp_dloss=tf.math.abs(tf.math.log(tf.linalg.trace(tf.linalg.expm(tf.multiply(adjp, adjp)))/dval)) 
        adjg_dloss=tf.math.abs(tf.math.log(tf.linalg.trace(tf.linalg.expm(tf.multiply(adjg, adjg)))/dval)) 

        adjm_dloss=tf.reduce_mean(adjm_dloss) #mean over batch_size*layer*num_head
        adjp_dloss=tf.reduce_mean(adjp_dloss)
        adjg_dloss=tf.reduce_mean(adjg_dloss)

        return adjm_dloss, adjp_dloss, adjg_dloss    


    #Cosine Decay with linear warm up learning schedule
    opt_learning_rate=WarmUpCosineDecaySchedule(learning_rate=hpdict["learning_rate"],
                                                         total_steps=hpdict["lr_total_steps"],
                                                         warmup_steps=hpdict["warmup_steps"],
                                                         alpha=hpdict["lr_alpha"]
                                                         )


    optimizer=tf.keras.optimizers.AdamW(learning_rate=opt_learning_rate, weight_decay=hpdict["adamw_decay"],
                                                      beta_1=0.9, beta_2=0.95, clipvalue=1.0, epsilon=1e-5)
    
    ############################ END MODEL INITIALIZATION #########################################################
    
    ############################# TRAIN PREPARATION ###############################################################

    train_ckpt_path=os.path.join(checkpoint_path, "train", hpdict["save_model_path"])

    if not os.path.isdir(train_ckpt_path):
        print(f"Creating train ckpt dir: {train_ckpt_path}")
        os.makedirs(train_ckpt_path)
    if hvd.rank()==0:
        print(f"PLDR model hyperparameters: {hpdict}")
        print(f"PLDR model configuration: {pldr_model.get_config()}")
        cm.pklsave(os.path.join(train_ckpt_path, hpdict["save_model_path"] + "_hparams.pkl"), hpdict)
    
    train_ckpt = tf.train.Checkpoint(pldr_model=pldr_model)
    train_ckpt_manager = tf.train.CheckpointManager(train_ckpt,
                                                            directory=train_ckpt_path,
                                                            checkpoint_name="train_"+hpdict["save_model_path"],
                                                            max_to_keep=10)

    pt_training=True
    pt_trainable=True
    train_loss1 = tf.keras.metrics.Mean(name='global_train_loss')
    train_accuracy1= tf.keras.metrics.Mean(name='global_train_accuracy')
    train_loss2 = tf.keras.metrics.Mean(name='running_train_loss')
    train_accuracy2= tf.keras.metrics.Mean(name='running_train_accuracy')
    train_loss_a=tf.keras.metrics.Mean(name='train_loss_a')
    train_loss_ap=tf.keras.metrics.Mean(name='train_loss_ap')
    train_loss_g=tf.keras.metrics.Mean(name='train_loss_g')
    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_accuracy= tf.keras.metrics.Mean(name='val_accuracy')
    val_loss_a=tf.keras.metrics.Mean(name='val_loss_a')
    val_loss_ap=tf.keras.metrics.Mean(name='val_loss_ap')
    val_loss_g=tf.keras.metrics.Mean(name='val_loss_g')

    train_step_signature = [
        (tf.TensorSpec(shape=(None, None), dtype=tf.int64),tf.TensorSpec(shape=(None, None), dtype=tf.int64), 
         tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32)) ,
        tf.TensorSpec(shape=(), dtype=tf.bool),
    ]

    @tf.function(input_signature=train_step_signature)
    def train_step(inp, first_batch):
        '''
        Train step for single token for pldr model with pretrain data input.
        Input has one component: text
        '''

        tar_inp, tar_real, combined_mask=inp

        with tf.GradientTape() as tape:
            predictions, _, att_weights = pldr_model([tar_inp, combined_mask], 
                                                training=pt_training, trainable=pt_trainable)
            loss_a, loss_ap, loss_g=dag_regularizer(att_weights=att_weights)
            loss = loss_function(tar_real, predictions) + lambda_dagA*loss_a+lambda_dagApG*(loss_ap+loss_g)

        tape=hvd.DistributedGradientTape(tape, sparse_as_dense=False)
        gradients = tape.gradient(loss, pldr_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, pldr_model.trainable_variables))

        if first_batch:
            hvd.broadcast_variables(pldr_model.variables, root_rank=0)
            hvd.broadcast_variables(optimizer.variables(), root_rank=0)

        if hvd.rank()==0:
            train_loss1.update_state(loss)
            train_accuracy1.update_state(accuracy_function(tar_real, predictions))
            train_loss2.update_state(loss)
            train_accuracy2.update_state(accuracy_function(tar_real, predictions))
            train_loss_a.update_state(loss_a)
            train_loss_ap.update_state(loss_ap)
            train_loss_g.update_state(loss_g)

        return loss
    
    validate_step_signature = [
        (tf.TensorSpec(shape=(None, None), dtype=tf.int64),tf.TensorSpec(shape=(None, None), dtype=tf.int64), 
         tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32))
    ]

    @tf.function(input_signature=validate_step_signature)
    def validate_step(inp):
        
        tar_inp, tar_real, combined_mask=inp

        predictions, _, att_weights = pldr_model([tar_inp, combined_mask], 
                                            training=False, trainable=False)
        loss_a, loss_ap, loss_g=dag_regularizer(att_weights=att_weights)
        loss = loss_function(tar_real, predictions) + lambda_dagA*loss_a+lambda_dagApG*(loss_ap+loss_g)

        if hvd.rank()==0:
            val_loss.update_state(loss)
            val_accuracy.update_state(accuracy_function(tar_real, predictions))
            val_loss_a.update_state(loss_a)
            val_loss_ap.update_state(loss_ap)
            val_loss_g.update_state(loss_g)
        
        return loss
    
    def validate_model(val_batches):

        for (batch, inp) in enumerate(val_batches):
            validate_step(inp)

    print(f"Rank {hvd.rank()} starting training")

    train_loss1_lst=[]
    train_acc1_lst=[]
    train_loss2_lst=[]
    train_acc2_lst=[]
    train_lossa_lst=[]
    train_lossap_lst=[]
    train_lossg_lst=[]
    val_loss_lst=[]
    val_acc_lst=[]
    val_lossa_lst=[]
    val_lossap_lst=[]
    val_lossg_lst=[]

    ############################ END TRAIN PREPARATION ########################################################

    ############################# START TRAINING ##############################################################
    for epoch in range(EPOCHS):
        if hvd.rank()==0:
            start = time.time()
            train_loss1.reset_states()
            train_accuracy1.reset_states()
            train_loss2.reset_states()
            train_accuracy2.reset_states()
            train_loss_a.reset_states()
            train_loss_ap.reset_states()
            train_loss_g.reset_states()

        for (batch, inp) in enumerate(dataset_per_rank):

            train_step(inp, batch==0)

            # HOROVOD: get metrics from rank 0 process only.
            if ((batch+1) % verbose_freq == 0 or batch==0) and hvd.rank()==0:
                time_so_far=time.time()-start
                loss1=train_loss1.result().numpy()
                acc1=train_accuracy1.result().numpy()
                loss2=train_loss2.result().numpy()
                acc2=train_accuracy2.result().numpy()
                lossa=train_loss_a.result().numpy()
                lossap=train_loss_ap.result().numpy()
                lossg=train_loss_g.result().numpy()
                print(f'{time_so_far:.2f}s Epoch {epoch + 1} Batch {batch+1} Loss(E) {loss1:.4f} Accuracy(E) {acc1:.4f} '+ 
                      f'Loss(R) {loss2:.4f} Accuracy(R) {acc2:.4f} LR {optimizer.learning_rate.numpy():.4e} ' +
                      f'Loss A(R) {lossa:.4e} Loss Ap(R) {lossap:.4e} Loss G(R) {lossg:.4e}')
                train_loss1_lst.append(loss1)
                train_acc1_lst.append(acc1)
                train_loss2_lst.append(loss2)
                train_acc2_lst.append(acc2)
                train_lossa_lst.append(lossa)
                train_lossap_lst.append(lossap)
                train_lossg_lst.append(lossg)          

                #reset running loss and accuracy averages                    
                train_loss2.reset_states()
                train_accuracy2.reset_states()
                train_loss_a.reset_states()
                train_loss_ap.reset_states()
                train_loss_g.reset_states()
            
            if ((batch+1) % val_verbose_freq == 0) and enable_val:
                if hvd.rank()==0:
                    val_loss.reset_states()
                    val_accuracy.reset_states()
                    val_loss_a.reset_states()
                    val_loss_ap.reset_states()
                    val_loss_g.reset_states()
                #run for all ranks
                validate_model(val_dataset_per_rank)
                if hvd.rank()==0:
                    val_lossv=val_loss.result().numpy()
                    val_accv=val_accuracy.result().numpy()
                    val_lossa=val_loss_a.result().numpy()
                    val_lossap=val_loss_ap.result().numpy()
                    val_lossg=val_loss_g.result().numpy()
                    time_so_far=time.time()-start
                    print(f"{time_so_far:.2f}s Epoch {epoch + 1} Batch {batch+1} Val Loss {val_lossv:.4f} Val Accuracy {val_accv:.4f} " +
                          f"Val Loss A {val_lossa:.4e} Val Loss Ap {val_lossap:.4e} Val Loss G {val_lossg:.4e}  at Rank {hvd.rank()}")
                    val_loss_lst.append(val_lossv)
                    val_acc_lst.append(val_accv)
                    val_lossa_lst.append(val_lossa)
                    val_lossap_lst.append(val_lossap)
                    val_lossg_lst.append(val_lossg)

            #HOROVOD: save checkpoint on rank 0 process to avoid it being corrupted.                   
            if hvd.rank()==0:
                if chkpt_batches is not None:
                    if (batch+1) in chkpt_batches:
                        ckpt_save_path = train_ckpt_manager.save()
                        print(f'Saving TRAIN checkpoint for batch {batch+1} in epoch {epoch+1} at {ckpt_save_path}')

        if hvd.rank()==0:
            time_so_far=time.time()-start
            loss1=train_loss1.result().numpy()
            acc1=train_accuracy1.result().numpy()
            print(f'{time_so_far:.2f}s End of Epoch {epoch + 1} Batch {batch+1} Loss(E) {loss1:.4f} Accuracy(E) {acc1:.4f} LR {optimizer.learning_rate.numpy():.4e}')
            train_loss1_lst.append(loss1)
            train_acc1_lst.append(acc1)

        if chkpt_epochs is not None and hvd.rank()==0:
            if (epoch + 1) in chkpt_epochs:
                ckpt_save_path = train_ckpt_manager.save()
                print(f'Saving train checkpoint for epoch {epoch + 1} at {ckpt_save_path}')
        print(f"Rank {hvd.rank()} finished epoch with batch count {batch+1}")

        #End of epoch validation run if final batch is not multiple of val_verbose_freq
        if enable_val and ((batch+1) % val_verbose_freq != 0):
            if hvd.rank()==0:
                val_loss.reset_states()
                val_accuracy.reset_states()
                val_loss_a.reset_states()
                val_loss_ap.reset_states()
                val_loss_g.reset_states()
            #run for all ranks
            validate_model(val_dataset_per_rank)
            if hvd.rank()==0:
                val_lossv=val_loss.result().numpy()
                val_accv=val_accuracy.result().numpy()
                val_lossa=val_loss_a.result().numpy()
                val_lossap=val_loss_ap.result().numpy()
                val_lossg=val_loss_g.result().numpy()
                time_so_far=time.time()-start
                print(f"{time_so_far:.2f}s End of Epoch {epoch + 1} Batch {batch+1} Val Loss {val_lossv:.4f} Val Accuracy {val_accv:.4f} " +
                      f"Val Loss A {val_lossa:.4e} Val Loss Ap {val_lossap:.4e} Val Loss G {val_lossg:.4e} at Rank {hvd.rank()}")
                val_loss_lst.append(val_lossv)
                val_acc_lst.append(val_accv)
                val_lossa_lst.append(val_lossa)
                val_lossap_lst.append(val_lossap)
                val_lossg_lst.append(val_lossg)

    if hvd.rank()==0:
        #save final checkpoint
        final_ckpt_save_path = train_ckpt_manager.save()
        print(f'Saving final train checkpoint for epoch {EPOCHS} at {final_ckpt_save_path}')

        #Print final results
        time_so_far=time.time()-start
        loss1=train_loss1.result().numpy()
        acc1=train_accuracy1.result().numpy()
        loss2=train_loss2.result().numpy()
        acc2=train_accuracy2.result().numpy()
        lossa=train_loss_a.result().numpy()
        lossap=train_loss_ap.result().numpy()
        lossg=train_loss_g.result().numpy()
        print(f'{time_so_far:.2f}s Epoch {epoch + 1} Batch {batch+1} Loss(E) {loss1:.4f} Accuracy(E) {acc1:.4f} '+ 
                f'Loss(R) {loss2:.4f} Accuracy(R) {acc2:.4f} LR {optimizer.learning_rate.numpy():.4e} ' +
                f'Loss A(R) {lossa:.4e} Loss Ap(R) {lossap:.4e} Loss G(R) {lossg:.4e}')
        
        #save loss/accuracy checkpoints to file
        cm.pklsave(train_ckpt_path+'/train_loss1.pkl', train_loss1_lst)
        cm.pklsave(train_ckpt_path+'/train_accuracies1.pkl', train_acc1_lst)
        cm.pklsave(train_ckpt_path+'/train_loss2.pkl', train_loss2_lst)
        cm.pklsave(train_ckpt_path+'/train_accuracies2.pkl', train_acc2_lst)
        cm.pklsave(train_ckpt_path+'/train_lossAApG.pkl', [train_lossa_lst, train_lossap_lst, train_lossg_lst] )
        cm.pklsave(train_ckpt_path+'/val_loss.pkl', val_loss_lst)
        cm.pklsave(train_ckpt_path+'/val_accuracies.pkl', val_acc_lst)
        cm.pklsave(train_ckpt_path+'/val_lossAApG.pkl', [val_lossa_lst, val_lossap_lst, val_lossg_lst] )

###################################### END TRAINING ########################################################################

#END OF MAIN


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
        cosine_decay = self.learning_rate*((1-self.alpha)*0.5*(1+tf.math.cos(pi*(decay_step/decay_rate)))+self.alpha)

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


@tf.function
def accuracy_function(real, pred):
    accuracies = tf.equal(real, tf.argmax(pred, axis=2))

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)

    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)

if __name__ == '__main__':
    if len(sys.argv) == 4:
        # run training through horovod.run
        npr = int(sys.argv[1])
        hosts = sys.argv[2]
        comm = sys.argv[3]
        print('Running training through horovod.run')
        horovod.run(main, np=npr, hosts=hosts, use_gloo=comm == 'gloo', use_mpi=comm == 'mpi', verbose=True)
    else:
        # this is running via horovodrun on single rank
        main()   


