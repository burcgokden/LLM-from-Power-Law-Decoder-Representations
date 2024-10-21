'''
Model Implementation for Large LAngiage Model from Power Law Decoder Representations v900 (PLDR-LLM-v900)
'''

import tensorflow as tf
import keras_nlp
import power_law_attention_layer_v900 as plgatt
import contextlib

@tf.keras.saving.register_keras_serializable(package="pldr_llm")
class plgMultiHeadAttention(tf.keras.layers.Layer):
    '''
    Power Law Multihead Attention Implementation for PLDR-LLM.
    '''
    def __init__(self, d_model, num_heads, att_dropout_rate_in=0.0,
                 att_dropout_rate_eij=0.0, Adropout_rate=0.0, A_dff_depth=None, A_dff=None,
                 num_reslayerA=None, num_denseA=None, activation=tf.nn.silu, **kwargs):

        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model
        self.att_dropout_rate_in=att_dropout_rate_in
        self.att_dropout_rate_eij = att_dropout_rate_eij
        self.Adropout_rate=Adropout_rate
        self.A_dff = A_dff if A_dff is not None else self.d_model
        self.A_dff_depth=A_dff_depth if A_dff_depth is not None else self.d_model
        self.num_denseA = num_denseA if num_denseA is not None else 1
        self.num_reslayerA = num_reslayerA if num_reslayerA is not None else 1
        self.activation=tf.keras.activations.get(activation)

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model, name='wq')
        self.wk = tf.keras.layers.Dense(d_model, name='wk')
        self.wv = tf.keras.layers.Dense(d_model, name='wv')

        self.plgatt_layer= plgatt.plga_layer(F_hidden=self.depth,
                                       activation=None,
                                       att_activation= None,
                                       pw_regularizer=None,
                                       in_dropout_prob=self.att_dropout_rate_in,
                                       eij_dropout_prob=self.att_dropout_rate_eij,
                                       name='plga_layer')

        self.dense = tf.keras.layers.Dense(d_model, name='dense')

        #residual layers for metric tensor learning
        #suppress layout_optimizer failed messages
        with meta_options({"layout_optimizer": False}):
            self.reslayerAs=[ResLayerA(A_dff_depth=self.A_dff_depth, A_dff=self.A_dff,
                                    Adropout_rate=self.Adropout_rate,
                                    num_denseA=self.num_denseA,
                                    index=str(i), activation=self.activation) for i in range(self.num_reslayerA)]

        #dense layers to scale up and down for residual layers.
        self.res_inp_ffn=tf.keras.layers.Dense(self.A_dff_depth, name='A_dff_dense1')
        self.res_final_ffn=tf.keras.layers.Dense(self.depth, name='A_dff_dense2')
        
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='mha_layernorm1')
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='mha_layernorm2')
        self.rotary_embedding=keras_nlp.layers.RotaryEmbedding(max_wavelength=10000, scaling_factor=1.0, sequence_axis=2, feature_axis=-1)


    def split_heads(self, x, batch_size):
        '''
        Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        '''
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, training=None, **kwargs):
        '''
        Args:
            inputs: [q,k,v,mask]
            training
        Returns:
            inductive and deductive task outputs.
        '''
        q, k, v, mask = inputs
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)
        v = self.wv(v)


        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len, depth)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        q = self.rotary_embedding(q)
        k = self.rotary_embedding(k)

        #Calculate density matrix using linear self attention
        qt = tf.transpose(q, perm=[0,1, 3, 2])
        A = tf.matmul(qt, q)  # (batch_size, num_head, depth, depth)

        #Deep residual network for learning metric tensor
        A=self.res_inp_ffn(A) #change layer size to A_dff_depth
        A=self.layernorm1(A)
        for i in range(self.num_reslayerA):
            A=self.reslayerAs[i]([A], training=training)
        A=self.res_final_ffn(A) #change layer size to depth
        A=self.layernorm2(A)

        #Apply multi-head power law attention
        Hnext, Elst, Alst, pwlst, attvlst, balst, avAplst, Eplst = self.plgatt_layer([q, k, v, A, mask], training=training)
        Hnext = tf.transpose(Hnext, perm=[0, 2, 1, 3])

        Hnext= tf.reshape(Hnext, (batch_size, -1, self.d_model)) # [batch_size, seq_len, d_model]

        output = self.dense(Hnext)

        return output, Elst, Alst, pwlst, attvlst, balst, avAplst, Eplst

    def get_config(self):
        config = super().get_config()
        config.update({
                    "d_model":self.d_model,
                    "num_heads":self.num_heads,
                    "att_dropout_rate_in":self.att_dropout_rate_in,
                    "att_dropout_rate_eij":self.att_dropout_rate_eij,
                    "Adropout_rate":self.Adropout_rate,
                    "A_dff":self.A_dff,
                    "A_dff_depth":self.A_dff_depth,
                    "num_reslayerA":self.num_reslayerA,
                    "num_denseA":self.num_denseA,
                    "activation":tf.keras.saving.serialize_keras_object(self.activation),
                    })
        return config
    
    @classmethod
    def from_config(cls, config):
        config["activation"]=tf.keras.saving.deserialize_keras_object(config["activation"])
        return cls(**config)


@tf.keras.saving.register_keras_serializable(package="pldr_llm")
class PLDR_DecoderLayer(tf.keras.layers.Layer):
    '''
    Single decoder layer implementation for PLDR-LLM with single masked multihead attention.
    '''
    def __init__(self, d_model, num_heads, dff, rate=0.0, att_dropout_rate_in=0.0, att_dropout_rate_eij=0.0,
                 Adropout_rate=0.0, A_dff_depth=None, A_dff=None, num_reslayerA=None, num_denseA=None, activation=tf.nn.silu, **kwargs):

        super().__init__(**kwargs)

        self.d_model=d_model
        self.num_heads=num_heads
        self.dff=dff
        self.rate=rate
        self.att_dropout_rate_in = att_dropout_rate_in
        self.att_dropout_rate_eij = att_dropout_rate_eij
        self.Adropout_rate=Adropout_rate
        self.A_dff=A_dff
        self.A_dff_depth=A_dff_depth
        self.num_denseA = num_denseA
        self.num_reslayerA = num_reslayerA
        self.activation=tf.keras.activations.get(activation)

        self.mha1 = plgMultiHeadAttention(d_model=self.d_model, num_heads=self.num_heads,att_dropout_rate_in=self.att_dropout_rate_in,
                                       att_dropout_rate_eij=self.att_dropout_rate_eij, Adropout_rate=self.Adropout_rate, 
                                       A_dff_depth=self.A_dff_depth, A_dff=self.A_dff,
                                       num_reslayerA=self.num_reslayerA, num_denseA=self.num_denseA, activation=self.activation)

        self.ffn = self.dec_point_wise_feed_forward_network()

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='layernorm1')
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='layernorm2')

        self.dropout1 = tf.keras.layers.Dropout(self.rate, name='dropout1')
        self.dropout2 = tf.keras.layers.Dropout(self.rate, name='dropout2')

    def call(self, inputs, training=None, **kwargs):
        '''
        inputs: [x, look_ahead_mask]
        Returns Decoder Layer output and deductive task outputs.
        '''

        x, look_ahead_mask = inputs

        attn1, Elst1, Alst1, pwlst1, attvlst1, balst1, avAplst1, Eplst1 = self.mha1([x,x,x, look_ahead_mask], training=training)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(ffn_output + out1)  # (batch_size, target_seq_len, d_model)

        return out2, [Elst1, Alst1, pwlst1, attvlst1, balst1, avAplst1, Eplst1]


    #swiGLU implementation for feedforward network, scale dff accordingly (i.e., 2/3 of original).
    def dec_point_wise_feed_forward_network(self):
        return GLUVariant(self.dff, self.d_model, name_sfx="decffn", activation=self.activation)


    def get_config(self):
        config = super().get_config()
        config.update({
                    "d_model":self.d_model,
                    "num_heads":self.num_heads,
                    "dff":self.dff,
                    "rate":self.rate,
                    "att_dropout_rate_in":self.att_dropout_rate_in,
                    "att_dropout_rate_eij":self.att_dropout_rate_eij,
                    "Adropout_rate":self.Adropout_rate,
                    "A_dff":self.A_dff,
                    "A_dff_depth":self.A_dff_depth,
                    "num_reslayerA":self.num_reslayerA,
                    "num_denseA":self.num_denseA,
                    "activation":tf.keras.saving.serialize_keras_object(self.activation),
                    })
        return config
    
    @classmethod
    def from_config(cls, config):
        config["activation"]=tf.keras.saving.deserialize_keras_object(config["activation"])
        return cls(**config)

@tf.keras.saving.register_keras_serializable(package="pldr_llm")
class PLDR_Decoder(tf.keras.layers.Layer):
    '''
    Multi layer decoder implementation for PLDR-LLM
    '''
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                 rate=0.0, att_dropout_rate_in=0.0, att_dropout_rate_eij=0.0,
                 Adropout_rate=0.0, A_dff_depth=None, A_dff=None, num_reslayerA=None, num_denseA=None, 
                  activation=tf.nn.silu, **kwargs):
        super().__init__(**kwargs)

        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads=num_heads
        self.dff=dff
        self.target_vocab_size = target_vocab_size
        self.rate = rate
        self.att_dropout_rate_in = att_dropout_rate_in
        self.att_dropout_rate_eij = att_dropout_rate_eij
        self.Adropout_rate=Adropout_rate
        self.A_dff=A_dff
        self.A_dff_depth=A_dff_depth
        self.num_denseA = num_denseA
        self.num_reslayerA = num_reslayerA
        self.activation=tf.keras.activations.get(activation)

        self.embedding = tf.keras.layers.Embedding(self.target_vocab_size, self.d_model, name='dec_embedding')

        self.dec_layers = [PLDR_DecoderLayer(d_model=self.d_model, num_heads=self.num_heads, dff=self.dff, 
                                             rate=self.rate, att_dropout_rate_in=self.att_dropout_rate_in,
                                        att_dropout_rate_eij=self.att_dropout_rate_eij, Adropout_rate=self.Adropout_rate,
                                         A_dff_depth=self.A_dff_depth, A_dff=self.A_dff,
                                        num_reslayerA=self.num_reslayerA, num_denseA=self.num_denseA, activation=self.activation) for _ in range(self.num_layers)]

        self.inp_dropout = tf.keras.layers.Dropout(self.rate, name='dec_inp_dropout')
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='dec_layernorm1')


    def call(self, inputs, training=None, **kwargs):
        '''
        inputs: [x, look_ahead_mask].
        Returns output of decoder and attention weights for PLDR_Decoder.
        '''

        x, look_ahead_mask= inputs

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        x=self.inp_dropout(x)
        x=self.layernorm1(x)

        dec_att_weigths1=[]
        for i in range(self.num_layers):
            x, dec_att_w1= self.dec_layers[i]([x, look_ahead_mask], training=training)
            dec_att_weigths1.append(dec_att_w1)

        return x, dec_att_weigths1


    def get_config(self):
        config = super().get_config()
        config.update({
                    "num_layers":self.num_layers,
                    "d_model":self.d_model,
                    "num_heads":self.num_heads,
                    "dff":self.dff,
                    "target_vocab_size":self.target_vocab_size,
                    "rate":self.rate,
                    "att_dropout_rate_in":self.att_dropout_rate_in,
                    "att_dropout_rate_eij":self.att_dropout_rate_eij,
                    "Adropout_rate":self.Adropout_rate,
                    "A_dff":self.A_dff,
                    "A_dff_depth":self.A_dff_depth,
                    "num_reslayerA":self.num_reslayerA,
                    "num_denseA":self.num_denseA,
                    "activation":tf.keras.saving.serialize_keras_object(self.activation),
                    })
        return config
    
    @classmethod
    def from_config(cls, config):
        config["activation"]=tf.keras.saving.deserialize_keras_object(config["activation"])
        return cls(**config)

@tf.keras.saving.register_keras_serializable(package="pldr_llm")
class PLDR_Model(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,  
                  rate=0.0, att_dropout_rate_in=0.0, att_dropout_rate_eij=0.0,
                 Adropout_rate=0.0, A_dff_depth=None, A_dff=None, num_reslayerA=None, num_denseA=None, 
                  activation=tf.nn.silu, **kwargs):
        '''
        Args:
            num_layers: Number of Decoder Layers.
            d_model: Embedding/LM feature dimension
            num_heads: Number of power law attention heads
            dff: Number of neurons on single layer of fully connected network
            input_vocab_size: Vocabulary size for the embedding layer (target_vocab_size for PLDR_Decoder class)
            rate: Drop out rate for embeddings and output of fully connected networks.
            att_dropout_rate_in: Drop out rate for power law attention query and key inputs
            att_dropout_rate_eij: Drop out rate for power law attention weight
            Adropout_rate: Drop out rate for each unit in residual network for metric tensor learning
            A_dff_depth: Number of neurons in linear layer of residual unit for metric tensor learning
            A_dff: Number of neurons in gated-linear layer of residual unit for metric tensor learning
            num_reslayerA: Number of residual units
            num_denseA: Number of dense layers in each residual unit.
        Returns:
            Logit probabilities for predicted sentence and power law attention weights for deductive task.

        '''
        super().__init__(**kwargs)

        self.num_layers = num_layers
        self.d_model=d_model
        self.num_heads=num_heads
        self.dff=dff
        self.input_vocab_size = input_vocab_size
        self.rate=rate
        self.att_dropout_rate_in = att_dropout_rate_in
        self.att_dropout_rate_eij = att_dropout_rate_eij
        self.Adropout_rate=Adropout_rate
        self.A_dff = A_dff
        self.A_dff_depth=A_dff_depth
        self.num_denseA = num_denseA
        self.num_reslayerA = num_reslayerA
        self.activation=tf.keras.activations.get(activation)

        self.decoder = PLDR_Decoder(num_layers=self.num_layers, d_model=self.d_model, 
                                    num_heads=self.num_heads, dff=self.dff,
                               target_vocab_size=self.input_vocab_size, 
                               rate=self.rate, att_dropout_rate_in=self.att_dropout_rate_in,
                               att_dropout_rate_eij=self.att_dropout_rate_eij, Adropout_rate=self.Adropout_rate, 
                               A_dff_depth=self.A_dff_depth, A_dff=self.A_dff,
                               num_reslayerA=self.num_reslayerA, num_denseA=self.num_denseA,  activation=self.activation) #use_position_emb,

        self.final_layer = tf.keras.layers.Dense(self.input_vocab_size,name='dense_final_layer')

    def call(self, inputs, training=None, trainable=True, **kwargs):

        inp, look_ahead_mask=inputs

        self.trainable=trainable

        dec_output, dec_att_weights1 = self.decoder([inp, look_ahead_mask], training=training )

        final_output = self.final_layer(dec_output)

        return final_output, dec_output, [dec_att_weights1]


    def get_config(self):
        config = super().get_config()
        config.update({
                    "num_layers":self.num_layers,
                    "d_model":self.d_model,
                    "num_heads":self.num_heads,
                    "dff":self.dff,
                    "input_vocab_size":self.input_vocab_size,
                    "rate":self.rate,
                    "att_dropout_rate_in":self.att_dropout_rate_in,
                    "att_dropout_rate_eij":self.att_dropout_rate_eij,
                    "Adropout_rate":self.Adropout_rate,
                    "A_dff":self.A_dff,
                    "A_dff_depth":self.A_dff_depth,
                    "num_reslayerA":self.num_reslayerA,
                    "num_denseA":self.num_denseA,
                    "activation":tf.keras.saving.serialize_keras_object(self.activation),
                    })
        return config
    
    @classmethod
    def from_config(cls, config):
        config["activation"]=tf.keras.saving.deserialize_keras_object(config["activation"])
        return cls(**config)


@tf.keras.saving.register_keras_serializable(package="pldr_llm")
class ResLayerA(tf.keras.layers.Layer):
    '''
    Residual Layer implementation for metric learner of PLDR-LLM
    '''
    def __init__(self, A_dff_depth, A_dff, Adropout_rate=0.0, num_denseA=None, index='0', activation=tf.nn.silu, **kwargs):
        super().__init__(**kwargs)
        self.A_dff_depth=A_dff_depth
        self.A_dff = A_dff
        self.Adropout_rate = Adropout_rate
        self.num_denseA = num_denseA if num_denseA is not None else 1
        self.index=index
        self.activation=tf.keras.activations.get(activation)

        self.denseAs = [GLUVariant(self.A_dff, self.A_dff_depth, name_sfx="gluA"+self.index+str(i), activation=self.activation)
                        for i in range(self.num_denseA)]
        self.dropoutA = tf.keras.layers.Dropout(rate=self.Adropout_rate, name="Adropout"+self.index)

        self.layernormA = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='layernormA'+self.index)

    
    def ResUnit(self, A, training):
        Ain = tf.identity(A)
        for i in range(self.num_denseA):          
            A = self.denseAs[i](A)
        A = self.dropoutA(A, training=training)
        A = self.layernormA(A + Ain)
        return A

    def call(self, inputs, training=None, **kwargs):
        A=inputs[0]
        return self.ResUnit(A, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({
                    "A_dff_depth":self.A_dff_depth,
                    "A_dff":self.A_dff,
                    "Adropout_rate":self.Adropout_rate,
                    "num_denseA":self.num_denseA,                    
                    "index":self.index,
                    "activation":tf.keras.saving.serialize_keras_object(self.activation),
                    })
        return config
    
    @classmethod
    def from_config(cls, config):
        config["activation"]=tf.keras.saving.deserialize_keras_object(config["activation"])
        return cls(**config)
    
@contextlib.contextmanager
def meta_options(options):
    old_opts = tf.config.optimizer.get_experimental_options()
    tf.config.optimizer.set_experimental_options(options)
    try:
        yield
    finally:
        tf.config.optimizer.set_experimental_options(old_opts)


@tf.keras.saving.register_keras_serializable(package="pldr_llm")
class GLUVariant(tf.keras.layers.Layer):
    '''
    Implementation of GLU variants with default activation for SwiGLU configuration 
    For the hidden layer dff, to match size with non-SwiGLU FFN version scaling with 2/3 may be useful.
    '''
    def __init__(self, dff, depth, name_sfx, activation=tf.nn.silu, **kwargs):
        super().__init__(**kwargs)
        self.dff=dff
        self.depth=depth
        self.name_sfx=name_sfx
        self.activation=tf.keras.activations.get(activation)
        self.gluw1=tf.keras.layers.Dense(self.dff, activation=self.activation, name="gW1"+self.name_sfx)
        self.gluw2=tf.keras.layers.Dense(self.dff, activation=None, name="gW2"+self.name_sfx)
        self.gluw3=tf.keras.layers.Dense(self.depth, activation=None, name="gW3"+self.name_sfx)

    def call(self, input, **kwargs):
        x1=self.gluw1(input)
        x2=self.gluw2(input)
        return self.gluw3(tf.multiply(x1, x2))
    
    def get_config(self):
        config = super().get_config()
        config.update({
                    "depth":self.depth,
                    "dff":self.dff,                   
                    "name_sfx":self.name_sfx,
                    "activation":tf.keras.saving.serialize_keras_object(self.activation),
                    })
        return config
    
    @classmethod
    def from_config(cls, config):
        config["activation"]=tf.keras.saving.deserialize_keras_object(config["activation"])
        return cls(**config)
    






