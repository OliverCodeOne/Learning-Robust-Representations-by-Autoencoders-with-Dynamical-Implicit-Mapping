import tensorflow as tf
import numpy as np
import itertools
import functools
from fjcommon import functools_ext

_HARD_SIGMA = 1e7
_HARD_TEMP = 1e-5

###################### quantizer utils
def transpose_NCHW_to_NHWC(t):
    return tf.transpose(t, (0, 2, 3, 1), name='to_NHWC')


def transpose_NHWC_to_NCHW(t):
    return tf.transpose(t, (0, 3, 1, 2), name='to_NCHW')


def quantize(x, centers, sigma, data_format):
    """ :return qsoft, qhard, symbols """
    with tf.name_scope('quantize'):
        return _quantize1d(x, centers, sigma, data_format=data_format)


def _quantize1d(x, centers, sigma, data_format):
    #print('Quantizer inside x:',x.shape)
    #we assume the data order is NHWC
    """
    :return: (softout, hardout, symbols_vol)
        each of same shape as x, softout, hardout will be float32, symbols_vol will be int64
    """
    assert tf.float32.is_compatible_with(x.dtype), 'x should be float32'
    assert tf.float32.is_compatible_with(centers.dtype), 'centers should be float32'
    assert len(x.get_shape()) == 4, 'x should be NCHW or NHWC, got {}'.format(x.get_shape())
    assert len(centers.get_shape()) == 1, 'centers should be (L,), got {}'.format(centers.get_shape())

    if data_format == 'NHWC':
        print('data format: NHWC')
        x_t = transpose_NHWC_to_NCHW(x)
        softout, hardout, symbols_hard = _quantize1d(x_t, centers, sigma, data_format='NCHW')
        return tuple(map(transpose_NCHW_to_NHWC, (softout, hardout, symbols_hard)))

    # Note: from here on down, x is NCHW ---

    # count centers
    num_centers = centers.get_shape().as_list()[-1]

    with tf.name_scope('reshape_BCm1'):
        # reshape (B, C, w, h) to (B, C, m=w*h)
        x_shape_BCwh = tf.shape(x)
        B = x_shape_BCwh[0]  # B is not necessarily static
        C = int(x.shape[1])  # C is static
        x = tf.reshape(x, [B, C, -1])

        # make x into (B, C, m, 1)
        x = tf.expand_dims(x, axis=-1)

    with tf.name_scope('dist'):
        # dist is (B, C, m, L), contains | x_i - c_j | ^ 2
        dist = tf.square(tf.abs(x - centers))

    with tf.name_scope('phi_soft'):
        # (B, C, m, L)
        phi_soft = tf.nn.softmax(-sigma       * dist, dim=-1)
    with tf.name_scope('phi_hard'):
        # (B, C, m, L) probably not necessary due to the argmax!
        phi_hard = tf.nn.softmax(-_HARD_SIGMA * dist, dim=-1)

        symbols_hard = tf.argmax(phi_hard, axis=-1)
        phi_hard = tf.one_hot(symbols_hard, depth=num_centers, axis=-1, dtype=tf.float32)

    with tf.name_scope('softout'):
        softout = phi_times_centers(phi_soft, centers,3)
    with tf.name_scope('hardout'):
        hardout = phi_times_centers(phi_hard, centers,3)

    def reshape_to_BCwh(t_):
        with tf.name_scope('reshape_BCwh'):
            return tf.reshape(t_, x_shape_BCwh)
    return tuple(map(reshape_to_BCwh, (softout, hardout, symbols_hard)))

def phi_times_centers(phi, centers,axis):
    matmul_innerproduct = phi * centers  # (B, C, m, L)
    return tf.reduce_sum(matmul_innerproduct, axis=axis)  # (B, C, m)



def assert_dtype(x, dtype):
    assert dtype.is_compatible_with(x.dtype), 'Expected {} == {}'.format(dtype.name, x.dtype.name)


def assert_ndims(x, ndims):
    assert x.shape.ndims == ndims, 'Expected ndims == {}, got shape {}'.format(ndims, x.shape)

def assert_equal_shape(x, y):
    assert x.shape == y.shape, 'Expected {} == {}'.format(x.shape, y.shape)


################
#pobability class
def bitcost_to_bpp(bit_cost, input_batch):
    """
    :param bit_cost: NChw
    :param input_batch: N3HW
    :return: Chw / HW, i.e., num_bits / num_pixels
    """
    assert bit_cost.shape.ndims == input_batch.shape.ndims == 4, 'Expected NChw and N3HW, got {} and {}'.format(
        bit_cost, input_batch)
    with tf.name_scope('bitcost_to_bpp'):
        num_bits = tf.reduce_sum(bit_cost, name='num_bits')
        return num_bits / tf.to_float(num_pixels_in_input_batch(input_batch))


def num_pixels_in_input_batch(input_batch):
    assert int(input_batch.shape[1]) == 3, 'Expected N3HW, got {}'.format(input_batch)
    with tf.name_scope('num_pixels'):
        return tf.reduce_prod(tf.shape(input_batch)) / 3


def context_shape_from_context_size(context_size):
    """ :return context shape as DHW """
    return context_size // 2 + 1, context_size, context_size


def context_size_from_context_shape(context_shape):
    return context_shape[-1]



class _Network3D(object):
    def __init__(self,kernel_size, num_centers):
        """
        :param pc_config: Expected to contain
                - kernel_size: int
                - arch: str, see get_network_cls
        """
        self.kernel_size = kernel_size
        self.reuse = False
        self.L = num_centers
        self.arch_param__k = 24
        self.arch_param__non_linearity = 'relu'
        self.arch_param__fc = 64

        self.first_mask = None
        self.other_mask = None
        self.use_centers_for_padding = True 
#        _PROBCLASS_SCOPE = 'probclass3d'
        self._NUM_RESIDUAL =1
    def get_num_layers(self):
        num_conv = 2
        per_residual = 2
        return num_conv + self._NUM_RESIDUAL * per_residual


    def get_context_size(self, kernel_size):
        """
        width / height of the receptive field
        """
        return self.get_num_layers() * (kernel_size - 1) + 1

    def get_context_shape(self, kernel_size):
        """ Shape as DHW """
        return context_shape_from_context_size(self.get_context_size(kernel_size))

    def auto_pad_value(self, centers):
        return (0 if not self.use_centers_for_padding else
                centers[0])

    def bitcost(self, q, target_symbols, is_training, pad_value=0):
        """
        Pads q, creates PC network, calculates cross entropy between output of PC network and target_symbols
        :param q: NCHW
        :param target_symbols:
        :param is_training:
        :return: bitcost per symbol: NCHW
        """
        #we may want to use data format as NCHW
        assert_ndims(q, 4)

        #with self._building_ctx(self.reuse):
        if self.first_mask is None:
            self.first_mask = self.create_first_mask()  # DHWio
            self.other_mask = self.create_other_mask()  # DHWio

        self.reuse = True

        targets_one_hot = tf.one_hot(target_symbols, depth=self.L, axis=-1, name='target_symbols')

        q_pad = pad_for_probclass3d(
                q, context_size=self.get_context_size(self.kernel_size),
                pad_value=pad_value, learn_pad_var=False)
        with tf.variable_scope('logits'):
            # make it into NCHWT, where T is the channel dim of the conv3d
            q_pad = tf.expand_dims(q_pad, -1, name='NCHWT')
            logits = self._logits(q_pad, is_training)

        if targets_one_hot.shape.is_fully_defined() and logits.shape.is_fully_defined():
            assert_equal_shape(targets_one_hot, logits)

        with tf.name_scope('bitcost'):
            # softmax_cross_entropy_with_logits is basis e, change base to 2
            log_base_change_factor = tf.constant(np.log2(np.e), dtype=tf.float32)
            bc = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=targets_one_hot) * log_base_change_factor  # NCHW

        return bc

    def logits(self, q, is_training):
        """ For accessing logits, mimics the structure of bitcost(...) """
        assert self.reuse, 'Make sure to call bitcost(...) before calling logits(...)'
        with tf.variable_scope('logits'):
            return self._logits(q, is_training)

    def _logits(self, q, is_training):
        """
        :param q: input features, NCHW1, where 1 is the channel dimension of the 3d conv
        :param is_training:
        :return:
        """
        k = self.arch_param__k
        net = q
        #print('*****************************input to 3D conv:',net.shape)
        net = conv3d('conv0', net, k, self.filter_shape, filter_mask=self.first_mask)
        for res_i in range(self._NUM_RESIDUAL):
            net = self.residual_block(net, name='res{}'.format(res_i + 1))
            
        #print('***************************** after residual:',net.shape)
        net = conv3d('conv2', net, self.L, self.filter_shape, filter_mask=self.other_mask)
        #print('***************************** logits:',net.shape)
        return net
    @property
    def filter_shape(self):
        K = self.kernel_size
        return K // 2 + 1, K, K  # CHW

    def create_first_mask(self):
        with tf.name_scope('first_mask'):
            K = self.kernel_size
            # mask is DHW
            mask = np.ones(self.filter_shape, dtype=np.float32)
            # zero out D=1,
            # - everything to the right of the central pixel, including the central pixel
            mask[-1, K // 2, K // 2:] = 0
            # - all rows below the central row
            mask[-1, K // 2 + 1:, :] = 0

            mask = np.expand_dims(np.expand_dims(mask, -1), -1)  # Make into DHWio, for broadcasting with 3D filters
            return _Network3D._make_tf_conv3d_mask(mask)

    def create_other_mask(self):
        with tf.name_scope('other_mask'):
            K = self.kernel_size
            # mask is DHW
            mask = np.ones(self.filter_shape, dtype=np.float32)
            # zero out D=1,
            # - everything to the right of the central pixel, except the central pixel
            mask[-1, K // 2, K // 2 + 1:] = 0
            # - all rows below the central row
            mask[-1, K // 2 + 1:, :] = 0

            mask = np.expand_dims(np.expand_dims(mask, -1), -1)  # Make into DHWio, for broadcasting with 3D filters
            return _Network3D._make_tf_conv3d_mask(mask)

    @staticmethod
    def _make_tf_conv3d_mask(mask):
        assert mask.ndim == 5, 'Expected DHWio'
        mask = tf.constant(mask)
        mask = tf.stop_gradient(mask)
        return mask

    def residual_block(self, x, num_conv2d=2, name=None):
        num_outputs = x.shape.as_list()[-1]
        residual_input = x
        activation_fn = tf.nn.relu
        with tf.variable_scope(name, 'res'):
            for conv_i in range(num_conv2d):
                if conv_i == (num_conv2d - 1):  # no relu after final conv
                    activation_fn = None
                x = conv3d('conv{}'.format(conv_i + 1),
                           x, num_outputs, self.filter_shape, self.other_mask,
                           activation_fn=activation_fn)
            return x + residual_input[..., 2:, 2:-2, 2:-2, :]  # for padding



# =============================================================================

# Helpers ----------------------------------------------------------------------


def conv3d(name,
           x,  # NCHWD
           num_outputs,
           filter_shape,  # (C, H, W)
           filter_mask=None,
           strides=None,
           activation_fn=tf.nn.relu,
           padding='VALID',
           weights_initializer=tf.contrib.layers.xavier_initializer(),
           biases_initializer=tf.zeros_initializer(),
           ):
    assert name is not None, 'Need name'
    assert len(filter_shape) == 3
    if not strides:
        strides = [1, 1, 1, 1, 1]
    if not activation_fn:
        activation_fn = functools_ext.identity

    num_inputs = x.shape.as_list()[-1]
    filter_shape = tuple(filter_shape) + (num_inputs, num_outputs)

    masked = filter_mask is not None
    scope_name = 'conv3d_{}'.format(name) + ('_mask' if masked else '')
    with tf.variable_scope(scope_name):
        weights = tf.get_variable('weights_{}'.format(name), shape=filter_shape, dtype=tf.float32,
                                  initializer=weights_initializer)
        if filter_mask is not None:
            weights = weights * filter_mask

        biases = tf.get_variable('biases', shape=(num_outputs,), dtype=tf.float32,
                                 initializer=biases_initializer)
        out = tf.nn.conv3d(x, weights, strides, padding, name='conv3d')
        out = tf.nn.bias_add(out, biases, name='bias3d')
        out = activation_fn(out)
        return out



def _get_all_conv3d_weights_in_scope(scope):
    return [v for v in tf.trainable_variables(scope=scope) if 'weights' in v.name]


def pad_for_probclass3d(x, context_size, pad_value=0, learn_pad_var=False):
    """
    :param x: NCHW tensorflow Tensor or numpy array
    """
    input_is_tf = not isinstance(x, np.ndarray)
    if not input_is_tf and x.ndim == 3:  # for bit_counter
        return remove_batch_dim(pad_for_probclass3d(
                add_batch_dim(x), context_size, pad_value, learn_pad_var))

    with tf.name_scope('pad_cs' + str(context_size)):
        pad = context_size // 2
        assert pad >= 1
        if learn_pad_var:
            if not isinstance(pad_value, tf.Variable):
                print('Warn: Expected tf.Variable for padding, got {}'.format(pad_value))
            return pc_pad_grad(x, pad, pad_value)

        pads = [[0, 0],  # don't pad batch dimension
                [pad, 0],  # don't pad depth_future, it's not seen by any filter
                [pad, pad],
                [pad, pad]]
        assert len(pads) == _get_ndims(x), '{} != {}'.format(len(pads), x.shape)

        pad_fn = tf.pad if input_is_tf else get_np_pad_fn()
        return pad_fn(x, pads, constant_values=pad_value)


def get_np_pad_fn():
    return functools.partial(np.pad, mode='constant')


def add_batch_dim(a):
    return np.expand_dims(a, 0)


def remove_batch_dim(a):
    return a[0, ...]


def pc_pad_grad(x, pad, pad_var):
    """
    Like tf.pad but gradients flow to pad_var
    :param x: NCHW
    :param pad: will pad with
        pads = [[0, 0],     don't pad batch dimension
                [pad, 0],   don't pad depth_future, it's not seen by any filter
                [pad, pad],
                [pad, pad]]
    :param pad_var: value to use for padding
    :return:
    """
    with tf.name_scope('pc_pad_grad'):
        n, c, h, w = x.shape.as_list()
        with tf.name_scope('pad_var_NCHW'):
            pad_var = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.expand_dims(pad_var, 0), 0), 0), 0)

        with tf.name_scope('front'):
            front = tf.tile(pad_var, [n, pad, h, w])
            x_front = tf.concat((front, x), 1)

        with tf.name_scope('left_right'):
            left = tf.tile(pad_var, [n, c + pad, h, pad])
            right = left
            x_front_left_right = tf.concat((left, x_front, right), 3)

        with tf.name_scope('top_bottom'):
            top = tf.tile(pad_var, [n, c + pad, pad, w + 2 * pad])
            bottom = top
            x_fron_left_right_top_bottom = tf.concat((top, x_front_left_right, bottom), 2)

        return x_fron_left_right_top_bottom


def undo_pad_for_probclass3d(x, context_size):
    """
    :param x: NCHW tensorflow Tensor or numpy array
    """
    if isinstance(x, np.ndarray) and x.ndim == 3:
        return remove_batch_dim(undo_pad_for_probclass3d(add_batch_dim(x), context_size))

    with tf.name_scope('undo_pad_cs' + str(context_size)):
        pad = context_size // 2
        assert pad >= 1
        return x[:, pad:, pad:-pad, pad:-pad]


def _get_ndims(t_or_a):
    try:
        return t_or_a.shape.ndims
    except AttributeError:
        return t_or_a.ndim


def iter_over_blocks(syms, block_sizes):
    """
    Iterate over symbols in blocks of size in block_sizes.
    :param syms: CHW
    :return: blocks, iterating in order W, H, C
    """
    for c_slice, h_slice, w_slice in _iter_block_idices(syms.shape, block_sizes):
        yield syms[c_slice, h_slice, w_slice]


def num_blocks(syms_shape, block_sizes):
    """ Number of blocks iterated over when iter_over_blocks is called. """
    return sum(1 for _ in _iter_block_idices(syms_shape, block_sizes))


def _iter_block_idices(syms_shape, block_sizes):
    C, H, W = syms_shape
    bC, bH, bW = block_sizes
    last_indices = (C - bC + 1, H - bH + 1, W - bW + 1)
    for c, h, w in itertools.product(*map(range, last_indices)):
        yield slice(c, c+bC), slice(h, h+bH), slice(w, w+bW)





class ProbclassNetworkTesting(object):
    """
    Used to get the bit_cost fully convolutionally for an arbitrariy input. To compare to actual bpp calculation.
    """
    def __init__(self, pc: _Network3D, ae, sess):
        """
        :param pc: Probability classifier network
        :param ae: Auotencoder
        :param sess: session to run
        """
        self.input_ph_symbols = tf.placeholder(tf.int64, shape=(None, None, None, None))  # NCHW

        # get q from symbols
        centers = ae.get_centers_variable()
        q = tf.gather(centers, self.input_ph_symbols)
        bit_cost_per_symbol = pc.bitcost(q, self.input_ph_symbols, is_training=False,
                                         pad_value=pc.auto_pad_value(ae))
        self.bit_cost_total = tf.reduce_sum(bit_cost_per_symbol)  # single value
        self.sess = sess

    def get_total_bit_cost(self, symbols):
        """
        :param symbols: CHW or NCHW, numpy, all symbols for a given image
        :return:
        """
        if symbols.ndim == 3:
            return self.get_total_bit_cost(np.expand_dims(symbols, 0))
        assert symbols.ndim == 4
        return self.sess.run(self.bit_cost_total, feed_dict={self.input_ph_symbols: symbols})



class PredictionNetwork(object):
    """
    Used for prediction given a slice of the symbols volume
    """
    def __init__(self, pc: _Network3D, config, centers, sess, freqs_resolution=1e9):
        """
        :param sess: Must be set at the latest before using get_pr or get_freqs
        """
        self.pc_class = pc.__class__
        self.config = config
        self.input_ctx_shape = self.pc_class.get_context_shape(config)
        self.input_ctx = tf.placeholder(tf.int64, self.input_ctx_shape)  # symbols!
        input_ctx_batched = tf.expand_dims(self.input_ctx, 0)  # add batch dimension, 1DHW
        input_ctx_batched = tf.expand_dims(input_ctx_batched, -1)  # add T dimension for 3d conv, now 1CHW1
        # Here, in contrast to pc.bitcost(...), q does not need to be padded, as it is part of some context.
        # Logits will be a 1111L vector, i.e., prediction of the next pixel
        q = tf.gather(centers, input_ctx_batched)
        logits = pc.logits(q, is_training=False)
        self.pr = tf.nn.softmax(logits)
        self.freqs = tf.squeeze(tf.cast(self.pr * freqs_resolution, tf.int64))
        self.sess = sess

        self._get_freqs = None

    def pad_symbols_volume(self, symbols):
        assert symbols.ndim == 3
        return pad_for_probclass3d(symbols, self.pc_class.get_context_size(self.config))

    def undo_pad_symbols_volume(self, symbols):
        assert symbols.ndim == 3
        return undo_pad_for_probclass3d(symbols, self.pc_class.get_context_size(self.config))

    def get_pr(self, input_ctx):
        """
        NOTE: input_ctx is expected to be CHW!
        :param input_ctx: symbols as CHW
        :return: pr
        """
        return np.squeeze(self._run(self.pr, input_ctx))

    def get_freqs(self, input_ctx):
        """
        NOTE: input_ctx is expected to be CHW!
        :param input_ctx: symbols as CHW
        :return: freqs, int64
        """
        if not self._get_freqs:  # makes it slightly faster
            self._get_freqs = self.sess.make_callable(self.freqs, feed_list=[self.input_ctx])
        f = self._get_freqs(input_ctx)
        f = np.maximum(f, 1)
        assert np.all(f > 0), 'We do not want zero frequencies!: {}'.format(f)
        return f

    def _run(self, t, input_ctx):
        assert self.sess is not None, 'Set sess before using get_pr or get_freqs'
        assert input_ctx.shape == self.input_ctx.shape, '{} != {}'.format(
                input_ctx.shape, self.input_ctx.shape)
        return self.sess.run(t, feed_dict={self.input_ctx: input_ctx})


def denormalize(x):
    return x*64.15 + 120.707

def clip_to_image_range(x):
        return tf.clip_by_value(x, 0, 255, name='clip')