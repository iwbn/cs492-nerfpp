import tensorflow as tf

"""
batchfy function
"""
def batchfy(fn, inputs, output_signature, chunk, parallel_iterations=10):
    orig_inputs = inputs
    orig_output_signature = output_signature

    is_single_input = False
    if isinstance(inputs, tf.Tensor):
        B = tf.shape(inputs)[0]
        inputs = [inputs]
        is_single_input = True
    else:
        inputs = tf.nest.flatten(inputs)
        B = tf.shape(inputs[0])[0]

    is_single_output = False
    if isinstance(output_signature, tf.DType):
        output_signature = [output_signature]
        is_single_output = True
    else:
        output_signature = tf.nest.flatten(output_signature)

    ta_size = tf.cast(tf.math.ceil(tf.cast(B, tf.float32) / chunk), tf.int32)
    remainder = tf.cond(ta_size > 1, lambda: ta_size * chunk - B, lambda: 0)
    
    inputs_padded = []
    for inp in inputs:
        s = tf.unstack(tf.shape(inp))
        c = tf.pad(inp, [[0,remainder]] + [[0,0]] * (len(s)-1), mode="REFLECT")
        inputs_padded.append(c)

    tas = []
    for dtype in output_signature:
        ta = tf.TensorArray(dtype, size=ta_size, dynamic_size=False, infer_shape=True)
        tas.append(ta)
    i = tf.constant(0)

    def loop_body(i, *tas):
        st = i * chunk
        if is_single_input:
            inp_packed = [inp[st:st+chunk] for inp in inputs_padded]
        else:
            inp_packed = tf.nest.pack_sequence_as(orig_inputs, [inp[st:st+chunk] for inp in inputs_padded])
        outputs = fn(inp_packed)

        if is_single_output:
            outputs = [outputs]
        else:
            outputs = tf.nest.flatten(outputs)

        new_tas = []
        for ta, output in zip(tas, outputs):
            ta = ta.write(i, output)
            new_tas.append(ta)
        out = [i+1] + new_tas
        return tuple(out)

    i, *tas = tf.while_loop(lambda *args: args[0] < ta_size, loop_body, tuple([i] + tas), parallel_iterations=parallel_iterations)
    outputs = [ta.concat()[0:B] for ta in tas]


    if is_single_output:
        return outputs[0]
    else:
        outputs = tf.nest.pack_sequence_as(orig_output_signature, outputs)
        return outputs
        
