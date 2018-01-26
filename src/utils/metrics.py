import tensorflow as tf


def masked_sigmoid_binary_cross_entropy(preds, labels, mask, wce, multilabel):
    """Softmax cross-entropy loss with masking."""

    if multilabel:
        label_sigmoid = tf.nn.sigmoid(preds)
        loss = -tf.reduce_sum((labels * tf.log(label_sigmoid + 1e-10) + (1-labels) * tf.log((1-label_sigmoid) + 1e-10)) * wce, 1)
    else:
        label_sigmoid = tf.nn.softmax(preds)
        loss = -tf.reduce_sum((labels * tf.log(label_sigmoid + 1e-10)) * wce, 1)

    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def sigmoid_binary_cross_entropy(preds, labels, wce, multilabel):
    if multilabel:
        label_sigmoid = tf.nn.sigmoid(preds)
        loss = -tf.reduce_sum((labels * tf.log(label_sigmoid + 1e-10) + (1-labels) * tf.log((1-label_sigmoid) + 1e-10)) * wce, 1)
    else:
        label_sigmoid = tf.nn.softmax(preds)
        loss = -tf.reduce_sum((labels * tf.log(label_sigmoid + 1e-10)) * wce, 1)
    return tf.reduce_mean(loss)


def mc_accuracy(preds, labels):
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def ml_accuracy(preds, labels):
    correct_prediction = tf.equal(tf.round(tf.nn.sigmoid(preds)), tf.round(labels))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def true_positives(preds, labels):
    temp = tf.logical_and(tf.equal(labels, tf.ones_like(preds)), tf.equal(preds, tf.ones_like(preds)))
    return tf.reduce_sum(tf.cast(temp, dtype=tf.float32))


def compute_f1(preds, labels, n_labels, f1_type='micro'):
    if f1_type == 'micro':
        axis = None
    else:
        axis = 0
    tp = tf.count_nonzero(preds * labels, axis=axis, dtype=tf.float32)
    # tn = tf.count_nonzero((predictions - 1) * (truth - 1), axjs=axis, dtype=tf.float32)
    fp = tf.count_nonzero(preds * (labels - 1), axis=axis, dtype=tf.float32)
    fn = tf.count_nonzero((preds - 1) * labels, axis=axis, dtype=tf.float32)

    num = 2 * tp
    den = num + fp + fn + tf.constant(1e-10)
    f1 = num / den
    if f1_type == 'micro':
        f1 = f1
    else:
        f1 = tf.reduce_mean(f1)
    return f1


def compute_accuracy(preds, labels, multilabel):

    if multilabel:
        preds = tf.round(preds)
        preds = tf.cast(preds, dtype=tf.bool)
        labels = tf.cast(labels, dtype=tf.bool)
        intersect = tf.count_nonzero(tf.logical_and(preds, labels), axis=1, dtype=tf.float32)
        union = tf.count_nonzero(tf.logical_or(preds, labels), axis=1, dtype=tf.float32) + tf.constant(1e-5)
        accuracy = tf.reduce_mean(intersect / union, axis=0)
    else:
        correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return accuracy


def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)


def masked_bae(preds, labels, mask, n_labels):
    labels = tf.multiply(tf.cast(tf.expand_dims(mask, dim=1), tf.float32), labels)
    preds = tf.one_hot(tf.argmax(preds, axis=1), on_value=1.0, depth=n_labels, dtype=tf.float32)
    abs_error = tf.multiply((1 - preds), labels)
    freq = tf.reduce_sum(labels, axis=0) + 1e-15
    bae = tf.reduce_sum(tf.cast(tf.reduce_sum(abs_error, axis=0), tf.float32) / freq)/n_labels
    return bae


def get_bae(preds, labels, n_labels):
    preds = tf.one_hot(tf.argmax(preds, axis=1), on_value=1.0, depth=n_labels, dtype=tf.float32)
    abs_error = tf.multiply((1 - preds), labels)
    freq = tf.reduce_sum(labels, axis=0) + 1e-15
    bae = tf.reduce_sum(tf.cast(tf.reduce_sum(abs_error, axis=0), tf.float32) / freq)/n_labels
    return bae


def evaluate(predictions, labels, multi_label=False):
    from sklearn.metrics import coverage_error
    from sklearn.metrics import label_ranking_loss
    from sklearn.metrics import hamming_loss
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
    import numpy as np

    #predictions are logits here and binarized labels
    multi_label = False
    if np.sum(labels) > np.shape(labels)[0]:
        multi_label = True

    # n_ids, n_labels = np.shape(labels)
    assert predictions.shape == labels.shape, "Shapes: %s, %s" % (predictions.shape, labels.shape,)
    metrics = dict()

    # metrics['cross_entropy'] = 0  # -np.mean(labels * np.log(predictions + 1e-10))
    # metrics['accuracy'] = accuracy_score(np.argmax(labels, axis=1), np.argmax(predictions, axis=1))
    for i in range(predictions.shape[0]):
        k = np.sum(labels[i])
        pos = predictions[i].argsort()
        predictions[i].fill(0)
        predictions[i][pos[-int(k):]] = 1
    # predictions = np.round(predictions)

    # metrics['bae'] = 0
    # metrics['average_precision'] = 0#label_ranking_average_precision_score(labels, predictions)
    # metrics['pak'] = 0#patk(predictions, labels)

    # metrics['coverage'] = coverage_error(labels, predictions)
    # metrics['ranking_loss'] = label_ranking_loss(labels, predictions)
    # metrics['hamming_loss'] = hamming_loss(labels, predictions)
    # # metrics['micro_precision'], metrics['micro_recall'], metrics['micro_f1'], _ = precision_recall_fscore_support(labels, predictions, average='micro')
    # metrics['macro_precision'], metrics['macro_recall'], metrics['macro_f1'], _ = precision_recall_fscore_support(labels, predictions, average='macro')

    metrics['micro_f1'] = f1_score(labels, predictions, average='micro')
    metrics['macro_f1'] = f1_score(labels, predictions, average='macro')

    return predictions, metrics

