from models import Config, BatchSkipped
from basic_model import CTCModel
from quantized_model import QuantizedCTCModel
from utils import load_dataset, split_train_and_val, make_batches, pad_sequences

from time import gmtime, strftime
import json
import math
import numpy as np
import os
import random
import tensorflow as tf
import time

def choose_model(config):
    if config.model == 'basic':
        return CTCModel(config)
    elif config.model == 'quantized':
        return QuantizedCTCModel(config)
    return None

def pad_all_batches(batch_feature_array):
    for batch_num in range(len(batch_feature_array)):
        batch_feature_array[batch_num] = pad_sequences(batch_feature_array[batch_num])[0]
    return batch_feature_array
        
def train_model(config):
    logs_path = "tensorboard/" + strftime("train_%Y_%m_%d_%H_%M_%S", gmtime())

    train_dataset = load_dataset(config.train_path)
    
    train_dataset, val_dataset = split_train_and_val(train_dataset)

    train_feature_minibatches, train_labels_minibatches, train_seqlens_minibatches = make_batches(train_dataset, batch_size=Config.batch_size)
    val_feature_minibatches, val_labels_minibatches, val_seqlens_minibatches = make_batches(val_dataset, batch_size=len(val_dataset[0]))

    train_feature_minibatches = pad_all_batches(train_feature_minibatches)
    val_feature_minibatches = pad_all_batches(val_feature_minibatches)

    num_examples = np.sum([batch.shape[0] for batch in train_feature_minibatches])
    num_batches_per_epoch = int(math.ceil(num_examples / Config.batch_size))

    with tf.Graph().as_default():
        model = choose_model(config)
        init = tf.global_variables_initializer()

        saver = tf.train.Saver(tf.global_variables())

        with tf.Session() as session:
            # Initializate the weights and biases
            session.run(init)
            if config.load_from_file is not None:
                new_saver = tf.train.import_meta_graph('%s.meta'%config.load_from_file, clear_devices=True)
                new_saver.restore(session, config.load_from_file)

            train_writer = tf.summary.FileWriter(logs_path + '/train', session.graph)

            step_ii = 0

            for curr_epoch in range(config.num_epochs):
                total_train_cost = total_train_wer = 0
                start = time.time()

                for batch in random.sample(range(num_batches_per_epoch),num_batches_per_epoch):
                    cur_batch_size = len(train_seqlens_minibatches[batch])

                    try:
                        batch_cost, batch_ler, summary = model.train_on_batch(session, train_feature_minibatches[batch], train_labels_minibatches[batch], train_seqlens_minibatches[batch], train=True)
                    except BatchSkipped:
                        continue

                    total_train_cost += batch_cost * cur_batch_size
                    total_train_wer += batch_ler * cur_batch_size

                    train_writer.add_summary(summary, step_ii)
                    step_ii += 1

                train_cost = total_train_cost / num_examples
                train_wer = total_train_wer / num_examples

                val_batch_cost, val_batch_ler, _ = model.train_on_batch(session, val_feature_minibatches[0], val_labels_minibatches[0], val_seqlens_minibatches[0], train=False)
                
                log = "Epoch {}/{}, train_cost = {:.3f}, train_ed = {:.3f}, val_cost = {:.3f}, val_ed = {:.3f}, time = {:.3f}"
                print(log.format(curr_epoch+1, config.num_epochs, train_cost, train_wer, val_batch_cost, val_batch_ler, time.time() - start))

                # Write out status to JSON for CodaLab table display
                with open('status.json', 'w') as fp:
                    json.dump({
                        'epoch': curr_epoch + 1,
                        'train_cost': float(train_cost),
                        'train_wer': float(train_wer),
                        'val_batch_cost': float(val_batch_cost),
                        'val_batch_ler': float(val_batch_ler),
                    }, fp)

                if config.print_every > 0 and (curr_epoch + 1) % config.print_every == 0:
                    batch_ii = 0
                    model.print_results(session, train_feature_minibatches[batch_ii], train_labels_minibatches[batch_ii], train_seqlens_minibatches[batch_ii])

                if config.save_every > 0 and config.save_to_file and (curr_epoch + 1) % config.save_every == 0:
                    os.makedirs(os.path.dirname(config.save_to_file), exist_ok=True)
                    saver.save(session, config.save_to_file, global_step=curr_epoch + 1)

def test_model(config):
    logs_path = "tensorboard/" + strftime("test_%Y_%m_%d_%H_%M_%S", gmtime())

    test_dataset = load_dataset(config.test_path)
    test_feature_minibatches, test_labels_minibatches, test_seqlens_minibatches = make_batches(test_dataset, batch_size=Config.batch_size)

    test_feature_minibatches = pad_all_batches(test_feature_minibatches)

    num_examples = np.sum([batch.shape[0] for batch in test_feature_minibatches])
    num_batches_per_epoch = int(math.ceil(num_examples / Config.batch_size))
    print(num_examples, num_batches_per_epoch)

    with tf.Graph().as_default():
        model = choose_model(config)
        with tf.Session() as session:
            # Initializate the weights and biases
            
            if config.load_from_file is not None:
                new_saver = tf.train.Saver(tf.global_variables()) # tf.train.import_meta_graph('%s.meta'%config.load_from_file, clear_devices=True)
                new_saver.restore(session, tf.train.latest_checkpoint('models/'))
            else:
                print("No checkpoint to load for test.")
                return
            test_writer = tf.summary.FileWriter(logs_path + '/test', session.graph)

            step_ii = 0
            total_test_cost = total_test_wer = 0
            start = time.time()

            for batch in range(num_batches_per_epoch):
                cur_batch_size = len(test_feature_minibatches[batch])

                try:
                    batch_cost, batch_ler, summary = model.train_on_batch(session, test_feature_minibatches[batch], test_labels_minibatches[batch], test_seqlens_minibatches[batch], train=False)
                except BatchSkipped:
                    continue
                print(batch_cost, batch_ler)
                total_test_cost += batch_cost * cur_batch_size
                total_test_wer += batch_ler * cur_batch_size

                test_writer.add_summary(summary, step_ii)
                step_ii += 1

            batch_ii = 0
            model.print_results(session, test_feature_minibatches[batch_ii], test_labels_minibatches[batch_ii], test_seqlens_minibatches[batch_ii])
            test_cost = total_test_cost / num_examples
            test_wer = total_test_wer / num_examples
            
            log = "test_cost = {:.3f}, test_ed = {:.3f}, time = {:.3f}"
            print(log.format(test_cost, test_wer, time.time() - start))

            # Write out status to JSON for CodaLab table display
            with open('status.json', 'w') as fp:
                json.dump({
                    'test_cost': float(test_cost),
                    'test_ed': float(test_wer),
                }, fp)
    
def main(args):
    config = Config()
    config.save('config.json')
    print(config)
    if config.phase == 'train':
        train_model(config)
    elif config.phase == 'test':
        test_model(config)
    
if __name__ == "__main__":
    tf.app.run()