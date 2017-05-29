from basic_model import *
from utils import *

def choose_model(config):
    if config.model == 'basic':
        return CTCModel(config)
    elif config.model == 'quantized':
        pass
    return None

def run_model():
    config = Config()
    config.save('config.json')
    print(config)

    logs_path = "tensorboard/" + strftime("%Y_%m_%d_%H_%M_%S", gmtime())

    train_dataset = load_dataset(config.train_path)
    test_dataset = load_dataset(config.test_path)
    
    train_dataset, val_dataset = split_train_and_val(train_dataset)

    train_feature_minibatches, train_labels_minibatches, train_seqlens_minibatches = make_batches(train_dataset, batch_size=Config.batch_size)
    val_feature_minibatches, val_labels_minibatches, val_seqlens_minibatches = make_batches(val_dataset, batch_size=len(val_dataset[0]))

    def pad_all_batches(batch_feature_array):
        for batch_num in range(len(batch_feature_array)):
            batch_feature_array[batch_num] = pad_sequences(batch_feature_array[batch_num])[0]
        return batch_feature_array

    train_feature_minibatches = pad_all_batches(train_feature_minibatches)
    val_feature_minibatches = pad_all_batches(val_feature_minibatches)

    num_examples = np.sum([batch.shape[0] for batch in train_feature_minibatches])
    num_batches_per_epoch = int(math.ceil(num_examples / Config.batch_size))

    with tf.Graph().as_default():
        model = choose_model(config)
        init = tf.global_variables_initializer()

        saver = tf.train.Saver(tf.trainable_variables())

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

def main(_):
    run_model()
    
if __name__ == "__main__":
    tf.app.run()