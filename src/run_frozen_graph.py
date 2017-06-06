import tensorflow as tf
import argparse
from utils import load_dataset, pad_sequences, compute_wer
from data import construct_char_string_to_index_mapping as construct_string_to_index_mapping
import editdistance

def pad_all_batches(batch_feature_array):
    for batch_num in range(len(batch_feature_array)):
        batch_feature_array[batch_num] = pad_sequences(batch_feature_array[batch_num])[0]
    return batch_feature_array

def print_predicted(pred):
    inv_index_mapping = {v: k for k, v in
                         construct_string_to_index_mapping().items()}
    predicted_label = "".join(
            [inv_index_mapping[ch] for ch in pred if ch != -1])

    print("Predicted: {}\n".format(predicted_label))
                                                      
def load_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def, 
            input_map=None, 
            return_elements=None, 
            name="prefix", 
            op_dict=None, 
            producer_op_list=None
        )
    return graph

def perform_inference(session, feature_input, seq_len):
    return session.run(decode_sequence, feed_dict={
            inputs: feature_input,
            seq_lens : seq_len
        })

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="models/frozen_model.pb", type=str)
    parser.add_argument("--path", default='./data/train/train.dat', type=str)
    parser.add_argument("--print_every", default=500, type=int)
    args = parser.parse_args()

    graph = load_graph(args.model_name)
    inputs = graph.get_tensor_by_name('prefix/InputNode:0')
    seq_lens = graph.get_tensor_by_name('prefix/SeqLenNode:0')
    decode_sequence = graph.get_tensor_by_name('prefix/DecodedSequence:0')
    
    features, labels, seqlens = load_dataset(args.path)
    total_test_wer = 0

    with tf.Session(graph=graph) as session:
        for i in range(len(features)):
            output_phones = perform_inference(session, [features[i]], [seqlens[i]])
            ed = editdistance.eval(output_phones, labels[i]) / len(labels[i])
            total_test_wer += ed
            if i % args.print_every == 0:
                print_predicted(output_phones)
                print_predicted(labels[i])
    wer = total_test_wer / float(len(features))
    print(wer)