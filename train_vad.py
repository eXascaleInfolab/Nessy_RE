import argparse
from bunch import Bunch
import json
import numpy as np
import os
import random

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

from utils.data_loader import DataLoader, load_json, load_hidden
from vad_model.multi_text_vad_transfer import MultiTextVADTransfer
from vad_model.multi_text_vad import MultiTextVAD
from utils import constant, helper, scorer
from utils.vocab import Vocab
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve


def get_dims(params, vocab_size, num_classes):
    p_theta_dims = params['p_theta_dims_1'] + [vocab_size]
    p_gamma_dims = [p_theta_dims[0], num_classes]
    if params['model'] == "MultiTextVAD":
        q_psi_dims = [constant.MAX_LEN]
        q_phi_dims = [constant.MAX_LEN]
    else:
        q_psi_dims = [1]
        q_phi_dims = [1]
    q_psi_dims.extend(params['q_psi_dims_1'])
    q_psi_dims.append(num_classes)
    q_phi_dims.extend(p_theta_dims[::-1][1:])
    return p_theta_dims, p_gamma_dims, q_phi_dims, q_psi_dims


def calc_best_accuracy(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred[:, 1])
    best_acc = 0
    best_t = 0
    for t in thresholds:
        acc = accuracy_score(y_true, y_pred[:, 1] >= t)
        if acc > best_acc:
            best_acc = acc
            best_t = t
    print(f"Best accuracy: {best_acc:.6} with a threshold {best_t:.3}") 
    return best_acc, best_t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, help="Path to the dataset")
    parser.add_argument("dataset_name", type=str, help="TITLE, TOP_MEMBERS or EMPLOYEE")
    parser.add_argument("config", type=str, help="Path to the config")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--prior", type=str, default="uniform")
    parser.add_argument("--rules", type=str, default="")

    parser.add_argument('--emb_dim', type=int, default=300, help='Word embedding dimension.')
    parser.add_argument('--ner_dim', type=int, default=30, help='NER embedding dimension.')
    parser.add_argument('--pos_dim', type=int, default=30, help='POS embedding dimension.')
    parser.add_argument('--pe_dim', type=int, default=30, help='Position encoding dimension.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Input and RNN dropout rate.')
    parser.add_argument('--word_dropout', type=float, default=0.,
                        help='The rate at which randomly set a word to UNK.')
    parser.add_argument('--lower', dest='lower', action='store_true', help='Lowercase all words.')
    parser.add_argument('--no-lower', dest='lower', action='store_false')
    parser.set_defaults(lower=False)

    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--print_every", type=int, default=100, help="Print log every k steps.")
    parser.add_argument("--model_name", type=str, default="VAD", help="Model name")

    parser.add_argument("--seed", type=int, default=98765)

    args = parser.parse_args()
    random.seed(31)
    np.random.seed(args.seed)
    opt = vars(args)
    with open(args.config) as f:
        config = Bunch(json.load(f))
    vocab_file = os.path.join(opt['data_dir'], "vocab", 'vocab.pkl')
    vocab = Vocab(vocab_file, load=True)
    emb_file = os.path.join(opt['data_dir'], "vocab", 'embedding.npy')
    emb_matrix = np.load(emb_file)
    assert emb_matrix.shape[0] == vocab.size
    print("Loading data from {} with batch size {}...".format(
        opt['data_dir'], opt['batch_size']))

    if args.dataset_name == "TITLE":
        label2id = constant.TITLE_LABEL_TO_ID
        all_rules = constant.TITLE_RULES
    elif args.dataset_name == "TOP_MEMBERS":
        label2id = constant.TOP_MEMBERS_LABEL_TO_ID
        all_rules = constant.TOP_MEMBERS_RULES
    elif args.dataset_name == "EMPLOYEE":
        label2id = constant.EMPLOYEE_LABEL_TO_ID
        all_rules = constant.EMPLOYEE_RULES
    else:
        raise FileNotFoundError

    opt['num_classes'] = len(label2id)
    rules_list = []
    if opt['rules']:
        for rule in opt['rules'].split(';'):
            if rule in all_rules:
                rules_list.append(rule)
            else:
                raise AttributeError(f"Rule {rule} does not exist for dataset {args.dataset_name}")

    train_data = load_json(opt['data_dir'] + '/train.json')
    hidden_repr = load_hidden(opt['data_dir'], config.vad_model['model'], 'train')
    if hidden_repr is not None:
        opt['pretrained_size'] = hidden_repr.shape[1]
    train_batch = DataLoader(
        train_data, opt['batch_size'], opt, vocab, label2id, hidden_repr, all_rules, evaluation=False, rules=rules_list)
    dev_data = load_json(opt['data_dir'] + '/dev.json')
    hidden_repr = load_hidden(opt['data_dir'], config.vad_model['model'], 'dev')
    dev_batch = DataLoader(
        dev_data, opt['batch_size'], opt, vocab, label2id, hidden_repr, all_rules, evaluation=True)
    # print(len(train_batch), len(dev_batch))

    id2label = dict([(v, k) for k, v in train_batch.label2id.items()])
    if args.prior == "uniform":
        prior = np.array([1 / opt['num_classes'] for _ in range(opt['num_classes'])])
    elif args.prior == "labels":
        prior = train_batch.prior
    else:
        raise NotImplementedError
    # print("Prior:", prior)
    p_theta_dims, p_gamma_dims, q_phi_dims, q_psi_dims = \
        get_dims(config.vad_model, vocab.size, opt['num_classes'])
    opt['p_theta'] = p_theta_dims
    opt['p_gamma'] = p_gamma_dims
    opt['q_phi'] = q_phi_dims
    opt['q_psi'] = q_psi_dims
    opt['lr'] = config.vad_model['lr']
    opt['beta'] = config.vad_model['beta']
    model_dir = opt['model_name'] + \
                     " Q_psi {}".format("-".join([str(d) for d in q_psi_dims])) + \
                     " P_theta {}".format("-".join([str(d) for d in p_theta_dims])) + \
                     " A {}".format(config.vad_model['alpha']) + \
                     " B {}".format(config.vad_model['beta'])
    checkpoint_dir = os.path.join("checkpoint", opt['log_dir'], model_dir)
    model_dir = os.path.join(opt['log_dir'], model_dir)
    helper.ensure_dir(model_dir, verbose=True)
    summary_dir = os.path.join(model_dir, "summary")
    helper.ensure_dir(checkpoint_dir, verbose=True)
    opt['checkpoint_dir'] = checkpoint_dir
    helper.save_config(opt, os.path.join(model_dir, "config.json"), verbose=True)

    # Train
    tf.compat.v1.reset_default_graph()
    with tf.Graph().as_default():
        tf.compat.v1.set_random_seed(args.seed)
        session_config = tf.compat.v1.ConfigProto(
            log_device_placement=False, allow_soft_placement=False)
        with tf.compat.v1.Session(config=session_config) as sess:
            summary_writer = tf.compat.v1.summary.FileWriter(
                summary_dir, graph=tf.compat.v1.get_default_graph())
            if config.vad_model['model'] == "MultiTextVADTransfer":
                vad_model = MultiTextVADTransfer(opt,
                                                 device=config.device,
                                                 alpha=config.vad_model['alpha'],
                                                 beta=config.vad_model['beta'],
                                                 lam=1e-2)
            elif config.vad_model['model'] == "MultiTextVAD":
                vae_params = {"embedding/E": emb_matrix}
                vad_model = MultiTextVAD(opt,
                                         device=config.device,
                                         vae_params=vae_params,
                                         alpha=config.vad_model['alpha'],
                                         beta=config.vad_model['beta'],
                                         lam=1e-2)
            vad_saver, sampled_z, q_y, gamma_y, loss_var, train_op_var, merged_var = \
                vad_model.build_graph()
            sess.run(tf.compat.v1.global_variables_initializer())
            global_step = 0
            p_dev_var, r_dev_var, f1_dev_var = tf.Variable(0.0), tf.Variable(0.0), tf.Variable(0.0)
            rocauc_dev_var, acc_dev_var = tf.Variable(0.0), tf.Variable(0.0)
            loss_dev_var = tf.Variable(0.0)
            precision_summary = tf.compat.v1.summary.scalar("micro_precision_dev", p_dev_var)
            recall_summary = tf.compat.v1.summary.scalar("micro_recall_dev", r_dev_var)
            f1_summary = tf.compat.v1.summary.scalar("micro_f1_dev", f1_dev_var)
            loss_summary = tf.compat.v1.summary.scalar("loss_dev", loss_dev_var)
            roc_auc_summary = tf.compat.v1.summary.scalar("rocauc_dev", rocauc_dev_var)
            acc_summary = tf.compat.v1.summary.scalar("acc_dev", acc_dev_var)
            merged_dev_var = tf.compat.v1.summary.merge(
                [precision_summary, recall_summary, f1_summary, loss_summary, roc_auc_summary,
                    acc_summary])

            best_rocauc, best_acc = 0, 0
            for epoch in range(1, opt['num_epochs'] + 1):
                print("Starting epoch {}".format(epoch))
                epoch_loss = 0
                for i, batch in enumerate(train_batch):
                    if i > 200:
                        break
                    feed_dict = {
                        vad_model.input_ph: batch[0],
                        vad_model.weak_labels: batch[-1],
                        vad_model.is_training_ph: 1,
                        vad_model.prior: batch[-3],
                        vad_model.keep_prob_ph: opt['dropout']
                    }
                    if config.vad_model['model'] == "MultiTextVAD":
                        feed_dict[vad_model.subj_pos] = np.array(batch[2])
                        feed_dict[vad_model.obj_pos] = np.array(batch[3])
                    if config.vad_model['model'] == "MultiTextVADTransfer":
                        # Dependency path
                        feed_dict[vad_model.input_ph] = batch[1]
                        # Sentence embeddings
                        feed_dict[vad_model.input_emb] = batch[-2]
                    _, batch_loss, q_probs, gamma_probs = sess.run(
                        [train_op_var, loss_var, q_y, gamma_y],
                        feed_dict=feed_dict)
                    epoch_loss += batch_loss
                    if global_step % opt['print_every'] == 0:
                        # print(f"Step {global_step}, Loss {batch_loss:.6}")
                        summary_train = sess.run(merged_var, feed_dict=feed_dict)
                        summary_writer.add_summary(summary_train, global_step=global_step)
                    global_step += 1
                print("Evaluating on dev set")
                dev_loss: float = 0
                q_all_probs = np.empty((0, opt['num_classes']))
                gamma_all_probs = np.empty((0, opt['num_classes']))
                dev_sampled_z = np.empty((0, q_phi_dims[-1]))

                # Evaluation on the dev set
                y_true = []
                for i, batch in enumerate(dev_batch):
                    feed_dict = {
                        vad_model.input_ph: np.array(batch[0]),
                        vad_model.weak_labels: np.array(batch[-1]),
                        vad_model.prior: batch[-3],
                        vad_model.is_training_ph: 0
                    }
                    y_true.extend(batch[-1][:, 1])
                    if config.vad_model['model'] == "MultiTextVAD":
                        feed_dict[vad_model.subj_pos] = np.array(batch[2])
                        feed_dict[vad_model.obj_pos] = np.array(batch[3])
                    if config.vad_model['model'] == "MultiTextVADTransfer":
                        # Dependency path
                        # feed_dict[vad_model.input_ph] = batch[1]
                        # Sentence embeddings
                        feed_dict[vad_model.input_emb] = batch[-2]
                    batch_loss, gamma_probs, q_probs, batch_sampled_z = sess.run(
                        [loss_var, gamma_y, q_y, sampled_z], feed_dict=feed_dict)
                    dev_loss += batch_loss

                    q_all_probs = np.append(q_all_probs, q_probs, axis=0)
                    gamma_all_probs = np.append(gamma_all_probs, gamma_probs, axis=0)
                    dev_sampled_z = np.append(dev_sampled_z, batch_sampled_z, axis=0)
                
                # true_labels = np.array(dev_batch.labels)
                # y_true = np.array(true_labels == id2label[1], dtype=np.int)
                dev_rocauc = roc_auc_score(y_true, q_all_probs[:, 1])
                if dev_rocauc < 0.5:
                    q_all_probs = 1 - q_all_probs
                    dev_rocauc = roc_auc_score(y_true, q_all_probs[:, 1])
                dev_acc, threshold = calc_best_accuracy(y_true, q_all_probs)
                q_predictions = np.array(q_all_probs[:, 1] >= threshold, dtype=np.int)
                gamma_predictions = np.argmax(gamma_all_probs, axis=1)
                print("Psi predictions: {}".format(np.bincount(q_predictions)))
                print("Gamma predictions: {}".format(np.bincount(gamma_predictions)))
                rel_predictions = [id2label[p] for p in q_predictions]
                true_labels = [id2label[p] for p in y_true]
                dev_p, dev_r, dev_f1 = scorer.score(true_labels, rel_predictions)
                if dev_rocauc > best_rocauc:
                    best_rocauc = dev_rocauc
                    vad_saver.save(sess, '{}/model'.format(checkpoint_dir))
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    vad_saver.save(sess, '{}/acc_model'.format(checkpoint_dir))
                print(f"Epoch {epoch}, Dev Loss: {dev_loss:.6}, ROC AUC score: {dev_rocauc:.6}, Accuracy score: {dev_acc:.6}")
                merged_dev = sess.run(merged_dev_var,
                                      feed_dict={
                                          p_dev_var: dev_p,
                                          r_dev_var: dev_r,
                                          f1_dev_var: dev_f1,
                                          loss_dev_var: dev_loss,
                                          rocauc_dev_var: dev_rocauc,
                                          acc_dev_var: dev_acc
                                      })
                summary_writer.add_summary(merged_dev, epoch)
            print(f"=== Best ROC AUC: {best_rocauc:.6}; Best accuracy: {best_acc:.6} see {summary_dir} ===")


if __name__ == "__main__":
    main()
