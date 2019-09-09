import argparse
import os
import time
import numpy as np

from datetime import datetime

import tensorflow as tf

#import inception_resnet_v1
from data_input import inputs_data, inputs_multifile_data
import resnet

HEIGHT = 48
WIDTH = 48
CHANNEL = 3

def run_training(image_path, batch_size, epoch, model_path, log_dir, start_lr):
    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False, name='global_step')
        # Create a session for running operations in the Graph.
        sess = tf.Session()

        # Input images and labels.
        #images, labels = inputs(path=get_files_name(image_path), batch_size=batch_size, num_epochs=epoch)
        #images, labels = inputs(path=image_path, train=True, batch_size=batch_size, num_epochs=epoch)
        #train_images, train_labels = inputs(path=image_path, train=True, batch_size=batch_size, num_epochs=epoch)
        #test_images, test_labels = inputs(path=image_path, train=False, batch_size=batch_size, num_epochs=epoch)
        
        #record_file_names = ['./record_save/train-resave.tfrecords', './record_save/train-flip.tfrecords']
        record_file_names = ['./record_save/train-resize.tfrecords']
        train_images, train_labels = inputs_multifile_data(record_file_names=record_file_names, train=True, batch_size=batch_size, num_epochs=epoch)
        test_images, test_labels = inputs_data(record_file_path=image_path, train=False, batch_size=batch_size, num_epochs=None)
        
        
        images = tf.placeholder(tf.float32, [batch_size, HEIGHT, WIDTH, CHANNEL])
        labels = tf.placeholder(tf.int32, [batch_size])
        
        # train_mode = tf.placeholder(tf.bool)
        # load network

        decay_step = 10000  #10 * 190000 / 128
        hp = resnet.HParams(batch_size=batch_size,
                            num_classes=73,
                            num_residual_units=2,#2
                            k=4,   #caffe output k = 4
                            weight_decay=0.0005,
                            initial_lr=0.001,
                            decay_step=decay_step,
                            decay_rate=0.9,
                            momentum=0.9,
                            drop_prob = 0.5)

        net = resnet.WResNet(hp, images, labels, global_step)#, is_train=True
        net.build_model()
        net.build_train_op()
        
        #net = resnet.WResNet(hp, images, labels, global_step, train_mode)
        #net.network()
        #logits = net._logits
        
        #ans = tf.argmax(tf.nn.softmax(logits),1)
        # Build a Graph that computes predictions from the inference model.

        # Add to the Graph the loss calculation.
        #age_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        #age_loss = tf.reduce_mean(age_cross_entropy) #age_cross_entropy_mean


#        age_ = tf.cast(tf.constant([i for i in range(0, 89)]), tf.float32)
#        age = tf.reduce_sum(tf.multiply(tf.nn.softmax(logits), age_), axis=1)
#        abs_age_error = tf.losses.absolute_difference(labels, age)
#
#
#        tf.summary.scalar("age_cross_entropy", age_loss)
#        tf.summary.scalar("train_abs_age_error", abs_age_error)


        # Add to the Graph operations that train the model.
        
#        lr = tf.train.exponential_decay(start_lr, global_step=global_step, decay_steps=10000, decay_rate=0.9, staircase=True)
#        optimizer = tf.train.AdamOptimizer(lr)
#        tf.summary.scalar("lr", lr)
#        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # update batch normalization layer
#        with tf.control_dependencies(update_ops):
#            train_op = optimizer.minimize(net.loss, global_step)

        # if you want to transfer weight from another model,please comment below codes
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess = tf.Session(config=tf.ConfigProto(
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.45)))
        sess.run(init_op)
        #tf.global_variables_initializer().run()
        #tf.local_variables_initializer().run()


        merged = tf.summary.merge_all()
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        train_writer = tf.summary.FileWriter(log_dir, sess.graph)

        # if you want to transfer weight from another model,please comment below codes
        init_step = 0
        saver = tf.train.Saver(max_to_keep=10000)
        ckpt = tf.train.get_checkpoint_state(model_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            init_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
            print('init_step: ',init_step)
            print("restore and continue training!")
        else:
            print('No checkpoint file found. No old saved network, start from the scratch.')
        # if you want to transfer weight from another model, please comment above codes


        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            
        #Training
        test_best_acc = 0.0
        train_best_acc = 0.0
        test_interval = 1000
        max_steps = 1000000
        test_good_step = 0
        train_good_step = 0
          
        
        for step in range(init_step, max_steps):
            # Test
            if step % test_interval == 0:
                test_loss, test_acc = 0.0, 0.0
                for i in range(test_interval):
                    test_images_val, test_labels_val = sess.run([test_images, test_labels])
                    loss_value, acc_value = sess.run([net.loss, net.acc],
                                feed_dict={images:test_images_val, labels:test_labels_val, net.is_train:False})
                    test_loss += loss_value
                    test_acc += acc_value
                test_loss /= test_interval
                test_acc /= test_interval
                #test_best_acc = max(test_best_acc, test_acc)
                
                if test_best_acc < test_acc:
                    test_best_acc = test_acc
                    test_good_step = step
                print('!!!!!! test_good_step: ', test_good_step)
                format_str = ('%s: (Test)     step %d, loss=%.4f, acc=%.4f ')
                print(format_str % (datetime.now(), step, test_loss, test_acc))

                test_summary = tf.Summary()
                test_summary.value.add(tag='test/loss', simple_value=test_loss)
                test_summary.value.add(tag='test/acc', simple_value=test_acc)
                test_summary.value.add(tag='test/best_acc', simple_value=test_best_acc)
                train_writer.add_summary(test_summary, step)

                train_writer.flush()

            # Train
            start_time = time.time()
            train_images_val, train_labels_val = sess.run([train_images, train_labels])
            _, lr_value, train_loss, train_acc, train_summary_str = sess.run([net.train_op, net.lr, net.loss, net.acc, merged],
                                             feed_dict={images:train_images_val, labels:train_labels_val, net.is_train:True})#net.train_op
            duration = time.time() - start_time

            assert not np.isnan(train_loss)
            
            # Display & Summary(training)
            display = 100
            batch_size = 128
            if step % display == 0:
                num_examples_per_step = batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)
                format_str = ('%s: (Training) step %d, loss=%.4f, acc=%.4f, lr=%f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print(format_str % (datetime.now(), step, train_loss, train_acc, lr_value,
                                     examples_per_sec, sec_per_batch))
                print(sess.run([net.train_op, net.preds],feed_dict={images:train_images_val, labels:train_labels_val, net.is_train:True}))
                train_writer.add_summary(train_summary_str, step)
                
                if (train_best_acc <= train_acc)  and (train_best_acc > 0.95):
                    train_best_acc = train_acc
                    train_good_step = step
                    print('!!!!!! train_good_step: ', train_good_step)

            # Save the model checkpoint periodically.
            #if (step > init_step and step % display == 0) or (step + 1) == max_steps:
            if (step % display == 0) or (step + 1) == max_steps:
                if (step == test_good_step):
                    checkpoint_path_test = os.path.join(model_path+'/test_good', 'model_age.ckpt')
                    saver.save(sess, checkpoint_path_test, global_step=step)
                else:
                    checkpoint_path = os.path.join(model_path, 'model_age.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)
            
            if(step == train_good_step) and (train_best_acc > 0.95):
                checkpoint_path = os.path.join(model_path, 'model_age.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
         # Wait for threads to finish.
        coord.join(threads)
        sess.close()

        
###################################################################################################################        
        
        
#        try:
#            step = sess.run(global_step)
#            start_time = time.time()
#            while not coord.should_stop():
#                # start_time = time.time()
#                # Run one step of the model.  The return values are
#                # the activations from the `train_op` (which is
#                # discarded) and the `loss` op.  To inspect the values
#                # of your ops or variables, you may include them in
#                # the list passed to sess.run() and the value tensors
#                # will be returned in the tuple from the call.
#                _, summary = sess.run([train_op, merged])#, {net.is_train: True}
#                train_writer.add_summary(summary, step)
#                # duration = time.time() - start_time
#                # # Print an overview fairly often.
#                if step % 100 == 0:
#                    duration = time.time() - start_time
#                    print('Step = %d: (%.3f sec)' % (step, duration))
#                    print(sess.run([train_op, net.loss, abs_age_error,ans,labels]))#, {net.is_train: True}
#                    #print('%.3f sec' % duration)
#                    start_time = time.time()
#                if step % 100 == 0:
#                    save_path = new_saver.save(sess, os.path.join(model_path, "model.ckpt"), global_step=step)
#                    print("Model saved in file: %s" % save_path)
#                step = sess.run(global_step)
#        except tf.errors.OutOfRangeError:
#            print('Done training for %d epochs, %d steps.' % (epoch, step))
#        finally:
#            # When done, ask the threads to stop.
#            save_path = new_saver.save(sess, os.path.join(model_path, "model.ckpt"), global_step=global_step)
#            print("Model saved in file: %s" % save_path)
#            coord.request_stop()
#        # Wait for threads to finish.
#        coord.join(threads)
#        sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_learning_rate", "--lr", type=float, default=0.1, help="Init learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0005, help="Set 0 to disable weight decay")
    parser.add_argument("--model_path", type=str, default="./saved_checkpoint", help="Path to save models")
    parser.add_argument("--log_path", type=str, default="./train_log", help="Path to save logs")
    parser.add_argument("--epoch", type=int, default=100, help="Epoch")
    parser.add_argument("--images", type=str, default="./record_save", help="Path of tfrecords")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    #parser.add_argument("--keep_prob", type=float, default=0.8, help="Used by dropout")
    parser.add_argument("--cuda", default=False, action="store_true",
                        help="Set this flag will use cuda when testing.")
    args = parser.parse_args()
    if not args.cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    run_training(image_path=args.images, batch_size=args.batch_size, epoch=args.epoch, model_path=args.model_path,
                 log_dir=args.log_path, start_lr=args.init_learning_rate)
