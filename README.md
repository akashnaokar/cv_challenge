# Whats about?
The challenge aims to get you close to an interesting cv task: Object detection and classification.
<h3>Note: All code is to be implemented in jupyter notebook file(s) and should contain comments and print-outs to ensure a reviewer is able to understand what you did and why.</h3>

## Dataset
Download the dataset from kaggle at: https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database <br>
The dataset is already divided in train and test set. At the later model section you have to use these respective for training and test/evaluation.

## Task 1 - Preparing the data in the requested format
Your first task is to transfer the downloaded dataset to a <code>tf.data.TFRecordDataset</code> dataset (see: https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset)

For the tfrecord.example use the following feature definition:
````
feature = {
        "image/width": tf.io.FixedLenFeature([], tf.int64),
        "image/height": tf.io.FixedLenFeature([], tf.int64),
        'image/filename': tf.io.FixedLenFeature([], tf.string),
        'image/source_id': tf.io.FixedLenFeature([], tf.string),
        'image/encodedrawdata': tf.io.FixedLenFeature([], tf.string),
        'image/format': tf.io.FixedLenFeature([], tf.string),
        'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
        'image/object/class/text': tf.io.VarLenFeature(tf.string),
        'image/object/class/label': tf.io.VarLenFeature(tf.int64),
        'image/object/class/single': tf.io.FixedLenFeature([], tf.int64),
        'image/object/difficult': tf.io.VarLenFeature(tf.int64),
        'image/object/truncated': tf.io.VarLenFeature(tf.int64),
        'image/object/view': tf.io.VarLenFeature(tf.string)
    }
````
Where the field `image/encodedrawdata` contains the converted np.ndarray image representation of a image file.

The result should be two folders: `train` and `test` holding n-tfrecord files, one for each sample.

# Tasks
You have to fullfill only one of the two tasks below - but of course you can work on both :-)

## Task 2 - Classification
Using `tensorflow (Version: 2.15) with highlevel api Keras` you create a model to perform a image classification (ignoring bboxes for now, just classify the target label to the hole image).
For the classification you will use a `Xception` `DNN` pretrained on the `imagenet` dataset and change the head to be fine-tuned on the data.
Keras helps you to get started with the `Xception` model.
```
base_model = tf.keras.applications.Xception(input_shape=(*[image_height,
                                                           image_width ],
                                                           n_color_channels),
                                            include_top=False,
                                            weights="imagenet")
```
Note that `image_height, image_width and n_color_channels` correspond to the shape dimensions of the image data of your dataset from *#1*.

The pre-trained base-model is to be extended with the following layer structure (ordered as the layers are to be appended to the base model):
<ol>
  <li><code>BatchNormalization</code></li>
  <li><code>GlobalAveragePooling2D</code></li>
  <li><code>Dense Layer with 8 units and relu activation</code></li>
  <li><code>40% dropout</code></li>
  <li><code>Dense Layer with n_classes as units and relu activation</code></li>
</ol>
Compile the model with the <code>SparseCategoricalCrossentropy</code> with <code>from_logits=True</code>.
For the remaining hyperparameters (optimizers, learning_rate,...) you are free to select/choose what you find suitable.

Train and test the model on the train/test dataset(s) from *#1*.<br><br>
<h4>Note: It does not matter how "good" your model tackles the task - but your validation accuracy should be >60% to ensure that the pipeline runs as expected.</h4>

## Task 3 - Object detection
Using `tensorflow (Version: 2.15) with highlevel api Keras` you create a model to perform an object detection on the dataset from *#1*.
For this task, you are completely free which model architecture you choose - but note that we might ask you why you choose the decision as you did.
<strong>Again: It does not matter how "good" your model tackles the task - but your validation accuracy should be >60% to ensure that the pipeline runs as expected.</strong>

# Submission
You submit your challenge result by sending us a link to a github repo, containing all your code (as the requested jupyter notebook files!). Send the link to your HR contact person. Note: To verify that section 1 was done as expected, please include only ONE dataset sample/files in the repo but not the whole dataset. Ensure that your code is well commented and formatted. 