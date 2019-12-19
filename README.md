# Road Segmentation

This project consists of identifying roads given aerial satellite images as an
input. We participate to an ML challenge with the final submission of our code which can be accessed with this [link](https://www.aicrowd.com/challenges/epfl-ml-road-segmentation-2019). Our method uses a Deep Convolutional Neural Network called
[U-Net](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)
developed at the University of Freiburg for image segmentation.
For the implementation we use an external Python library called [Keras](https://keras.io/) which is a
high level user friendly API capable of running on top of [TensorFlow](https://www.tensorflow.org/). This
library is useful to run on either CPU's or GPU's, the code being adapted for one or the other very easily.

### Installation
The installation for TensorFlow and Keras can be
done through ```pip install tensorflow``` ```pip install keras``` under MacOS, Linux and Anaconda on Windows.
Our submission file can be run with Python 3.7, after the external installations mentioned above and the installations of the standard python libraries ```numpy```, ```pandas``` and ```matplotlib```.

### File description
* ``` run.py ```: File loading the saved model and creating the submission file.
* ```u_net_model.h5```: File where the U-Net model is stored.
* ```segment_aerial_images.ipynb```: implementation of logistic regression model (given by the project)
* ```tf_aerial_images.py```: implementation of a CNN with two convolutional+pooling layers with a soft-max loss (given by the project)

### How to run our code

The Python code explaining how we generated the model of our final submission is on the file ```u_net_model.py```.  We recommend the code to be run on GPU's since it is computationally expensive on a CPU architecture.

The model of our final submission which has an F1 score of **0.906** on the platform can be downloaded through the following link: https://drive.google.com/file/d/1zubq9x5m0TZ0Fe2o6_SlyNfhEQVQQIXq/view?usp=sharingand and must be placed on the same folder as ```run.py```.

To generate the submission file, ```run.py``` has to be run: it takes the aforementioned model (has to be dowloaded due to space requirements) and creates the submission file called ```submission.csv```.
