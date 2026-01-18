Data preprocessing

The images were scaled down to a size of 224×224 pixels by utilizing bilinear interpolation. This process of standardization is essential to conform to the input size requirements of the model. The pixel values were scaled to the range between 0 and 1 and transformed into PyTorch tensors using the ToTensor() method. This process guarantees alignment with PyTorch computational architecture and enhances efficient GPU processing. The images were divided into separate training and validation sets, maintaining a rough split of 70% for training and 30% for validation. This division reduces the risk of data leakage and allows for an impartial assessment of the model performance throughout the training process. The classes designated as "Flooding" and "Normal" were determined from the dataset's directory organization utilizing PyTorch ImageFolder utility. These classifications were recorded in a classes.txt file to maintain reproducibility and facilitate deployment.


Utilization of checkpoint and early stopping

Employing checkpoint and early stopping strategies can significantly enhance the training process, mitigate overfitting, and lead to the creation of high-performing deep learning models while minimizing time investment. These methods involve saving the model's weights at intervals and stopping the training, which results in considerable savings in both computational power and time. Additionally, they effectively curb overfitting by retaining only the optimal weights, ensuring the model does not learn excessively through prolonged training epochs. As a result, there is no longer a need to manually choose a potentially arbitrary number of epochs, which could risk either underfitting or overfitting the model.

The validation loss is tracked after every epoch, and the model weights are saved to “best.pt” only when a new lowest validation loss is achieved. This approach guarantees the preservation of the most generalizable model, even if later epochs show a decline in performance due to overfitting. Additionally, the latest model state is periodically saved to “last.pt”, which serves as a backup option to continue training in the event of interruptions. This checkpoint enables future updates for recovering interrupted training. The “best.pt” checkpoint signifies the model with the best validation performance, making it ideal for direct integration into the flood detection system frontend interface.

Early stopping is utilized as a regularization method to end the training process when the model performance on the validation set stops increasing. A counter is increased each time there is no reduction in validation loss, and if this counter surpasses a preset limit which referred to as patience, the training is halted. This approach helps to prevent the model from overfitting to the training data by stopping further epochs once convergence on the validation set has been reached. It efficiently saves computational resources and ensures that the model does not continue to pick up noise found in the training data.

Hyperparameter tuning refers to the procedure of identifying the best values for a model's hyperparameters in deep learning. It entails methodically investigating various combinations of hyperparameter values to discover the settings that enhance the model's effectiveness on a specific task or dataset. Examples of hyperparameters include learning rate, batch size, optimizer, and dropout rate. Hyperparameters used in this project are batch size, learning rate, number of epoch, momentum, and patience. The selection of suitable hyperparameters can greatly affect the model’s accuracy, ability to generalize, and rate of convergence. Furthermore, the tuning tackles these concerns by looking for the hyperparameter values that increase the model’s performance on the validation set. This process helps in refining the model for improved performance, reducing overfitting, and enhancing its capacity to generalize to new, previously unseen data.

The model attained its highest accuracy when trained with a learning rate of 0.001. This indicates that 0.001 is the optimal value for training the model. Additionally, the true negative rate reached a perfect score of 1.000 with this learning rate, as did for precision. Notably, the value of recall falls to second highest when using a learning rate of 0.001. Overall, 0.001 emerges as the ideal learning rate for training MobileNetV2.
The batch size was adjusted while the optimal learning rate was kept constant. Various batch size values were employed to train the model, aiming to determine the most effective one. The trained models underwent testing on the validation set as well.

When batch size of 16 was employed, MobileNetV2 exhibited the highest accuracy, recall, true negative rate, and precision. This indicates that batch size of 16 is the optimal value for training the model. Consequently, MobileNetV2 was trained utilizing a learning rate of 0.001 and a batch size of 16, as they yielded the most favourable results.


Confusion matrix for validation data

X-axis (True) is the actual (ground truth) labels whereas Y-axis (Predicted) is what the model predicted. The model correctly predicted 42 pieces of flooding images when it really was flooding in true positive column. The model correctly predicted 38 pieces of normal images when it really was normal in true negative column. The model predicted 1 piece of flooding image but the actual was normal in false positive column. The model predicted 0 piece of flooding images in normal label.


Model performance evaluation

The model performance is evaluated using test data whereas validation data is used for the evaluation of hyperparameter tuning. Fine-tuned parameter in validation is used for testing and a confusion matrix for test data is obtained.
Accuracy - 0.9859, Precision - 0.9655, Recall - 1.0000, True negative rate - 0.9767
 

Training and validation loss and validation accuracy graph

Training loss line illustrates how effectively the model is adapting to the training data. It begins at a high value, representing significant error but then declines, indicating that the model is picking up on the training patterns. After epoch 10, the training loss exhibits some fluctuations while remaining low, suggesting a good fit. Validation loss line indicates how the model performs on unseen data. It starts at a moderate value and then decreases. It shows more fluctuation compared to the training loss. The variability observed after epoch 10 may suggest slight overfitting. The model is effectively learning the training dataset but its ability to generalize to new data is somewhat low. The decline in training and validation loss that stabilizes at a low level indicates effective learning. The fluctuations suggest some variability in the performance on the validation set due to a limited dataset or variability within the data. Overfitting appears to be minimal as training and validation loss remain low.
Validation accuracy quite high suggests the pretrained model had useful features. The drops may be due to random sampling, noisy data, or class imbalance.
