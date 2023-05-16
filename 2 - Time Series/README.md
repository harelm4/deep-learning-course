<!-- Output copied to clipboard! -->

<!-----

You have some errors, warnings, or alerts. If you are using reckless mode, turn it off to see inline alerts.
* ERRORs: 0
* WARNINGs: 0
* ALERTS: 14

Conversion time: 2.96 seconds.


Using this Markdown file:

1. Paste this output into your source file.
2. See the notes and action items below regarding this conversion run.
3. Check the rendered output (headings, lists, code blocks, tables) for proper
   formatting and use a linkchecker before you publish this page.

Conversion notes:

* Docs to Markdown version 1.0β34
* Tue May 16 2023 04:48:10 GMT-0700 (PDT)
* Source doc: deep learning hw2
* This document has images: check for >>>>>  gd2md-html alert:  inline image link in generated source and store images to your server. NOTE: Images in exported zip file from Google Docs may not appear in  the same order as they do in your doc. Please check the images!

----->


<p style="color: red; font-weight: bold">>>>>>  gd2md-html alert:  ERRORs: 0; WARNINGs: 0; ALERTS: 14.</p>
<ul style="color: red; font-weight: bold"><li>See top comment block for details on ERRORs and WARNINGs. <li>In the converted Markdown or HTML, search for inline alerts that start with >>>>>  gd2md-html alert:  for specific instances that need correction.</ul>

<p style="color: red; font-weight: bold">Links to alert messages:</p><a href="#gdcalert1">alert1</a>
<a href="#gdcalert2">alert2</a>
<a href="#gdcalert3">alert3</a>
<a href="#gdcalert4">alert4</a>
<a href="#gdcalert5">alert5</a>
<a href="#gdcalert6">alert6</a>
<a href="#gdcalert7">alert7</a>
<a href="#gdcalert8">alert8</a>
<a href="#gdcalert9">alert9</a>
<a href="#gdcalert10">alert10</a>
<a href="#gdcalert11">alert11</a>
<a href="#gdcalert12">alert12</a>
<a href="#gdcalert13">alert13</a>
<a href="#gdcalert14">alert14</a>

<p style="color: red; font-weight: bold">>>>>> PLEASE check and correct alert issues and delete this message and the inline alerts.<hr></p>



    **<span style="text-decoration:underline;">Assignment 2 – Practical Deep Learning Workshop</span>**


    <span style="text-decoration:underline;">abstract</span>


    The data we dealt with in this assignment came from a series of experiments carried out with experimenters with Parkinson's disease. The phenomenon of Freezing of gait (FOG) is a common phenomenon among Parkinson's patients (statistics show that half of the patients experience the phenomenon). As part of the phenomenon, the patient feels stuck to the floor and is unable to perform simple tasks including turning, walking in a straight line and opening doors. The above experiments were carried out in 3 situations. The first experiment was in the laboratory, when the experimenters performed a test called the FOG-provoking protocol. This experiment was recorded in a dataset called `tdcsfog`. The second experiment was carried out in the experimenters' home and the same test was performed there, this experiment was recorded in a dataset called `defog`. The third experiment We will last a week, during the experiment the activity of the experimenters was monitored with the help of sensors attached to their lower backs (these sensors were also used in the first 2 experiments). The third experiment was recorded in a dataset called `daily` . The first experiments were photographed and observed by experts. For moments when the experimenter experienced FOG according to the expert, the phenomenon was recorded At that moment. The goal of our work is to predict when there was a FOG event according to the sensor data and what kind of event it is (between general hesitation, stalling while turning or stalling while walking).


    <span style="text-decoration:underline;">Part 1 - data exploration</span>


    a. 


    **i**.  The data is taken from the series of experiments mentioned above, it is not homogeneous. For example, in the notype folder, the FOG events are recorded but it is not specified which event it is. Unlike defog and tdcsfog where the type of event was recorded (StartHesitation, Turn, Walking). Furthermore, the information in noType is taken in an experiment done in the subjects homes and noType/defog was in a lab.


    **ii**. The data was tagged with the help of experts on the phenomenon who watched video recordings collected from the experiments.


    **iii**. As mentioned above, in some folders the information is labeled differently and in a less detailed form than the others, therefore more weight should be given to more detailed labeling.


    In addition, there are labels that were not unambiguous, therefore in the notype folder there is the Valid column that describes whether the labeling is unambiguous or not. That's why not all labeling is valid.


    **iv**. 65 people participated in the experiments. The information is divided into train/test in such a way that in test there are 250 series of records in which there are subjects who are not in the training set.


    b.


    The task we are facing is classification ,we have 3 classes that are 3 different FOG events.lets explore the data:


    distribution of the classes for each dataset:

<p dir="rtl">


<p id="gdcalert1" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image1.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert2">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


<img src="images/image1.png" width="" alt="alt_text" title="image_tooltip">
</p>



        

<p id="gdcalert2" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image2.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert3">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image2.png "image_tooltip")



        

<p id="gdcalert3" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image3.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert4">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image3.png "image_tooltip")


We can see that the most common phenomenon of FOG events is Turning .

Another interesting thing that we have seen is the ratio of events to non events records in defog+tdcsfog datasets



<p id="gdcalert4" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image4.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert5">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image4.png "image_tooltip")


We can see the massive imbalance of the classes. We will need to handle it in pre-processing.

More analytics and charts can be found in the `data exploration` notebook and in the pre-processing section below.

c.



1. Average each feature (AccV,AccML,AccAP) and try to predict for each feature if its above or below his average.
2. Delete sections of the time series and try to predict them.

2.

In this section we will try to train models to perform said task. To do so we will first need to pre process the data.

<span style="text-decoration:underline;">Pre processing</span>



* We have noticed that the data measured at the subject homes and the data from the lab are measured differently (g vs a) . Due to that  we have scaled the defog data by 1/g .
* The data of deFog and tscsFog has different ranges so we have activated min max norm on all the data.
* The data is very big. training a model on a dataset this big can take a lot of time. In addition, we have an imbalance data problem,the rate of events is 0.31. To handle these 2 problems we have constructed the following solution. we have located the records indexes that had 1 label at least for one of the labels. for each located section (that is a section that a certain event has occurred),we took a window of 5 seconds in the past and 2.5 seconds ( we took into account that in defog 1 sec is 100 records and in tscsfog 1 sec is 128 records) in the future so our model can focus on the important events and the surrounding time. After this filtering the rate of events was 0.64 so the data is balanced,and more focused.

    Another thing that we checked is the correlation between the features before and after the filtering .


    

<p id="gdcalert5" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image5.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert6">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image5.png "image_tooltip")


<p id="gdcalert6" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image6.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert7">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image6.png "image_tooltip")



    We found that after the filtering, the correlation between the features and the targets increased. which means that the filter made it so the feature express better the targets.


    

<p id="gdcalert7" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image7.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert8">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image7.png "image_tooltip")


<p id="gdcalert8" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image8.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert9">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image8.png "image_tooltip")



    After analyzing the acceleration axes and studying video demonstrations of freezing of gait, we observed that the subjects tend to lean forward during the event. 


    Based on this finding, we decided to incorporate an additional feature into our model, which calculates the inverse tangent of AccAP and AccV to provide the angle of the subject. This will enable the model to use this angle as a significant input in its training process.

1. 

    As each dataset was divided by the experiment's ID, and each experiment had only one subject, we decided to split our data in a way that kept each experiment as a whole. This meant that no part of the same experiment would appear in both the training and validation sets. Additionally, we ensured that no subject appeared in both sets.


    We divided our data in a way that preserved the distribution of our target


    columns while splitting the data to around 75% for training and 25% for validation. 

2. We created a naïve baseline solution by using the class distribution of the target columns. We predicted the majority class for all samples in the training and validation sets.

    To see how this solution performs we calculated the MAP score for predicting the entire datasets, and got the value of 0.0898.

3. We tested the results of two different classical machine learning models: Random Forest Classifier and KNeighbors Classifier. For each one of these models we used the same data sets as we used for our neural network models, so we could compare the results and evaluate the networks accordingly.

    Results for Random Forest - MAP score: 0.2731


    Results for KNeighbors  -  MAP score: 0.2761


    The training was done using the train data frame and the testing was done on our validation data frame. We can see that these models gave very similar scores, both better than the naïve baseline.

4. In this section we have trained a LSTM model and a 1 dimensional CNN

    **<span style="text-decoration:underline;">1D-CNN</span>**


    Our CNN includes 3 convolutional layers with batch normalization after each layer,2 linear layers and a dropout layer. We have noticed that even after we filtered the data , the training time is very long so we have trained our model with 5 epochs,128 batch size and 1e-4 lr .


    We have decided to try our training with 2 visions, one with a training loader that is set on shuffle,and one when the loader doesn't.


    .


    

<p id="gdcalert9" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image9.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert10">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image9.png "image_tooltip")



    As you can clearly see, the model with the shuffle loader (pink) is converging faster then the non-shuffle one (red) and also gets better performance on the validation set.


    

<p id="gdcalert10" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image10.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert11">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image10.png "image_tooltip")



     Our model (the shuffled one) showed good performance after only 1 epoch, we assumed that the batch normalization contributed to this quick convergence time.


    **<span style="text-decoration:underline;">LSTM</span>**


    Our LSTM-model architecture consists of an LSTM layer followed by two fully connected layers. The LSTM has a hidden-size of 128 and two layers with a dropout of 0.5. The output of the LSTM is passed through the two linear layers.  The model uses cross-entropy as the loss function, Adam optimizer and a learning rate of 1e-4. 

5.  

    We decided to implement our first self-supervised task, in which we calculated the average of all three acceleration axes and used the values as targets in each time sample.


    **<span style="text-decoration:underline;">1D-CNN</span>**


    For this model, we fine tuned the model on the task mentioned above.


    We took the CNN model that we have trained with our objective classification task.


    We have set a new classifier for the model so he can predict the mean value of each feature (3 features,3 targets).


    After convergence, we have compared the result of the cnn model (original one) with the cnn_mean model (fine tuned one) with precision metric. The cnn model achieved 0.299 precision and the fine tuned model (cnn_mean) got 0.86 (!).


    **<span style="text-decoration:underline;">LSTM</span>**


    We started training our model on the self-supervised data set, and then changed our classifier layer to match the main classification targets. The LSTM model without the additional supervised task achieved 0.2775 precision. With pre-training the model achieved an accuracy 0.2766, meaning that the self supervised task of this kind may not be the best choice for this model.


    These graphs show the training process on the new task. while the loss appears to be converging during training, the validation results are not great.


    In this graph we compare the training loss with pre training and without. Although using the self supervised task did help with the learning rate and started from a lower loss, it didn’t end up with better results than using the normal model without the task.


f.


    we have analyzed our models abilities to predict each class:


    <span style="text-decoration:underline;">Normal CNN</span>


    

<p id="gdcalert11" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image11.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert12">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image11.png "image_tooltip")



    <span style="text-decoration:underline;">Fine tuned CNN</span>


    

<p id="gdcalert12" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image12.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert13">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image12.png "image_tooltip")



    <span style="text-decoration:underline;">LSTM</span>


    when 0 is no event ,1 is StartHesitation ,2 is Turn and 3 is Walking



1. We noticed that our models are succeeding to converge rather quickly, so <span style="text-decoration:underline;">increasing the learning rate</span> might be a way to accelerate the learning even more.
2. <span style="text-decoration:underline;">Increasing the batch size</span> of the training could help with efficient computation of gradients and speed up the learning process. 
3. Since we trained our models on data windows based on the position of the events, there is a chance that the model would learn these positions which would cause overfitting. <span style="text-decoration:underline;">Adding regularization</span> to the model could help prevent this happening.

g. 


    **<span style="text-decoration:underline;">1D-CNN</span>**


    We have tried to improve our model performance with 2 other versions.



1. increase batch size from 128 to 256 (yellow)
2. increase learning rate to 1e-3,batch size 128 (green)



<p id="gdcalert13" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image13.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert14">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image13.png "image_tooltip")


As we can see , increasing the learning rate improved the convergence time.

	

	

	**<span style="text-decoration:underline;">LSTM</span>**

**	**We tried to improve our performance in a similar way in our LSTM model:



1. Increase batch size from 1e-4 to 1e-3 (red).
2. Adding regularization to the model (green).



<p id="gdcalert14" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image14.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert15">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image14.png "image_tooltip")

