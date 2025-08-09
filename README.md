<h2>TensorFlow-FlexUNet-Image-Segmentation-Gastrointestinal-Polyp (2025/08/10)</h2>

This is the first experiment of Image Segmentation for Gastrointestinal-Polyp (Kvasir-Polyp) ,
 based on our 
 <a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model">
<b>TensorFlowFlexUNet (TensorFlow Flexible UNet Image Segmentation Model for Multiclass)</b></a>
 
and a 512x512 pixels 
<a href="https://drive.google.com/file/d/1HL0Ybt-zfya_1_o8NwoaIwtS6BkFKDWn/view?usp=sharing">
<b>Augmented-Kvasir-Polyp-PNG-ImageMask-Dataset.zip</b></a>, 
which was derived by us from 
<a href="https://www.kaggle.com/datasets/debeshjha1/kvasirseg"><b>Kvasir-SEG Data (Polyp segmentation & detection)</b></a>
<br><br>


As demonstrated in <a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-STARE-Retinal-Vessel">
TensorFlow-FlexUNet-Image-Segmentation-STARE-Retinal-Vessel</a> ,
 our Multiclass TensorFlowFlexUNet, which uses categorized masks, can also be applied to 
single-class image segmentation models. 
This is because it inherently treats the background as one category and your single-class mask data as 
a second category. In essence, your single-class segmentation model will operate with two categorized classes within our Multiclass UNet framework.
<br>
<br>
Please see also our experiment 
<a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-Gastrointestinal-Polyp">
Tensorflow-Image-Segmentation-Gastrointestinal-Polyp</a>.
<br><br>

<b>Acutual Image Segmentation for 512x512 Kvasir-Polyp images</b><br>

As shown below, the inferred masks predicted by our segmentation model, which was trained on 
the augmented dataset, appear similar to the ground truth masks.
<br>
<br>

<br>
<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Kvasir-Polyp/mini_test/images/11.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Kvasir-Polyp/mini_test/masks/11.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Kvasir-Polyp/mini_test_output/11.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Kvasir-Polyp/mini_test/images/142.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Kvasir-Polyp/mini_test/masks/142.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Kvasir-Polyp/mini_test_output/142.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Kvasir-Polyp/mini_test/images/362.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Kvasir-Polyp/mini_test/masks/362.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Kvasir-Polyp/mini_test_output/362.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>1. Dataset Citation</h3>
The image dataset used here has been taken from the following kaggle web site.<br>
<a href="https://www.kaggle.com/datasets/debeshjha1/kvasirseg">Kvasir-SEG Data (Polyp segmentation & detection)</a>
<br><br>
<b>About Dataset</b>

Kvasir-SEG information:
The Kvasir-SEG dataset (size 46.2 MB) contains 1000 polyp images and their corresponding ground truth 
from the Kvasir Dataset v2. The images' resolution in Kvasir-SEG varies from 332x487 to 1920x1072 pixels. 
The images and its corresponding masks are stored in two separate folders with the same filename. 
The image files are encoded using JPEG compression, facilitating online browsing. 
The open-access dataset can be easily downloaded for research and educational purposes.
<br>
<br>
<b>Applications of the Dataset</b><br>

The Kvasir-SEG dataset is intended to be used for researching and developing new and improved methods 
for segmentation, detection, localization, and classification of polyps. 
Multiple datasets are prerequisites for comparing computer vision-based algorithms, and this dataset 
is useful both as a training dataset or as a validation dataset. These datasets can assist the 
development of state-of-the-art solutions for images captured by colonoscopes from different manufacturers. 
Further research in this field has the potential to help reduce the polyp miss rate and thus improve 
examination quality. The Kvasir-SEG dataset is also suitable for general segmentation and bounding box 
detection research. In this context, the datasets can accompany several other datasets from a wide 
range of fields, both medical and otherwise.
<br>
<br>
<b>Ground Truth Extraction</b><br>

We uploaded the entire Kvasir polyp class to Labelbox and created all the segmentations using this application. 
The Labelbox is a tool used for labeling the region of interest (ROI) in image frames, i.e., the polyp regions 
for our case. We manually annotated and labeled all of the 1000 images with the help of medical experts. 
After annotation, we exported the files to generate masks for each annotation. 
The exported JSON file contained all the information about the image and the coordinate points for generating 
the mask. To create a mask, we used ROI coordinates to draw contours on an empty black image and fill the 
contours with white color. The generated masks are a 1-bit color depth images. The pixels depicting polyp tissue, 
the region of interest, are represented by the foreground (white mask), while the background (in black) does not 
contain positive pixels. Some of the original images contain the image of the endoscope position marking probe, 
ScopeGuide TM, Olympus Tokyo Japan, located in one of the bottom corners, seen as a small green box. 
As this information is superfluous for the segmentation task, we have replaced these with black boxes in the 
Kvasir-SEG dataset.
<br>
<br>
See also:Kvasir-SEG 
<a href="https://paperswithcode.com/dataset/kvasir-seg">
https://paperswithcode.com/dataset/kvasir-seg
</a>

<br>

<h3>
<a id="2">
2 Kvasir-Polyp ImageMask Dataset
</a>
</h3>
 If you would like to train this Kvasir-Polyp Segmentation model by yourself,
 please download the dataset from the google drive  
<a href="https://drive.google.com/file/d/1HL0Ybt-zfya_1_o8NwoaIwtS6BkFKDWn/view?usp=sharing">
<b>Augmented-Kvasir-Polyp-PNG-ImageMask-Dataset.zip</b></a>.
<br>
, expand the downloaded ImageMaskDataset and put it under <b>./dataset</b> folder to be
<pre>
./dataset
└─Kvasir-Polyp
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>
<br>
<b>Kvasir-Polyp Statistics</b><br>
<img src ="./projects/TensorFlowFlexUNet/Kvasir-Polyp/Kvasir-Polyp_Statistics.png" width="512" height="auto"><br>
<br>

On the derivation of the dataset, please refer to the following Python scripts:<br>
<li><a href="./generator/ImageMaskDatasetGenerator.py">ImageMaskDatasetGenerator.py</a></li>
<li><a href="./generator/split_master.py">split_master.py</a></li>
<br>

As shown above, the number of images of train and valid datasets is large enough to use for a training set of our segmentation model.
<br>
<br>
<b>Train_images_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/Kvasir-Polyp/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/Kvasir-Polyp/asset/train_masks_sample.png" width="1024" height="auto">
<br>
<h3>
3 Train TensorFlowFlexUNet Model
</h3>
 We trained Kvasir-Polyp TensorFlowFlexUNet Model by using the following
<a href="./projects/TensorFlowFlexUNet/Kvasir-Polyp/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorFlowFlexUNet/Kvasir-Polyp and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorFlowFlexUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small <b>base_filters = 16 </b> and large <b>base_kernels = (9,9)</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
;You may specify your own UNet class derived from our TensorFlowFlexModel
model         = "TensorFlowFlexUNet"
generator     =  False
image_width    = 512
image_height   = 512
image_channels = 3
num_classes    = 2

base_filters   = 16
base_kernels   = (9,9)
num_layers     = 8
dropout_rate   = 0.04
dilation       = (1,1)
</pre>
<b>Learning rate</b><br>
Defined a very small learning rate.  
<pre>
[model]
learning_rate  = 0.00007
</pre>
<b>Loss and metrics functions</b><br>
Specified "categorical_crossentropy" and <a href="./src/dice_coef_multiclass.py">"dice_coef_multiclass"</a>.<br>
<pre>
[model]
loss           = "categorical_crossentropy"
metrics        = ["dice_coef_multiclass"]
</pre>
<b>Dataset class</b><br>
Specifed <a href="./src/ImageCategorizedMaskDataset.py">ImageCategorizedMaskDataset</a> class.<br>
<pre>
[dataset]
class_name    = "ImageCategorizedMaskDataset"
</pre>
<br>
<b>Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.4
reducer_patience   = 4
</pre>
<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>

<b>RGB Color map</b><br>
rgb color map dict for Kvasir-Polyp 1+1 classes.<br>
<pre>
[mask]
mask_file_format = ".png"
;
;RGB colors          (Polyp:white)
rgb_map = {(0,0,0):0,(255,255,255):1, }



</pre>

<b>Epoch change inference callback</b><br>
Enabled <a href="./src/EpochChangeInfereuncer.py">epoch_change_infer callback</a></b>.<br>
<pre>
[train]
epoch_change_infer       = True
epoch_change_infer_dir   =  "./epoch_change_infer"
num_infer_images         = 6
</pre>

By using this callback, on every epoch_change, the inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 

<b>Epoch_change_inference output at starting (epoch 1,2,3)</b><br>
<img src="./projects/TensorFlowFlexUNet/Kvasir-Polyp/asset/epoch_change_infer_at_start.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at middlepoint (epoch 28,29,30)</b><br>
<img src="./projects/TensorFlowFlexUNet/Kvasir-Polyp/asset/epoch_change_infer_at_middlepoint.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at ending (epoch 58,59,60)</b><br>
<img src="./projects/TensorFlowFlexUNet/Kvasir-Polyp/asset/epoch_change_infer_at_end.png" width="1024" height="auto"><br>
<br>
In this experiment, the training process was terminated at epoch 60.<br><br>
<img src="./projects/TensorFlowFlexUNet/Kvasir-Polyp/asset/train_console_output_at_epoch60.png" width="920" height="auto"><br>
<br>

<a href="./projects/TensorFlowFlexUNet/Kvasir-Polyp/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Kvasir-Polyp/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorFlowFlexUNet/Kvasir-Polyp/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Kvasir-Polyp/eval/train_losses.png" width="520" height="auto"><br>

<br>

<h3>
4 Evaluation
</h3>
Please move to <b>./projects/TensorFlowFlexUNet/Kvasir-Polyp</b> folder,<br>
and run the following bat file to evaluate TensorFlowFlexUNet model for Kvasir-Polyp.<br>
<pre>
./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetEvaluator.py ./train_eval_infer_aug.config
</pre>

Evaluation console output:<br>
<img src="./projects/TensorFlowFlexUNet/Kvasir-Polyp/asset/evaluate_console_output_at_epoch60.png" width="920" height="auto">
<br><br>

<a href="./projects/TensorFlowFlexUNet/Kvasir-Polyp/evaluation.csv">evaluation.csv</a><br>
The loss (categorical_crossentropy) to this Kvasir-Polyp/test was very low and dice_coef_multiclass 
very high as shown below.
<br>
<pre>
categorical_crossentropy,0.0084
dice_coef_multiclass,0.9958
</pre>
<br>

<h3>
5 Inference
</h3>
Please move <b>./projects/TensorFlowFlexUNet/Kvasir-Polyp</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorFlowFlexUNet model for Kvasir-Polyp.<br>
<pre>
./3.infer.bat
</pre>
This simply runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetInferencer.py ./train_eval_infer_aug.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorFlowFlexUNet/Kvasir-Polyp/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorFlowFlexUNet/Kvasir-Polyp/asset/mini_test_masks.png" width="1024" height="auto"><br>

<hr>
<b>Inferred test masks</b><br>
<img src="./projects/TensorFlowFlexUNet/Kvasir-Polyp/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks of 512x512 pixels</b><br>
<table>
<tr>
<th>Image</th>
<th>Mask (ground_truth)</th>
<th>Inferred-mask</th>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Kvasir-Polyp/mini_test/images/24.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Kvasir-Polyp/mini_test/masks/24.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Kvasir-Polyp/mini_test_output/24.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Kvasir-Polyp/mini_test/images/58.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Kvasir-Polyp/mini_test/masks/58.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Kvasir-Polyp/mini_test_output/58.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Kvasir-Polyp/mini_test/images/79.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Kvasir-Polyp/mini_test/masks/79.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Kvasir-Polyp/mini_test_output/79.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Kvasir-Polyp/mini_test/images/241.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Kvasir-Polyp/mini_test/masks/241.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Kvasir-Polyp/mini_test_output/241.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Kvasir-Polyp/mini_test/images/263.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Kvasir-Polyp/mini_test/masks/263.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Kvasir-Polyp/mini_test_output/263.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Kvasir-Polyp/mini_test/images/362.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Kvasir-Polyp/mini_test/masks/362.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Kvasir-Polyp/mini_test_output/362.png" width="320" height="auto"></td>
</tr>
</table>
<hr>

<br>
<h3>
References
</h3>

<b>1. Kvasir-SEG Data (Polyp segmentation & detection)</b><br>
<a href="https://www.kaggle.com/datasets/debeshjha1/kvasirseg">
https://www.kaggle.com/datasets/debeshjha1/kvasirseg</a>
<br>
<br>
<b>2. Kvasir-SEG: A Segmented Polyp Dataset</b><br>
Debesh Jha, Pia H. Smedsrud, Michael A. Riegler, P˚al Halvorsen,<br>
Thomas de Lange, Dag Johansen, and H˚avard D. Johansen<br>
<a href="https://arxiv.org/pdf/1911.07069v1.pdf">
https://arxiv.org/pdf/1911.07069v1.pdf
</a>
<br>
<br>
<b>3. DeepLabV3Plus-Tf2.x</b><br>
TanyaChutani<br>
<a href="https://github.com/TanyaChutani/DeepLabV3Plus-Tf2.x/blob/master/notebook/DeepLab_V3_Plus.ipynb">
https://github.com/TanyaChutani/DeepLabV3Plus-Tf2.x/blob/master/notebook/DeepLab_V3_Plus.ipynb
</a>
<br>
<br>
<b>4. TensorflowSwinUNet-Image-Segmentation-Augmented-GastrointestinalPolyp</b><br>
Toshiyuki Arai @antillia.com<br>
<a href="https://github.com/sarah-antillia/TensorflowSwinUNet-Image-Segmentation-Augmented-GastrointestinalPolyp">
https://github.com/sarah-antillia/TensorflowSwinUNet-Image-Segmentation-Augmented-GastrointestinalPolyp
</a>
<br>
<br>
<b>5. TensorflowMultiResUNet-Segmentation-Gastrointestinal-Polyp</b><br>
Toshiyuki Arai @antillia.com<br>
<a href="https://github.com/sarah-antillia/TensorflowMultiResUNet-Segmentation-Gastrointestinal-Polyp">
https://github.com/sarah-antillia/TensorflowMultiResUNet-Segmentation-Gastrointestinal-Polyp
</a>
<br>
<br>
<b>6. TensorflowUNet3Plus-Segmentation-Gastrointestinal-Polyp</b><br>
Toshiyuki Arai @antillia.com<br>
<a href="https://github.com/sarah-antillia/TensorflowUNet3Plus-Segmentation-Gastrointestinal-Polyp">
https://github.com/sarah-antillia/TensorflowUNet3Plus-Segmentation-Gastrointestinal-Polyp
</a>
<br>
<br>
<b>7. TensorflowEfficientUNet-Segmentation-Gastrointestinal-Polyp</b><br>
Toshiyuki Arai @antillia.com<br>
<a href="https://github.com/sarah-antillia/TensorflowEfficientUNet-Image-Segmentation-Augmented-GastrointestinalPolyp">
https://github.com/sarah-antillia/TensorflowEfficientUNet-Image-Segmentation-Augmented-GastrointestinalPolyp
</a>
<br>
<br>
<b>8. TensorflowSharpUNet-Segmentation-Gastrointestinal-Polyp</b><br>
Toshiyuki Arai @antillia.com<br>
<a href="https://github.com/sarah-antillia/TensorflowSharpUNet-Image-Segmentation-Augmented-GastrointestinalPolyp">
https://github.com/sarah-antillia/TensorflowSharpUNet-Image-Segmentation-Augmented-GastrointestinalPolyp
</a>
<br>
<br>
<b>9. TensorflowU2Net-Segmentation-Gastrointestinal-Polyp</b><br>
Toshiyuki Arai @antillia.com<br>
<a href="https://github.com/sarah-antillia/TensorflowU2Net-Image-Segmentation-Augmented-GastrointestinalPolyp">
https://github.com/sarah-antillia/TensorflowU2Net-Image-Segmentation-Augmented-GastrointestinalPolyp
</a>
<br>

