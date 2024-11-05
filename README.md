## Attention as Explanation

The aim of this project is to test how attention fares as a model explainer. Based on the weight of an attentin layer, feature importance scores are calculated that represent the contribution that each feature makes for the prediction. Or so the idea. I test whether Attention can serve this function in a number of experiments and in comparison to the SHAP explainer. The experimental set up and its rationale is detailed in the report. 

To run the code the following is useful. All scripts are designed to run on a single cuda device (if specified). 

main.py trains the base models. It has an optional argument for adding a dataset (if no huggingface dataset is used) and requires a path to save the trained model to: 
usage: main.py [-h] [--cuda_device CUDA_DEVICE] [--dataset DATASET]
               path_to_save
               
Module.py contains the classes for all three attention layers, dense, uniform, and sparse. 

explainer_experiments.py runs the experiments. It requies a path to load the trained model from. 
usage: explainer_experiments.py [-h] [--file_path FILE_PATH]
                                [--cuda_device CUDA_DEVICE]
                                path_to_load

Both, main.py and explainer_experiments.py contain code for training and experimenting with models on four different datasets. When using one of those datasets, the respective parts of the code needs to be commented out. If using multiclass or multinomial predictions, the correlations in explainer_experiments.py on lines 292, 293 need to be commented used when plotting correlation distributions.  
