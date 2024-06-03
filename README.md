# Torque-Regression-and-Misfire-Classification

This git contains the files used for the paper "Torque Regression Using Machine Learning Techniques in Automotive ECUs", by Rafael Canal, João Paulo Bonomo, Rodrigo Santos de Carvalho, and Giovani Gracioli.

## Data Structure

In our work, the datasets were structured having one separate file for each performed experiment. The server exported the data in the `parquet` format, which may be easily imported as a DataFrame by the `pandas` library. These experiments are set to be saved in `data/testes`.

After importing, the data consists of a DataFrame where the rows are the data records and each column is the respective feature.

The results of the algorithm’s returns are set to be saved in the `data` folder.

## Instructions for Feature Selection

On `feature_selectors/fs_utils.py` we joined all the routines used to apply feature selection to our work. We split one function for every method applied to get a uniform formatted return for all of them: a `list` with the keys for each feature. Beyond these, we congregate the used methods in two functions which perform the selection with all methods and save the formatted results, given some dataset and/or a model: `model_feature_selection` and `filter_feature_selection`.


We created two different routines because of the differences in the types of feature selector's work principles. This distinction is mainly on the necessity or not of using the predictor in the feature selection process.

Briefly, the main types are:

- Filter Type feature selectors, which perform the selection based on statistical measures between train and target features, without the need for the model. In our case, we use the Select Percentile and Select K-Best filter methods.
- Wrapped and Embedded feature selectors, which use iterative training processes and testing the model with different sets of available features, evaluating and ranking the features based on these train/test results. In this case, the final prediction model is needed to perform the selection. On our context, the used Wrapped and Embedded methods were Select from Model (SFM), Sequential Feature Selection(SFS), Recursive Feature Elimination (RFE) and Recursive Feature Elimination with Cross-Validation (RFECV).

All implementations used were the ones provided by the `sklearn.feature_selection` library. Below, is a description of the received parameters in both methods. Both return a dictionary, where each key is the used selection method and the respective list of selected features.

```
filter_feature_selection
    x_train: DataFrame or np.array.    The array of features;
    y_train: DataFrame ou np.array.    Target value array;
    save_jason=true: Boolean    If you want to save a JSON file with the dictionary information.
    save_excel=true: Boolean    If you want to save a sheet file with the dictionary information.
    filename = "fs": String.     File name to be saved, same for JSON and Excel.


modelbased_feature_selection:
    model: Object   	 Predictor model object, not fitted. It needs to be on the Estimator API format of Sklearn.
    x_train: DataFrame or np.array.    The array of features;
    y_train: DataFrame ou np.array.    Target value array;
    save_jason=true: Boolean    If you want to save a JSON file with the dictionary information.
    save_excel=true: Boolean    If you want to save a sheet file with the dictionary information.
    filename = "fs": String.     File name to be saved, same for JSON and Excel.
```

An example of how these methods were applied is on the `feature_selectors/fs.py` file.

## Instructions for model evaluation

As described in our work, the experiments were divided into Stages, because of the differences in the available features between each Stage of experiments. With this, we performed a cross-validation for both classifier and regression methods consisting of taking one experiment as a test set, and the remaining experiment's data of the respective Stage as a training set, performing a train/test in each round for every experiment set in the Stage. Furthermore, in the early stages of development, we considered the case where there was only one experiment in a Stage. In that case, we performed a usual Cross-Validation, splitting the dataset into random fold sets using the `sklearn.model_selection.cross_validate` method.

We implemented routines for performing the described process independent of the dataset and informed model: `cross_tests_validate`. Besides having the same cross-validation process, we split the classifier and regression into two different methods because of the different score outputs between each type of prediction. Both have the same name `cross_tests_validate`, but are available in two different files: `classifiers/classifier_utils.py` and `regression/regression_utils.py`. Beyond this method, each file contains other useful methods, such as automated fitting/prediction, plotting, and scoring.


Below, is the description of the parameters informed to `cross_tests_validate` methods. Examples of applications are available in `classifiers` and `regression` folders, splitting one file for each model used.

```
cross_tests_validate:
    path_list: list   		 List with the paths for each experiment file.
    features: list or array,   	 List with the keys for features, relative to the informed dataset
    target: str or int,   		 Key for the target value on each dataset
    model: Object:   				 Predictor model object, not fitted. It needs to be on the Estimator API format of Sklearn
    results_name: str=f"log_{time.ctime(time. time())}",    the Name of the output file.
    savefile: Boolean=True   	 If you want to save an Excel file with the results. The default is True, being saved to file data/validation_logs/results_name.xls
```

