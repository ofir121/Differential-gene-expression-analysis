# Differential gene expression analysis and classification
## By: Ofir Shliefer
##### The analysis script is divided into 2 parts:
1. Using RNA-seq data to find genes which are significantly differentially expressed between two conditions
2. Solving classification problem

##### A few notes about the solutions:
1. The functions in the code are divided into 2 parts which are separated by a line comment (## Part x ##)
2. Each function has its own documentation using "javadoc" format.
3. All important variables are located in the top of the script, they can later be put out as arguments.


### Output
All plots are saved into 2 folders:
- Part 1 plots are saved to Output/Plots
- Part 2 plots are saved to Output/Classification_Plots.

Additional files (All files are located in the Output folder):
- Part 1 - Question 4 file: counts_normalized_log_filtered.pkl - Saved as pickle file
- Part 2-1 - Question 2a: Features information - features-info.csv, features-statistics.csv
- Part 2 - classification results - cross-validation-classification-results.csv (Part 2-1), meta-samples-classification-results.csv (Part 2-2)

##### Note
PCA for part 1 was plotted before and after cleaning data. 
In addition, PCA was plotted for all genes and genes that are ot lowly expressed.

### Decisions and thresholds explanations
#### Statistical Thresholds and tests
##### Part 1 - Question 3 - Filtering lowly-expressed genes
"Filter the count data for lowly-expressed genes, for example, only keep genes with a
CPM >= X in at least Y% samples, in at least Z of the groups (X, Y, Z to be defined by
you)."
I've used log(CPM) because it is the normalized form, it makes the difference way more convenient to define the cutoff.
I've chosen the following variables arbitrary by logic. if time would allow I would draw an histogram of how many features remaining after defining each variable and than defining the cutoff.
1. X - 1 - Was defined by observing the data. (If time would allow I would use an histogram to define this cutfoff)
2. Y - 20% - Cutoff that don't require all the samples to be above cutoff but still a sufficient amount to know this gene can be highly expressed.
3. Z - 1 - I've chosen at least 1 of the groups because if it highly expressed in one group and not another it is a great way to separate the data and we wouldn't want to discard it.

##### Part 1 - Question 7
I've made 2 descisions in this part:
1. Using FDR - false discovery rate - to adjust p-value. Because were making many statistical tests we need to adjust the p-value accordingly in order not to a get a false positive.
2. Using Independent t-test - To differentiate the groups I've used t-test. We have a sufficiant number of samples in each group (Over 30) to use a parametric test which require normal distribution of the mean (By Central Limit Theorem the means of samples from a population with finite variance approach a normal distribution regardless of the distribution of the population)

#### Classifications decisions
##### Classifiers
3 Classifiers  were tested: Random forest, AdaBoost and Multilayer perceptron.
Multilayer perceptron used regularization (In alpha parameter) - Although the results were already really good in all models so there was no need for messing with this parameter.

###### Classification Results - 5 fold cross validation: 

| classifier            | precision | recall   | f1       | roc auc score | tn   | fp  | fn  | tp   |
|-----------------------|-----------|----------|----------|---------------|------|-----|-----|------|
| Random Forest         | 1         | 1        | 1        | 1             | 16.6 | 0   | 0   | 18.8 |
| AdaBoost              | 0.980952  | 0.989474 | 0.984595 | 0.982237      | 16.2 | 0.4 | 0.2 | 18.6 |
| Multilayer Perceptron | 1         | 1        | 1        | 1             | 16.6 | 0   | 0   | 18.8 |

###### Classification Results using meta-samples training:

| classifier            | precision | recall   | f1       | roc auc score | tn | fp | fn | tp |
|-----------------------|-----------|----------|----------|---------------|----|----|----|----|
| Random Forest         | 1         | 1        | 1        | 1             | 83 | 0  | 0  | 94 |
| AdaBoost              | 0.989362  | 0.989362 | 0.989362 | 0.988657      | 82 | 1  | 1  | 93 |
| Multilayer Perceptron | 1         | 1        | 1        | 1             | 83 | 0  | 0  | 94 |

##### Meta-samples creation
Each feature was tested for the best matching distribution in each target group {"norm", "genextreme", "weibull_max", "weibull_min"}.
The best distribution and its parameters was used to create this feature in the meta-samples in each group.



### Part 2 - Questions:
##### Evaluate models performance and compare to those in the first part.

Training using regular and meta-samples preformed equally well. While Adaboost did do a bit worse with meta-samples.

##### Discuss (in short) the effect (if any) of training on the meta-samples. 
 - Who performed better?
 
 Training on meta-samples got almost the exact same result as training on regular samples.
 In our case it did not have any effect.
 
 - What are the potential implications of training on meta-samples?
 
 Training on meta-samples loses some of the information in the data. The meta-samples features are created from a different distribution that is only dependent on the group and not on one another. But in real data the value may be dependent on one another, for example feature A will be high only if feature B will be high as well. When creating meta-samples this isn't taken into account.
 Therefore, we may get meta-samples that don't completely represents our data.
 
##### (bonus) can you think of a way to use the meta-samples, and train better performing models?
Yes, we can add meta-samples to the training set in order to increase the number of samples. It will create new combinations of features that are not in our data and may improve the preformance of our model.


#### Additional notes
In part 2-2 - finding the distribution of the different features - the process takes some time, therefore it loads it from file to avoid redoing heavy computation (There is an option to redo the computation).



#### External python packages used
mapply, scipy, bioinfokit, matplotlib, seaborn, sklearn, pandas, numpy