## LearningMktStruct: A method to learn a latent characteristic space 
This repository contains the code used to estimate latent product characteristics and consumer preferences from "Learning Product Characteristics and Consumer Preferences from Search Data" by Luis Armona, Greg Lewis, and Giorgos Zervas. The repository contains the actual functions used to construct the model (bpr.py) and a minimum working example on how to estimate the model using Keras (example.py)


Please cite the above paper ([link to SSRN Version](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3858377)) if you use this method/code in your own work.



### Methodology
This package allows users to estimate unobserved product characteristics, along with user preferences over both observed and unobserved product characteristics. It builds on the Bayesian Personalized Ranking model ([Rendle et al. 2009](https://arxiv.org/abs/1205.2618))  to allow for 1. fixed observable characteristics associated with products available to the researcher and 2. attributes that are allowed to vary within product (in particular, price). One can use the unobserved product characteristics to better characterize the market structure / product differentiation in a particular setting, or use the recovered latent preferences to improve demand estimation (See paper for details). 

### Computational Requirements
To use the code, one needs the following python libraries:

```keras```  (ran on version 2.13.1)

```tensorflow``` (ran on version 2.13.1)

```numpy``` (ran on version 1.23.5)

### Data Requirements
To use the method, one should have a dataset with some sort of revealed preferences, (e.g. product A preferred to product B by consumer i). In our context, and the most direct use case, this arises from clickstream data, which represent searches that indicate products have higher expected utility from the perspective of consumers. 
In principle though, any auxiliary information indicating preferences over products, such as survey data, could be used to estimate the latent characteristics and consumer preferences.


The data for the example is from a shopping website offering clothes to pregnant women, and contains clickstream data of the different products searched (clicked) in individual search sessions, analogous to the (proprietary) hotel data we use in our paper.
The data can be obtained from the UCI machine learning repository [here](https://archive.ics.uci.edu/dataset/553/clickstream+data+for+online+shopping). The original paper attributable for this data can be found in this link: https://cejsh.icm.edu.pl/cejsh/element/bwmeta1.element.desklight-aba572ac-3144-40a9-9fb6-4b3b0df0eda3
### Usage 
The primary file of importance is ```bpr.py```, which contains three functions that build the model in keras (```build_bpr_model```), provide a sampler for large datasets (```bpr_triplet_impression_sampler```), and export the estimated parameters for usage in downstream analysis (```export_embeddings```). See the documentation under each function for information on how to use them, along with the example for a step-by-step implementation of the methodology. See the paper's model section for details 

