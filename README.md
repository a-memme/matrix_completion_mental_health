# Low-rank approximation in mental health self-report measures
*Exploring the suitability of low-rank approximation vs leading imputation methods for predicting missing data in self-report mental health measures. Downstream effects of imputed datasets are also explored and analytical approaches discussed.*

## Overview
### Purpose 
Quite a large body of research has explored the statistical process of imputation within the domain of medical diagnostics - specifically Electronic Health Records (EHR) that are susceptible to data sparseness - however, a contrastingly small portion has investigated said approaches for the more qualitative, yet comparably critical field of mental health. Given ongoing advancements in technology, mental health awareness, and general resources, we have become increasingly better positioned to collect longitudinal self-report mental health records at scale, however, at the risk of substantial sparseness (Wu et al, 2022). Uncovering or exploring effective approaches at imputing these datasets reliabily could thus provide a notable contribution to the framework of study and diagnosis in modern mental-health diagnostics research and practice. 

Given the above, the current project aims at exploring the suitability of low-rank approximation as a method of imputation in sparse mental health records, specifically in comparison to popular
imputation techniques that rely more heavily on local patterns rather than a dataset’s global
structure. The intuition here is to explore whether the success found in applying matrix completion to traditional medical diagnostics can be (somewhat) replicated in the domain of mental health, where the data in question is much more susceptible to random variability due to its inherent highly qualitative nature.

### Data
Three different forms of mental health datasets are leveraged in this analysis to get at some understanding of generalizability. For the purpose of time and comprehensiveness, only the **Student Dataset** is used to explore in-depth analytics, but all resources are considered for overall implications of study.

- **Student Dataset**
  - Student survey exploring the relationship of demographic, lifestyle and academic
factors on depression.
  - https://www.kaggle.com/datasets/ikynahidwin/depression-student-dataset
- **Synthetic Dataset**
  -  Synthetic mental health dataset containing simulated patient conditions related to
mental health with diagnostic and treatment recommendation information
  - https://www.kaggle.com/datasets/anweshaghosh123/mental-health-synthetic-dataset
- **DSM Dataset**
  -  DSM-5 based questionnaire used to diagnose PTSD, Bi-Polar Disorder,
Depression, Anxiety Disorder, and Schizophrenia based on specified symptoms
  - https://www.kaggle.com/datasets/karanbakshi1/mental-illness-dataset

### Methodology 
Our analysis focuses on understanding whether local or global patterns in the data are more influential in helping approximate missing data.  
