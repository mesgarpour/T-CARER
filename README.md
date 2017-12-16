<div align="center">
  <img src="https://github.com/mesgarpour/T-CARER/blob/master/Documents/Logo/logo_tcarer.png">
</div>
<br>

-----------------
| **`Linux Debian`** | **`Linux Fedora`** | **`Mac OS`** | **`Windows, except TensorFlow`** |
|-----------------|---------------------|------------------|-------------------|
| [![CircleCI](https://img.shields.io/circleci/project/github/RedSparr0w/node-csgo-parser.svg)](PASSING) | [![CircleCI](https://img.shields.io/circleci/project/github/RedSparr0w/node-csgo-parser.svg)](PASSING) | [![CircleCI](https://img.shields.io/circleci/project/github/RedSparr0w/node-csgo-parser.svg)](PASSING) | [![CircleCI](https://img.shields.io/circleci/project/github/RedSparr0w/node-csgo-parser.svg)](PASSING) |


<br>
Version: 1.1 (2017.10.29)
<br>
Supported Python Version: 3.5
<br>
Supported TensorFlow Version: 1.4

------
# [T-CARER](https://github.com/mesgarpour/T-CARER)
[Temporal-Comorbidity Adjusted Risk of Emergency Readmission (T-CARER)](https://github.com/mesgarpour/T-CARER) is a comorbidity risk index that incorporates temporal dimensions, operations and procedures groups, demographics, and admission details, as well as diagnoses groups. The features of the model are generated using the [Healthcare Pre-Processing Framework](https://github.com/mesgarpour/Healthcare_PreProcessing_Framework), but it is partially integrated into the [T-CARER](https://github.com/mesgarpour/T-CARER) development toolkit, in order to preserve the tool's generic structure. The [T-CARER](https://github.com/mesgarpour/T-CARER) development toolkit is a generic, user-friendly and open-source software package that can be used for development of temporal comorbidity index independent of source of healthcare data.



# Introduction
Patients' [comorbidities](https://en.wikipedia.org/wiki/Comorbidity), operations and complications can be associated with reduced long-term survival probability and increased healthcare utilisation. The aim of this research [Ref !!!] was to produce an adjusted case-mix model of comorbidity risk and develop a user-friendly software tool to encourage public adaptation and incremental development.

In previous literature, there have been two streams of work on risk scoring comorbidities to estimate future resource utilisation, emergency admission and mortality. Firstly, one stream of research looks at the odds ratio of major diagnoses groups, like [Charlson Comorbidity Index (CCI)](https://academic.oup.com/aje/article/173/6/676/182985/Updating-and-Validating-the-Charlson-Comorbidity) which rely on twenty-two comorbidity groups. The second stream uses a case-mix model or a diagnoses classification approach based on similarities, type of care, likelihood or duration, like [Elixhauser Comorbidity Index (ECI)](https://www.hcup-us.ahrq.gov/toolssoftware/comorbidity/comorbidity.jsp), [Diagnosis-related Groups (DRG)](https://en.wikipedia.org/wiki/Diagnosis-related_group), which is developed by and John Hopkin's [Adjusted Clinical Groups (ACG)](https://www.hopkinsacg.org/).

[Temporal-Comorbidity Adjusted Risk of Emergency Readmission (T-CARER)](https://github.com/mesgarpour/T-CARER) is a generic development toolkit for designing temporal comorbidity risk index. The [T-CARER](https://github.com/mesgarpour/T-CARER) model, which was proposed by Mesgarpour et al (2017) [Ref !!!], predicts [hospital emergency admission](https://www.nao.org.uk/report/emergency-admissions-hospitals-managing-demand/) within 30- and 365-day using a generic set of variables. 



# Performance
The proposed model in this study incorporates temporal dimensions, operations and procedures groups, demographics, and admission details, as well as diagnoses groups. The research resulted in the [T-CARER](https://github.com/mesgarpour/T-CARER) model using routinely collected hospital inpatient data. The [T-CARER](https://github.com/mesgarpour/T-CARER) model is published publicly as an interactive [IPython](https://ipython.org/) Notebook, with generic inputs, features and population settings for general purpose use.

Moreover, several stages of analysis have been carried out to test and benchmark the [T-CARER](https://github.com/mesgarpour/T-CARER). Firstly, two data-frames across 10-year of HES inpatient records were selected (1999-2004 & 2004-2009). Afterwards, three different modelling approaches were developed: a [Logistic Regression](https://en.wikipedia.org/wiki/Logistic_regression), a [Random Forest](https://en.wikipedia.org/wiki/Random_forest), and a [Wide and Deep Neural Network (WDNN)](https://arxiv.org/abs/1606.07792). Then, the models were benchmarked against the [HSCIC](http://content.digital.nhs.uk/) implementation of the Charlson Comorbidity Index ([HSCIC-CCI](http://content.digital.nhs.uk/SHMI)), and the reported performance of [CCI](https://academic.oup.com/aje/article/173/6/676/182985/Updating-and-Validating-the-Charlson-Comorbidity) and [ECI](https://www.hcup-us.ahrq.gov/toolssoftware/comorbidity/comorbidity.jsp) implementations.
 
The [WDNN](https://arxiv.org/abs/1606.07792) and the [Random Forest](https://en.wikipedia.org/wiki/Random_forest) methods outperform in terms of the [Area Under the Curve (AUC) of Receiver-Operating Characteristic (ROC)](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) against the [Logistic Regression](https://en.wikipedia.org/wiki/Logistic_regression), as well as [HSCIC-CCI](http://content.digital.nhs.uk/SHMI), [CCI](https://academic.oup.com/aje/article/173/6/676/182985/Updating-and-Validating-the-Charlson-Comorbidity) and [ECI](https://www.hcup-us.ahrq.gov/toolssoftware/comorbidity/comorbidity.jsp) models. For 30- and 365-day emergency admissions, [ROCs](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) of different modelling approaches were from 0.772% to 0.804% the two sampled time-frames.

The [WDNN](https://arxiv.org/abs/1606.07792) method produced predictions with high precision, and the [Random Forest](https://en.wikipedia.org/wiki/Random_forest) method outperformed in terms of micro-average of [F1-score](https://en.wikipedia.org/wiki/F1_score). The [precisions](https://en.wikipedia.org/wiki/Precision_and_recall) were 0.582% to 0.639%, and the micro-average of [F1-score](https://en.wikipedia.org/wiki/F1_score) was 0.730% to 0.790% for the best modelling methods across different sampled time-frames.



# Related Publications
+  Mesgarpour, M., Chaussalet, T. & Chahed, S. (2017) Temporal-Comorbidity Adjusted Risk of Emergency Readmission. (In review!)
+  [Mesgarpour, M. (2017) Using Machine Learning Techniques in Predictive Risk Modelling in Healthcare. Ph.D. Thesis, University of Westminster, London, UK.](http://westminsterresearch.wmin.ac.uk/20306/1/Mesgarpour_Mohsen_thesis.pdf)



# License
[Apache License, Version 2.0.](https://www.apache.org/licenses/LICENSE-2.0.html)
Enjoy!



# Creadits
Original Author: [Mohsen Mesgarpour](https://uk.linkedin.com/in/mesgarpour), [Health and Social Care Modelling Group (HSCMG)](http://www.healthcareanalytics.co.uk/), [University of Westminster](https://www.westminster.ac.uk/).

Most Recent Author: [Mohsen Mesgarpour](https://uk.linkedin.com/in/mesgarpour), [Health and Social Care Modelling Group (HSCMG)](http://www.healthcareanalytics.co.uk/), [University of Westminster](https://www.westminster.ac.uk/).
