Loading data from: data\Merged_Featured_DATA.xlsx
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 574289 entries, 0 to 574288
Data columns (total 33 columns):
 #   Column                        Non-Null Count   Dtype
---  ------                        --------------   -----
 0   cdmMissDistance               574289 non-null  int64
 1   cdmPc                         574289 non-null  float64
 2   creationTsOfCDM               574289 non-null  datetime64[ns]
 3   SAT1_CDM_TYPE                 574289 non-null  category
 4   SAT2_CDM_TYPE                 574289 non-null  category
 5   cdmTca                        574289 non-null  datetime64[ns]
 6   rso1_noradId                  574289 non-null  int64
 7   rso1_objectType               574289 non-null  category
 8   org1_displayName              574289 non-null  category
 9   rso2_noradId                  574289 non-null  int64
 10  rso2_objectType               574289 non-null  category
 11  org2_displayName              574289 non-null  category
 12  condition_cdmType=EPHEM:HAC   574289 non-null  bool
 13  condition_24H_tca_72H         574289 non-null  bool
 14  condition_Pc>1e-6             574289 non-null  bool
 15  condition_missDistance<2000m  574289 non-null  bool
 16  condition_Radial_100m         574289 non-null  bool
 17  condition_Radial<50m          574289 non-null  bool
 18  condition_InTrack_500m        574289 non-null  bool
 19  condition_CrossTrack_500m     574289 non-null  bool
 20  condition_sat2posUnc_1km      574289 non-null  bool
 21  condition_sat2Obs_25          574289 non-null  bool
 22  hours_to_tca                  574289 non-null  float64
 23  HighRisk                      574289 non-null  int64
 24  log_cdmPc                     574289 non-null  float64
 25  inv_miss_distance             574289 non-null  float64
 26  tca_bin                       574289 non-null  int64
 27  same_sat_type                 574289 non-null  int64
 28  is_debris_pair                574289 non-null  int64
 29  close_all_axes                574289 non-null  int64
 30  risky_uncertainty             574289 non-null  int64
 31  distance_ratio                574289 non-null  float64
 32  object_type_match             574289 non-null  int64
dtypes: bool(10), category(6), datetime64[ns](2), float64(5), int64(10)
memory usage: 83.2 MB
None

[LGB_Boost] scale_pos_weight: 144.7585659898477

================= LGB_Boost @ Threshold = 0.5 =================
Confusion Matrix:
 [[114060     10]
 [     4    784]]

Classification Report:
               precision    recall  f1-score   support

           0     1.0000    0.9999    0.9999    114070
           1     0.9874    0.9949    0.9912       788

    accuracy                         0.9999    114858
   macro avg     0.9937    0.9974    0.9955    114858
weighted avg     0.9999    0.9999    0.9999    114858

Recall: 0.9949
Precision: 0.9874
F1-score: 0.9912
Accuracy: 0.9999
AUC-PR: 0.999722
AUC-ROC: 0.999998

Best threshold for LGB_Boost = 0.1113

================= LGB_Boost @ BEST Threshold =================
Confusion Matrix:
 [[114059     11]
 [     0    788]]

Classification Report:
               precision    recall  f1-score   support

           0     1.0000    0.9999    1.0000    114070
           1     0.9862    1.0000    0.9931       788

    accuracy                         0.9999    114858
   macro avg     0.9931    1.0000    0.9965    114858
weighted avg     0.9999    0.9999    0.9999    114858

Recall: 1.0000
Precision: 0.9862
F1-score: 0.9931
Accuracy: 0.9999

Feature importances:
cdmPc: 25537774.6283
cdmMissDistance: 1395704.5828
hours_to_tca: 7198.0567
condition_InTrack_500m: 1198.1848
condition_Radial_100m: 695.0468
condition_24H_tca_72H: 288.3870
rso2_objectType: 285.3985
SAT1_CDM_TYPE: 269.8828
condition_CrossTrack_500m: 231.4388
condition_sat2posUnc_1km: 157.4696
condition_sat2Obs_25: 138.5562
org2_displayName: 67.4105
rso1_objectType: 0.0000
SAT2_CDM_TYPE: 0.0000
org1_displayName: 0.0000

Saved LightGBM evaluation results to results\LGB_Boost.json

LightGBM model saved to models\LGB_Boost_LGB.txt

[LGB_Boost_NoLeak] scale_pos_weight: 144.7585659898477

================= LGB_Boost_NoLeak @ Threshold = 0.5 =================
Confusion Matrix:
 [[108671   5399]
 [   196    592]]

Classification Report:
               precision    recall  f1-score   support

           0     0.9982    0.9527    0.9749    114070
           1     0.0988    0.7513    0.1747       788

    accuracy                         0.9513    114858
   macro avg     0.5485    0.8520    0.5748    114858
weighted avg     0.9920    0.9513    0.9694    114858

Recall: 0.7513
Precision: 0.0988
F1-score: 0.1747
Accuracy: 0.9513
AUC-PR: 0.440542
AUC-ROC: 0.894805

Best threshold for LGB_Boost_NoLeak = 0.0000

================= LGB_Boost_NoLeak @ BEST Threshold =================
Confusion Matrix:
 [[  3883 110187]
 [     0    788]]

Classification Report:
               precision    recall  f1-score   support

           0     1.0000    0.0340    0.0658    114070
           1     0.0071    1.0000    0.0141       788

    accuracy                         0.0407    114858
   macro avg     0.5036    0.5170    0.0400    114858
weighted avg     0.9932    0.0407    0.0655    114858

Recall: 1.0000
Precision: 0.0071
F1-score: 0.0141
Accuracy: 0.0407

Feature importances:
condition_InTrack_500m: 12103074.8438
hours_to_tca: 5573372.3739
SAT1_CDM_TYPE: 2882736.9326
condition_Radial_100m: 1408882.6681
condition_CrossTrack_500m: 1405848.8638
rso2_objectType: 462502.7471
condition_sat2posUnc_1km: 317803.0524
condition_sat2Obs_25: 294206.4814
org1_displayName: 181215.1893
org2_displayName: 165741.2692
condition_24H_tca_72H: 51236.5562
rso1_objectType: 1809.6323
SAT2_CDM_TYPE: 0.0000

Saved LightGBM evaluation results to results\LGB_Boost_NoLeak.json

LightGBM model saved to models\LGB_Boost_NoLeak_LGB.txt

[LGB_Boost_NoLeak_Featured] scale_pos_weight: 144.7585659898477

================= LGB_Boost_NoLeak_Featured @ Threshold = 0.5 =================   
Confusion Matrix:
 [[112976   1094]
 [   116    672]]

Classification Report:
               precision    recall  f1-score   support

           0     0.9990    0.9904    0.9947    114070
           1     0.3805    0.8528    0.5262       788

    accuracy                         0.9895    114858
   macro avg     0.6897    0.9216    0.7605    114858
weighted avg     0.9947    0.9895    0.9915    114858

Recall: 0.8528
Precision: 0.3805
F1-score: 0.5262
Accuracy: 0.9895
AUC-PR: 0.605733
AUC-ROC: 0.994964

Best threshold for LGB_Boost_NoLeak_Featured = 0.0003

================= LGB_Boost_NoLeak_Featured @ BEST Threshold =================    
Confusion Matrix:
 [[111630   2440]
 [    11    777]]

Classification Report:
               precision    recall  f1-score   support

           0     0.9999    0.9786    0.9891    114070
           1     0.2415    0.9860    0.3880       788

    accuracy                         0.9787    114858
   macro avg     0.6207    0.9823    0.6886    114858
weighted avg     0.9947    0.9787    0.9850    114858

Recall: 0.9860
Precision: 0.2415
F1-score: 0.3880
Accuracy: 0.9787

Feature importances:
hours_to_tca: 53148388.3398
distance_ratio: 22566620.8078
risky_uncertainty: 2400964.5996
SAT1_CDM_TYPE: 1219442.8325
condition_Radial_100m: 184371.4384
condition_CrossTrack_500m: 166854.2830
condition_InTrack_500m: 126500.9471
tca_bin: 84347.7293
condition_sat2posUnc_1km: 61824.0420
rso2_objectType: 55932.9894
condition_sat2Obs_25: 53344.8531
same_sat_type: 29980.5838
condition_24H_tca_72H: 16817.0564
org1_displayName: 12254.3703
org2_displayName: 10796.3152
close_all_axes: 3064.4188
rso1_objectType: 120.4732
is_debris_pair: 16.7818
object_type_match: 2.8359
SAT2_CDM_TYPE: 0.0000

Saved LightGBM evaluation results to results\LGB_Boost_NoLeak_Featured.json       

LightGBM model saved to models\LGB_Boost_NoLeak_Featured_LGB.txt

[LGB_Boost_Featured] scale_pos_weight: 144.7585659898477

================= LGB_Boost_Featured @ Threshold = 0.5 =================
Confusion Matrix:
 [[114037     33]
 [     3    785]]

Classification Report:
               precision    recall  f1-score   support

           0     1.0000    0.9997    0.9998    114070
           1     0.9597    0.9962    0.9776       788

    accuracy                         0.9997    114858
   macro avg     0.9798    0.9980    0.9887    114858
weighted avg     0.9997    0.9997    0.9997    114858

Recall: 0.9962
Precision: 0.9597
F1-score: 0.9776
Accuracy: 0.9997
AUC-PR: 0.947863
AUC-ROC: 0.999875

Best threshold for LGB_Boost_Featured = 0.0066

================= LGB_Boost_Featured @ BEST Threshold =================
Confusion Matrix:
 [[114036     34]
 [     0    788]]

Classification Report:
               precision    recall  f1-score   support

           0     1.0000    0.9997    0.9999    114070
           1     0.9586    1.0000    0.9789       788

    accuracy                         0.9997    114858
   macro avg     0.9793    0.9999    0.9894    114858
weighted avg     0.9997    0.9997    0.9997    114858

Recall: 1.0000
Precision: 0.9586
F1-score: 0.9789
Accuracy: 0.9997

Feature importances:
log_cdmPc: 25442225.7005
inv_miss_distance: 1556415.5292
cdmPc: 348717.9199
distance_ratio: 68915.3748
hours_to_tca: 49030.2348
condition_Radial_100m: 25532.7172
cdmMissDistance: 10719.9108
condition_sat2posUnc_1km: 2299.2922
tca_bin: 1860.4789
rso2_objectType: 846.6436
condition_InTrack_500m: 392.4174
same_sat_type: 199.3278
condition_24H_tca_72H: 57.7939
condition_sat2Obs_25: 57.0820
risky_uncertainty: 27.3249
condition_CrossTrack_500m: 24.1023
org2_displayName: 15.5109
SAT1_CDM_TYPE: 1.6952
close_all_axes: 0.0070
SAT2_CDM_TYPE: 0.0000
rso1_objectType: 0.0000
org1_displayName: 0.0000
is_debris_pair: 0.0000
object_type_match: 0.0000

Saved LightGBM evaluation results to results\LGB_Boost_Featured.json