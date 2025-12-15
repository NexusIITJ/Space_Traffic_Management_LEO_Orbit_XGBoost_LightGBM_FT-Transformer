Loading data from: data\Merged_Featured_DATA.xlsx
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 574289 entries, 0 to 574288
Data columns (total 33 columns):
 #   Column                        Non-Null Count   Dtype
---  ------                        --------------   -----
 0   cdmMissDistance               574289 non-null  int64
 1   cdmPc                         574289 non-null  float64
 2   creationTsOfCDM               574289 non-null  datetime64[ns]
 3   SAT1_CDM_TYPE                 574289 non-null  object
 4   SAT2_CDM_TYPE                 574289 non-null  object
 5   cdmTca                        574289 non-null  datetime64[ns]
 6   rso1_noradId                  574289 non-null  int64
 7   rso1_objectType               574289 non-null  object
 8   org1_displayName              574289 non-null  object
 9   rso2_noradId                  574289 non-null  int64
 10  rso2_objectType               574289 non-null  object
 11  org2_displayName              574289 non-null  object
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
dtypes: bool(10), datetime64[ns](2), float64(5), int64(10), object(6)
memory usage: 106.3+ MB
None

=== Training XG_Boost on 15 features ===

Chosen threshold = 0.4800 (precision >= 0.50 enforced)
Recall = 1.0000, Precision = 0.9752, Score = 0.9975

--- Evaluation @ chosen threshold ---
Confusion Matrix:
 [[114050     20]
 [     0    788]]

Classification Report:
               precision    recall  f1-score   support

           0     1.0000    0.9998    0.9999    114070
           1     0.9752    1.0000    0.9875       788

    accuracy                         0.9998    114858
   macro avg     0.9876    0.9999    0.9937    114858
weighted avg     0.9998    0.9998    0.9998    114858

AUC-PR (average_precision): 0.999616, AUC-ROC: 0.999997

Top feature importances:
cdmPc: 0.8970
cdmMissDistance: 0.1016
condition_sat2posUnc_1km: 0.0002
SAT1_CDM_TYPE: 0.0002
condition_24H_tca_72H: 0.0002

=== Training XG_Boost_NoLeak on 13 features ===

Chosen threshold = 0.9741 (precision >= 0.50 enforced)
Recall = 0.4581, Precision = 0.5042, Score = 0.4627

--- Evaluation @ chosen threshold ---
Confusion Matrix:
 [[113715    355]
 [   427    361]]

Classification Report:
               precision    recall  f1-score   support

           0     0.9963    0.9969    0.9966    114070
           1     0.5042    0.4581    0.4801       788

    accuracy                         0.9932    114858
   macro avg     0.7502    0.7275    0.7383    114858
weighted avg     0.9929    0.9932    0.9930    114858

AUC-PR (average_precision): 0.474058, AUC-ROC: 0.921175

Top feature importances:
condition_InTrack_500m: 0.6132
SAT1_CDM_TYPE: 0.1712
condition_Radial_100m: 0.0509
condition_CrossTrack_500m: 0.0430
condition_24H_tca_72H: 0.0357

=== Training XG_Boost_NoLeak_Featured on 20 features ===

AUC-PR: 0.672376, AUC-ROC: 0.996192

Chosen threshold = 0.9724 (precision >= 0.50 enforced)
Recall = 0.7246, Precision = 0.5009, Score = 0.7022

--- Evaluation @ chosen threshold ---
Confusion Matrix:
 [[113501    569]
 [   217    571]]

Classification Report:
               precision    recall  f1-score   support

           0     0.9981    0.9950    0.9965    114070
           1     0.5009    0.7246    0.5923       788

    accuracy                         0.9932    114858
   macro avg     0.7495    0.8598    0.7944    114858
weighted avg     0.9947    0.9932    0.9938    114858

AUC-PR (average_precision): 0.672376, AUC-ROC: 0.996192

Top feature importances:
distance_ratio: 0.6035
SAT1_CDM_TYPE: 0.1638
hours_to_tca: 0.0451
condition_Radial_100m: 0.0388
condition_CrossTrack_500m: 0.0361


=== Training XG_Boost_Featured on 24 features ===
scale_pos_weight: 144.7585659898477

Chosen threshold = 0.7332 (precision >= 0.50 enforced)
Recall = 1.0000, Precision = 0.9801, Score = 0.9980

--- Evaluation @ chosen threshold ---
Confusion Matrix:
 [[114054     16]
 [     0    788]]

Classification Report:
               precision    recall  f1-score   support

           0     1.0000    0.9999    0.9999    114070
           1     0.9801    1.0000    0.9899       788

    accuracy                         0.9999    114858
   macro avg     0.9900    0.9999    0.9949    114858
weighted avg     0.9999    0.9999    0.9999    114858

AUC-PR (average_precision): 0.999684, AUC-ROC: 0.999998

Top feature importances:
cdmPc: 0.8721
cdmMissDistance: 0.1261
inv_miss_distance: 0.0002
condition_sat2posUnc_1km: 0.0002
condition_24H_tca_72H: 0.0002
