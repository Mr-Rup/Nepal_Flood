# INTERPETING NOTEBOOK 3    
## Overview
This notebook implements a machine learning workflow to detect and map change in landscape (landscape shift) happened due to flood of August 2026 through multi temporal satellite imagery. We predict landscape alterations across a geographic region 
coordinates 84.96-85.00°E, 28.00-28.25°N (central Nepal, this exact region covers parts of the Dhading and Gorkha districts in the Bagmati and Gandaki provinces of Nepal. This mountainous area sits just to the west of Kathmandu.)
---
The notebook is designed to work with sentinel 1 data (SAR) and Sentinel 2 data for optical sensors.

Sentinel 1 imagery: SAR (Synthetic Aperture Radar) images captured before and after the change event. These provide microwave-based observations useful for detecting structural changes in landscapes (buildings, deforestation, urban expansion).
Sentinel-2 Change Mask: An optical satellite-derived reference mask indicating where changes occurred. This helps in model training. Values range from 0 (no change) to 1 (confirmed change).
---
Sentinel 2 change mark is converted into binary variable
Y=0 means change not found landscape is same
 =1 means change detected landscape altered
Supervised Learning for change detection

`S1_pre.tif` - Radar image BEFORE the flood
`S1_post.tif` - Radar image AFTER the flood
`S2_indices_change.tif` - Vegetation health changes (green areas that turned brown)
`DEM.tif` - Digital elevation model (terrain/topography)
`random_forest_final.joblib` - The trained model that learned to spot flood damage
`RF_spatial_test_predictions.csv` - Model's predictions on test regions

In EO_change_intensity.tif we used satellite images to estimate real flood damage as high radar change showed water, low vegetation index showed destroyed crops. As the pixel value increases more alterations are detected. This gave us quantitative measure of flood impact from satellites.

MAP 1: EO DERIVED LANDSCAPE CHANGE INTENSITY 
Dark purple (0.0)=No change detected (landscape looks the same as before flood)
Orange (0.6 to 0.8)=Strong change detected (likely flood impact)
Yellow (1.0)=Maximum change detected

The yellow colored patches running through the dark purple areas are the river valleys and flood-affected zones.
The river network can be seen through the line running in between.
Flood impact zones have yellow patches. Mountainous terrain are in purple and didnt get much affected
Strong changes in these specific areas were seen.

MAP 2: EO DERIVED REFERENCE CHANGE MASK
This is a binary map using 0 (gray=no change) and 1 (black=change detected through flood damage)
River valleys (black lines)
Flood-affected fields or flood plains (patches of black)
Vegetation loss zones (black clusters)
Basically we checked for changed or not changed through satellite data.

MAP 3: RANDOM FOREST PREDICTED CHANGE
This is based on ml model decision which was trained on the entire region. It learned the things from labels in Y and lso predicts where damage occured but this is prediction by the model.

*Map 2=What satellites directly measured*
*Map 3=What the ML model inferred based on learned patterns*
So we can conclude that the model successfully adopted the satellite's logic. It can now be trusted for areas the satellites don't clearly show damage.


MAP 4: RANDOM FOREST SHOWING PROBABILITY OF CHANGE
Here we want to find how much confident we are about the predictions. It can be used for making decisions.
Purple (0.0 to 0.2)=Model very low confident
Cyan (0.4 to 0.6)=Model little bit confident
Green (0.7 to 0.8)=Model fairly confident
Yellow (0.9 to 1.0)=Model highly confident there's damage present

In simple words disaster management can be done like yellow zones are having high probability of damage so rescue teams are to be sent, teal zones are needed to be verified and purple zones actually need verification of damage. This can be done in order to ensure proper allocation of resourcs.

MAP 5: SPATIAL TEST AGREEMENT AND ERROR
Here the colour indices are important and to observe this we have to zoom out the tif,

Gray (TN)=True Negative
Model said:No damage
Satellite said:No damage
Result:Correct
Orange (FP)=False Positive 
Model said:Damage
Satellite said:No damage
Result:Model falsely alarmed
Light blue(FN)=False Negative 
Model said:No damage
Satellite said:Damage
Result:Model missed damage
Green(TP)=True Positive
Model said:Damage
Satellite said:Damage
Result:Correct

If we look closely we can see for the boxes:
Top box:Mostly gray+some orange=some correct,some false alarms
Middle box:Some orange (false alarms) and a bit of blue (missed some)
Bottom box:Mostly gray=working well on no-damage areas

---

To be noted

*Y is an EO-derived/rule-based change target, not independent CEMS ground truth.*

We're using satellite data to define what "flood damage" means. The model learns satellite patterns of damage and can then REPEAT this pattern (generalize) to unseen areas,
but we can't claim the model independently found real flood damage.
