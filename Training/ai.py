import joblib
import pandas as pd
pipeline = joblib.load( 
"house_rent_pipeline.pkl" 
)
new_house = pd.DataFrame({ 
"BHK":[2], 
"Size":[1200], 
"Area Type":["Super Area"], 
"City":["Delhi"], 
"Furnishing Status":["Semi-Furnished"], 
"Tenant Preferred":["Family"], 
"Bathroom":[2], 
"Point of Contact":["Contact Owner"] 
})
prediction = pipeline.predict(new_house) 
print(prediction)