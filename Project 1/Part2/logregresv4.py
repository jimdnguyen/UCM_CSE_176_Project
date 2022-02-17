import pandas as pd
import matplotlib.pyplot as plt
from pandas.core.indexes import numeric
import seaborn as sns
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import make_scorer, mean_squared_error
from rgf.sklearn import RGFRegressor
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import time
start = time.time()
airbnb_london_listing = './londonairbnb/listings_detailed.csv'
airbnb_data = pd.read_csv(airbnb_london_listing)
#
airbnb_data['price'] = airbnb_data['price'].str.replace('$', '', regex=True)
airbnb_data['price'] = airbnb_data['price'].str.replace(
    ',', '', regex=True).astype(float)
airbnb_data['weekly_price'] = airbnb_data['weekly_price'].str.replace(
    '$', '', regex=True)
airbnb_data['weekly_price'] = airbnb_data['weekly_price'].str.replace(
    ',', '', regex=True).astype(float)
airbnb_data['monthly_price'] = airbnb_data['monthly_price'].str.replace(
    '$', '', regex=True)
airbnb_data['monthly_price'] = airbnb_data['monthly_price'].str.replace(
    ',', '', regex=True).astype(float)
airbnb_data['security_deposit'] = airbnb_data['security_deposit'].str.replace(
    '$', '', regex=True)
airbnb_data['security_deposit'] = airbnb_data['security_deposit'].str.replace(
    ',', '', regex=True).astype(float)
airbnb_data['cleaning_fee'] = airbnb_data['cleaning_fee'].str.replace(
    '$', '', regex=True)
airbnb_data['cleaning_fee'] = airbnb_data['cleaning_fee'].str.replace(
    ',', '', regex=True).astype(float)
airbnb_data['extra_people'] = airbnb_data['extra_people'].str.replace(
    '$', '', regex=True)
airbnb_data['extra_people'] = airbnb_data['extra_people'].str.replace(
    ',', '', regex=True).astype(float)
# this section is just changing the price column into a float value

#
columntf = ['require_guest_profile_picture', 'require_guest_phone_verification', 'host_is_superhost', 'host_has_profile_pic',
            'host_identity_verified', 'is_location_exact', 'has_availability', 'requires_license', 'instant_bookable', 'is_business_travel_ready']

airbnb_data[columntf] = airbnb_data[columntf].replace({'t': 1, 'f': 0})
# this section is replacing true and false (t and f) with 1 and 0

# print(airbnb_data.columns[airbnb_data.isnull().mean() > 0.50])

#
missing_more_than_50 = list(
    airbnb_data.columns[airbnb_data.isnull().mean() > 0.50])

airbnb_data = airbnb_data.drop(missing_more_than_50, axis=1)
# this section is dropping columns that are missing > 50% of their data.

#
numerical_ix = airbnb_data.select_dtypes(include=['int64', 'float64']).columns
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(missing_values=np.NaN, strategy='mean'))
])

airbnb_data[numerical_ix] = numeric_transformer.fit_transform(
    airbnb_data[numerical_ix])

categorical_ix = airbnb_data.select_dtypes(include=['object']).columns
# airbnb_data = airbnb_data.drop(categorical_ix, axis=1)
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(missing_values=np.NaN, strategy='most_frequent'))
])

airbnb_data[categorical_ix] = categorical_transformer.fit_transform(
    airbnb_data[categorical_ix])
# This section is filling in the data missing for the columns with less than 50% missing

#
airbnb_data['host_acceptance_rate'] = airbnb_data['host_acceptance_rate'].str.replace(
    '%', '', regex=True).astype(float)
airbnb_data['host_acceptance_rate'] = airbnb_data['host_acceptance_rate'] * 0.01
# this section is just changing the percentage into a float value

#
ids = ['id', 'scrape_id', 'host_id']
airbnb_data = airbnb_data.drop(ids, axis=1)

urls = ['listing_url', 'picture_url',
        'host_url', 'host_thumbnail_url', 'host_picture_url']

airbnb_data = airbnb_data.drop(urls, axis=1)
# this section, we are just getting rid of columns that are unneccessary for us

# summary and description have same info, but description has less null values
# same with street and city
columnsimiliarinfo = ['summary', 'street', 'neighbourhood']
airbnb_data = airbnb_data.drop(columnsimiliarinfo, axis=1)

columnunecessary = ['market', 'country_code',
                    'country', 'smart_location', 'name', 'state', 'city', 'latitude', 'longitude', 'zipcode']
airbnb_data = airbnb_data.drop(columnunecessary, axis=1)

columndates = ['last_scraped', 'calendar_last_scraped',
               'first_review', 'last_review', 'host_since', 'calendar_updated']
airbnb_data = airbnb_data.drop(columndates, axis=1)

randomcolumns = ['space', 'description', 'experiences_offered', 'neighborhood_overview', 'transit',
                 'access', 'interaction', 'house_rules', 'host_name', 'host_location', 'host_about', 'host_neighbourhood']
airbnb_data = airbnb_data.drop(randomcolumns, axis=1)

airbnb_data.amenities = airbnb_data.amenities.str.replace(
    '[{""}]', "", regex=True)
amenity_df = airbnb_data.amenities.str.get_dummies(sep=",")
print(amenity_df.info())
airbnb_data = airbnb_data.drop('amenities', axis=1)
airbnb_data = pd.concat([airbnb_data, amenity_df], axis=1)

# Drop amenities with translation missing
trans_miss = ['translation missing: en.hosting_amenity_50']
airbnb_data.drop(trans_miss, axis=1)

# Encode host verification
airbnb_data.host_verifications = airbnb_data.host_verifications.str.replace(
    "['']", "", regex=True)
verification_df = airbnb_data.host_verifications.str.get_dummies(
    sep=",")
airbnb_data = airbnb_data.drop('host_verifications', axis=1)
airbnb_data = pd.concat([airbnb_data, verification_df], axis=1)

for categorical_feature in ['neighbourhood_cleansed', 'property_type', 'room_type', 'bed_type',
                            'cancellation_policy']:
    airbnb_data = pd.concat([airbnb_data, pd.get_dummies(
        airbnb_data[categorical_feature])], axis=1)

# Drop original categorical columns
airbnb_data = airbnb_data.drop(['neighbourhood_cleansed', 'property_type', 'room_type', 'bed_type',
                                'cancellation_policy'], axis=1)

# input(airbnb_data[airbnb_data['price'] > 140]['price'])

plt.figure(figsize=(15, 10))
plt.style.use('seaborn-white')
ax = plt.subplot(221)
plt.boxplot(airbnb_data['price'])
ax.set_title('Price')

airbnb_data = airbnb_data[airbnb_data['price'] <= 140]
airbnb_price = airbnb_data['price']

ax = plt.subplot(222)
plt.boxplot(airbnb_data['price'])
ax.set_title('Price')

airbnb_data = airbnb_data.drop(columns=['price'], axis=1)
print(airbnb_price.describe())
print(airbnb_data.info())
print(airbnb_data.head())

X_train, X_test, y_train, y_test = train_test_split(
    airbnb_data, airbnb_price, test_size=0.1, random_state=42)


print("Starting rgf")

rgf = RGFRegressor(max_leaf=10000,
                   algorithm="RGF_Sib",
                   test_interval=500,
                   loss="LS",
                   verbose=True)

my_pipeline = Pipeline(steps=[('model', rgf)])

n_folds = 3
rgf_scores = cross_val_score(my_pipeline,
                             X_train,
                             y_train,
                             scoring=make_scorer(mean_squared_error),
                             cv=n_folds)

rgf_score = sum(rgf_scores)/n_folds
print(rgf_scores)
print(
    'RGF Regressor Average Cross Validation Train MSE: {0:.5f}'.format(rgf_score))

y_pred_rgf = rgf.fit(X_train, y_train).predict(X_test)
mean_squared_error_pred = mean_squared_error(y_test, y_pred_rgf)
print('RGF Mean_Abs_Square_error : {0:.5f}'.format(mean_squared_error_pred))
# rsquared_score = r2_score(y_test, y_pred_rgf)
# mean_abs_error = mean_absolute_error(y_test, y_pred_rgf)

# print('RGF RSquared Score is : {0:.5f}'.format(rsquared_score))
# print('RGF Mean_Abs_error : {0:.5f}'.format(mean_abs_error))

end = time.time()
taken = end - start
print(f"Time taken : {taken}")
# with outliers in price values
# [max_leaf = 10000, algorithm = 'RGF_Sib', test_interval = 500, loss = 'LS', verbose = True]
# RSquared score is 0.61393 took approx ~290.39 seconds, ~302.61 second, ~272.97 seconds, ~420 seconds
# RGF Regressor MSE: 44724.17397
# RSquared Score is : 0.61393


# get rid of outliers in price values on the high end
# RGF Regressor MSE: 366.36099
# RSquared Score is : 0.66581

# get rid of outliers in price values on both low and high end
# https://github.com/Rawan-Alharbi/Boston-Airbnb-Data-Analysis/blob/master/Boston%20Airbnb%20Data%20Analysis.ipynb
# https://github.com/Dima806/Airbnb_project/blob/master/airbnb_final_analysis_v3.ipynb
# https://github.com/samuelklam/airbnb-pricing-prediction/blob/master/data-cleaning/data-cleaning-listings.ipynb
plt.show()
