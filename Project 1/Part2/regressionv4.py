import pandas as pd
import matplotlib.pyplot as plt
from pandas.core.indexes import numeric
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import (
    make_scorer,
    mean_squared_error,
    r2_score,
    mean_absolute_error,
)
from rgf.sklearn import RGFRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, OneHotEncoder
from timeit import default_timer as timer
from datetime import timedelta
import math

airbnb_london_listing = "./londonairbnb/listings detailed.csv"
airbnb_data = pd.read_csv(airbnb_london_listing)
# print(airbnb_data.info())
pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)
print("Initial Overview")
print(airbnb_data.isnull().sum().sort_values(ascending=False))
pd.reset_option("all")
missing_more_than_50 = list(airbnb_data.columns[airbnb_data.isnull().mean() > 0.50])
for x in missing_more_than_50:
    print(x)
airbnb_data = airbnb_data.drop(missing_more_than_50, axis=1)
print("After removing columns missing 50 percent of data")
print(airbnb_data.info())

airbnb_data["price"] = airbnb_data["price"].str.replace("$", "", regex=True)
airbnb_data["price"] = (
    airbnb_data["price"].str.replace(",", "", regex=True).astype(float)
)
airbnb_data["security_deposit"] = airbnb_data["security_deposit"].str.replace(
    "$", "", regex=True
)
airbnb_data["security_deposit"] = (
    airbnb_data["security_deposit"].str.replace(",", "", regex=True).astype(float)
)
airbnb_data["cleaning_fee"] = airbnb_data["cleaning_fee"].str.replace(
    "$", "", regex=True
)
airbnb_data["cleaning_fee"] = (
    airbnb_data["cleaning_fee"].str.replace(",", "", regex=True).astype(float)
)
airbnb_data["extra_people"] = airbnb_data["extra_people"].str.replace(
    "$", "", regex=True
)
airbnb_data["extra_people"] = (
    airbnb_data["extra_people"].str.replace(",", "", regex=True).astype(float)
)

airbnb_data["host_acceptance_rate"] = (
    airbnb_data["host_acceptance_rate"].str.replace("%", "", regex=True).astype(float)
)

airbnb_data["host_acceptance_rate"] = airbnb_data["host_acceptance_rate"] * 0.01


columntf = [
    "require_guest_profile_picture",
    "require_guest_phone_verification",
    "host_is_superhost",
    "host_has_profile_pic",
    "host_identity_verified",
    "is_location_exact",
    "has_availability",
    "requires_license",
    "instant_bookable",
    "is_business_travel_ready",
]

airbnb_data[columntf] = airbnb_data[columntf].replace({"t": 1, "f": 0})


ids = ["id", "scrape_id", "host_id"]
airbnb_data = airbnb_data.drop(ids, axis=1)

urls = [
    "listing_url",
    "picture_url",
    "host_url",
    "host_thumbnail_url",
    "host_picture_url",
]

airbnb_data = airbnb_data.drop(urls, axis=1)

# summary and description have same info, but description has less null values
# same with street and city
columnsimiliarinfo = ["summary", "street", "neighbourhood"]
airbnb_data = airbnb_data.drop(columnsimiliarinfo, axis=1)

columnunecessary = [
    "market",
    "country_code",
    "country",
    "smart_location",
    "name",
    "state",
    "city",
    "latitude",
    "longitude",
    "zipcode",
]
airbnb_data = airbnb_data.drop(columnunecessary, axis=1)

columndates = [
    "last_scraped",
    "calendar_last_scraped",
    "first_review",
    "last_review",
    "host_since",
    "calendar_updated",
]
airbnb_data = airbnb_data.drop(columndates, axis=1)

randomcolumns = [
    "space",
    "description",
    "experiences_offered",
    "neighborhood_overview",
    "transit",
    "access",
    "interaction",
    "house_rules",
    "host_name",
    "host_location",
    "host_about",
    "host_neighbourhood",
]
airbnb_data = airbnb_data.drop(randomcolumns, axis=1)
print("After removing unneccessary columns")
print(airbnb_data.info())

numerical_ix = airbnb_data.select_dtypes(include=["int64", "float64"]).columns
categorical_ix = airbnb_data.select_dtypes(include=["object"]).columns

numeric_transformer = SimpleImputer(missing_values=np.NaN, strategy="mean")
categorical_transformer = SimpleImputer(missing_values=np.NaN, strategy="most_frequent")
LB = LabelBinarizer()
OHE = OneHotEncoder()

airbnb_data[numerical_ix] = numeric_transformer.fit_transform(airbnb_data[numerical_ix])
airbnb_data[columntf] = airbnb_data[columntf].astype(np.int64)

airbnb_data[categorical_ix] = categorical_transformer.fit_transform(
    airbnb_data[categorical_ix]
)

print("After using SimpleImputer to replace missing values")
print(airbnb_data.info())

airbnb_data["amenities"] = airbnb_data["amenities"].str.replace(
    '[{""}]', "", regex=True
)

airbnb_data["host_verifications"] = airbnb_data["host_verifications"].str.replace(
    "['']", "", regex=True
)
airbnb_data["host_verifications"] = airbnb_data["host_verifications"].str.replace(
    "[", "", regex=True
)
airbnb_data["host_verifications"] = airbnb_data["host_verifications"].str.replace(
    "]", "", regex=True
)
airbnb_data["host_verifications"] = airbnb_data["host_verifications"].str.replace(
    " ", "", regex=True
)

for x in categorical_ix:
    print(x)
    if x == "host_verifications":
        tempdf = airbnb_data["host_verifications"].str.get_dummies(sep=",")
        tempdf = tempdf.drop(["None"], axis=1)
    elif x == "amenities":
        tempdf = airbnb_data["amenities"].str.get_dummies(sep=",")
        tempdf = tempdf.drop("translation missing: en.hosting_amenity_50", axis=1)
        tempdf = tempdf.drop("translation missing: en.hosting_amenity_49", axis=1)
        tempdf = tempdf.drop("translation missing: en.hosting_amenity_105", axis=1)
    else:
        tempdf = pd.get_dummies(airbnb_data[x])
    airbnb_data = pd.concat([airbnb_data, tempdf], axis=1)
    airbnb_data = airbnb_data.drop(columns=x, axis=1)


# plt.figure(figsize=(15, 10))
# plt.style.use("seaborn-white")
# ax = plt.subplot(221)
# plt.boxplot(airbnb_data["price"])
# ax.set_title("Price")

#airbnb_data = airbnb_data[airbnb_data["price"] <= 140]
airbnb_price = airbnb_data["price"]
print("Price mean was {}".format(airbnb_price.mean()))
# ax = plt.subplot(222)
# plt.boxplot(airbnb_data['price'])
# ax.set_title('Price')

airbnb_data = airbnb_data.drop(columns=["price"], axis=1)
print("Prints info about Price")
print(airbnb_price.describe())
print("Before sending the data off to the forest")
print(airbnb_data.info())

X_train, X_test, y_train, y_test = train_test_split(
    airbnb_data, airbnb_price, test_size=0.1, random_state=42
)

print("Starting rgf")
start = timer()
rgf = RGFRegressor(
    max_leaf=10000, algorithm="RGF_Sib", test_interval=500, loss="LS", verbose=True
)
rgf.fit(X_train, y_train)
# rgf_train_score = rgf.score(X_train, y_train)
# rgf_test_score = rgf.score(X_test, y_test)
rgf_y_train_pred = rgf.predict(X_train)
rgf_y_test_pred = rgf.predict(X_test)
train_sqrt_mean_squared_error_pred = math.sqrt(
    mean_absolute_error(y_train, rgf_y_train_pred)
)
test_sqrt_mean_squared_error_pred = math.sqrt(
    mean_absolute_error(y_test, rgf_y_test_pred)
)


# print("RGF Regressor Train Score: {0:.5f}".format(rgf_train_score))
# print("RGF Regressor Test Score: {0:.5f}".format(rgf_test_score))
print("Price mean was {}".format(airbnb_price.mean()))
print(
    "RGF Regressor Train Squared Root Mean_Square_error : {0:.5f}".format(
        train_sqrt_mean_squared_error_pred
    )
)
print(
    "RGF Regressor Test Squared Root Mean_Square_error : {0:.5f}".format(
        test_sqrt_mean_squared_error_pred
    )
)

end = timer()
taken = end - start
timein = "tmp"
if taken > 3600:
    timein = "hours"
elif taken > 60 < 3600:
    timein = "minutes"
else:
    timein = "seconds"
print(f"It took us {timedelta(seconds=taken)} {timein} to run this program")


# get rid of outliers in price values on both low and high end
# https://github.com/Rawan-Alharbi/Boston-Airbnb-Data-Analysis/blob/master/Boston%20Airbnb%20Data%20Analysis.ipynb
# https://github.com/Dima806/Airbnb_project/blob/master/airbnb_final_analysis_v3.ipynb
# https://github.com/samuelklam/airbnb-pricing-prediction/blob/master/data-cleaning/data-cleaning-listings.ipynb
# plt.show()
