import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# read data and parse datetime

df = pd.read_csv("Incidents_Responded_to_by_Fire_Companies.csv", parse_dates = ["INCIDENT_DATE_TIME", "ARRIVAL_DATE_TIME", "LAST_UNIT_CLEARED_DATE_TIME"], low_memory = False)


# ##### 1. What proportion of FDNY responses in this dataset correspond to the most common type of incident?

# count the number of different types and convert to a dataframe
df["INCIDENT_TYPE_DESC"].value_counts().to_frame(name = "counts")

# calculate the fraction of the most common type
fraction = df["INCIDENT_TYPE_DESC"].value_counts()[0]/len(df)

print("{:.10f} of calls are of the most common type of incident".format(fraction))


# ##### 2. What is the ratio of the average number of units that arrive to a scene of an incident classified as '111 - Building fire' to the number that arrive for '651 - Smoke scare, odor of smoke'?

# Calculate the average number of units that arrive to a scene of an incident classified as '111 - Building fire'
unit_111 = df[df["INCIDENT_TYPE_DESC"] == "111 - Building fire"]["UNITS_ONSCENE"].mean()

# Calculate the average number of units that arrive to a scene of an incident classified as '651 - Smoke scare, odor of smoke'
unit_651 = df[df["INCIDENT_TYPE_DESC"] == "651 - Smoke scare, odor of smoke"]["UNITS_ONSCENE"].mean()

# Calculate the ratio of the two avearge number of units
unit_ratio = unit_111/unit_651

print("the ratio of the two types of incidents is {:.10f}".format(unit_ratio))


# ##### 3. How many times more likely is an incident in Staten Island a false call compared to in Manhattan? The answer should be the ratio of Staten Island false call rate to Manhattan false call rate. A false call is an incident for which 'INCIDENT_TYPE_DESC' is '710 - Malicious, mischievous false call, other'.

# Calculate the false call rate of Staten Island
rate_staten = (df[df["BOROUGH_DESC"] == "3 - Staten Island"]["INCIDENT_TYPE_DESC"] == "710 - Malicious, mischievous false call, other").mean()

# Calculate the false call rate of Manhattan
rate_manhattan = (df[df["BOROUGH_DESC"] == "1 - Manhattan"]["INCIDENT_TYPE_DESC"] == "710 - Malicious, mischievous false call, other").mean()

# Calculate the ratio of Staten Island false call rate to Manhattan false call rate
falsecall_ratio = rate_staten/rate_manhattan

print("{:.10f} times more likely is an incident in Staten Island a false call compared to in Manhattan".format(falsecall_ratio))


# ##### 4. Check the distribution of the number of minutes it takes between the time a '111 - Building fire' incident has been logged into the Computer Aided Dispatch system and the time at which the first unit arrives on scene. What is the third quartile of that distribution. Note: the number of minutes can be fractional (ie, do not round).

# Calculate the response time between incident logged into computer and time of arrival
response_time = (df["ARRIVAL_DATE_TIME"] - df["INCIDENT_DATE_TIME"]).dt.seconds/60

# Calculate the third quartile of the distribution
time_quartile = response_time.quantile(q = 0.75)

print("The third quartile of the distribution is {:.10f}".format(time_quartile))


# ##### 5. We can use the FDNY dataset to investigate at what time of the day people cook most. Compute what proportion of all incidents are cooking fires for every hour of the day by normalizing the number of cooking fires in a given hour by the total number of incidents that occured in that hour. Find the hour of the day that has the highest proportion of cooking fires and submit that proportion of cooking fires. A cooking fire is an incident for which 'INCIDENT_TYPE_DESC' is '113 - Cooking fire, confined to container'. Note: round incident times down. For example, if an incident occured at 22:55 it occured in hour 22.

# Calculate the proportion of cook fire to all incidents in each hour 
proportion = df.groupby(df["INCIDENT_DATE_TIME"].dt.hour)["INCIDENT_TYPE_DESC"].agg(lambda x: np.mean(x == "113 - Cooking fire, confined to container"))

# Find the hour of day with the highest proportion
cook_proportion = proportion.sort_values(ascending = False).iloc[0]

print("The hour 18 has the highest proportion of cooking fires with the proportion of {:.10f}".format(cook_proportion))


# #####  6. What is the coefficient of determination (R squared) between the number of residents at each ZIP code and the number of inicidents whose type is classified as '111 - Building fire' at each of those zip codes. Note: the population for each ZIP code in New York state can be found here. Ignore ZIP codes that do not appear on the website.

# read the file for population for each ZIP in New York
df_zip = pd.read_csv("NYC_zip.csv")

# change zip column to float data type
df_zip["Zip Code"] = df_zip["Zip Code"].astype(float)

# change the zip column to the same data type
df["ZIP_CODE_new"] = df["ZIP_CODE"].str.replace('-\d+', '')
df["ZIP_CODE_new"] = df["ZIP_CODE_new"].astype(float)

# subset the original incident datasets to only keep type and zipcode
df_reg = df[["INCIDENT_TYPE_DESC", "ZIP_CODE_new"]]

# get the data only for building fire
df_bf = df_reg[df_reg["INCIDENT_TYPE_DESC"] == "111 - Building fire"]

# use groupby to find the count of incidents for each zip code
df_group = df_bf.groupby("ZIP_CODE_new")["INCIDENT_TYPE_DESC"].count().to_frame().reset_index()

# concatnate the two dataframes based on zip code
df_regression = df_group.merge(df_zip, left_on = "ZIP_CODE_new", right_on = "Zip Code", how = "left")

# change the column name to count
df_regression.rename(columns = {"INCIDENT_TYPE_DESC": "Count"}, inplace = True)

# remove nan
df_regression.dropna(inplace = True)

# calculate the coefficient of OLS using np.polyfit
coefficient, intercept = np.polyfit(df_regression["Population"], df_regression["Count"], deg = 1)

print("the coefficient between he number of residents at each ZIP code and the number of inicident is {:.10f}".format(coefficient))


# ##### 7. Calculate the chi-square test statistic for testing whether an incident is more likely to last longer than 60 minutes when CO detector is not present. Again only consider incidents that have information about whether a CO detector was present or not.

# subset the data for only the incidents with CO detector information
df_CO = df[df["CO_DETECTOR_PRESENT_DESC"].notnull()]

# calculate the time duration for each instance
df_CO["duration"] = (df_CO["LAST_UNIT_CLEARED_DATE_TIME"] - df_CO["ARRIVAL_DATE_TIME"]).dt.seconds

# only keep the duration and CO detector information
df_CO = df_CO[["duration" , "CO_DETECTOR_PRESENT_DESC"]]

# calculate the frequency of each case
# with CO detector and less than 60 minute
chi_11 = df_CO[(df_CO["duration"] <= 3600) & (df_CO["CO_DETECTOR_PRESENT_DESC"] == "Yes")].shape[0]

# with CO detector and longer than 60 minute
chi_12 = df_CO[(df_CO["duration"] > 3600) & (df_CO["CO_DETECTOR_PRESENT_DESC"] == "Yes")].shape[0]

# without CO detector and less than 60 minute
chi_21 = df_CO[(df_CO["duration"] <= 3600) & (df_CO["CO_DETECTOR_PRESENT_DESC"] == "No")].shape[0]

# without CO detector and longer than 60 minute
chi_22 = df_CO[(df_CO["duration"] > 3600) & (df_CO["CO_DETECTOR_PRESENT_DESC"] == "No")].shape[0]

# run the chi-square test
chi2, p, dof, ex = stats.chi2_contingency(np.array([[25498, 738],[5287,829]]), correction=False)

print("the test statistic of chi-squre for CO detector and response duration is {:.10f}".format(chi2))


