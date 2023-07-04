import tensorflow as tf
import numpy as np


#loop to load all the models from the same directory
models = [
    "Expected_years_of_schooling(overall)", 
    "Mean_years_of_schooling(overall)", 
    "Gross_national_income_per_capita(overall)", 
    "Life_expectancy_at_birth(overall)", 
    "carbon", 
    "Material_footprint_per_capita", 
    "Expected_years_of_schooling(female)", 
    "Mean_years_of_schooling(female)", 
    "Gross_national_income_per_capita(female)", 
    "Life_expectancy_at_birth(female)", 
    "Expected_years_of_schooling(male)", 
    "Mean_years_of_schooling(male)", 
    "Gross_national_income_per_capita(male)", 
    "Life_expectancy_at_birth(male)"
    ] # list all the model names
model_dict = {} # create an empty dictionary to store the models
for model in models: # loop through the model names
    model_dict[model] = tf.keras.models.load_model(f'static/models/{model}/{model}_model.h5') # load the model and store it in the dictionary with the same name

# Function to do prediction
def predict_indicator(model_name, x_input, years, max_value:None, reshape:tuple): # define a function that takes a model name and an input array as parameters
    model = model_dict[model_name] # get the model from the dictionary
    temp_input=list(x_input) # create a temporary list from the input array
    data_dict={} # create an empty dictionary to store the predictions
    i=0 # initialize a counter
    while(i<years): # loop for the number of years
        if(len(temp_input)>3): # if the temporary list has more than 3 elements
            x_input=np.array(temp_input[1:]) # use the last 3 elements as the input array
            x_input = x_input.reshape(reshape) # reshape the input array to match the model input shape
            yhat = tf.keras.activations.relu(model.predict(x_input, verbose=0), max_value=max_value) # predict the output using the model and apply a relu activation with a max value of 18
            temp_input.append(yhat[0][0]) # append the prediction to the temporary list
            temp_input=temp_input[1:] # remove the first element from the temporary list
            data_dict[(2022+i)]=tf.cast((yhat[0][0]), tf.float64).numpy() # store the prediction in the data dictionary with the year as the key
            i=i+1 # increment the counter
        else: # if the temporary list has 3 or less elements
            x_input = x_input.reshape(reshape) # reshape the input array to match the model input shape
            yhat = tf.keras.activations.relu(model.predict(x_input, verbose=0), max_value=max_value) # predict the output using the model and apply a relu activation with a max value of 18
            temp_input.append(yhat[0][0]) # append the prediction to the temporary list
            data_dict[(2022+i)]=tf.cast((yhat[0][0]), tf.float64).numpy() # store the prediction in the data dictionary with the year as the key
            i=i+1 # increment the counter
    return data_dict # return the data dictionary

############### Function to calculate metrics from predicted values ####################
def predict_HDI(years):
    HDI_dict={}
    female_HDI_dict={}
    male_HDI_dict={}
    PHDI_dict={}
    GDI_dict={}
    i=0
    ######### overall Indices #########
    XYOS = predict_indicator("Expected_years_of_schooling(overall)", np.array([10.69855835, 10.69855835, 10.69855835]), years, 18, (1, 3))
    GNI = predict_indicator("Gross_national_income_per_capita(overall)", np.array([4381.487855, 4266.967466, 4473.570344]), years, 75000, (1, 3))
    LE = predict_indicator("Life_expectancy_at_birth(overall)", np.array([62.9432, 62.6755, 61.427]), years, 85, (1, 3))
    MYOS = predict_indicator("Mean_years_of_schooling(overall)", np.array([6.652, 6.652, 6.652]), years, 15, (1, 3))
    ######### female Indices #########
    female_XYOS = predict_indicator("Expected_years_of_schooling(female)", np.array([10.34585953, 10.34585953, 10.34585953]), years, 18, (1, 3))
    female_GNI = predict_indicator("Gross_national_income_per_capita(female)", np.array([3983.236518, 3696.272467, 3873.191056]), years, 75000, (1, 3))
    female_LE = predict_indicator("Life_expectancy_at_birth(female)", np.array([65.2753, 65.0621, 64.0899]), years, 85, (1, 3))
    female_MYOS = predict_indicator("Mean_years_of_schooling(female)", np.array([6.064, 6.064, 6.064]), years, 15, (1, 3))
    ######### male Indices #########
    male_XYOS = predict_indicator("Expected_years_of_schooling(male)", np.array([11.05279836, 11.05279836, 11.05279836]), years, 18, (1, 3))
    male_GNI = predict_indicator("Gross_national_income_per_capita(male)", np.array([4786.153902, 4847.056851, 5084.167477]), years, 75000, (1, 3))
    male_LE = predict_indicator("Life_expectancy_at_birth(male)", np.array([60.6781, 60.3723, 58.9354]), years, 85, (1, 3))
    male_MYOS = predict_indicator("Mean_years_of_schooling(male)", np.array([7.26, 7.26, 7.26]), years, 15, (1, 3))
    ######################## for PHDI#########################
    carbon_emissions = predict_indicator("carbon", np.array([0.34897306, 0.30027359, 0.300273588]), years, None, (1, 3, 1))
    material_footprint = predict_indicator("Material_footprint_per_capita", np.array([4.62, 4.62, 4.62]), years, None, (1, 3, 1))
    ##########################################################
    while(i<years):
        ################ overall HDI indices #############
        health_index = (LE[(2022+i)]-20)/65
        education_index = ((XYOS[(2022+i)]/18)+(MYOS[(2022+i)]/15))/2
        income_index = ((np.log(GNI[(2022+i)])-np.log(100))/(np.log(75000)-np.log(100)))
        HDI = ((health_index)*(education_index)*(income_index))**(1/3)
        HDI_dict[(2022+i)] = HDI
        ################ female HDI indices #############
        female_health_index = (female_LE[(2022+i)]-20)/65
        female_education_index = ((female_XYOS[(2022+i)]/18)+(female_MYOS[(2022+i)]/15))/2
        female_income_index = ((np.log(female_GNI[(2022+i)])-np.log(100))/(np.log(75000)-np.log(100)))
        female_HDI = ((female_health_index)*(female_education_index)*(female_income_index))**(1/3)
        female_HDI_dict[(2022+i)] = female_HDI
        ################ female HDI indices #############
        male_health_index = (male_LE[(2022+i)]-20)/65
        male_education_index = ((male_XYOS[(2022+i)]/18)+(male_MYOS[(2022+i)]/15))/2
        male_income_index = ((np.log(male_GNI[(2022+i)])-np.log(100))/(np.log(75000)-np.log(100)))
        male_HDI = ((male_health_index)*(male_education_index)*(male_income_index))**(1/3)
        male_HDI_dict[(2022+i)] = male_HDI
        ################### PHDI calculation ########
        carbon_emissions_index = (69.85-(carbon_emissions[(2022+i)]))/69.85
        material_footprint_index = (152.58-(material_footprint[(2022+i)]))/152.58
        A_index = (carbon_emissions_index+material_footprint_index)/2
        PHDI = (A_index*HDI)
        PHDI_dict[(2022+i)] = PHDI
        ################### GDI calculation ##########
        GDI = female_HDI/male_HDI
        GDI_dict[(2022+i)] = GDI
        i=i+1
    all_vals = {"Life Expectancy at Birth": LE, "Expected Years of Schooling": XYOS, "Mean Years of Schooling": MYOS, "Gross National Income Per Capita (2017 PPP$)": GNI, "Human Development Index": HDI_dict, "Carbon dioxide emissions per capita (production) (tonnes)": carbon_emissions, "Material footprint per capita (tonnes)": material_footprint, "PHDI": PHDI_dict, "GDI": GDI_dict, "Female HDI": female_HDI_dict, "Male HDI": male_HDI_dict}
    return all_vals


################## Web backend with flask ############################
from flask import Flask, request, jsonify, render_template, redirect, url_for
import json
import os


app = Flask(__name__)
# predict_HDI(years)
@app.route("/", methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        years = int(request.form['years']) # Your form's
        final_year=2021+years
        results = predict_HDI(years)
        return render_template("dashboard.html", data=json.dumps(results), years=years)
    else:
        return """
        <!Doctype html>
        
    <h1>WELCOME to test!</h1>
    <form id='form' method='post'>
      <label for="year">Predict years into the future:</label><br>
      <input type="number" name="years"><br>
      <input type="submit" value="Submit">
    </form>
    </html>
    """
@app.route("/end-home")
def end_home():
    return render_template("ILINAF frontend/index.html")
@app.route("/end-dashboard")
def end_dashboard():
    return render_template("ILINAF frontend/dashboard.html")


# main driver function
if __name__ == '__main__':
 
    # run() method of Flask class runs the application
    # on the local development server.
    app.run()