import tensorflow as tf
import numpy as np

####### Load models and create their  respective prediction functions #######
##### Expected years of schooling #######
expected_YOS_model = tf.keras.models.load_model('static/models/Expected_years_of_schooling(overall)/Expected_years_of_schooling(overall)_model.h5')
def predict_expected_YOS(years):
    x_input = np.array([10.69855835, 10.69855835, 10.69855835])
    temp_input=list(x_input)
    data_dict={}
    i=0
    while(i<years):

        if(len(temp_input)>3):
            x_input=np.array(temp_input[1:])
            x_input = x_input.reshape((1, 3))
            yhat = tf.keras.activations.relu(expected_YOS_model.predict(x_input, verbose=0), max_value=18)
            temp_input.append(yhat[0][0])
            temp_input=temp_input[1:]
            data_dict[(2022+i)]=tf.cast((yhat[0][0]), tf.float64).numpy()
            i=i+1
        else:
            x_input = x_input.reshape((1, 3))
            yhat = tf.keras.activations.relu(expected_YOS_model.predict(x_input, verbose=0), max_value=18)
            temp_input.append(yhat[0][0])
            data_dict[(2022+i)]=tf.cast((yhat[0][0]), tf.float64).numpy()
            i=i+1
    return data_dict


###### Mean Years of schooling #######
MYOS_model = tf.keras.models.load_model('static/models/Mean_years_of_schooling(overall)/Mean_years_of_schooling(overall)_model.h5')
def predict_MYOS(years):
    x_input = np.array([6.652, 6.652, 6.652])
    temp_input=list(x_input)
    data_dict={}
    i=0
    while(i<years):

        if(len(temp_input)>3):
            x_input=np.array(temp_input[1:])
            x_input = x_input.reshape((1, 3))
            yhat = tf.keras.activations.relu(MYOS_model.predict(x_input, verbose=0), max_value=15)
            temp_input.append(yhat[0][0])
            temp_input=temp_input[1:]
            data_dict[(2022+i)]=tf.cast((yhat[0][0]), tf.float64).numpy()
            i=i+1
        else:
            x_input = x_input.reshape((1, 3))
            yhat = tf.keras.activations.relu(MYOS_model.predict(x_input, verbose=0), max_value=15)
            temp_input.append(yhat[0][0])
            data_dict[(2022+i)]=tf.cast((yhat[0][0]), tf.float64).numpy()
            i=i+1
    return data_dict


###### Gross national income per capita ####
GNI_model = tf.keras.models.load_model('static/models/Gross_national_income_per_capita(overall)/Gross_national_income_per_capita(overall)_model.h5')
def predict_GNI(years):
    x_input = np.array([4381.487855, 4266.967466, 4473.570344])
    temp_input=list(x_input)
    data_dict={}
    i=0
    while(i<years):

        if(len(temp_input)>3):
            x_input=np.array(temp_input[1:])
            x_input = x_input.reshape((1, 3))
            yhat = tf.keras.activations.relu(GNI_model.predict(x_input, verbose=0), max_value=75000)
            temp_input.append(yhat[0][0])
            temp_input=temp_input[1:]
            data_dict[(2022+i)]=tf.cast((yhat[0][0]), tf.float64).numpy()
            i=i+1
        else:
            x_input = x_input.reshape((1, 3))
            yhat = tf.keras.activations.relu(GNI_model.predict(x_input, verbose=0), max_value=75000)
            temp_input.append(yhat[0][0])
            data_dict[(2022+i)]=tf.cast((yhat[0][0]), tf.float64).numpy()
            i=i+1
    return data_dict


###### Life expectancy at birth ######
LEAT_model = tf.keras.models.load_model('static/models/Life_expectancy_at_birth(overall)/Life_expectancy_at_birth(overall)_model.h5')
def predict_Life_expectancy(years):
    x_input = np.array([62.9432, 62.6755, 61.427])
    temp_input=list(x_input)
    data_dict={}
    i=0
    while(i<years):

        if(len(temp_input)>3):
            x_input=np.array(temp_input[1:])
            x_input = x_input.reshape((1, 3))
            yhat = tf.keras.activations.relu(LEAT_model.predict(x_input, verbose=0), max_value=85)
            temp_input.append(yhat[0][0])
            temp_input=temp_input[1:]
            data_dict[(2022+i)]=tf.cast((yhat[0][0]), tf.float64).numpy()
            i=i+1
        else:
            x_input = x_input.reshape((1, 3))
            yhat = tf.keras.activations.relu(LEAT_model.predict(x_input, verbose=0), max_value=85)
            temp_input.append(yhat[0][0])
            data_dict[(2022+i)]=tf.cast((yhat[0][0]), tf.float64).numpy()
            i=i+1
    return data_dict



###########  Carbon emissions ###############
carbon_emissions_model = tf.keras.models.load_model('static/models/carbon/carbon_model.h5')
def predict_carbon_emissions(years):
    x_input = np.array([0.34897306, 0.30027359, 0.300273588])
    temp_input=list(x_input)
    data_dict={}
    i=0
    while(i<years):

        if(len(temp_input)>3):
            x_input=np.array(temp_input[1:])
            x_input = x_input.reshape((1, 3, 1))
            yhat = carbon_emissions_model.predict(x_input, verbose=0)
            temp_input.append(yhat[0][0])
            temp_input=temp_input[1:]
            data_dict[(2022+i)]=tf.cast((yhat[0][0]), tf.float64).numpy()
            i=i+1
        else:
            x_input = x_input.reshape((1, 3, 1))
            yhat = carbon_emissions_model.predict(x_input, verbose=0)
            temp_input.append(yhat[0][0])
            data_dict[(2022+i)]=tf.cast((yhat[0][0]), tf.float64).numpy()
            i=i+1
    return data_dict


######### Material Footprint ##########
material_footprint_model = tf.keras.models.load_model('static/models/Material_footprint_per_capita/Material_footprint_per_capita_model.h5')
def predict_material_footprint(years):
    x_input = np.array([4.62, 4.62, 4.62])
    temp_input=list(x_input)
    data_dict={}
    i=0
    while(i<years):

        if(len(temp_input)>3):
            x_input=np.array(temp_input[1:])
            x_input = x_input.reshape((1, 3, 1))
            yhat = material_footprint_model.predict(x_input, verbose=0)
            temp_input.append(yhat[0][0])
            temp_input=temp_input[1:]
            data_dict[(2022+i)]=tf.cast((yhat[0][0]), tf.float64).numpy()
            i=i+1
        else:
            x_input = x_input.reshape((1, 3, 1))
            yhat = material_footprint_model.predict(x_input, verbose=0)
            temp_input.append(yhat[0][0])
            data_dict[(2022+i)]=tf.cast((yhat[0][0]), tf.float64).numpy()
            i=i+1
    return data_dict


############## Female indices #################
############## Female expected years of schooling ####
female_expected_YOS_model = tf.keras.models.load_model('static/models/Expected_years_of_schooling(female)/Expected_years_of_schooling(female)_model.h5')
def predict_female_expected_YOS(years):
    x_input = np.array([10.34585953, 10.34585953, 10.34585953])
    temp_input=list(x_input)
    data_dict={}
    i=0
    while(i<years):

        if(len(temp_input)>3):
            x_input=np.array(temp_input[1:])
            x_input = x_input.reshape((1, 3))
            yhat = tf.keras.activations.relu(female_expected_YOS_model.predict(x_input, verbose=0), max_value=18)
            temp_input.append(yhat[0][0])
            temp_input=temp_input[1:]
            data_dict[(2022+i)]=tf.cast((yhat[0][0]), tf.float64).numpy()
            i=i+1
        else:
            x_input = x_input.reshape((1, 3))
            yhat = tf.keras.activations.relu(female_expected_YOS_model.predict(x_input, verbose=0), max_value=18)
            temp_input.append(yhat[0][0])
            data_dict[(2022+i)]=tf.cast((yhat[0][0]), tf.float64).numpy()
            i=i+1
    return data_dict



############ Female mean years of schooling ##################
female_MYOS_model = tf.keras.models.load_model('static/models/Mean_years_of_schooling(female)/Mean_years_of_schooling(female)_model.h5')
def predict_female_MYOS(years):
    x_input = np.array([6.064, 6.064, 6.064])
    temp_input=list(x_input)
    data_dict={}
    i=0
    while(i<years):

        if(len(temp_input)>3):
            x_input=np.array(temp_input[1:])
            x_input = x_input.reshape((1, 3))
            yhat = tf.keras.activations.relu(female_MYOS_model.predict(x_input, verbose=0), max_value=15)
            temp_input.append(yhat[0][0])
            temp_input=temp_input[1:]
            data_dict[(2022+i)]=tf.cast((yhat[0][0]), tf.float64).numpy()
            i=i+1
        else:
            x_input = x_input.reshape((1, 3))
            yhat = tf.keras.activations.relu(female_MYOS_model.predict(x_input, verbose=0), max_value=15)
            temp_input.append(yhat[0][0])
            data_dict[(2022+i)]=tf.cast((yhat[0][0]), tf.float64).numpy()
            i=i+1
    return data_dict



####### Female gross national income per capita ########
female_GNI_model = tf.keras.models.load_model('static/models/Gross_national_income_per_capita(female)/Gross_national_income_per_capita(female)_model.h5')
def predict_female_GNI(years):
    x_input = np.array([3983.236518, 3696.272467, 3873.191056])
    temp_input=list(x_input)
    data_dict={}
    i=0
    while(i<years):

        if(len(temp_input)>3):
            x_input=np.array(temp_input[1:])
            x_input = x_input.reshape((1, 3))
            yhat = tf.keras.activations.relu(female_GNI_model.predict(x_input, verbose=0), max_value=75000)
            temp_input.append(yhat[0][0])
            temp_input=temp_input[1:]
            data_dict[(2022+i)]=tf.cast((yhat[0][0]), tf.float64).numpy()
            i=i+1
        else:
            x_input = x_input.reshape((1, 3))
            yhat = tf.keras.activations.relu(female_GNI_model.predict(x_input, verbose=0), max_value=75000)
            temp_input.append(yhat[0][0])
            data_dict[(2022+i)]=tf.cast((yhat[0][0]), tf.float64).numpy()
            i=i+1
    return data_dict



####### Female life expectancy at birth ##############
female_LEAT_model = tf.keras.models.load_model('static/models/Life expectancy at birth(female)/Life_expectancy_at_birth(female)_model.h5')
def predict_female_Life_expectancy(years):
    x_input = np.array([65.2753, 65.0621, 64.0899])
    temp_input=list(x_input)
    data_dict={}
    i=0
    while(i<years):

        if(len(temp_input)>3):
            x_input=np.array(temp_input[1:])
            x_input = x_input.reshape((1, 3))
            yhat = tf.keras.activations.relu(female_LEAT_model.predict(x_input, verbose=0), max_value=85)
            temp_input.append(yhat[0][0])
            temp_input=temp_input[1:]
            data_dict[(2022+i)]=tf.cast((yhat[0][0]), tf.float64).numpy()
            i=i+1
        else:
            x_input = x_input.reshape((1, 3))
            yhat = tf.keras.activations.relu(female_LEAT_model.predict(x_input, verbose=0), max_value=85)
            temp_input.append(yhat[0][0])
            data_dict[(2022+i)]=tf.cast((yhat[0][0]), tf.float64).numpy()
            i=i+1
    return data_dict

########### Function to calculate Female HDI #################
def predict_female_HDI(years):
    data_dict={}
    i=0
    XYOS = predict_female_expected_YOS(years)
    GNI = predict_female_GNI(years)
    LE = predict_female_Life_expectancy(years)
    MYOS = predict_female_MYOS(years)
    while(i<years):
        #####indices#####
        health_index = (LE[(2022+i)]-20)/65
        education_index = ((XYOS[(2022+i)]/18)+(MYOS[(2022+i)]/15))/2
        income_index = ((np.log(GNI[(2022+i)])-np.log(100))/(np.log(75000)-np.log(100)))
        HDI = ((health_index)*(education_index)*(income_index))**(1/3)
        data_dict[(2022+i)] = HDI
        i=i+1
    all_vals = {"Female Life Expectancy at Birth": LE, "Female Expected Years of Schooling": XYOS, "Female Mean Years of Schooling": MYOS, "Female Gross National Income Per Capita (2017 PPP$)": GNI, "Female Human Development Index": data_dict}
    return all_vals



############### Male indices #####################
############### Expected years of schooling ######
male_expected_YOS_model = tf.keras.models.load_model('static/models/Expected_years_of_schooling(male)/Expected_years_of_schooling(male)_model.h5')
def predict_male_expected_YOS(years):
    x_input = np.array([11.05279836, 11.05279836, 11.05279836])
    temp_input=list(x_input)
    data_dict={}
    i=0
    while(i<years):

        if(len(temp_input)>3):
            x_input=np.array(temp_input[1:])
            x_input = x_input.reshape((1, 3))
            yhat = tf.keras.activations.relu(male_expected_YOS_model.predict(x_input, verbose=0), max_value=18)
            temp_input.append(yhat[0][0])
            temp_input=temp_input[1:]
            data_dict[(2022+i)]=tf.cast((yhat[0][0]), tf.float64).numpy()
            i=i+1
        else:
            x_input = x_input.reshape((1, 3))
            yhat = tf.keras.activations.relu(male_expected_YOS_model.predict(x_input, verbose=0), max_value=18)
            temp_input.append(yhat[0][0])
            data_dict[(2022+i)]=tf.cast((yhat[0][0]), tf.float64).numpy()
            i=i+1
    return data_dict



############ Male mean years of schooling ############
male_MYOS_model = tf.keras.models.load_model('static/models/Mean_years_of_schooling(male)/Mean_years_of_schooling(male)_model.h5')
def predict_male_MYOS(years):
    x_input = np.array([7.26, 7.26, 7.26])
    temp_input=list(x_input)
    data_dict={}
    i=0
    while(i<years):

        if(len(temp_input)>3):
            x_input=np.array(temp_input[1:])
            x_input = x_input.reshape((1, 3))
            yhat = tf.keras.activations.relu(male_MYOS_model.predict(x_input, verbose=0), max_value=15)
            temp_input.append(yhat[0][0])
            temp_input=temp_input[1:]
            data_dict[(2022+i)]=tf.cast((yhat[0][0]), tf.float64).numpy()
            i=i+1
        else:
            x_input = x_input.reshape((1, 3))
            yhat = tf.keras.activations.relu(male_MYOS_model.predict(x_input, verbose=0), max_value=15)
            temp_input.append(yhat[0][0])
            data_dict[(2022+i)]=tf.cast((yhat[0][0]), tf.float64).numpy()
            i=i+1
    return data_dict





######### Male gross national income per capita #############3
male_GNI_model = tf.keras.models.load_model('static/models/Gross_national_income_per_capita(male)/Gross_national_income_per_capita(male)_model.h5')
def predict_male_GNI(years):
    x_input = np.array([4786.153902, 4847.056851, 5084.167477])
    temp_input=list(x_input)
    data_dict={}
    i=0
    while(i<years):

        if(len(temp_input)>3):
            x_input=np.array(temp_input[1:])
            x_input = x_input.reshape((1, 3))
            yhat = tf.keras.activations.relu(male_GNI_model.predict(x_input, verbose=0), max_value=75000)
            temp_input.append(yhat[0][0])
            temp_input=temp_input[1:]
            data_dict[(2022+i)]=tf.cast((yhat[0][0]), tf.float64).numpy()
            i=i+1
        else:
            x_input = x_input.reshape((1, 3))
            yhat = tf.keras.activations.relu(male_GNI_model.predict(x_input, verbose=0), max_value=75000)
            temp_input.append(yhat[0][0])
            data_dict[(2022+i)]=tf.cast((yhat[0][0]), tf.float64).numpy()
            i=i+1
    return data_dict



######## Male life expectancy at birth ###########
male_LEAT_model = tf.keras.models.load_model('static/models/Life expectancy at birth(male)/Life_expectancy_at_birth(male)_model.h5')
def predict_male_Life_expectancy(years):
    x_input = np.array([60.6781, 60.3723, 58.9354])
    temp_input=list(x_input)
    data_dict={}
    i=0
    while(i<years):

        if(len(temp_input)>3):
            x_input=np.array(temp_input[1:])
            x_input = x_input.reshape((1, 3))
            yhat = tf.keras.activations.relu(male_LEAT_model.predict(x_input, verbose=0), max_value=85)
            temp_input.append(yhat[0][0])
            temp_input=temp_input[1:]
            data_dict[(2022+i)]=tf.cast((yhat[0][0]), tf.float64).numpy()
            i=i+1
        else:
            x_input = x_input.reshape((1, 3))
            yhat = tf.keras.activations.relu(male_LEAT_model.predict(x_input, verbose=0), max_value=85)
            temp_input.append(yhat[0][0])
            data_dict[(2022+i)]=tf.cast((yhat[0][0]), tf.float64).numpy()
            i=i+1
    return data_dict



########### Function to calculate male HDI ##############
def predict_male_HDI(years):
    data_dict={}
    i=0
    XYOS = predict_male_expected_YOS(years)
    GNI = predict_male_GNI(years)
    LE = predict_male_Life_expectancy(years)
    MYOS = predict_male_MYOS(years)
    while(i<years):
        #####indices#####
        health_index = (LE[(2022+i)]-20)/65
        education_index = ((XYOS[(2022+i)]/18)+(MYOS[(2022+i)]/15))/2
        income_index = ((np.log(GNI[(2022+i)])-np.log(100))/(np.log(75000)-np.log(100)))
        HDI = ((health_index)*(education_index)*(income_index))**(1/3)
        data_dict[(2022+i)] = HDI
        i=i+1
    all_vals = {"Male Life Expectancy at Birth": LE, "Male Expected Years of Schooling": XYOS, "Male Mean Years of Schooling": MYOS, "Male Gross National Income Per Capita (2017 PPP$)": GNI, "Male Human Development Index": data_dict}
    return all_vals



############### Function to calculate metrics from predicted values ####################
def predict_HDI(years):
    HDI_dict={}
    female_HDI_dict={}
    male_HDI_dict={}
    PHDI_dict={}
    GDI_dict={}
    i=0
    ######### overall Indices #########
    XYOS = predict_expected_YOS(years)
    GNI = predict_GNI(years)
    LE = predict_Life_expectancy(years)
    MYOS = predict_MYOS(years)
    ######### female Indices #########
    female_XYOS = predict_female_expected_YOS(years)
    female_GNI = predict_female_GNI(years)
    female_LE = predict_female_Life_expectancy(years)
    female_MYOS = predict_female_MYOS(years)
    ######### male Indices #########
    male_XYOS = predict_male_expected_YOS(years)
    male_GNI = predict_male_GNI(years)
    male_LE = predict_male_Life_expectancy(years)
    male_MYOS = predict_male_MYOS(years)
    ######################## for PHDI#########################
    carbon_emissions = predict_carbon_emissions(years)
    material_footprint = predict_material_footprint(years)
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
