#imports

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
import pickle
import numpy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from xgboost.sklearn import XGBClassifier

#functions
def apply_regression(data,previa):
    model=LinearRegression()
    y=data.loc[: , previa]
    x=data.loc[: , data.drop(previa, axis=1).columns]
    X_train, X_test, Y_train, Y_test= train_test_split(x,y,train_size=0.8)

    model_fit=model.fit(X_train, Y_train)
    predict = model_fit.predict(X_test)

    relative_error=np.abs(predict-Y_test)/np.abs(Y_test)
    relative_error_mean=relative_error.mean()

    return model, X_test, x, Y_test,relative_error_mean,predict


def insert_zipper(idx,dataframe):
    inputs=[]
    columnes=dataframe.columns
    for i in range(dataframe.shape[1]):
        if columnes[i]=='Familia' or columnes[i]=='Stopers' or columnes[i]=='Sliders' or columnes[i]=='Teeth' or columnes[i]=='Color' or columnes[i]=='Llargada' or columnes[i]=='Label' :
            keys=str(idx+1)+ str(i+1)
            #input=st.number_input()
            text=st.text_input('Insert the'+ columnes[i]+ 'Code:',key=keys)
            inputs.append(text)
    df=pd.DataFrame([inputs], columns=['Familia', 'Stopers', 'Sliders', 'Label','Teeth', 'Color', 'Llargada'])    
    return df



def get_quartil1(aux, quartils,id,num):
    q=[]
    a=0
    if id==0 or num==1: 
        if (int(aux)) > (int(quartils[3])):
            q.append('Més de ' + str(quartils[3]))
        if (int(aux)> int(quartils[2])) and (int(aux)<= int(quartils[3])):
            if (int(aux)> int(quartils[2])) and (int(aux)<= int(quartils[2]*3)):
                q.append(str(quartils[2])+'-'+ str((quartils[2]*3)))
            if (int(aux)> int(quartils[2]*3)) and (int(aux)<= (int(quartils[2])*6)):
                q.append(str(quartils[2]*3)+'-'+ str((quartils[2]*6)))
            if (int(aux) > int(quartils[2]*6)) and (int(aux)<= int(quartils[2]*9)):
                q.append(str(quartils[2]*6)+ '-'+ str((quartils[2]*9)))
            if (int(aux) > int(quartils[2]*9)) and (int(aux)<= int(quartils[2]*12)):
                q.append(str(quartils[2]*9)+ '-'+ str((quartils[2]*12)))
            if (int(aux) > int(quartils[2]*12)) and (int(aux)<= int(quartils[2]*15)):           
                q.append(str(quartils[2]*12)+'-'+str(quartils[2]*15))
            if (int(aux) > int(quartils[2]*15)) and (int(aux)<= int(quartils[3])): 
                q.append(str(quartils[2]*15) + '-' + str(quartils[3]))  
        if (int(aux)> int(quartils[1])) and (int(aux)<= int(quartils[2])):
            q.append(str(quartils[1])+'-'+str(quartils[2]))
        if (int(aux)> int(quartils[0])) and (int(aux)<= int(quartils[1])):
            q.append(str(quartils[0])+'-'+str(quartils[1]))
        if (int(aux)>=int(a)) and (int(aux) <=int(quartils[0])):
            q.append('0'+ '-' +str(quartils[0]))

    return q


def rename_columns(anterior,dataset_principi,quartilss,id,num,):
    if id==0:
        if num==1:
            q=[]
            new_cols={}
            llista=[]
            for i in range(anterior.shape[1]):
                q.append(get_quartil1(anterior.iloc[0][i], quartilss,0,num))
            for k in range(len(q)):
                llista.append(q[k])
            d = pd.DataFrame(llista)
            mitad = len(d) // num
            d=pd.DataFrame(d.values.reshape(mitad, num))
        else: 
            quart=[]
            for i in range(anterior.shape[1]):
                for j in range(anterior.shape[0]):
                    quart.append(get_quartil1(anterior.iloc[j][i],quartilss,0,num))
            filas = []
            for elemento in quart:
                fila = [elemento]
                filas.append(fila)
            d = pd.DataFrame(filas)
            
            if d.shape[1]>1:
                d=d.drop(d.columns[1],axis=1)
                mitad = len(d) // num
                d=pd.DataFrame(d.values.reshape(mitad, num))
            else: 
                mitad = len(d) // num
                d=pd.DataFrame(d.values.reshape(mitad, num))

    else:
        d = pd.DataFrame(anterior)
        d=d.transpose()

    return d

def cambiar_cols(dataset):
    new_cols = {}
    for i, col in enumerate(dataset.columns[7:]):
        new_col_name = int(i)+7
        new_cols[col] = new_col_name 
    dataset = dataset.rename(columns=new_cols)

    return dataset


def convert_to_int(indexs_cremalleres):
    indexs_cremalleres1=[]
    for i in range(len(indexs_cremalleres)):
        indexs_cremalleres1.append(int(indexs_cremalleres[i]))
    return indexs_cremalleres1

def get_tot(indexs_cremalleres,df,previa,idx):
    
    aux_aux=[]
    prova=[]
    fila=pd.DataFrame()
    for i in range(len(indexs_cremalleres)):
        fila = df.iloc[(indexs_cremalleres[i])]
        prova.append(fila)
    
    aux=pd.DataFrame()
    for i in range(len(prova)):
        aux[i]=prova[i]
    
    new_data = aux[:7]
    new_data1= aux[7:]
    new_data=new_data.transpose()
    new_data1=new_data1.transpose()
    
    valors=[]
    valors1=[]
    for i in range(new_data.shape[0]):
        valors.append(new_data.iloc[i])
    for i in range(new_data1.shape[0]):
        valors1.append(new_data1.iloc[i])

    new_data=pd.DataFrame(valors)
    new_data1=pd.DataFrame(valors1)
    
    aux=pd.concat([new_data, new_data1], axis=1)
    if idx==0:
        aux_aux.append(aux[int(previa)])
    else: 
        aux_aux.append(aux[previa])
    
    return new_data,aux_aux

def get_frames(indexs_cremalleres,interes1,dataset,idx,num):    
    anterior=[]
    if idx==1:
        for i,interes in enumerate(indexs_cremalleres):
            fila=dataset[interes1].iloc[indexs_cremalleres[i]]
            anterior.append(fila)
            
    else:
        for j in range(len(interes1)):
            for i,interes in enumerate(indexs_cremalleres):
                fila=dataset[int(interes1[j])].iloc[int(indexs_cremalleres[i])]
                anterior.append(fila)
 
    a=pd.DataFrame(anterior) 
    mitad = len(a) // num
    a=a.transpose()
    
    dataset1=pd.DataFrame(a.values.reshape(num, mitad))
    return dataset1

def change_name_rows(d):
    llista=[]
    for i in range(len(d)):
        llista.append(i)
    new_row_names = {i: llista[i] for i in range(len(llista))}
    r = d.rename(index=new_row_names)
    return r
        
def change_name_cols(d,id):
    if id==1: 
        j=1
        llista=[]
        for i in range(d.shape[1]):
            llista.append('Quantitat cremallers vengudes del' +' '+  str(j) +' '+ 'mes seguent del any anterior')
            j=j+1
        new_row_names = {i: llista[i] for i in range(len(llista))}
        r = d.rename(columns=new_row_names)
    else: 
        j=1
        llista=[]
        for i in range(d.shape[1]):
            llista.append(str(j) +'' +'Prediction next month')
            j=j+1
        new_row_names = {i: llista[i] for i in range(len(llista))}
        r = d.rename(columns=new_row_names)
    return r

def zippers_model(df_not_encoded11,df, dataset,df_not_encoded, important ,predict, quartils,relative_error_mean_reduced,previa,dataset_principi,num,dataset_principi_modified,df_encoded1):
    caracteristicas = st.text_input("Introdueix las características que vols fer la producció (separadas por una coma i amb MAJÚSCULES): ")
    caracteristicas = caracteristicas.split(',')

    cremalleres_filtrades = pd.DataFrame()
    indexs_cremalleres=[]
    for index, row in df_not_encoded.iterrows():
        contains_all_characteristics = True
        for caracteristica in caracteristicas:
            if (str(caracteristica) not in str(row['Familia'])) and (str(caracteristica) not in str(row['Stopers'])) and (str(caracteristica) not in str(row['Sliders'])) and (str(caracteristica) not in str(row['Label'])) and (str(caracteristica) not in str(row['Teeth'])) and (str(caracteristica) not in str(row['Color'])) and (str(caracteristica) not in str(row['Llargada'])):
                contains_all_characteristics = False
                break    
        if contains_all_characteristics:
            cremalleres_filtrades = cremalleres_filtrades.append(row)
            indexs_cremalleres.append(index)

    cremalleres_filtrades = cremalleres_filtrades.reset_index(drop=True)
    indexs_cremalleres=convert_to_int(indexs_cremalleres)

    aux,aux_aux=get_tot(indexs_cremalleres,df_not_encoded,previa,1)
    aux1,aux_aux1=get_tot(indexs_cremalleres,dataset,previa,1)

  
    df_not_encoded11_per_funcio=df_not_encoded11.copy()
    new_cols = {}
    for i, col in enumerate(df_not_encoded11_per_funcio.columns[7:]):
        new_col_name = int(i)+7
        new_cols[col] = new_col_name 
    df_not_encoded11_per_funcio = df_not_encoded11_per_funcio.rename(columns=new_cols)
    aux2,aux_aux2=get_tot(indexs_cremalleres,df_not_encoded11_per_funcio,previa,0)
    
    dataset_principi=cambiar_cols(dataset_principi)
    dataset_principi_modified=cambiar_cols(dataset_principi_modified)
    df_encoded1=cambiar_cols(df_encoded1)

    
    ultima=dataset_principi.shape[1]
    s=ultima-1
    s=int(s)
    dataset_principi_prova=dataset_principi.copy()
    for i, col in enumerate(dataset_principi_prova.columns[7:]):
        new_col_name = int(i)+7
        new_cols[col] = new_col_name 
    dataset_principi_prova = dataset_principi_prova.rename(columns=new_cols)


    if num==1:
        interes=s-12
        anterior=get_frames(indexs_cremalleres,interes,dataset_principi,1,num)
        #anterior1=get_frames(indexs_cremalleres,interes,dataset_principi_modified,1,num)
        anterior2=get_frames(indexs_cremalleres,interes,df_encoded1,1,num)

        d=rename_columns(anterior,dataset_principi,quartils,1,num)
        d1=rename_columns(anterior,dataset_principi_modified,quartils,0,num)
        d2=rename_columns(anterior2,df_encoded1,quartils,1,num)

        r=change_name_rows(d)
        r=change_name_cols(r,1)
        r1=change_name_rows(d1)
        r1=change_name_cols(r1,0)
        r2=change_name_rows(d2)
        r2=change_name_cols(r2,0)

    if num==2:
        interes=[s-12,s-13]
        anterior=get_frames(indexs_cremalleres,interes,dataset_principi,0,num)
        #anterior1=get_frames(indexs_cremalleres,interes,dataset_principi_modified,0,num)
        anterior2=get_frames(indexs_cremalleres,interes,df_encoded1,0,num)

        d=rename_columns(anterior,dataset_principi,quartils,1,num)
        d1=rename_columns(anterior,dataset_principi_modified,quartils,0,num)
        d2=rename_columns(anterior2,df_encoded1,quartils,1,num)


        r=change_name_rows(d)
        r=change_name_cols(r,1)
        r1=change_name_rows(d1)
        r1=change_name_cols(r1,0)
        r2=change_name_rows(d2)
        r2=change_name_cols(r2,0)
        

    if num==3: 
        interes=[s-12,s-13,s-14]
        
        anterior=get_frames(indexs_cremalleres,interes,dataset_principi,0,num)
        #anterior1=get_frames(indexs_cremalleres,interes,dataset_principi_modified,0,num)
        anterior2=get_frames(indexs_cremalleres,interes,df_encoded1,0,num)
        
        d=rename_columns(anterior,dataset_principi,quartils,1,num)
        d1=rename_columns(anterior,dataset_principi_modified,quartils,0,num)
        d2=rename_columns(anterior2,df_encoded1,quartils,1,num)

        r=change_name_rows(d)
        r=change_name_cols(r,1)
        r1=change_name_rows(d1)
        r1=change_name_cols(r1,0)
        r2=change_name_rows(d2)
        r2=change_name_cols(r2,0)

    if num==4:
        interes=[s-12,s-13,s-14, s-15]
        anterior=get_frames(indexs_cremalleres,interes,dataset_principi,0,num)
        #anterior1=get_frames(indexs_cremalleres,interes,dataset_principi_modified,0,num)
        anterior2=get_frames(indexs_cremalleres,interes,df_encoded1,0,num)
        
        d=rename_columns(anterior,dataset_principi,quartils,1,num)
        d1=rename_columns(anterior,dataset_principi_modified,quartils,0,num)
        d2=rename_columns(anterior2,df_encoded1,quartils,1,num)

        r=change_name_rows(d)
        r=change_name_cols(r,1)
        r1=change_name_rows(d1)
        r1=change_name_cols(r1,0)
        r2=change_name_rows(d2)
        r2=change_name_cols(r2,0)
        
    if num==6:
        interes=[s-12,s-13, s-14, s-15, s-16, s-17]
        anterior=get_frames(indexs_cremalleres,interes,dataset_principi,0,num)
        #anterior1=get_frames(indexs_cremalleres,interes,dataset_principi_modified,0,num)
        anterior2=get_frames(indexs_cremalleres,interes,df_encoded1,0,num)
        
        d=rename_columns(anterior,dataset_principi,quartils,1,num)
        d1=rename_columns(anterior,dataset_principi_modified,quartils,0,num)
        d2=rename_columns(anterior2,df_encoded1,quartils,1,num)

        r=change_name_rows(d)
        r=change_name_cols(r,1)
        r1=change_name_rows(d1)
        r1=change_name_cols(r1,0)
        r2=change_name_rows(d2)
        r2=change_name_cols(r2,0)
        

    if num==12: 
        interes=[s-12,s-13, s-14, s-15, s-16, s-17,s-18,s-19,s-20,s-21,s-22,s-23]
        anterior=get_frames(indexs_cremalleres,interes,dataset_principi,0,num)
        #anterior1=get_frames(indexs_cremalleres,interes,dataset_principi_modified,0,num)
        anterior2=get_frames(indexs_cremalleres,interes,df_encoded1,0,num)
        d=rename_columns(anterior,dataset_principi,quartils,1,num)
        d1=rename_columns(anterior,dataset_principi_modified,quartils,0,num)
        d2=rename_columns(anterior2,df_encoded1,quartils,1,num)

        r=change_name_rows(d)
        r=change_name_cols(r,1)
        r1=change_name_rows(d1)
        r1=change_name_cols(r1,0)
        r2=change_name_rows(d2)
        r2=change_name_cols(r2,0)


    result = pd.concat([aux, r], axis=1)
    result1= pd.concat([aux, r1],axis=1)
    result2=pd.concat([aux, r2],axis=1)

    if num!=1:        
        suma_filas = result2.iloc[:, 7:].sum(axis=1)
        result2['Total']=suma_filas  
        suma_files=result.iloc[:,7:].sum(axis=1)
        result['Total']=suma_files

        quart=[]
        for i in range(len(suma_files)):
            quart.append(get_quartil1(suma_files.iloc[i],quartils,0,num))

        filas = []
        for elemento in quart:
            fila = [elemento]
            filas.append(fila)
        aux_aux_aux1 = pd.DataFrame(filas)
        result1['Total']=aux_aux_aux1

        st.write('Ventes del any passat pels mateixo mesos de la predicció: ', result)
        st.write('Predicció utilitzant MULTICLASS pels mesos següents', result1)
        st.write('Predicció de la Regressió Lineal pels següents mesos', result2)

    else: 
        st.write('Ventes del any passat pels mateixo mesos de la predicció: ', result)
        st.write('Predicció utilitzant MULTICLASS pels mesos següents', result1)
        st.write('Predicció de la Regressió Lineal pels següents mesos', result2)
    
    return result1

  


@st.cache_data
def truncate(number, max_decimals):
    int_part, dec_part = str(number).split(".")
    return float(".".join((int_part, dec_part[:max_decimals])))


def convert_df(df):
   return df.to_csv().encode('utf-8')

def convert_to_string(dataframe):
    num_initial_cols = 7
    # Get the number of columns that are integers
    num_int_cols = dataframe.shape[1] - num_initial_cols
    # Create a list of new column names that are strings
    new_col_names = [str(i+num_initial_cols) for i in range(num_int_cols)]
    # Rename the integer columns using the new column names
    dataframe.columns.values[num_initial_cols:num_initial_cols + num_int_cols] = new_col_names
    for i in range(len(new_col_names)):
        df=dataframe.rename(columns={num_initial_cols:new_col_names[0]})
        num_initial_cols=num_initial_cols+1
    return df



dataset=st.session_state.data
dataset_grouped=st.session_state.data_encoded

num=st.session_state.numero
df_not_encoded=st.session_state.data_not_encoded
df_regression=st.session_state.df_regression
array_quartils=st.session_state.array_quartils
df=st.session_state.df
quartilss=st.session_state.quartils
values_dict=st.session_state.diccionari
previa1=st.session_state.previa1
df_not_encoded11=st.session_state.df_model
dataset_principi1=st.session_state.dataset_principi1
dataset_principi_modified=st.session_state.dataset_principi_modified
d_especial=st.session_state.d_especial



df_encoded1=st.session_state.df_not_encoded1
s=dataset_principi_modified.shape[1]
df_not_encoded_copy=df_not_encoded.copy()
columnas_a_convertir = df.columns[7:]
df[columnas_a_convertir] = df[columnas_a_convertir].astype(str)

#important=st.session_state.important
important=1
previa=st.session_state.num_mes_previ
previa_per_quartils=int(previa[0])-7

dataset=convert_to_string(dataset_grouped)

#starting plotting
st.title('Zipper prediction with ML')
st.write('')
st.write("We are going to see some predictions of our datasets using Machine Learning and Shap plots")
st.write('You will be able to see do two different types of prediction:')
st.write('1-You will be able to see the prediction of one zipper for the next month.')
st.write('2-Prediction for all the data. ')
st.write('')
st.write('')
st.write('Here you can find the prediction and the explainability of the model. ')


model_regression,X_test_regression_reduced,x_regression_reduced,Y_test_regression_reduced, relative_error_mean_reduced,predict=apply_regression(df_regression,previa1)


with st.expander("Prediction"):
    tab1,tab2= st.tabs(["Diferent Predictions", " Downloading data"])
    with tab1: 
        st.write('You will do differents predictions')
        
        result1=zippers_model(df_not_encoded11,df,dataset_grouped,df_not_encoded_copy, important,predict, quartilss, relative_error_mean_reduced,previa1,dataset_principi1,num,dataset_principi_modified,df_encoded1)

    with tab2:  
        st.write('Prediction for the next months: ')
        tab1,tab2= st.tabs(["Download MULTICLASS prediction", " Download LINEAR REGRESSION prediction"])
        with tab1: 
            st.write(' ')
            st.write('The MULTICLASS prediction is: ')
            st.write(result1)
            csv = convert_df(result1)
            st.download_button(label="Download data as CSV",data=csv,file_name='MULTICLASS_PREDICTION.csv',mime='text/csv')


        with tab2: 
            st.write('')
            st.write('The LINEAR REGRESSION prediction is: ')
            df_not_encoded1=df_not_encoded.copy()
            st.write(df_not_encoded11)
            csv = convert_df(df_not_encoded1)
            st.download_button(label="Download data as CSV",data=csv,file_name='Regression_predictions.csv',mime='text/csv')