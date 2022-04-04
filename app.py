import pandas as pd
import streamlit as st
import pickle
from datetime import date

pickle_in = open("breast_cancer_classifier.pkl","rb")
cancer_classifier=pickle.load(pickle_in)

#Date time, logger
today = date.today()
#date_time = datetime.fromtimestamp(1887639468)
f = open("file_logger.txt", "a")

def main():
    
    html_temp = """
   <div style="background-color:tomato;padding:10px">
   <h2 style="color:white;text-align:center;">Streamlit Breast Cancer Prediction ML App </h2>
   </div>
   """
   
   
    st.markdown(html_temp,unsafe_allow_html=True)
    st.info("")
    
    
    
    #code for file uploader to train.
    data = st.file_uploader("Choose a csv file for Traning", ["csv"])
    if data is not None:
        f.write(f"{today} : Dataset uploaded")
        f.write("\n")
        
        df = pd.read_csv(data)
        st.markdown("Dataset you have uploaded:-")
        st.dataframe(df)
        
        f.write(f"{today} : Dataset Shown")
        f.write("\n")
        
        try:
            f.write(f"{today} : Traning Started")
            f.write("\n")
            
            f.write(f"{today} : preprocessing for traning data Started!")
            f.write("\n")
            
            for c in df:
                if c == "Unnamed: 32":
                    df.drop(["Unnamed: 32"], axis=1, inplace=True)
            
            df.drop(["id"], axis=1, inplace=True)
            
            df.diagnosis.replace({"M":1, "B":0}, inplace=True)
            
            X = df.iloc[:, 1:]
            y = df.iloc[:, 0]
            
            
            from imblearn.over_sampling import RandomOverSampler
            os = RandomOverSampler()
            X_res, y_res = os.fit_resample(X, y)
            
            f.write(f"{today} : preprocessing for traning data Complete!")
            f.write("\n")
            
            f.write(f"{today} : Spliting traning and testing data")
            f.write("\n")
                        
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.20, random_state=42)
            
            cancer_classifier.fit(X_train, y_train)
            
            prediction = cancer_classifier.predict(X_test)
            
            f.write(f"{today} : Prediction on traning data Done!")
            f.write("\n")
            
            from sklearn.metrics import accuracy_score
            asc = accuracy_score(y_test, prediction)
            
            f.write(f"{today} : accuracy_score : {asc}")
            f.write("\n")
            
            st.success("Traning Complete")
            
            f.write(f"{today} : Traning Complete")
            f.write("\n")
            
            
            st.markdown("Dataset after Traning:-")
            st.dataframe(df)
            
            f.write(f"{today} : Dataset Shown after Traning")
            f.write("\n")
          
        except Exception as e:
            st.error("Inavlid Dataset, Please! Try Again....")
            
            f.write(f"{today} : {e}")
            f.write("\n")
          
    
    
    
    
    #code for file uploader to predict.
    data = st.file_uploader("Choose a csv file for Prediction", ["csv"])
    if data is not None:
        
        f.write(f"{today} : Dataset uploaded")
        f.write("\n")
        
        df = pd.read_csv(data)
        st.markdown("Dataset you have uploaded:-")
        st.dataframe(df)
        
        f.write(f"{today} : Dataset Shown")
        f.write("\n")
        
    
        try:
            
            f.write(f"{today} : Prediction Started")
            f.write("\n")
            
            f.write(f"{today} : preprocessing for testing data Started!")
            f.write("\n")
            
            for c in df:
                if c == "Unnamed: 32":
                    df.drop(["Unnamed: 32"], axis=1, inplace=True)
            
            ID = df["id"]
            
            df.drop(["id"], axis=1, inplace=True)
            
            f.write(f"{today} : preprocessing for testing data Complete!")
            f.write("\n")
            
            ans = cancer_classifier.predict(df)
            df["predcited_value"] = ans
            
            df["id"] = ID
            
            f.write(f"{today} : Prediction Complete")
            f.write("\n")
            
            st.markdown("Dataset after Prediction:-")
            st.dataframe(df)
            
            f.write(f"{today} : Dataset Shown after prediction")
            f.write("\n")
            
            if st.button("Download"):
                data.to_csv("Result.csv")
                st.success("Download Complete")
                
                f.write(f"{today} : Dataset Downloaded after prediction")
                f.write("\n")
               
        except Exception as e:
            st.error("Inavlid Dataset, Please! Try Again....")
            
            f.write(f"{today} : {e}")
            f.write("\n")
           
   
            
    
if __name__ == '__main__' :
    main()
    

