#  Frameworks_Assignment  

##  Overview  
This project analyzes the **CORD-19 COVID-19 research metadata dataset** and presents insights through a **Streamlit web application**.  
It demonstrates the use of Python frameworks for:  

- Data loading and cleaning  
- Exploratory data analysis (EDA)  
- Visualization of research trends  
- Building an interactive web app  

---

##  Objectives  
- Practice loading and exploring a real-world dataset  
- Apply data cleaning techniques with **Pandas**  
- Create visualizations with **Matplotlib & Seaborn**  
- Build an interactive app using **Streamlit**  
- Present data insights in an accessible way  

---

##  Repository Structure  
Frameworks_Assignment/
│── app.py # Streamlit web app
│── analysis.py # Data loading, cleaning, and visualization logic
│── requirements.txt # Python dependencies
│── metadata.csv # Dataset (sampled from CORD-19 metadata)
│── README.md # Project documentation


---

## ⚙ Installation  

Clone this repository:  

git clone https://github.com/Ikamunya-web/Frameworks_Assignment.git
cd Frameworks_Assignment

Install dependencies:
pip install -r requirements.txt

Running the Project
Run the Streamlit app:
streamlit run app.py

Features
 Load and clean CORD-19 metadata dataset
 Analyze publications by year, journal, and keywords
 Visualize trends with bar charts, histograms, and word clouds
 Interactive filters in the Streamlit app

 Tools & Libraries
Pandas → Data manipulation & cleaning

Matplotlib → Plotting

Seaborn → Prettier statistical plots

Streamlit → Web app framework

WordCloud → Generating word cloud of paper titles

Example Visualizations
 Number of publications per year

 Top publishing journals

 Word cloud of paper titles

 Distribution of papers by source

 Dataset

Source: CORD-19 Research Challenge (Kaggle)

File used: metadata.csv (subset for assignment)

 Reflection
This project shows how Python frameworks simplify the data science workflow:

Pandas → Clean & explore data

Matplotlib/Seaborn → Visualize insights

Streamlit → Share findings interactively
