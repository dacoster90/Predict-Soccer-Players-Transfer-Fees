# Predict Soccer Players Transfer Fees
 
### Description
As a Final Year Project of my Software Engineering course (or TCC in Portuguese), I've decided to create a work that combines my two biggest passions: Soccer and Data Science.
The script imports an Excel file with a list of transfers of the year 2019, which contains information such as the name of the player, current value, age, overall level, potential, contract duration, the player's position on the field and more. By applying statistical concepts such as **correlation** and **linear regression**, a model is created using the best available variables in order to get the best possible prediction.

This simple console tool, allows the user to input the variables and the tool will return the value of the player on the transfermarket.

On this repository, you will also find the Word document 'TCC I Kenneth De Coster.docx' in Portuguese, which provides more information on this project. Please note, this project is still a work in progress.

### Languages and Technologies
- Python
- Excel
- Jupyter Notebook

### Compatibility
- Python version 3.0 or above.

### Pre-requisites
1. Install Python
2. Install the packages: pandas, re, sklearn.linear_model, yellowbrick.regressor, numpy, matplotlib.pyplot and seaborn
3. Download the file DATASET_2019_OFFICIAL.xlsx (available for download on this repository)

### How to run
1. Open CMD and navigate to the directory that contains the TCC.py file.
2. Execute 'python TCC.py'
3. Insert the variables **age**, **overall level** and **potential level**.
4. The tool returns the predicted transfer fee of the player.

### Images
Prediction of player K.Mpabbé
![application](https://i.ibb.co/dcf0GsK/application-TCC.png)
The actual fee paid for K.Mpabbé, the tool predicted the fee with an accuracy of 91%.
![result](https://i.ibb.co/3RZNvvz/result-TCC.png)