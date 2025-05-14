import mysql.connector
import pandas as pd

connection = mysql.connector.connect(
    host="",
    user="",
    password="",
    database=""
)

query = "SELECT * FROM churn_predictions"
df = pd.read_sql(query, connection)
connection.close()

print(df)
