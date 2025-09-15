import pandas as pd
import sqlite3

# Step 1: Load your CSV file
df = pd.read_csv("train.csv")   # replace with your csv file name

# Step 2: Create a SQLite database (medical.db)
conn = sqlite3.connect("medical.db")

# Step 3: Save the CSV data into a table (let's call it 'medical_faq')
df.to_sql("medical_faq", conn, if_exists="replace", index=False)

# Step 4: Close the connection
conn.close()

print("✅ CSV saved as medical.db with table 'medical_faq'")
