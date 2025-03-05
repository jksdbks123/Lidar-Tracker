import mysql.connector

# Database connection settings
DB_HOST = "172.20.133.42"
DB_NAME = "edgedata"
DB_USER = "edgeuser"
DB_PASSWORD = "strongpassword"

# Sample data
data_entries = [
    {"sensor_id": 1, "value": 23.5},
    {"sensor_id": 2, "value": 19.8}
]

def upload_data(data_entries):
    """Uploads collected data to MySQL database."""
    try:
        conn = mysql.connector.connect(
            host=DB_HOST, user=DB_USER, password=DB_PASSWORD, database=DB_NAME
        )
        cur = conn.cursor()

        # Ensure the table exists
        cur.execute("""
            CREATE TABLE IF NOT EXISTS sensor_data (
                id INT AUTO_INCREMENT PRIMARY KEY,
                sensor_id INT NOT NULL,
                value FLOAT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Insert data
        for entry in data_entries:
            cur.execute(
                "INSERT INTO sensor_data (sensor_id, value) VALUES (%s, %s);",
                (entry["sensor_id"], entry["value"])
            )

        conn.commit()
        cur.close()
        conn.close()
        print("Data uploaded successfully!")

    except Exception as e:
        print("Error:", e)

# Run the function
upload_data(data_entries)
