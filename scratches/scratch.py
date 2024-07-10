import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from scapy.all import sniff, IP, TCP, UDP
import signal
import time

# Correct the file path using a raw string
train_data_path = r'D:\Train_data\Train_data.csv'
data = pd.read_csv(train_data_path)

# Display the first few rows of the dataset
print("Test Data Set")
print(data.head())
print("Train Describe")
print(data.describe())
print("Train Describe Object")
print(data.describe(include='object'))

# Function to check for missing data
def check_missing_data(data):
    total = data.shape[0]
    missing_columns = [col for col in data.columns if data[col].isnull().sum() > 0]
    for col in missing_columns:
        null_count = data[col].isnull().sum()
        per = (null_count / total) * 100
        print(f"{col}: {null_count} ({round(per, 3)}%)")

# Function to check for duplicated data
def check_duplicated_data(data):
    duplicate_count = data.duplicated().sum()
    print(f"Number of duplicate rows: {duplicate_count}")

# Check for missing and duplicated data
print("Checking for missing data...")
check_missing_data(data)

print("Checking for duplicated data...")
check_duplicated_data(data)

# Use 'class' as the label column
if 'class' in data.columns:
    # Encode categorical variables
    data = pd.get_dummies(data, columns=['duration', 'protocol_type', 'service', 'flag', 'hot'])

    X = data.drop('class', axis=1)
    y = data['class'].apply(lambda x: 1 if x == 'anomaly' else 0)  # Encode class as 0 (normal) and 1 (anomaly)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Normalize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize the KNN model
    knn = KNeighborsClassifier(n_neighbors=5)

    # Train the model
    knn.fit(X_train, y_train)

    # Evaluate the model
    y_pred = knn.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
else:
    print("The 'class' column is not found in the dataset. Please check the dataset.")

# Create a DataFrame to store packet statistics
packet_stats = pd.DataFrame(columns=list(X.columns) + ['prediction', 'actual'])

# Function to capture live network traffic and extract packet information
def extract_packet_info(packet):
    if IP in packet:
        ip = packet[IP]
        features = {
            'duration': packet.time,  # Timestamp of the packet capture
            'protocol_type': 'tcp' if TCP in packet else ('udp' if UDP in packet else 'other'),  # Protocol type
            'service': ip.sport if TCP in packet or UDP in packet else 0,  # Source port
            'flag': packet.sprintf('%TCP.flags%') if packet.haslayer('TCP') else 0,
            'src_bytes': len(packet[IP].payload),  # Source bytes
            'dst_bytes': len(packet),  # Destination bytes
            'land': int(packet[IP].src == packet[IP].dst),  # Check if source and destination are the same
            'wrong_fragment': int(packet[IP].frag != 0),  # Fragmentation flag
            'urgent': int(packet.haslayer('URG')),  # URG flag
            'hot': 0  # Placeholder, adjust based on your features
        }
        return features
    return None

# Real-time anomaly detection and alerting system
def detect_anomaly(packet):
    packet_info = extract_packet_info(packet)
    if packet_info:
        # Convert packet information to DataFrame
        packet_df = pd.DataFrame([packet_info])
        # Encode categorical variables in the same way as the training data
        packet_df = pd.get_dummies(packet_df, columns=['duration', 'protocol_type', 'service', 'flag', 'hot'])
        # Align packet_df columns with X_train columns, fill missing columns with 0
        packet_df = packet_df.reindex(columns=X.columns, fill_value=0)
        # Normalize the data using the same scaler used for training
        packet_scaled = scaler.transform(packet_df)
        # Predict using the trained KNN model
        prediction = knn.predict(packet_scaled)[0]
        # Add prediction to packet info
        packet_info['prediction'] = prediction
        # Append packet info to packet_stats DataFrame
        packet_stats.loc[len(packet_stats)] = packet_info
        # Check if the prediction is an anomaly
        if prediction == 1:
            print("Anomaly detected! Packet info:", packet_info)
            # Log the anomaly for retraining
            log_anomaly(packet_info, 1)
        else:
            print("Normal packet. Packet info:", packet_info)
            # Log the normal packet for retraining
            log_anomaly(packet_info, 0)

# Function to log detected packets for retraining
def log_anomaly(packet_info, label):
    packet_info['class'] = label
    df = pd.DataFrame([packet_info])
    df.to_csv('logged_data.csv', mode='a', header=False, index=False)

# Function to periodically retrain the model with new data
def retrain_model():
    while True:
        time.sleep(3600)  # Retrain every hour
        logged_data = pd.read_csv('logged_data.csv', names=list(X.columns) + ['class'])
        if not logged_data.empty:
            new_data = pd.concat([data, logged_data])
            new_X = new_data.drop('class', axis=1)
            new_y = new_data['class']
            new_X_train, new_X_test, new_y_train, new_y_test = train_test_split(new_X, new_y, test_size=0.3, random_state=42)
            new_X_train = scaler.fit_transform(new_X_train)
            new_X_test = scaler.transform(new_X_test)
            knn.fit(new_X_train, new_y_train)
            new_y_pred = knn.predict(new_X_test)
            print("Retrained Model Evaluation:")
            print(confusion_matrix(new_y_test, new_y_pred))
            print(classification_report(new_y_test, new_y_pred))

# Signal handler to gracefully stop the packet capture
def signal_handler(sig, frame):
    print('Stopping live capture...')
    global running
    running = False

signal.signal(signal.SIGINT, signal_handler)

# Ensure the models are trained before starting live capture
if 'class' in data.columns:
    print("Starting live capture...")
    global running
    running = True
    while running:
        sniff(prn=detect_anomaly, store=0, count=1)
    print("Packet Statistics:")
    print(packet_stats)

    # Calculate statistics
    mean_src_bytes = packet_stats['src_bytes'].mean()
    mean_dst_bytes = packet_stats['dst_bytes'].mean()
    anomaly_count = packet_stats['prediction'].sum()
    normal_count = len(packet_stats) - anomaly_count

    print(f"Mean src_bytes: {mean_src_bytes}")
    print(f"Mean dst_bytes: {mean_dst_bytes}")
    print(f"Anomalies detected: {anomaly_count}")
    print(f"Normal packets: {normal_count}")

    # Visualization
    plt.figure(figsize=(14, 7))

    # Subplot 1: src_bytes distribution
    plt.subplot(2, 2, 1)
    sns.histplot(packet_stats['src_bytes'], kde=True)
    plt.title('Source Bytes Distribution')

    # Subplot 2: dst_bytes distribution
    plt.subplot(2, 2, 2)
    sns.histplot(packet_stats['dst_bytes'], kde=True)
    plt.title('Destination Bytes Distribution')

    # Subplot 3: Prediction count
    plt.subplot(2, 2, 3)
    sns.countplot(x='prediction', data=packet_stats)
    plt.title('Prediction Count')
    plt.xticks(ticks=[0, 1], labels=['Normal', 'Anomaly'])

    # Subplot 4: src_bytes vs. dst_bytes
    plt.subplot(2, 2, 4)
    sns.scatterplot(x='src_bytes', y='dst_bytes', hue='prediction', data=packet_stats, palette={0: 'blue', 1: 'red'})
    plt.title('Source Bytes vs. Destination Bytes')

    plt.tight_layout()
    plt.show()
else:
    print("Training data is not correctly loaded.")
