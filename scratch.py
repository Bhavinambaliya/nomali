from flask import Flask, render_template, jsonify
import pandas as pd
import plotly.express as px
import threading
from scapy.all import sniff, IP, TCP, UDP
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

app = Flask(__name__)

# Load and preprocess data
train_data_path = r'D:\Train_data\Train_data.csv'
data = pd.read_csv(train_data_path)

# Separate numeric and categorical columns
numeric_cols = data.select_dtypes(include=['number']).columns
categorical_cols = data.select_dtypes(include=['object']).columns

# Fill missing values for numeric columns with the median
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())

# Fill missing values for categorical columns with the mode
data[categorical_cols] = data[categorical_cols].apply(lambda x: x.fillna(x.mode()[0]))

# Encode categorical variables
data = pd.get_dummies(data, columns=['duration', 'protocol_type', 'service', 'flag', 'hot'])

# Split features and target
X = data.drop('class', axis=1)
y = data['class'].apply(lambda x: 1 if x == 'anomaly' else 0)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train the KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Evaluate the model
y_pred = knn.predict(X_test)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# DataFrame to store packet statistics
packet_stats = pd.DataFrame(columns=list(X.columns) + ['prediction', 'actual'])
captured_packets = []

def extract_packet_info(packet):
    if IP in packet:
        ip = packet[IP]
        features = {
            'src_ip': packet[IP].src,
            'dst_ip': packet[IP].dst,
            'src_port': packet[IP].sport if TCP in packet or UDP in packet else 0,
            'dst_port': packet[IP].dport if TCP in packet or UDP in packet else 0,
            'protocol': 'TCP' if TCP in packet else 'UDP' if UDP in packet else 'Other',
            'service': packet[IP].sport if TCP in packet or UDP in packet else 0,
            'length': len(packet),
            'info': packet.summary(),
            'duration': packet.time,
            'flag': packet.sprintf('%TCP.flags%') if packet.haslayer('TCP') else 0,
            'src_bytes': len(packet[IP].payload),
            'dst_bytes': len(packet),
            'land': int(packet[IP].src == packet[IP].dst),
            'wrong_fragment': int(packet[IP].frag != 0),
            'urgent': int(packet.haslayer('URG')),
            'hot': 0
        }
        return features
    return None

def detect_anomaly(packet):
    packet_info = extract_packet_info(packet)
    print(packet_info)
    if packet_info:
        packet_df = pd.DataFrame([packet_info])
        packet_df = pd.get_dummies(packet_df, columns=['duration', 'protocol', 'service', 'flag', 'hot'])
        packet_df = packet_df.reindex(columns=X.columns, fill_value=0)
        packet_scaled = scaler.transform(packet_df)
        prediction = knn.predict(packet_scaled)[0]
        packet_info['prediction'] = prediction
        packet_stats.loc[len(packet_stats)] = packet_info
        packet_info['status'] = 'Anomaly' if prediction == 1 else 'Normal'
        captured_packets.append({
            'src_ip': packet_info['src_ip'],
            'dst_ip': packet_info['dst_ip'],
            'src_port': packet_info['src_port'],
            'dst_port': packet_info['dst_port'],
            'protocol': packet_info['protocol'],
            'service': packet_info['service'],
            'length': packet_info['length'],
            'info': packet_info['info'],
            'status': packet_info['status']
        })
        print(f"Captured packet: {packet_info}")  # Debugging line

running = False

def start_sniffing():
    global running
    running = True
    while running:
        sniff(prn=detect_anomaly, store=0, count=1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_capture')
def start_capture():
    global running
    if not running:
        threading.Thread(target=start_sniffing).start()
    return 'Packet capture started.'

@app.route('/stop_capture')
def stop_capture():
    global running
    running = False
    return 'Packet capture stopped.'

@app.route('/get_stats')
def get_stats():
    mean_src_bytes = packet_stats['src_bytes'].mean()
    mean_dst_bytes = packet_stats['dst_bytes'].mean()
    anomaly_count = packet_stats['prediction'].sum()
    normal_count = len(packet_stats) - anomaly_count

    return jsonify({
        'mean_src_bytes': mean_src_bytes,
        'mean_dst_bytes': mean_dst_bytes,
        'anomaly_count': anomaly_count,
        'normal_count': normal_count
    })

@app.route('/get_plots')
def get_plots():
    fig1 = px.histogram(packet_stats, x='src_bytes', title='Source Bytes Distribution')
    fig2 = px.histogram(packet_stats, x='dst_bytes', title='Destination Bytes Distribution')
    fig3 = px.histogram(packet_stats, x='prediction', title='Prediction Count', category_orders={"prediction": [0, 1]})
    fig4 = px.scatter(packet_stats, x='src_bytes', y='dst_bytes', color='prediction', title='Source Bytes vs. Destination Bytes')

    return jsonify({
        'fig1': fig1.to_json(),
        'fig2': fig2.to_json(),
        'fig3': fig3.to_json(),
        'fig4': fig4.to_json()
    })

@app.route('/get_packets')
def get_packets():
    return jsonify(captured_packets)

if __name__ == '__main__':
    app.run(debug=True)
