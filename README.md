# rPPG Real-Time Dashboard

A real-time web dashboard for remote Photoplethysmography (rPPG). This application captures video from your webcam, processes it to detect faces, and extracts physiological signals such as Heart Rate (HR) and Blood Volume Pulse (BVP) in real-time.

## Features
- **Real-Time Video Streaming:** Streams live video from your webcam to the browser.
- **Live Vitals Tracking:** Displays real-time Heart Rate (BPM) and Blood Volume Pulse (BVP) signals.
- **Face Detection:** Automatically detects and tracks the face in the video feed for accurate signal extraction.
- **Web-based Interface:** Easy to use dashboard served via Flask.

## Prerequisites

Before you begin, ensure you have met the following requirements:
- Python 3.7 or higher installed on your system.
- A working webcam connected to your computer.

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd rgb_rppg
   ```

2. **(Optional but recommended) Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required dependencies:**
   Install the necessary Python packages using pip:
   ```bash
   pip install flask flask-socketio opencv-python numpy
   ```
   *Note: Ensure the custom `rppg` library is correctly installed or located within your Python path.*

## Usage

1. **Start the application server:**
   Run the following command in your terminal:
   ```bash
   python run.py
   ```

2. **Access the dashboard:**
   Open your web browser and navigate to:
   [http://localhost:5050](http://localhost:5050)

3. **View your vitals:**
   The application will access your webcam (index 0). Please ensure your face is well-lit and visible. The dashboard will stream the video feed and display your live Heart Rate and BVP signals.

4. **Stop the application:**
   Press `Ctrl+C` in your terminal to stop the Flask server.

## Acknowledgments and Dependencies

This project was built using the following open-source libraries, packages, and modules. We gratefully acknowledge their developers and contributors:

- **[Flask](https://flask.palletsprojects.com/):** A lightweight WSGI web application framework used to serve the frontend dashboard.
- **[Flask-SocketIO](https://flask-socketio.readthedocs.io/):** Enables low-latency, bi-directional, and real-time communication over WebSockets between the server and the web client. Used for streaming base64-encoded video frames and live physiological data.
- **[OpenCV (cv2)](https://opencv.org/):** A powerful open-source computer vision library utilized for webcam video capture, color space conversions (RGB to BGR), and image frame encoding.
- **[NumPy](https://numpy.org/):** The fundamental package for scientific computing with Python, used extensively for array and matrix operations under the hood.
- **`rppg` (Custom Module):** The core implementation used in this project to perform remote photoplethysmography, handling facial detection bounding boxes and generating Heart Rate (HR) and Blood Volume Pulse (BVP) estimations.
- **Python Standard Libraries:** 
  - `threading`: Utilized to run the rPPG processing loop asynchronously, ensuring the main Flask thread isn't blocked.
  - `base64`: Used to encode the OpenCV image buffers into base64 strings for transmission over WebSockets to the HTML client.
  - `time`: Used for timestamping and managing data sampling windows.

---
*Disclaimer: This application is intended for demonstration, educational, and research purposes only. It is not intended for medical diagnosis, monitoring, or treatment.*
