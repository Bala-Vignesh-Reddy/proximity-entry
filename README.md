# Smart Imaging Device - Intrusion Detection System

A real-time security monitoring system built with Raspberry Pi, computer vision, and GPIO-controlled alerts. This system uses a camera to detect people, monitors a user-defined restricted area, and triggers visual and audio alerts when intrusions are detected.

## Features

- Real-time people detection using YOLOv5
- User-defined restricted area monitoring
- Visual and audio alerts via LED and buzzer
- People counting
- Interactive Streamlit dashboard
- Threaded video capture for improved performance

## Hardware Requirements

- Raspberry Pi 5 
- Camera (Raspberry Pi Camera Module or USB webcam)
- LED (any color)
- Active buzzer
- 330 ohm resistor (for LED)
- Jumper wires
- Breadboard

## Circuit Connections

### LED Connection
1. Connect the longer leg (anode) of the LED to GPIO pin 23 on the Raspberry Pi
2. Connect the shorter leg (cathode) to one end of the 330 ohm resistor
3. Connect the other end of the resistor to a GND pin on the Raspberry Pi

### Buzzer Connection
1. Connect the positive leg (longer leg) of the buzzer to GPIO pin 18 on the Raspberry Pi
2. Connect the negative leg (shorter leg) to a GND pin on the Raspberry Pi

### Circuit Diagram
```
Raspberry Pi GPIO:
    GPIO 18 ────────► Buzzer (+) ────► Buzzer (-) ────► GND
    
    GPIO 23 ────────► LED Anode (longer leg) 
                                  │
                                  ▼
                       LED Cathode (shorter leg)
                                  │
                                  ▼
                             330Ω Resistor
                                  │
                                  ▼
                                 GND
```

## Software Setup

### 1. Operating System
Install Raspberry Pi OS (64-bit recommended) on your Raspberry Pi.

### 2. Update System
```bash
sudo apt update
sudo apt upgrade -y
```

### 3. Clone the Repository
```bash
git clone https://github.com/yourusername/smart-imaging-device.git
cd smart-imaging-device
```

### 4. Create a Virtual Environment (Optional but Recommended)
```bash
python3 -m venv venv
source venv/bin/activate
```

### 5. Install Dependencies
```bash
pip install -r requirements.txt
```

## Running the Application

```bash
streamlit run app.py
```

The dashboard will be accessible at http://localhost:8501 or the IP address of your Raspberry Pi.

## Usage Guide

1. **Camera Selection**: Choose between Raspberry Pi Camera or USB Webcam
2. **USB Camera Index**: If using a USB webcam, you may need to try different indices (0, 1, 2, etc.)
3. **Restricted Area Settings**: Define the coordinates of the area to monitor
4. **Enable/Disable Alarm**: Toggle the alarm functionality

## Troubleshooting

### GPIO Access Issues
If you encounter GPIO permission errors:
```bash
sudo chmod 660 /dev/gpiomem
sudo chown root:gpio /dev/gpiomem
```

### Camera Not Found
- For USB webcams, try different camera indices in the dashboard
- Ensure the camera is properly connected
- Check if the camera is recognized by the system:
  ```bash
  ls /dev/video*
  ```
## Requirements.txt
```
streamlit==1.32.0
opencv-python-headless==4.8.1.78
torch==2.1.0
torchvision==0.16.0
numpy==1.24.3
Pillow==10.0.0
gpiozero==2.0.1
```

## Auto-start on Boot (Optional)

To make the application start automatically when the Raspberry Pi boots:

1. Create a systemd service file:
```bash
sudo nano /etc/systemd/system/smart-imaging.service
```

2. Add the following content (adjust paths as needed):
```
[Unit]
Description=Smart Imaging Device
After=network.target

[Service]
User=pi
WorkingDirectory=/home/pi/smart-imaging-device
ExecStart=/home/pi/smart-imaging-device/venv/bin/streamlit run /home/pi/smart-imaging-device/app.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

3. Enable and start the service:
```bash
sudo systemctl enable smart-imaging.service
sudo systemctl start smart-imaging.service
```

## Project Structure

```
smart-imaging-device/
├── app.py               # Main application file
├── requirements.txt     # Python dependencies
├── README.md           # Project documentation
```

## Future Enhancements

- Multiple detection zones with different alert levels
- Motion tracking and path analysis
- Time-based analytics and reporting
- Remote notifications via email or messaging services
- Integration with home automation systems

---
# AI written
