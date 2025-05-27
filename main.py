
from flask import Flask, request, render_template_string, jsonify
import time
import sqlite3
from datetime import datetime, timedelta


# Try to load the ML model with error handling
model = None
try:
    import joblib
    model = joblib.load('fire_model.pkl')
    print("✅ Fire detection model loaded successfully!")
except Exception as e:
    print(f"⚠️ Warning: Could not load fire model - {e}")
    print("🔄 Continuing without ML prediction...")

app = Flask(__name__)

def init_db():
    conn = sqlite3.connect('sensor_data.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS sensor_readings
                 (timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                  fire BOOLEAN,
                  temperature REAL,
                  smoke REAL,
                  co REAL,
                  lpg REAL,
                  gas_value INTEGER,
                  pressure REAL,
                  aqi INTEGER)''')
    conn.commit()
    conn.close()

init_db()

# Initial default sensor values
fire_detected = False
temperature = 0.0
smoke = 0.0
co = 0.0
lpg = 0.0
gasValue = 0
pressure = 0.0
aqi = 0
last_data_received = None

html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Fire Detection & Pollution Monitoring</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --card-bg: rgba(255,255,255,0.12);
            --card-border: rgba(255,255,255,0.15);
            --primary: #3498db;
            --secondary: #2c3e50;
            --text: #f8f9fa;
            --background: #1a1f35;
            --success: #00b894;
            --warning: #fdcb6e;
            --danger: #d63031;
            --accent: #00cec9;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, rgba(20, 25, 40, 0.95), rgba(30, 35, 55, 0.95)),
                        url('https://images.unsplash.com/photo-1517594422361-5eeb8ae275a9?auto=format&fit=crop&w=1920&q=100&brightness=120');
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
            background-repeat: no-repeat;
            color: var(--text);
            min-height: 100vh;
            position: relative;
        }

        body::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: radial-gradient(circle at 20% 80%, rgba(0, 184, 148, 0.1) 0%, transparent 50%),
                        radial-gradient(circle at 80% 20%, rgba(52, 152, 219, 0.1) 0%, transparent 50%);
            pointer-events: none;
            z-index: -1;
        }

        @media (max-width: 768px) {
            body {
                background-attachment: scroll; /* Better performance on mobile */
            }
        }

        nav {
            background: rgba(26, 31, 53, 0.98);
            backdrop-filter: blur(20px);
            padding: 1.5rem;
            position: sticky;
            top: 0;
            z-index: 100;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        }

        nav h1 {
            font-size: 1.8rem;
            letter-spacing: -0.5px;
            font-weight: 800;
            background: linear-gradient(135deg, var(--accent), var(--primary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0;
        }

        .nav-content {
            max-width: 1200px;
            margin: 0 auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .nav-links {
            display: flex;
            gap: 1rem;
        }

        .nav-link {
            color: var(--text);
            text-decoration: none;
            font-weight: 500;
            transition: all 0.3s ease;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            position: relative;
            overflow: hidden;
        }

        .nav-link:hover {
            background: rgba(255, 255, 255, 0.1);
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        }

        /* AI Dashboard Button with Moving Colors */
        .nav-link[href="/ai-dashboard"] {
            background: linear-gradient(270deg, #ff6b6b, #4ecdc4, #45b7d1, #f9ca24, #f0932b, #eb4d4b, #6c5ce7);
            background-size: 400% 400%;
            animation: movingColors 4s ease infinite;
            color: white;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
            border: none;
            border-radius: 8px;
            position: relative;
            padding: 0.75rem 1.5rem;
            text-shadow: 0 1px 3px rgba(0,0,0,0.3);
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            overflow: hidden;
        }

        .nav-link[href="/ai-dashboard"]::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
            transition: left 0.5s;
        }

        .nav-link[href="/ai-dashboard"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        }

        .nav-link[href="/ai-dashboard"]:hover::before {
            left: 100%;
        }

        @keyframes movingColors {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        .status-indicator {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 6px 12px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.15);
            margin-top: 8px;
            font-size: 0.85rem;
        }

        .status-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }

        .status-dot.online {
            background: var(--success);
            box-shadow: 0 0 10px rgba(0, 184, 148, 0.5);
        }

        .status-dot.offline {
            background: var(--danger);
            box-shadow: 0 0 10px rgba(214, 48, 49, 0.5);
        }

        .status-text {
            font-size: 0.9rem;
            font-weight: 600;
            letter-spacing: 0.5px;
        }

        .status-text.online {
            color: var(--success);
        }

        .status-text.offline {
            color: var(--danger);
        }

        @keyframes pulse {
            0% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.7; transform: scale(1.1); }
            100% { opacity: 1; transform: scale(1); }
        }

        .container {
            max-width: 1200px;
            margin: 2rem auto;
            padding: 0 1rem;
        }

        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 2rem;
            margin: 2rem 0;
        }

        .card {
            background: linear-gradient(145deg, var(--card-bg), rgba(255,255,255,0.05));
            backdrop-filter: blur(12px);
            border-radius: 20px;
            border: 1px solid var(--card-border);
            padding: 32px;
            text-align: center;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        }

        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 15px rgba(0,0,0,0.2);
        }

        .label {
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 0.5rem;
            color: var(--primary);
        }

        .value {
            font-size: 2.5rem;
            font-weight: 600;
            background: linear-gradient(135deg, var(--text), #a8b2d1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.5px;
        }

        .safe {
            background: linear-gradient(135deg, var(--success), #87d8a7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .danger {
            background: linear-gradient(135deg, var(--danger), var(--warning));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: pulse 1.5s ease-in-out infinite;
        }

        .chart-container {
            background: rgba(255,255,255,0.08);
            backdrop-filter: blur(12px);
            border-radius: 16px;
            border: 1px solid rgba(255,255,255,0.08);
            padding: 24px;
            height: 300px;
            margin: 6rem 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
            position: relative;
        }

        #mainGraphContainer {
            height: 500px;
            margin: 8rem 0 6rem 0;
            position: relative;
            z-index: 0;
        }

        .chart-header {
            position: absolute;
            top: -40px;
            right: 0;
            z-index: 10;
        }

        .history-btn, .analytics-btn {
            background: linear-gradient(135deg, var(--primary), var(--accent));
            border: none;
            color: white;
            padding: 12px 24px;
            border-radius: 12px;
            cursor: pointer;
            transition: all 0.3s ease;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-size: 0.9rem;
            z-index: 2;
        }

        .history-btn:hover, .analytics-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(0, 184, 148, 0.3);
            background: linear-gradient(135deg, var(--accent), var(--primary));
        }

        .timestamp {
            position: absolute;
            bottom: 10px;
            left: 10px;
            font-size: 0.8rem;
            color: #f5f6fa;
        }

        .normal-range-card {
            background: linear-gradient(145deg, rgba(46, 204, 113, 0.15), rgba(0, 184, 148, 0.15));
            border: 2px solid rgba(46, 204, 113, 0.3);
        }

        .normal-ranges {
            display: grid;
            grid-template-columns: 1fr;
            gap: 0.8rem;
            margin-top: 1rem;
        }

        .range-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.5rem 0.8rem;
            background: rgba(255, 255, 255, 0.08);
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(5px);
            transition: all 0.3s ease;
        }

        .range-item:hover {
            background: rgba(255, 255, 255, 0.12);
            transform: translateX(3px);
        }

        .range-label {
            font-size: 0.9rem;
            font-weight: 500;
            color: var(--text);
            opacity: 0.9;
        }

        .range-value {
            font-size: 0.9rem;
            font-weight: 600;
            color: var(--success);
            background: linear-gradient(135deg, var(--success), var(--accent));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .footer {
            background: rgba(26, 31, 53, 0.95);
            backdrop-filter: blur(20px);
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            margin-top: 4rem;
            padding: 3rem 1rem 2rem;
            color: var(--text);
        }

        .footer-content {
            max-width: 1200px;
            margin: 0 auto;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 2rem;
        }

        .footer-section h3 {
            color: var(--accent);
            margin-bottom: 1rem;
            font-size: 1.2rem;
            font-weight: 600;
        }

        .footer-section h4 {
            color: var(--primary);
            margin-bottom: 0.5rem;
            font-size: 1.1rem;
            font-weight: 500;
        }

        .footer-section p {
            line-height: 1.6;
            margin-bottom: 0.5rem;
            opacity: 0.9;
        }

        .footer-section a {
            color: var(--accent);
            text-decoration: none;
            transition: color 0.3s ease;
        }

        .footer-section a:hover {
            color: var(--primary);
            text-decoration: underline;
        }

        .back-to-top {
            position: fixed;
            bottom: 30px;
            right: 30px;
            background: linear-gradient(135deg, var(--primary), var(--accent));
            color: white;
            border: none;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            cursor: pointer;
            font-size: 1.2rem;
            font-weight: bold;
            display: none;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
            z-index: 1000;
        }

        .back-to-top:hover {
            transform: translateY(-3px) scale(1.1);
            box-shadow: 0 6px 20px rgba(0, 184, 148, 0.4);
        }

        /* Enhanced Visual Effects */
        .card {
            position: relative;
            overflow: hidden;
        }

        .card::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
            transition: left 0.5s;
        }

        .card:hover::before {
            left: 100%;
        }

        /* Enhanced status indicator */
        .status-indicator {
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(15px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }

        /* Enhanced chart containers */
        .chart-container {
            position: relative;
            overflow: visible;
        }

        .chart-container::after {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(45deg, rgba(0, 184, 148, 0.05), rgba(52, 152, 219, 0.05));
            pointer-events: none;
        }

        /* Mobile Responsiveness */
        @media (max-width: 768px) {
            nav {
                padding: 1rem 0.5rem;
            }

            .nav-content {
                flex-direction: column;
                gap: 1rem;
                align-items: center;
            }

            nav h1 {
                font-size: 1.5rem;
                text-align: center;
            }

            .nav-links {
                display: flex;
                flex-wrap: wrap;
                justify-content: center;
                gap: 0.5rem;
                margin: 0 !important;
                width: 100%;
            }

            .nav-link {
                padding: 0.5rem 0.8rem;
                font-size: 0.9rem;
                min-width: 80px;
                text-align: center;
            }

            .nav-link[href="/ai-dashboard"] {
                padding: 0.6rem 1rem;
                font-size: 0.85rem;
                letter-spacing: 0.8px;
            }

            .status-indicator {
                justify-content: center;
                margin-top: 0.5rem;
                font-size: 0.8rem;
            }

            .container {
                margin: 1rem auto;
                padding: 0 0.5rem;
            }

            .grid {
                grid-template-columns: 1fr;
                gap: 1rem;
                margin: 1rem 0;
            }

            .card {
                padding: 20px 16px;
                border-radius: 16px;
            }

            .label {
                font-size: 0.8rem;
                margin-bottom: 0.5rem;
            }

            .value {
                font-size: 1.6rem;
                line-height: 1.2;
            }

            .normal-range-card .label {
                font-size: 0.8rem;
                margin-bottom: 0.8rem;
            }

            .normal-ranges {
                gap: 0.6rem;
                margin-top: 0.8rem;
            }

            .range-item {
                padding: 0.4rem 0.6rem;
                border-radius: 6px;
                flex-direction: column;
                text-align: center;
                gap: 0.2rem;
            }

            .range-label, .range-value {
                font-size: 0.8rem;
            }

            .chart-container {
                height: 250px;
                margin: 2rem 0;
                padding: 16px;
            }

            #mainGraphContainer {
                height: 300px;
                margin: 3rem 0 2rem 0;
            }

            .chart-header {
                top: -30px;
                right: 10px;
            }

            .history-btn, .analytics-btn {
                position relative;
                padding: 8px 12px;
                font-size: 0.8rem;
                border-radius: 8px;
                z-index: 100000;
            }

            .timestamp {
                font-size: 0.7rem;
                bottom: 8px;
                left: 8px;
            }

            .footer {
                padding: 2rem 1rem 1.5rem;
            }

            .footer-content {
                grid-template-columns: 1fr;
                gap: 1.5rem;
                text-align: center;
            }

            .back-to-top {
                bottom: 20px;
                right: 20px;
                width: 45px;
                height: 45px;
                font-size: 1.1rem;
            }
        }

        /* Extra small devices */
        @media (max-width: 480px) {
            nav h1 {
                font-size: 1.3rem;
            }

            .nav-links {
                gap: 0.3rem;
            }

            .nav-link {
                padding: 0.4rem 0.6rem;
                font-size: 0.8rem;
                min-width: 70px;
            }

            .nav-link[href="/ai-dashboard"] {
                padding: 0.5rem 0.8rem;
                font-size: 0.75rem;
                letter-spacing: 0.5px;
            }

            .container {
                padding: 0 0.25rem;
            }

            .card {
                padding: 16px 12px;
            }

            .value {
                font-size: 1.4rem;
            }

            .chart-container {
                height: 200px;
                padding: 12px;
            }

            #mainGraphContainer {
                height: 250px;
            }
        }

        /* Report section styles */
        .history-btn {
            min-width: 160px;
            font-size: 0.9rem;
            transition: all 0.3s ease;
            display: inline-flex;
            align-items: center;
            justify-content: center;
        }

        .history-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }

        #reportStatus {
            color: var(--text);
            font-weight: 500;
        }

        #reportStatus.success {
            color: var(--success);
        }

        #reportStatus.error {
            color: var(--danger);
        }

        /* Touch-friendly improvements */
        @media (hover: none) and (pointer: coarse) {
            .nav-link {
                min-height: 44px;
                display: flex;
                align-items: center;
                justify-content: center;
            }

            .nav-link[href="/ai-dashboard"] {
                min-height: 48px;
            }

            .card:hover::before {
                left: 0; /* Disable hover effect on touch devices */
            }

            .card:hover {
                transform: none;
            }
        }

        /* Chatbot Styles */
        .chat-float-button {
            position: fixed;
            bottom: 30px;
            left: 30px;
            width: 60px;
            height: 60px;
            background: rgba(52, 152, 219, 0.4);
            border: 2px solid rgba(52, 152, 219, 0.6);
            border-radius: 50%;
            cursor: pointer;
            z-index: 1000;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
            color: rgba(255, 255, 255, 0.7);
            transition: all 0.3s ease;
            backdrop-filter: blur(10px);
        }

        .chat-float-button:hover {
            background: rgba(52, 152, 219, 0.8);
            color: white;
            border-color: rgba(52, 152, 219, 1);
            transform: scale(1.05);
            box-shadow: 0 4px 20px rgba(52, 152, 219, 0.4);
        }

        .chat-float-button.active {
            background: rgba(52, 152, 219, 1);
            color: white;
            border-color: rgba(52, 152, 219, 1);
            box-shadow: 0 0 20px rgba(52, 152, 219, 0.6);
        }

        .chatbot-container {
            position: fixed;
            bottom: 100px;
            left: 30px;
            width: 300px;
            height: 400px;
            background: rgba(26, 31, 53, 0.95);
            backdrop-filter: blur(20px);
            border: 2px solid rgba(255, 20, 147, 0.3);
            border-radius: 20px;
            z-index: 999;
            display: none;
            flex-direction: column;
            box-shadow: 0 0 30px rgba(0, 0, 0, 0.5);
        }

        .chat-header {
            background: linear-gradient(45deg, #ff1493, #00ff7f);
            color: white;
            padding: 15px;
            border-radius: 18px 18px 0 0;
            font-weight: 600;
            text-align: center;
            position: relative;
        }

        .chat-close {
            position: absolute;
            right: 15px;
            top: 50%;
            transform: translateY(-50%);
            background: none;
            border: none;
            color: white;
            font-size: 18px;
            cursor: pointer;
            font-weight: bold;
        }

        .chat-messages {
            flex: 1;
            padding: 15px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }

        .chat-message {
            max-width: 80%;
            padding: 10px 15px;
            border-radius: 15px;
            word-wrap: break-word;
        }

        .chat-message.bot {
            background: linear-gradient(135deg, var(--primary), var(--accent));
            color: white;
            align-self: flex-start;
        }

        .chat-message.user {
            background: rgba(255, 255, 255, 0.1);
            color: var(--text);
            align-self: flex-end;
        }

        .chat-input-container {
            padding: 15px;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            display: flex;
            gap: 10px;
        }

        .chat-input {
            flex: 1;
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 25px;
            padding: 10px 15px;
            color: white;
            outline: none;
        }

        .chat-input::placeholder {
            color: rgba(255, 255, 255, 0.6);
        }

        .chat-send {
            background: linear-gradient(45deg, #ff1493, #00ff7f);
            border: none;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            cursor: pointer;
            color: white;
            font-size: 16px;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        /* Mobile responsive for chatbot */
        @media (max-width: 768px) {
            .chat-float-button {
                bottom: 80px;
                left: 20px;
                width: 50px;
                height: 50px;
                font-size: 18px;
                z-index: 1001;
            }

            .chatbot-container {
                bottom: 140px;
                left: 10px;
                right: 10px;
                width: auto;
                height: 300px;
                max-width: calc(100vw - 20px);
                max-height: calc(100vh - 200px);
            }
        }

        /* Extra small devices chatbot fix */
        @media (max-width: 480px) {
            .chat-float-button {
                bottom: 70px;
                left: 15px;
                width: 45px;
                height: 45px;
                font-size: 16px;
            }

            .chatbot-container {
                bottom: 125px;
                left: 5px;
                right: 5px;
                height: 250px;
                max-width: calc(100vw - 10px);
                max-height: calc(100vh - 150px);
            }

            .chat-header {
                padding: 12px;
                font-size: 0.9rem;
            }

            .chat-messages {
                padding: 10px;
            }

            .chat-input-container {
                padding: 10px;
            }

            .chat-input {
                padding: 8px 12px;
                font-size: 0.9rem;
            }
        }
    </style>
</head>
<body>
    <nav>
        <div class="nav-content">
            <div style="display: flex; align-items: center; justify-content: space-between; width: 100%;">
                <h1>Fire Detection & Monitoring</h1>
                <div style="display: flex; align-items: center; gap: 1rem;">
                    <div class="nav-links">
                        <a href="/" class="nav-link">Home</a>
                        <a href="/ai-dashboard" class="nav-link">AI Dashboard</a>
                        <a href="/about" class="nav-link">About</a>
                        <a href="/features" class="nav-link">Features</a>
                    </div>
                    <div class="status-indicator" id="statusIndicator">
                        <span class="status-dot" id="statusDot"></span>
                        <span class="status-text" id="statusText">OFFLINE</span>
                    </div>
                </div>
            </div>
        </div>
    </nav>

    <div class="container">
        <div class="grid">
            <div class="card">
                <div class="label">Fire Status</div>
                <div class="value {{ 'danger' if fire else 'safe' }}">
                    {{ "🔥 DANGER" if fire else "✅ SAFE" }}
                </div>
                <div style="font-size: 0.8rem; margin-top: 0.5rem; opacity: 0.8;">
                    <span id="predictionSource">Sensor Detection</span>
                </div>
            </div>

            <div class="card">
                <div class="label">Temperature</div>
                <div class="value">{{ "%.1f°C" | format(temperature) }}</div>
            </div>

            <div class="card">
                <div class="label">Smoke Level</div>
                <div class="value">{{ "%.1f ppm" | format(smoke) }}</div>
            </div>

            <div class="card">
                <div class="label">CO Level</div>
                <div class="value">{{ "%.1f ppm" | format(co) }}</div>
            </div>

            <div class="card">
                <div class="label">LPG Level</div>
                <div class="value">{{ "%.1f ppm" | format(lpg) }}</div>
            </div>

            <div class="card">
                <div class="label">Gas Value</div>
                <div class="value">{{ gasValue }}</div>
            </div>

            <div class="card">
                <div class="label">Air Quality Index</div>
                <div class="value {{ 'danger' if aqi > 150 else 'safe' }}">{{ aqi }}</div>
            </div>

            <div class="card normal-range-card">
                <div class="label">📊 Normal Range Values</div>
                <div class="normal-ranges">
                    <div class="range-item">
                        <span class="range-label">🌡️ Temperature:</span>
                        <span class="range-value">15-35°C</span>
                    </div>
                    <div class="range-item">
                        <span class="range-label">💨 Smoke:</span>
                        <span class="range-value">< 50 ppm</span>
                    </div>
                    <div class="range-item">
                        <span class="range-label">⚠️ CO:</span>
                        <span class="range-value">< 9 ppm</span>
                    </div>
                    <div class="range-item">
                        <span class="range-label">🔥 LPG:</span>
                        <span class="range-value">< 1000 ppm</span>
                    </div>
                    <div class="range-item">
                        <span class="range-label">🌬️ Gas:</span>
                        <span class="range-value">< 500</span>
                    </div>
                    <div class="range-item">
                        <span class="range-label">🌍 AQI:</span>
                        <span class="range-value">0-150</span>
                    </div>
                </div>
            </div>
        </div>

        <!-- Bomma AI Prediction - Separate Section -->
        <div class="grid" style="margin-top: 2rem;">
            <div class="card" style="background: linear-gradient(145deg, rgba(0, 184, 148, 0.15), rgba(52, 152, 219, 0.15)); border: 2px solid rgba(0, 184, 148, 0.3);">
                <div class="label" style="color: var(--accent); font-weight: 700;">🤖 Bomma AI Prediction</div>
                <div id="aiPredictionValue" class="value safe" style="font-size: 3rem; margin: 1rem 0;">
                    🤖 Analyzing...
                </div>
                <div style="font-size: 1.1rem; margin-top: 1rem; font-weight: 600;">
                    <span id="aiConfidence" style="color: var(--primary);">Confidence: --%</span>
                </div>
                <div style="font-size: 0.9rem; margin-top: 0.5rem; opacity: 0.9;">
                    <span id="predictionSource">AI-Powered Detection</span>
                </div>
            </div>
        </div>

        <div class="grid">
            <div style="position: relative; width: 100%;">
                <button onclick="toggleAnalytics()" class="analytics-btn">SHOW COMBINED ANALYTICS</button>
            </div>
        </div>

        <div id="mainGraphContainer" class="chart-container" style="display: none;">
            <div class="chart-header">
                <button class="history-btn" data-chart="combined">Show History</button>
            </div>
            <canvas id="combinedChart"></canvas>
            <div class="timestamp" id="combinedTime"></div>
        </div>
            <div class="chart-container">
                <div class="chart-header">
                    <button class="history-btn" data-chart="temp">Show History</button>
                </div>
                <canvas id="tempChart"></canvas>
                <div class="timestamp" id="tempTime"></div>
            </div>
            <div class="chart-container">
                <div class="chart-header">
                    <button class="history-btn" data-chart="smoke">Show History</button>
                </div>
                <canvas id="smokeChart"></canvas>
                <div class="timestamp" id="smokeTime"></div>
            </div>
            <div class="chart-container">
                <div class="chart-header">
                    <button class="history-btn" data-chart="co">Show History</button>
                </div>
                <canvas id="coChart"></canvas>
                <div class="timestamp" id="coTime"></div>
            </div>
            <div class="chart-container">
                <div class="chart-header">
                    <button class="history-btn" data-chart="lpg">Show History</button>
                </div>
                <canvas id="lpgChart"></canvas>
                <div class="timestamp" id="lpgTime"></div>
            </div>
            <div class="chart-container">
                <div class="chart-header">
                    <button class="history-btn" data-chart="gasValue">Show History</button>
                </div>
                <canvas id="gasChart"></canvas>
                <div class="timestamp" id="gasTime"></div>
            </div>
            <div class="chart-container">
                <div class="chart-header">
                    <button class="history-btn" data-chart="pressure">Show History</button>
                </div>
                <canvas id="pressureChart"></canvas>
                <div class="timestamp" id="pressureTime"></div>
            </div>
        </div>

        <!-- Generate Report Section -->
        <div class="grid" style="margin-top: 3rem;">
            <div class="card" style="background: linear-gradient(145deg, rgba(52, 152, 219, 0.15), rgba(0, 184, 148, 0.15)); border: 2px solid rgba(52, 152, 219, 0.3); text-align: center;">
                <div class="label" style="color: var(--primary); font-weight: 700; font-size: 1.2rem;">📊 Generate Data Report</div>
                <div style="margin: 2rem 0;">
                    <p style="font-size: 1rem; opacity: 0.9; margin-bottom: 1.5rem;">
                        Download a comprehensive CSV report of all sensor readings and historical data
                    </p>
                    <div style="display: flex; gap: 1rem; justify-content: center; flex-wrap: wrap;">
                        <button onclick="downloadReport('all')" class="history-btn" style="background: linear-gradient(135deg, var(--primary), var(--accent)); margin: 0;">
                            <i class="fas fa-download" style="margin-right: 0.5rem;"></i>
                            Download Full Report
                        </button>
                        <button onclick="downloadReport('today')" class="history-btn" style="background: linear-gradient(135deg, var(--accent), var(--primary)); margin: 0;">
                            <i class="fas fa-calendar-day" style="margin-right: 0.5rem;"></i>
                            Today's Data
                        </button>
                        <button onclick="downloadReport('week')" class="history-btn" style="background: linear-gradient(135deg, var(--warning), var(--primary)); margin: 0;">
                            <i class="fas fa-calendar-week" style="margin-right: 0.5rem;"></i>
                            Last 7 Days
                        </button>
                    </div>
                    <div style="margin-top: 1rem; font-size: 0.9rem; opacity: 0.8;">
                        <span id="reportStatus">Click a button above to generate and download your report</span>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Footer Section -->
    <footer class="footer">
        <div class="footer-content">
            <!-- Project Info -->
            <div class="footer-section">
                <h3>Project Info</h3>
                <h4>"Fire Detection & Monitoring System using IoT & AI"</h4>
                <p>Real-time monitoring of fire, smoke, and temperature.</p>
                <p>Project Link: <a href="https://www.linkedin.com/posts/jagan-s-04795b32a_iot-ai-firedetection-activity-7318241937084309508-FJFt?utm_source=share&utm_medium=member_desktop&rcm=ACoAAFMUK3cBDJIibWcQA8aGCL5tLwIi1hmjBus" target="_blank">View on LinkedIn</a></p>
            </div>

            <!-- Contact Info -->
            <div class="footer-section">
                <h3>Contact Info</h3>
                <p>Email: <a href="mailto:jaganrmkcet@gmail.com">jaganrmkcet@gmail.com</a></p>
                <p>GitHub: <a href="https://github.com/Jagan-Srinivasan" target="_blank">github.com/Jagan-Srinivasan</a></p>
            </div>

            <!-- Live Status -->
            <div class="footer-section">
                <h3>Live Status / Last Updated</h3>
                <p id="lastUpdateTime">Last data received at: --</p>
            </div>
        </div>
        <div style="text-align: center; margin-top: 2rem; padding: 1.5rem; background: linear-gradient(135deg, rgba(0, 184, 148, 0.2), rgba(52, 152, 219, 0.2)); border-radius: 15px; border: 1px solid rgba(0, 184, 148, 0.3);">
            <h2 style="font-size: 1.8rem; margin: 0; background: linear-gradient(135deg, var(--accent), var(--primary)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 700;">
                🤖 Powered by Bomma AI 
            </h2>
        </div>
    </footer>

    <!-- Back to Top Button -->
    <button id="backToTop" class="back-to-top" onclick="scrollToTop()">
        ↑ Top
    </button>

    <!-- Chatbot Float Button -->
    <button class="chat-float-button" onclick="toggleChatbot()">
        🤖
    </button>

    <!-- Chatbot Container -->
    <div id="chatbotContainer" class="chatbot-container">
        <div class="chat-header">
            🤖 Bomma AI Assistant
            <button class="chat-close" onclick="toggleChatbot()">×</button>
        </div>
        <div id="chatMessages" class="chat-messages">
            <div class="chat-message bot">
                Hello! I'm Bomma AI, your fire detection assistant. I can help you with:
                <br>• Understanding sensor readings
                <br>• Fire safety tips
                <br>• System status information
                <br>• Data analysis
                <br><br>How can I help you today?
            </div>
        </div>
        <div class="chat-input-container">
            <input type="text" id="chatInput" class="chat-input" placeholder="Ask me about fire detection..." onkeypress="handleChatKeyPress(event)">
            <button class="chat-send" onclick="sendMessage()">▶</button>
        </div>
    </div>

    <script>
        // Store historical data
        const historicalData = {
            temp: [],
            smoke: [],
            co: [],
            lpg: [],
            gasValue: [],
            pressure: []
        };

        function createChart(ctx, label, color) {
            return new Chart(ctx, {
                type: 'line',
                data: {
                    labels: Array(20).fill(''),
                    datasets: [{
                        label: label,
                        data: Array(20).fill(null),
                        borderColor: color,
                        tension: 0.4,
                        fill: true,
                        backgroundColor: color.replace(')', ', 0.1)').replace('rgb', 'rgba')
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            grid: { color: 'rgba(255, 255, 255, 0.1)' },
                            ticks: { color: '#f5f6fa' }
                        },
                        x: {
                            grid: { color: 'rgba(255, 255, 255, 0.1)' },
                            ticks: { color: '#f5f6fa' }
                        }
                    },
                    plugins: {
                        legend: {
                            labels: { color: '#f5f6fa' }
                        }
                    }
                }
            });
        }

        const tempChart = createChart(document.getElementById('tempChart').getContext('2d'), 'Temperature (°C)', 'rgb(74, 144, 226)');
        const smokeChart = createChart(document.getElementById('smokeChart').getContext('2d'), 'Smoke (ppm)', 'rgb(231, 76, 60)');
        const coChart = createChart(document.getElementById('coChart').getContext('2d'), 'CO (ppm)', 'rgb(243, 156, 18)');
        const lpgChart = createChart(document.getElementById('lpgChart').getContext('2d'), 'LPG (ppm)', 'rgb(46, 204, 113)');
        const gasChart = createChart(document.getElementById('gasChart').getContext('2d'), 'Gas Value', 'rgb(155, 89, 182)');
        const pressureChart = createChart(document.getElementById('pressureChart').getContext('2d'), 'Pressure (hPa)', 'rgb(26, 188, 156)');

        // Create combined chart
        const combinedChart = new Chart(document.getElementById('combinedChart').getContext('2d'), {
            type: 'line',
            data: {
                labels: Array(20).fill(''),
                datasets: [
                    {
                        label: 'Temperature (°C)',
                        data: Array(20).fill(null),
                        borderColor: 'rgb(74, 144, 226)',
                        tension: 0.4
                    },
                    {
                        label: 'Smoke (ppm)',
                        data: Array(20).fill(null),
                        borderColor: 'rgb(231, 76, 60)',
                        tension: 0.4
                    },
                    {
                        label: 'CO (ppm)',
                        data: Array(20).fill(null),
                        borderColor: 'rgb(243, 156, 18)',
                        tension: 0.4
                    },
                    {
                        label: 'LPG (ppm)',
                        data: Array(20).fill(null),
                        borderColor: 'rgb(46, 204, 113)',
                        tension: 0.4
                    },
                    {
                        label: 'Gas Value',
                        data: Array(20).fill(null),
                        borderColor: 'rgb(155, 89, 182)',
                        tension: 0.4
                    },
                    {
                        label: 'Pressure (hPa)',
                        data: Array(20).fill(null),
                        borderColor: 'rgb(26, 188, 156)',
                        tension: 0.4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: { color: 'rgba(255, 255, 255, 0.1)' },
                        ticks: { color: '#f5f6fa' }
                    },
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.1)' },
                        ticks: { color: '#f5f6fa' }
                    }
                },
                plugins: {
                    legend: {
                        labels: { color: '#f5f6fa' }
                    }
                }
            }
        });

        function updateChart(newData) {
            const date = new Date();
            const timeStr = date.toLocaleTimeString();

            function updateSingleChart(chart, value, timeElementId) {
                chart.data.labels.shift();
                chart.data.labels.push(timeStr);
                chart.data.datasets[0].data.shift();
                chart.data.datasets[0].data.push(value);
                chart.update();
                document.getElementById(timeElementId).innerText = 'Updated: ' + timeStr;
            }

            updateSingleChart(tempChart, newData.temp, 'tempTime');
            updateSingleChart(smokeChart, newData.smoke, 'smokeTime');
            updateSingleChart(coChart, newData.co, 'coTime');
            updateSingleChart(lpgChart, newData.lpg, 'lpgTime');
            updateSingleChart(gasChart, newData.gasValue, 'gasTime');
            updateSingleChart(pressureChart, newData.pressure, 'pressureTime');

            // Update combined chart
            combinedChart.data.labels.shift();
            combinedChart.data.labels.push(timeStr);
            combinedChart.data.datasets[0].data.shift();
            combinedChart.data.datasets[0].data.push(newData.temp);
            combinedChart.data.datasets[1].data.shift();
            combinedChart.data.datasets[1].data.push(newData.smoke);
            combinedChart.data.datasets[2].data.shift();
            combinedChart.data.datasets[2].data.push(newData.co);
            combinedChart.data.datasets[3].data.shift();
            combinedChart.data.datasets[3].data.push(newData.lpg);
            combinedChart.data.datasets[4].data.shift();
            combinedChart.data.datasets[4].data.push(newData.gasValue);
            combinedChart.data.datasets[5].data.shift();
            combinedChart.data.datasets[5].data.push(newData.pressure);
            combinedChart.update();
            document.getElementById('combinedTime').innerText = 'Updated: ' + timeStr;
        }

        // Alarm sound variables
        let audioContext = null;
        let isAlarmPlaying = false;
        let vigorousAlarmInterval = null;
        let flashInterval = null;

        // Initialize audio context
        function initAudio() {
            if (!audioContext) {
                try {
                    audioContext = new (window.AudioContext || window.webkitAudioContext)();
                } catch (e) {
                    console.log('Audio not supported');
                }
            }
        }

        // Create vigorous alarm sound pattern
        function playVigorousAlarmBeep() {
            if (!audioContext) return;

            // Triple beep pattern for urgency
            for (let i = 0; i < 3; i++) {
                setTimeout(() => {
                    const oscillator = audioContext.createOscillator();
                    const gainNode = audioContext.createGain();

                    oscillator.connect(gainNode);
                    gainNode.connect(audioContext.destination);

                    // Alternating high and low frequency for urgency
                    const freq = i % 2 === 0 ? 1400 : 900;
                    oscillator.frequency.setValueAtTime(freq, audioContext.currentTime);
                    oscillator.frequency.setValueAtTime(freq + 200, audioContext.currentTime + 0.1);
                    oscillator.frequency.setValueAtTime(freq, audioContext.currentTime + 0.2);

                    gainNode.gain.setValueAtTime(0.4, audioContext.currentTime);
                    gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.3);

                    oscillator.start(audioContext.currentTime);
                    oscillator.stop(audioContext.currentTime + 0.3);
                }, i * 150); // 150ms between each beep
            }
        }

        // Create screen flash effect
        function flashScreen() {
            const overlay = document.createElement('div');
            overlay.style.cssText = `
                position: fixed;
                top: 0;
                left: 0;
                width: 100vw;
                height: 100vh;
                background: rgba(214, 48, 49, 0.3);
                z-index: 9999;
                pointer-events: none;
                animation: flashEffect 0.3s ease-in-out;
            `;

            // Add flash animation
            const style = document.createElement('style');
            style.textContent = `
                @keyframes flashEffect {
                    0% { opacity: 0; }
                    50% { opacity: 1; }
                    100% { opacity: 0; }
                }
            `;
            document.head.appendChild(style);
            document.body.appendChild(overlay);

            setTimeout(() => {
                document.body.removeChild(overlay);
                document.head.removeChild(style);
            }, 300);
        }

        // Start vigorous alarm
        function startAlarm() {
            if (isAlarmPlaying) return;
            initAudio();
            if (!audioContext) return;

            isAlarmPlaying = true;

            // Immediate first alarm
            playVigorousAlarmBeep();
            flashScreen();

            // Set up repeated vigorous alarms every 800ms
            vigorousAlarmInterval = setInterval(() => {
                playVigorousAlarmBeep();
            }, 800);

            // Set up screen flash every 1.2 seconds
            flashInterval = setInterval(flashScreen, 1200);

            // Add visual pulsing to danger cards
            const dangerCards = document.querySelectorAll('.value.danger');
            dangerCards.forEach(card => {
                card.style.animation = 'vigorousPulse 0.5s infinite';
            });

            // Add vigorous pulse animation
            const pulseStyle = document.createElement('style');
            pulseStyle.id = 'vigorousPulseStyle';
            pulseStyle.textContent = `
                @keyframes vigorousPulse {
                    0% { 
                        transform: scale(1); 
                        text-shadow: 0 0 10px var(--danger);
                    }
                    25% { 
                        transform: scale(1.1); 
                        text-shadow: 0 0 20px var(--danger), 0 0 30px var(--danger);
                    }
                    50% { 
                        transform: scale(1.2); 
                        text-shadow: 0 0 30px var(--danger), 0 0 40px var(--danger);
                    }
                    75% { 
                        transform: scale(1.1); 
                        text-shadow: 0 0 20px var(--danger), 0 0 30px var(--danger);
                    }
                    100% { 
                        transform: scale(1); 
                        text-shadow: 0 0 10px var(--danger);
                    }
                }
            `;
            document.head.appendChild(pulseStyle);
        }

        // Stop alarm
        function stopAlarm() {
            if (vigorousAlarmInterval) {
                clearInterval(vigorousAlarmInterval);
                vigorousAlarmInterval = null;
            }
            if (flashInterval) {
                clearInterval(flashInterval);
                flashInterval = null;
            }

            // Remove visual effects
            const dangerCards = document.querySelectorAll('.value.danger');
            dangerCards.forEach(card => {
                card.style.animation = '';
            });

            // Remove vigorous pulse style
            const pulseStyle = document.getElementById('vigorousPulseStyle');
            if (pulseStyle) {
                document.head.removeChild(pulseStyle);
            }

            isAlarmPlaying = false;
        }

        // Function to update only card values
        function updateCards(data) {
            // Update fire status (first card)
            const fireStatus = document.querySelector('.card:nth-child(1) .value');
            fireStatus.textContent = data.fire ? '🔥 DANGER' : '✅ SAFE';
            fireStatus.className = `value ${data.fire ? 'danger' : 'safe'}`;

            // Find and update each card by looking for the label text
            const cards = document.querySelectorAll('.card');
            cards.forEach(card => {
                const label = card.querySelector('.label');
                const valueElement = card.querySelector('.value');
                
                if (label && valueElement) {
                    const labelText = label.textContent.toLowerCase();
                    
                    if (labelText.includes('temperature')) {
                        valueElement.textContent = `${data.temp.toFixed(1)}°C`;
                    } else if (labelText.includes('smoke')) {
                        valueElement.textContent = `${data.smoke.toFixed(1)} ppm`;
                    } else if (labelText.includes('co level')) {
                        valueElement.textContent = `${data.co.toFixed(1)} ppm`;
                    } else if (labelText.includes('lpg')) {
                        valueElement.textContent = `${data.lpg.toFixed(1)} ppm`;
                    } else if (labelText.includes('gas value')) {
                        valueElement.textContent = data.gasValue;
                    } else if (labelText.includes('air quality')) {
                        valueElement.textContent = data.aqi;
                        valueElement.className = `value ${data.aqi > 150 ? 'danger' : 'safe'}`;
                    }
                }
            });

            // Handle alarm based on fire status
            if (data.fire) {
                startAlarm();
            } else {
                stopAlarm();
            }
        }

        // Function to fetch sensor data from the /update endpoint
        function fetchSensorData() {
            fetch('/update')
                .then(response => {
                    if (!response.ok) {
                        throw new Error('Network response was not ok');
                    }
                    return response.json();
                })
                .then(data => {
                    if (data && data.current) {
                        updateCards(data.current);
                        updateChart(data.current);
                        // Store historical data
                        if (data.historical) {
                            Object.keys(data.historical).forEach(key => {
                                historicalData[key] = data.historical[key];
                            });
                        }
                    }
                })
                .catch(error => {
                    console.error('Error fetching sensor data:', error);
                });
        }

        // Function to fetch AI prediction from Bomma
        function fetchAIPrediction() {
            fetch('/fire-status')
                .then(response => {
                    if (!response.ok) {
                        throw new Error('Network response was not ok');
                    }
                    return response.json();
                })
                .then(data => {
                    updateAIPrediction(data);
                })
                .catch(error => {
                    console.error('Error fetching AI prediction:', error);
                    updateAIPredictionError();
                });
        }

        // Function to update AI prediction display
        function updateAIPrediction(data) {
            const aiPredictionValue = document.getElementById('aiPredictionValue');
            const aiConfidence = document.getElementById('aiConfidence');

            if (data.prediction === 'Fire Detected') {
                aiPredictionValue.textContent = '🔥 FIRE DETECTED';
                aiPredictionValue.className = 'value danger';
            } else {
                aiPredictionValue.textContent = '✅ NO FIRE';
                aiPredictionValue.className = 'value safe';
            }

            const confidencePercent = (data.confidence * 100).toFixed(1);
            aiConfidence.textContent = `Confidence: ${confidencePercent}%`;

            // Update prediction source to show if AI-powered or threshold-based
            const predictionSource = document.getElementById('predictionSource');
            if (data.ai_powered) {
                predictionSource.textContent = 'Bomma AI Detection';
            } else {
                predictionSource.textContent = 'Threshold Detection';
            }
        }

        // Function to handle AI prediction errors
        function updateAIPredictionError() {
            const aiPredictionValue = document.getElementById('aiPredictionValue');
            const aiConfidence = document.getElementById('aiConfidence');

            aiPredictionValue.textContent = '❌ Connection Error';
            aiPredictionValue.className = 'value';
            aiConfidence.textContent = 'Confidence: --%';
        }

        // Function to check and update status
        function checkStatus() {
            fetch('/status')
                .then(response => response.json())
                .then(data => {
                    const statusDot = document.getElementById('statusDot');
                    const statusText = document.getElementById('statusText');

                    statusText.textContent = data.status;

                    if (data.status === 'ONLINE') {
                        statusDot.className = 'status-dot online';
                        statusText.className = 'status-text online';
                    } else {
                        statusDot.className = 'status-dot offline';
                        statusText.className = 'status-text offline';
                    }
                })
                .catch(error => {
                    console.error('Error checking status:', error);
                    const statusDot = document.getElementById('statusDot');
                    const statusText = document.getElementById('statusText');
                    statusDot.className = 'status-dot offline';
                    statusText.className = 'status-text offline';
                    statusText.textContent = 'OFFLINE';
                });
        }

        // Update sensor data every 2 seconds
        setInterval(fetchSensorData, 2000);

        // Fetch AI prediction every 3 seconds
        setInterval(fetchAIPrediction, 3000);

        // Check status every 3 seconds
        setInterval(checkStatus, 3000);

        // Initial checks
        checkStatus();
        fetchAIPrediction();

        // Enable audio on first user interaction (required by some browsers)
        document.addEventListener('click', function enableAudio() {
            initAudio();
            if (audioContext && audioContext.state === 'suspended') {
                audioContext.resume();
            }
        }, { once: true });

        // Back to top functionality
        function scrollToTop() {
            window.scrollTo({
                top: 0,
                behavior: 'smooth'
            });
        }

        // Show/hide back to top button based on scroll position
        window.addEventListener('scroll', function() {
            const backToTopBtn = document.getElementById('backToTop');
            if (window.pageYOffset > 300) {
                backToTopBtn.style.display = 'block';
            } else {
                backToTopBtn.style.display = 'none';
            }
        });

        // Update last received time in footer
        function updateLastReceivedTime() {
            const now = new Date();
            const timeStr = now.toLocaleString('en-US', {
                hour: 'numeric',
                minute: '2-digit',
                hour12: true,
                month: 'long',
                day: 'numeric',
                year: 'numeric'
            });
            document.getElementById('lastUpdateTime').textContent = `Last data received at: ${timeStr}`;
        }

        // Update last received time when new data arrives
        const originalUpdateChart = updateChart;
        updateChart = function(newData) {
            originalUpdateChart(newData);
            updateLastReceivedTime();
        };

        // Hide main graph container initially
        document.getElementById('mainGraphContainer').style.display = 'none';

        // Handle analytics button using onclick
        function toggleAnalytics() {
            const graphContainer = document.getElementById('mainGraphContainer');
            const analyticsBtn = document.querySelector('.analytics-btn');
            const isHidden = graphContainer.style.display === 'none';
            graphContainer.style.display = isHidden ? 'block' : 'none';
            analyticsBtn.textContent = isHidden ? 'HIDE ANALYTICS' : 'ANALYTICS';
        }

        // Handle history buttons
        document.addEventListener('DOMContentLoaded', function() {
            document.querySelectorAll('.history-btn').forEach(button => {
                if (button && button.dataset && button.dataset.chart) {
                    button.addEventListener('click', function() {
                        const chartType = this.dataset.chart;
                        const chart = {
                            'temp': tempChart,
                            'smoke': smokeChart,
                            'co': coChart,
                            'lpg': lpgChart,
                            'gasValue': gasChart,
                            'pressure': pressureChart,
                            'combined': combinedChart
                        }[chartType];

                        if (!chart) return;

                        const showingHistory = this.textContent === 'Show History';
                        this.textContent = showingHistory ? 'Show Current' : 'Show History';

                        if (showingHistory && historicalData[chartType === 'combined' ? 'temp' : chartType]) {
                            const data = historicalData[chartType === 'combined' ? 'temp' : chartType];
                            const timestamps = data.map(d => new Date(d.timestamp).toLocaleTimeString());

                            if (chartType === 'combined') {
                                // Update all datasets in combined chart
                                Object.keys(historicalData).forEach((key, index) => {
                                    if (chart.data.datasets[index]) {
                                        chart.data.labels = timestamps;
                                        chart.data.datasets[index].data = historicalData[key].map(d => d.value);
                                    }
                                });
                            } else {
                                chart.data.labels = timestamps;
                                chart.data.datasets[0].data = data.map(d => d.value);
                            }
                        } else {
                            chart.data.labels = Array(20).fill('');
                            if (chartType === 'combined') {
                                chart.data.datasets.forEach(dataset => {
                                    dataset.data = Array(20).fill(null);
                                });
                            } else {
                                chart.data.datasets[0].data = Array(20).fill(null);
                            }
                        }
                        chart.update();
                    });
                }
            });
        });

        // Function to download CSV report
        async function downloadReport(timeRange) {
            const reportStatus = document.getElementById('reportStatus');
            reportStatus.textContent = 'Generating report...';
            reportStatus.className = '';

            try {
                const response = await fetch(`/download-report?range=${timeRange}`);
                
                if (!response.ok) {
                    throw new Error('Failed to generate report');
                }

                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                
                const filename = `fire_detection_report_${timeRange}_${new Date().toISOString().split('T')[0]}.csv`;
                a.download = filename;
                
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                window.URL.revokeObjectURL(url);

                reportStatus.textContent = `✅ Report downloaded successfully: ${filename}`;
                reportStatus.className = 'success';

                // Reset status after 5 seconds
                setTimeout(() => {
                    reportStatus.textContent = 'Click a button above to generate and download your report';
                    reportStatus.className = '';
                }, 5000);

            } catch (error) {
                console.error('Error downloading report:', error);
                reportStatus.textContent = '❌ Error generating report. Please try again.';
                reportStatus.className = 'error';

                // Reset status after 5 seconds
                setTimeout(() => {
                    reportStatus.textContent = 'Click a button above to generate and download your report';
                    reportStatus.className = '';
                }, 5000);
            }
        }

        // Chatbot functionality
        let isChatbotOpen = false;

        function toggleChatbot() {
            const chatbot = document.getElementById('chatbotContainer');
            const floatButton = document.querySelector('.chat-float-button');
            isChatbotOpen = !isChatbotOpen;
            chatbot.style.display = isChatbotOpen ? 'flex' : 'none';
            
            // Toggle active state
            if (isChatbotOpen) {
                floatButton.classList.add('active');
            } else {
                floatButton.classList.remove('active');
            }
        }

        function sendMessage() {
            const input = document.getElementById('chatInput');
            const message = input.value.trim();
            if (!message) return;

            addChatMessage(message, 'user');
            input.value = '';
            
            // Simulate bot response
            setTimeout(() => {
                const response = getBotResponse(message);
                addChatMessage(response, 'bot');
            }, 1000);
        }

        function addChatMessage(message, sender) {
            const messagesContainer = document.getElementById('chatMessages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `chat-message ${sender}`;
            messageDiv.innerHTML = message;
            messagesContainer.appendChild(messageDiv);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }

        function getBotResponse(message) {
            const lowerMessage = message.toLowerCase();
            
            if (lowerMessage.includes('temperature') || lowerMessage.includes('temp')) {
                return `🌡️ Current temperature is ${temperature.toFixed(1)}°C. Normal range is 15-35°C. ${temperature > 35 ? 'Temperature is above normal - please check for heat sources!' : 'Temperature is within normal range.'}`;
            } else if (lowerMessage.includes('smoke')) {
                return `💨 Current smoke level is ${smoke.toFixed(1)} ppm. Normal range is below 50 ppm. ${smoke > 50 ? 'Smoke levels are elevated - fire risk detected!' : 'Smoke levels are normal.'}`;
            } else if (lowerMessage.includes('fire') || lowerMessage.includes('danger')) {
                return `🔥 Fire status: ${fire_detected ? '⚠️ FIRE DETECTED! Please evacuate immediately and call emergency services!' : '✅ No fire detected. All systems normal.'}`;
            } else if (lowerMessage.includes('gas') || lowerMessage.includes('lpg')) {
                return `⛽ Current gas value is ${gasValue}. LPG level is ${lpg.toFixed(1)} ppm. Normal gas value should be below 500, and LPG below 1000 ppm.`;
            } else if (lowerMessage.includes('co') || lowerMessage.includes('carbon monoxide')) {
                return `⚠️ Current CO level is ${co.toFixed(1)} ppm. Safe level is below 9 ppm. ${co > 9 ? 'CO levels are dangerous - ensure proper ventilation!' : 'CO levels are safe.'}`;
            } else if (lowerMessage.includes('aqi') || lowerMessage.includes('air quality')) {
                return `🌬️ Current Air Quality Index is ${aqi}. Good air quality is 0-150. ${aqi > 150 ? 'Air quality is poor - consider using air purifiers.' : 'Air quality is acceptable.'}`;
            } else if (lowerMessage.includes('status') || lowerMessage.includes('system')) {
                return `📊 System Status: ${last_data_received ? 'Online - receiving data every few seconds' : 'Offline - no recent data'}. All sensors are monitoring continuously for your safety.`;
            } else if (lowerMessage.includes('help') || lowerMessage.includes('what can you do')) {
                return `🤖 I can help you with:\n• Real-time sensor readings\n• Fire safety explanations\n• Understanding normal vs dangerous levels\n• System status information\n• Emergency guidance\n\nJust ask me about any sensor or safety concern!`;
            } else if (lowerMessage.includes('emergency') || lowerMessage.includes('evacuation')) {
                return `🚨 EMERGENCY PROTOCOL:\n1. Stay calm and alert others\n2. Use nearest exit - don't use elevators\n3. Call emergency services (911/local)\n4. Meet at designated safe area\n5. Don't re-enter until cleared by authorities\n\nYour safety is the top priority!`;
            } else if (lowerMessage.includes('normal') || lowerMessage.includes('safe')) {
                return `✅ SAFE RANGES:\n🌡️ Temperature: 15-35°C\n💨 Smoke: <50 ppm\n⚠️ CO: <9 ppm\n⛽ LPG: <1000 ppm\n🌬️ Gas Value: <500\n🌍 AQI: 0-150\n\nCurrent readings are being monitored 24/7!`;
            } else {
                return `🤖 I understand you're asking about "${message}". I specialize in fire detection and safety. Try asking me about:\n• Sensor readings (temperature, smoke, gas, CO, AQI)\n• Fire status and safety\n• Emergency procedures\n• Normal vs dangerous levels\n\nWhat specific safety information would you like?`;
            }
        }

        function handleChatKeyPress(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        }

        </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(html_template,
                          fire=fire_detected,
                          temperature=temperature,
                          smoke=smoke,
                          co=co,
                          lpg=lpg,
                          gasValue=gasValue,
                          pressure=pressure,
                          aqi=aqi)

@app.route("/update", methods=["POST", "GET"])
def update():
    global fire_detected, temperature, smoke, co, lpg, gasValue, pressure, aqi, last_data_received
    if request.method == 'POST':
        data = request.get_json()
        if data:
            print("Received sensor data:", data)  # Print received data to console
            fire_detected = data.get("fire", data.get("lampIndicator", False))  # Use lampIndicator as fire if fire not present
            temperature = data.get("temp", data.get("temperature", 0.0))  # Accept both temp and temperature
            smoke = data.get("smoke", 0.0)
            co = data.get("co", 0.0)
            lpg = data.get("lpg", 0.0)
            gasValue = data.get("gasValue", 0)
            pressure = data.get("pressure", 0.0)
            aqi = data.get("aqi", 0)
            last_data_received = datetime.now()

            # Store in database
            conn = sqlite3.connect('sensor_data.db')
            c = conn.cursor()
            c.execute('''INSERT INTO sensor_readings                  (fire, temperature, smoke, co, lpg, gas_value, pressure, aqi)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                     (fire_detected, temperature, smoke, co, lpg, gasValue, pressure, aqi))
            conn.commit()
            conn.close()

            return {"status": "success"}, 200
        else:
            return {"status": "failed", "message": "No JSON data received"}, 400
    else: # GET Request
        # Get current and historical data
        conn = sqlite3.connect('sensor_data.db')
        c = conn.cursor()

        # Get latest reading
        c.execute('''SELECT * FROM sensor_readings ORDER BY timestamp DESC LIMIT 1''')
        latest = c.fetchone()

        # Get historical data (last 2 months)
        two_months_ago = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d %H:%M:%S')
        c.execute('''SELECT * FROM sensor_readings WHERE timestamp > ? ORDER BY timestamp''', (two_months_ago,))
        history = c.fetchall()

        conn.close()

        if latest:
            current_data = {
                "fire": latest[1],
                "temp": latest[2],
                "smoke": latest[3],
                "co": latest[4],
                "lpg": latest[5],
                "gasValue": latest[6],
                "pressure": latest[7],
                "aqi": latest[8]
            }
        else:
            current_data = {
                "fire": fire_detected,
                "temp": temperature,
                "smoke": smoke,
                "co": co,
                "lpg": lpg,
                "gasValue": gasValue,
                "pressure": pressure,
                "aqi": aqi
            }

        historical_data = {
            "temp": [{"timestamp": row[0], "value": row[2]} for row in history],
            "smoke": [{"timestamp": row[0], "value": row[3]} for row in history],
            "co": [{"timestamp": row[0], "value": row[4]} for row in history],
            "lpg": [{"timestamp": row[0], "value": row[5]} for row in history],
            "gasValue": [{"timestamp": row[0], "value": row[6]} for row in history],
            "pressure": [{"timestamp": row[0], "value": row[7]} for row in history]
        }

        return {"current": current_data, "historical": historical_data}

@app.route("/status")
def status():
    global last_data_received
    if last_data_received:
        time_diff = (datetime.now() - last_data_received).total_seconds()
        is_online = time_diff < 300  # Consider offline if no data for 5 minutes (300 seconds)
        return {"status": "ONLINE" if is_online else "OFFLINE", "last_update": last_data_received.isoformat()}
    return {"status": "OFFLINE", "last_update": None}

@app.route("/about")
def about():
    about_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>About - Fire Detection & Monitoring</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
        <style>
            :root {
                --card-bg: rgba(255,255,255,0.12);
                --card-border: rgba(255,255,255,0.15);
                --primary: #3498db;
                --text: #f8f9fa;
                --accent: #00cec9;
            }
            body {
                font-family: 'Inter', sans-serif;
                background: linear-gradient(135deg, rgba(30, 33, 48, 0.85), rgba(42, 48, 64, 0.85));
                color: var(--text);
                min-height: 100vh;
                margin: 0;
            }
            nav {
                background: rgba(26, 31, 53, 0.98);
                backdrop-filter: blur(20px);
                padding: 1.5rem;
                position: sticky;
                top: 0;
                z-index: 100;
            }
            nav h1 {
                font-size: 1.8rem;
                background: linear-gradient(135deg, var(--accent), var(--primary));
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin: 0;
            }
            .nav-content {
                max-width: 1200px;
                margin: 0 auto;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .nav-links {
                display: flex;
                gap: 1rem;
            }
            .nav-link {
                color: var(--text);
                text-decoration: none;
                font-weight: 500;
                padding: 0.5rem 1rem;
                border-radius: 6px;
            }
            .nav-link:hover {
                background: rgba(255, 255, 255, 0.1);
            }
            .container {
                max-width: 1200px;
                margin: 2rem auto;
                padding: 0 1rem;
            }
            .card {
                background: linear-gradient(145deg, var(--card-bg), rgba(255,255,255,0.05));
                backdrop-filter: blur(12px);
                border-radius: 20px;
                border: 1px solid var(--card-border);
                padding: 32px;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
            }
        </style>
    </head>
    <body>
        <nav>
            <div class="nav-content">
                <div style="display: flex; align-items: center; justify-content: space-between; width: 100%;">
                    <h1>Fire Detection & Monitoring</h1>
                    <div class="nav-links" style="margin-left: auto;">
                        <a href="/" class="nav-link">Home</a>
                        <a href="/ai-dashboard" class="nav-link">AI Dashboard</a>
                        <a href="/about" class="nav-link">About</a>
                        <a href="/features" class="nav-link">Features</a>
                    </div>
                </div>
            </div>
        </nav>
        <div class="container">
            <div class="card" style="max-width: 800px; margin: 2rem auto;">
                <h2 style="margin-bottom: 1.5rem; color: var(--accent);">About Fire Detection & Monitoring System</h2>
                <p style="line-height: 1.8; margin-bottom: 1rem;">
                    This advanced fire detection and environmental monitoring system uses ESP8266 microcontroller 
                    technology to provide real-time monitoring of critical environmental parameters.
                </p>
                <p style="line-height: 1.8; margin-bottom: 1rem;">
                    Our system continuously monitors temperature, smoke levels, carbon monoxide (CO), 
                    liquefied petroleum gas (LPG), air quality index (AQI), and atmospheric pressure 
                    to ensure early detection of potential fire hazards and environmental concerns.
                </p>
                <p style="line-height: 1.8;">
                    Data is transmitted wirelessly to this web dashboard, providing instant alerts 
                    and historical data visualization for comprehensive safety monitoring.
                </p>
            </div>
        </div>
    </body>
    </html>
    """
    return render_template_string(about_template)

@app.route("/ai-dashboard")
def ai_dashboard():
    ai_dashboard_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Fire Detection Dashboard-BM</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        body { font-family: 'Inter', sans-serif; }
        
        @keyframes glow {
            0%, 100% { text-shadow: 0 0 5px #3b82f6, 0 0 10px #3b82f6, 0 0 15px #3b82f6; }
            50% { text-shadow: 0 0 10px #3b82f6, 0 0 20px #3b82f6, 0 0 30px #3b82f6; }
        }
        
        @keyframes pulse-ai {
            0%, 100% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.05); opacity: 0.8; }
        }
        
        @keyframes gradient {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        .ai-glow { animation: glow 2s ease-in-out infinite; }
        .ai-pulse { animation: pulse-ai 2s ease-in-out infinite; }
        .gradient-bg { 
            background: linear-gradient(-45deg, #667eea, #764ba2, #667eea, #764ba2);
            background-size: 400% 400%;
            animation: gradient 3s ease infinite;
        }
        
        .neon-revolving {
            background: linear-gradient(45deg, #ff0080, #00ff80, #8000ff, #ff8000, #0080ff, #ff0080);
            background-size: 600% 600%;
            animation: neonRevolution 3s ease infinite;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            filter: drop-shadow(0 0 10px rgba(255, 0, 128, 0.5));
        }
        
        @keyframes neonRevolution {
            0% { 
                background-position: 0% 50%;
                filter: drop-shadow(0 0 10px rgba(255, 0, 128, 0.8));
            }
            16% { 
                background-position: 16% 50%;
                filter: drop-shadow(0 0 15px rgba(0, 255, 128, 0.8));
            }
            33% { 
                background-position: 33% 50%;
                filter: drop-shadow(0 0 15px rgba(128, 0, 255, 0.8));
            }
            50% { 
                background-position: 50% 50%;
                filter: drop-shadow(0 0 20px rgba(255, 128, 0, 0.8));
            }
            66% { 
                background-position: 66% 50%;
                filter: drop-shadow(0 0 15px rgba(0, 128, 255, 0.8));
            }
            83% { 
                background-position: 83% 50%;
                filter: drop-shadow(0 0 15px rgba(255, 0, 128, 0.8));
            }
            100% { 
                background-position: 100% 50%;
                filter: drop-shadow(0 0 10px rgba(255, 0, 128, 0.8));
            }
        }
        
        .fire-danger {
            background: linear-gradient(-45deg, #ff6b6b, #ee5a24, #ff6b6b, #ee5a24);
            background-size: 400% 400%;
            animation: gradient 1s ease infinite;
        }
        
        .glass-effect {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .dark-glass {
            background: rgba(0, 0, 0, 0.2);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
    </style>
</head>
<body class="bg-gradient-to-br from-gray-900 via-blue-900 to-purple-900 min-h-screen text-white">
    <!-- Header -->
    <header class="glass-effect p-6 mb-8">
        <div class="max-w-7xl mx-auto flex items-center justify-between">
            <div class="flex items-center space-x-4">
                <i class="fas fa-fire text-3xl text-orange-500"></i>
                <h1 class="text-3xl font-bold neon-revolving"> AI Fire Detection System</h1>
            </div>
            <div class="flex items-center space-x-2">
                <span class="text-sm opacity-75">Powered by</span>
                <span class="gradient-bg text-white px-3 py-1 rounded-full font-semibold ai-pulse">
                    🤖 Bomma AI
                </span>
            </div>
        </div>
    </header>

    <div class="max-w-7xl mx-auto px-6 space-y-8">
        <!-- Real-time Prediction Card -->
        <div class="glass-effect rounded-2xl p-8 shadow-2xl">
            <div class="text-center">
                <div class="flex items-center justify-center mb-6">
                    <i class="fas fa-brain text-4xl text-blue-400 mr-4"></i>
                    <h2 class="text-4xl font-bold ai-glow">AI Fire Prediction</h2>
                </div>
                
                <div id="predictionStatus" class="mb-6">
                    <div id="statusCard" class="inline-block px-8 py-4 rounded-xl font-bold text-2xl transition-all duration-500">
                        <i class="fas fa-spinner fa-spin mr-2"></i>
                        Loading...
                    </div>
                </div>
                
                <div class="grid grid-cols-1 md:grid-cols-4 gap-4 mt-8">
                    <div class="dark-glass rounded-xl p-4">
                        <div class="text-sm opacity-75 mb-1">Temperature</div>
                        <div id="tempValue" class="text-2xl font-bold text-orange-400">--°C</div>
                    </div>
                    <div class="dark-glass rounded-xl p-4">
                        <div class="text-sm opacity-75 mb-1">Smoke Level</div>
                        <div id="smokeValue" class="text-2xl font-bold text-gray-300">-- ppm</div>
                    </div>
                    <div class="dark-glass rounded-xl p-4">
                        <div class="text-sm opacity-75 mb-1">Gas Level</div>
                        <div id="gasValue" class="text-2xl font-bold text-purple-400">--</div>
                    </div>
                    <div class="dark-glass rounded-xl p-4">
                        <div class="text-sm opacity-75 mb-1">AI Confidence</div>
                        <div id="confidenceValue" class="text-2xl font-bold text-blue-400">--%</div>
                    </div>
                </div>
                
                <div class="mt-6 text-sm opacity-75">
                    <span id="lastUpdate">Last updated: --</span>
                    <span class="ml-4">
                        <i id="aiIcon" class="fas fa-robot text-green-400"></i>
                        <span id="aiStatus">Bomma AI Active</span>
                    </span>
                </div>
            </div>
        </div>

        <!-- Prediction History -->
        <div class="glass-effect rounded-2xl p-8 shadow-2xl">
            <div class="flex items-center justify-between mb-6">
                <h3 class="text-2xl font-bold flex items-center">
                    <i class="fas fa-history text-blue-400 mr-3"></i>
                    Prediction History
                </h3>
                <button id="downloadCsv" class="bg-blue-600 hover:bg-blue-700 px-6 py-2 rounded-lg font-semibold transition-colors duration-200 flex items-center space-x-2">
                    <i class="fas fa-download"></i>
                    <span>Download CSV</span>
                </button>
            </div>
            
            <div class="overflow-x-auto">
                <table class="w-full text-left">
                    <thead>
                        <tr class="border-b border-gray-600">
                            <th class="pb-3 font-semibold">Timestamp</th>
                            <th class="pb-3 font-semibold">Temperature</th>
                            <th class="pb-3 font-semibold">Smoke</th>
                            <th class="pb-3 font-semibold">Gas</th>
                            <th class="pb-3 font-semibold">Prediction</th>
                            <th class="pb-3 font-semibold">Confidence</th>
                        </tr>
                    </thead>
                    <tbody id="historyTable">
                        <!-- History rows will be added here -->
                    </tbody>
                </table>
            </div>
            
            <div id="noHistory" class="text-center py-8 text-gray-400">
                <i class="fas fa-clock text-4xl mb-2"></i>
                <p>No prediction history yet. Data will appear as AI makes predictions.</p>
            </div>
        </div>
    </div>

    <script>
        let predictionHistory = [];
        const maxHistoryLength = 50;

        async function fetchFireStatus() {
            try {
                const response = await fetch('/fire-status');
                const data = await response.json();
                
                updatePredictionDisplay(data);
                addToHistory(data);
                updateLastUpdateTime();
                
            } catch (error) {
                console.error('Error fetching fire status:', error);
                showError();
            }
        }

        function updatePredictionDisplay(data) {
            const statusCard = document.getElementById('statusCard');
            const tempValue = document.getElementById('tempValue');
            const smokeValue = document.getElementById('smokeValue');
            const gasValue = document.getElementById('gasValue');
            const confidenceValue = document.getElementById('confidenceValue');
            const aiIcon = document.getElementById('aiIcon');
            const aiStatus = document.getElementById('aiStatus');

            // Update sensor values
            tempValue.textContent = `${data.temperature.toFixed(1)}°C`;
            smokeValue.textContent = `${data.smoke.toFixed(1)} ppm`;
            gasValue.textContent = data.gas.toString();
            confidenceValue.textContent = `${(data.confidence * 100).toFixed(1)}%`;

            // Update AI status
            if (data.ai_powered) {
                aiIcon.className = 'fas fa-robot text-green-400';
                aiStatus.textContent = 'Bomma AI Active';
            } else {
                aiIcon.className = 'fas fa-cog text-yellow-400';
                aiStatus.textContent = 'Threshold-based';
            }

            // Update prediction status
            if (data.prediction === 'Fire Detected') {
                statusCard.className = 'inline-block px-8 py-4 rounded-xl font-bold text-2xl transition-all duration-500 fire-danger text-white shadow-2xl';
                statusCard.innerHTML = '<i class="fas fa-fire mr-2"></i>🔥 FIRE DETECTED!';
            } else {
                statusCard.className = 'inline-block px-8 py-4 rounded-xl font-bold text-2xl transition-all duration-500 bg-green-600 text-white shadow-2xl';
                statusCard.innerHTML = '<i class="fas fa-shield-alt mr-2"></i>✅ NO FIRE DETECTED';
            }
        }

        function addToHistory(data) {
            const historyItem = {
                timestamp: new Date(data.timestamp).toLocaleString(),
                temperature: data.temperature.toFixed(1),
                smoke: data.smoke.toFixed(1),
                gas: data.gas,
                prediction: data.prediction,
                confidence: (data.confidence * 100).toFixed(1)
            };

            predictionHistory.unshift(historyItem);
            
            if (predictionHistory.length > maxHistoryLength) {
                predictionHistory = predictionHistory.slice(0, maxHistoryLength);
            }

            updateHistoryTable();
        }

        function updateHistoryTable() {
            const historyTable = document.getElementById('historyTable');
            const noHistory = document.getElementById('noHistory');

            if (predictionHistory.length === 0) {
                noHistory.style.display = 'block';
                return;
            }

            noHistory.style.display = 'none';
            
            historyTable.innerHTML = predictionHistory.map(item => `
                <tr class="border-b border-gray-700 hover:bg-gray-800 transition-colors duration-200">
                    <td class="py-3 text-sm">${item.timestamp}</td>
                    <td class="py-3 text-orange-400 font-semibold">${item.temperature}°C</td>
                    <td class="py-3 text-gray-300">${item.smoke} ppm</td>
                    <td class="py-3 text-purple-400">${item.gas}</td>
                    <td class="py-3">
                        <span class="px-3 py-1 rounded-full text-sm font-semibold ${
                            item.prediction === 'Fire Detected' 
                                ? 'bg-red-600 text-white' 
                                : 'bg-green-600 text-white'
                        }">
                            ${item.prediction}
                        </span>
                    </td>
                    <td class="py-3 text-blue-400 font-semibold">${item.confidence}%</td>
                </tr>
            `).join('');
        }

        function updateLastUpdateTime() {
            const now = new Date();
            document.getElementById('lastUpdate').textContent = 
                `Last updated: ${now.toLocaleTimeString()}`;
        }

        function showError() {
            const statusCard = document.getElementById('statusCard');
            statusCard.className = 'inline-block px-8 py-4 rounded-xl font-bold text-2xl transition-all duration-500 bg-red-600 text-white';
            statusCard.innerHTML = '<i class="fas fa-exclamation-triangle mr-2"></i>Connection Error';
        }

        function downloadCSV() {
            if (predictionHistory.length === 0) {
                alert('No data to download');
                return;
            }

            const headers = ['Timestamp', 'Temperature (°C)', 'Smoke (ppm)', 'Gas', 'Prediction', 'Confidence (%)'];
            const csvContent = [
                headers.join(','),
                ...predictionHistory.map(item => 
                    [item.timestamp, item.temperature, item.smoke, item.gas, item.prediction, item.confidence].join(',')
                )
            ].join('\\n');

            const blob = new Blob([csvContent], { type: 'text/csv' });
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `fire_detection_history_${new Date().toISOString().split('T')[0]}.csv`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
        }

        // Event listeners
        document.getElementById('downloadCsv').addEventListener('click', downloadCSV);

        // Start fetching data
        fetchFireStatus();
        setInterval(fetchFireStatus, 3000);
    </script>
</body>
</html>
    """
    return render_template_string(ai_dashboard_template)

@app.route("/features")
def features():
    features_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Features - Fire Detection & Monitoring</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
        <style>
            :root {
                --card-bg: rgba(255,255,255,0.12);
                --card-border: rgba(255,255,255,0.15);
                --primary: #3498db;
                --text: #f8f9fa;
                --accent: #00cec9;
            }
            body {
                font-family: 'Inter', sans-serif;
                background: linear-gradient(135deg, rgba(30, 33, 48, 0.85), rgba(42, 48, 64, 0.85));
                color: var(--text);
                min-height: 100vh;
                margin: 0;
            }
            nav {
                background: rgba(26, 31, 53, 0.98);
                backdrop-filter: blur(20px);
                padding: 1.5rem;
                position: sticky;
                top: 0;
                z-index: 100;
            }
            nav h1 {
                font-size: 1.8rem;
                background: linear-gradient(135deg, var(--accent), var(--primary));
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin: 0;
            }
            .nav-content {
                max-width: 1200px;
                margin: 0 auto;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .nav-links {
                display: flex;
                gap: 1rem;
            }
            .nav-link {
                color: var(--text);
                text-decoration: none;
                font-weight: 500;
                padding: 0.5rem 1rem;
                border-radius: 6px;
            }
            .nav-link:hover {
                background: rgba(255, 255, 255, 0.1);
            }
            .container {
                max-width: 1200px;
                margin: 2rem auto;
                padding: 0 1rem;
            }
            .grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
                gap: 2rem;
                margin: 2rem 0;
            }
            .card {
                background: linear-gradient(145deg, var(--card-bg), rgba(255,255,255,0.05));
                backdrop-filter: blur(12px);
                border-radius: 20px;
                border: 1px solid var(--card-border);
                padding: 32px;
                text-align: left;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
            }
        </style>
    </head>
    <body>
        <nav>
            <div class="nav-content">
                <div style="display: flex; align-items: center; justify-content: space-between; width: 100%;">
                    <h1>Fire Detection & Monitoring</h1>
                    <div class="nav-links" style="margin-left: auto;">
                        <a href="/" class="nav-link">Home</a>
                        <a href="/ai-dashboard" class="nav-link">AI Dashboard</a>
                        <a href="/about" class="nav-link">About</a>
                        <a href="/features" class="nav-link">Features</a>
                    </div>
                </div>
            </div>
        </nav>
        <div class="container">
            <div class="grid">
                <div class="card">
                    <h3 style="color: var(--accent); margin-bottom: 1rem;">Real-time Monitoring</h3>
                    <p>Continuous monitoring of temperature, smoke, CO, LPG, and air quality with instant updates every 2 seconds.</p>
                </div>
                <div class="card">
                    <h3 style="color: var(--accent); margin-bottom: 1rem;">Fire Detection</h3>
                    <p>Advanced fire detection algorithms that analyze multiple sensor inputs for accurate fire hazard identification.</p>
                </div>
                <div class="card">
                    <h3 style="color: var(--accent); margin-bottom: 1rem;">Data Visualization</h3>
                    <p>Interactive charts and graphs showing current readings and historical trends for comprehensive analysis.</p>
                </div>
                <div class="card">
                    <h3 style="color: var(--accent); margin-bottom: 1rem;">Status Monitoring</h3>
                    <p>Online/offline status indicator with automatic detection of sensor connectivity issues.</p>
                </div>
                <div class="card">
                    <h3 style="color: var(--accent); margin-bottom: 1rem;">Historical Data</h3>
                    <p>Store and retrieve up to 2 months of sensor data with timestamp information for trend analysis.</p>
                </div>
                <div class="card">
                    <h3 style="color: var(--accent); margin-bottom: 1rem;">Wireless Communication</h3>
                    <p>ESP8266-based wireless data transmission with JSON API endpoints for easy integration.</p>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    return render_template_string(features_template)

# ML prediction endpoint for sensor data
@app.route('/sensor', methods=['POST'])
def sensor_data():
    global fire_detected, temperature, smoke, co, lpg, gasValue, pressure, aqi, last_data_received

    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No JSON data received'}), 400

        temp = float(data.get('temperature', 0))
        smoke_val = float(data.get('smoke', 0))
        gas_val = float(data.get('gas', 0))

        print(f"🔥 Sensor data received - Temperature: {temp}°C, Smoke: {smoke_val}ppm, Gas: {gas_val}")

        # Predict fire using ML model if available
        prediction_result = 'No Fire'
        prediction_confidence = 0.0

        if model is not None:
            try:
                input_data = [[temp, smoke_val, gas_val]]
                prediction = model.predict(input_data)

                # Try to get prediction probability if available
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(input_data)
                    prediction_confidence = float(probabilities[0][1]) if len(probabilities[0]) > 1 else 0.0

                prediction_result = 'Fire Detected' if prediction[0] == 1 else 'No Fire'
                print(f"🤖 Bomma AI Prediction: {prediction_result} (Confidence: {prediction_confidence:.2f})")

            except Exception as e:
                print(f"❌ Prediction error: {e}")
                prediction_result = 'Prediction Error'
        else:
            print("⚠️ No ML model available - using threshold-based detection")
            # Fallback threshold-based detection
            if temp > 50 or smoke_val > 300 or gas_val > 400:
                prediction_result = 'Fire Detected'

        # Update global variables
        fire_detected = prediction_result == 'Fire Detected'
        temperature = temp
        smoke = smoke_val
        last_data_received = datetime.now()

        # Store in database
        conn = sqlite3.connect('sensor_data.db')
        c = conn.cursor()
        c.execute('''INSERT INTO sensor_readings 
                    (fire, temperature, smoke, co, lpg, gas_value, pressure, aqi)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                 (fire_detected, temperature, smoke, co, lpg, gas_val, pressure, aqi))
        conn.commit()
        conn.close()

        return jsonify({
            'status': 'Data Received',
            'prediction': prediction_result,
            'confidence': prediction_confidence,
            'fire_detected': fire_detected,
            'timestamp': last_data_received.isoformat()
        })

    except Exception as e:
        print(f"❌ Error in sensor endpoint: {e}")
        return jsonify({'error': str(e)}), 500

# Simple POST route to receive data from ESP8266
@app.route('/simple-update', methods=['POST'])
def simple_update():
    global fire_detected, temperature, smoke, co, lpg, gasValue, pressure, aqi, last_data_received

    data = request.get_json()
    print("Simple endpoint - Received data:", data)

    # Update in-memory variables
    fire_detected = data.get("fire", False)
    temperature = data.get("temperature", 0.0)
    smoke = data.get("smoke", 0.0)
    co = data.get("co", 0.0)
    lpg = data.get("lpg", 0.0)
    gasValue = data.get("gasValue", 0)
    pressure = data.get("pressure", 0.0)
    aqi = data.get("aqi", 0)
    last_data_received = datetime.now()

    # Optional: save to database
    conn = sqlite3.connect('sensor_data.db')
    c = conn.cursor()
    c.execute('''INSERT INTO sensor_readings 
                 (fire, temperature, smoke, co, lpg, gas_value, pressure, aqi) 
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
              (fire_detected, temperature, smoke, co, lpg, gasValue, pressure, aqi))
    conn.commit()
    conn.close()

    return {"status": "success"}, 200

@app.route('/fire-status', methods=['GET'])
def fire_status():
    global fire_detected, temperature, smoke, co, lpg, gasValue, last_data_received

    # Get current sensor readings (you can modify this to get real sensor data)
    current_temp = temperature
    current_smoke = smoke
    current_gas = gasValue

    # Make AI prediction if model is available
    prediction_result = 'No Fire'
    confidence = 0.0

    if model is not None:
        try:
            input_data = [[current_temp, current_smoke, current_gas]]
            prediction = model.predict(input_data)

            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(input_data)
                confidence = float(probabilities[0][1]) if len(probabilities[0]) > 1 else 0.0

            prediction_result = 'Fire Detected' if prediction[0] == 1 else 'No Fire'

        except Exception as e:
            print(f"Prediction error: {e}")
            prediction_result = 'Model Error'
    else:
        # Fallback threshold-based detection
        if current_temp > 50 or current_smoke > 300 or current_gas > 400:
            prediction_result = 'Fire Detected'
            confidence = 0.85
        else:
            confidence = 0.15

    return jsonify({
        'timestamp': datetime.now().isoformat(),
        'temperature': current_temp,
        'smoke': current_smoke,
        'gas': current_gas,
        'prediction': prediction_result,
        'confidence': confidence,
        'ai_powered': model is not None
    })

# CSV report download endpoint
@app.route('/download-report')
def download_report():
    time_range = request.args.get('range', 'all')
    
    try:
        conn = sqlite3.connect('sensor_data.db')
        c = conn.cursor()
        
        # Determine date filter based on time range
        if time_range == 'today':
            date_filter = datetime.now().strftime('%Y-%m-%d')
            c.execute('''SELECT * FROM sensor_readings 
                        WHERE DATE(timestamp) = ? 
                        ORDER BY timestamp DESC''', (date_filter,))
        elif time_range == 'week':
            week_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d %H:%M:%S')
            c.execute('''SELECT * FROM sensor_readings 
                        WHERE timestamp >= ? 
                        ORDER BY timestamp DESC''', (week_ago,))
        else:  # all data
            c.execute('''SELECT * FROM sensor_readings 
                        ORDER BY timestamp DESC''')
        
        data = c.fetchall()
        conn.close()
        
        if not data:
            return jsonify({'error': 'No data found for the selected time range'}), 404
        
        # Create CSV content
        csv_content = "Timestamp,Fire Status,Temperature (°C),Smoke (ppm),CO (ppm),LPG (ppm),Gas Value,Pressure (hPa),AQI\n"
        
        for row in data:
            timestamp = row[0]
            fire_status = "FIRE DETECTED" if row[1] else "SAFE"
            temperature = row[2] if row[2] is not None else 0
            smoke = row[3] if row[3] is not None else 0
            co = row[4] if row[4] is not None else 0
            lpg = row[5] if row[5] is not None else 0
            gas_value = row[6] if row[6] is not None else 0
            pressure = row[7] if row[7] is not None else 0
            aqi = row[8] if row[8] is not None else 0
            
            csv_content += f"{timestamp},{fire_status},{temperature:.2f},{smoke:.2f},{co:.2f},{lpg:.2f},{gas_value},{pressure:.2f},{aqi}\n"
        
        # Create response with CSV file
        from flask import Response
        
        response = Response(
            csv_content,
            mimetype='text/csv',
            headers={
                'Content-Disposition': f'attachment; filename=fire_detection_report_{time_range}_{datetime.now().strftime("%Y%m%d")}.csv'
            }
        )
        
        return response
        
    except Exception as e:
        print(f"Error generating CSV report: {e}")
        return jsonify({'error': 'Failed to generate report'}), 500

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))  # Default to 5000 for local testing
    app.run(host='0.0.0.0', port=port)
