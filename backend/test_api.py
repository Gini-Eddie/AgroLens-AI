import requests
import os

# 1. SETUP
url = 'http://127.0.0.1:5000/predict'
filename = 'RS_Rust_1566.JPG'  # <--- Make sure this matches your file EXACTLY

# 2. DEBUGGING INFO
print(f"📍 I am currently running in this folder: {os.getcwd()}")
print(f"👀 I am looking for: {filename}")

# Check if file exists in the current folder
if filename in os.listdir():
    print("✅ FOUND IT! The file is right here.")

    # 3. SEND REQUEST
    try:
        with open(filename, 'rb') as img:
            files = {'file': img}
            response = requests.post(url, files=files)
        print("------------------------------------------------")
        print("🤖 AI SAYS:", response.json())
        print("------------------------------------------------")
    except Exception as e:
        print(f"❌ Connection Error: {e}")

else:
    print("❌ ERROR: File NOT found in this folder.")
    print("📂 Here are the files I actually see in this folder:")
    print(os.listdir())  # This lists all files so you can see if you are in the wrong place