import requests
import os
import time
from datetime import datetime
# นำเข้าฟังก์ชันหลักจากสคริปต์เดิมของคุณ
from automation_pipeline import main_pipeline 

# Configuration
WIKI_URL = "https://helldivers.wiki.gg/wiki/Stratagems"
VERSION_FILE = "../../version.txt"

def get_remote_last_modified(url):
    """ตรวจสอบเวลาอัปเดตล่าสุดจาก Header ของเว็บไซต์"""
    try:
        response = requests.head(url)
        last_modified = response.headers.get('Last-Modified')
        if last_modified:
            # แปลง string เวลาของ HTTP เป็น datetime object
            return datetime.strptime(last_modified, '%a, %d %b %Y %H:%M:%S %Z')
    except Exception as e:
        print(f"Error checking wiki: {e}")
    return None

def get_local_last_update():
    """อ่านเวลาที่อัปเดตครั้งล่าสุดจากไฟล์ในเครื่อง"""
    if os.path.exists(VERSION_FILE):
        with open(VERSION_FILE, 'r') as f:
            try:
                return datetime.fromisoformat(f.read().strip())
            except:
                return None
    return None

def update_local_version(dt):
    """บันทึกเวลาอัปเดตล่าสุดลงเครื่อง"""
    with open(VERSION_FILE, 'w') as f:
        f.write(dt.isoformat())

def check_and_update():
    print(f"[{datetime.now()}] Checking for Helldivers 2 Wiki updates...")
    
    remote_time = get_remote_last_modified(WIKI_URL)
    local_time = get_local_last_update()
    
    if remote_time is None:
        print("Could not retrieve wiki information. Skipping...")
        return

    # ถ้าไม่มีข้อมูลในเครื่อง หรือ ข้อมูลบนเว็บใหม่กว่า
    if local_time is None or remote_time > local_time:
        print(f"New Update Found! (Wiki: {remote_time} > Local: {local_time})")
        print("Starting Automation Pipeline...")
        
        # รัน Pipeline ที่รวมการ Fetch และ Extract
        main_pipeline() 
        
        # บันทึกเวลาใหม่
        update_local_version(remote_time)
        print("Update Completed Successfully.")
    else:
        print("Everything is up to date.")

if __name__ == "__main__":
    # รันทันทีเมื่อเรียกสคริปต์
    check_and_update()