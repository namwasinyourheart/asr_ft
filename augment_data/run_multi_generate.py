#!/usr/bin/env python3
import subprocess
from multiprocessing import Process
import time

# Danh sách các lệnh muốn chạy song song
COMMANDS = [
    [
        "python", "/home/nampv1/projects/asr/asr_ft/augment_data/generate_addess_minimax_selenium.py",
        "--text_path", "/media/nampv1/hdd/data/vn_commune_district_province/raw/text/commune_list_with_no_prefix.txt",
        "--output_dir", "/media/nampv1/hdd/data/Voice-of-Address/raw/generated/commune_list_with_no_prefix/",
        "--provider", "minimax_selenium",
        "--delay", "3.0",
        "--batch_size", "10",
        "--continue_from_text_id", "4700"
    ],
    [
        "python", "/home/nampv1/projects/asr/asr_ft/augment_data/generate_addess_minimax_selenium.py",
        "--text_path", "/media/nampv1/hdd/data/vn_commune_district_province/raw/text/commune_list_with_no_prefix.txt",
        "--output_dir", "/media/nampv1/hdd/data/Voice-of-Address/raw/generated/commune_list_with_no_prefix/",
        "--provider", "minimax_selenium",
        "--delay", "3.0",
        "--batch_size", "10",
        "--continue_from_text_id", "5500"
    ],
    # thêm args khác tại đây nếu muốn
]

def run_command(cmd):
    """Chạy 1 instance script với args riêng"""
    print(f"🚀 Starting process: {' '.join(cmd)}")
    subprocess.run(cmd)
    print(f"✅ Finished process: {' '.join(cmd)}")

if __name__ == "__main__":
    processes = []

    # khởi động song song
    for i, cmd in enumerate(COMMANDS):
        p = Process(target=run_command, args=(cmd,))
        p.start()
        processes.append(p)
        time.sleep(2)  # tránh mở Chrome cùng lúc (tăng ổn định)

    # đợi tất cả kết thúc
    for p in processes:
        p.join()

    print("🎉 All processes finished.")
