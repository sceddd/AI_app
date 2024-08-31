import subprocess
import re

# Chạy pip freeze và lấy kết quả
# result = subprocess.run(["pip", "freeze"], stdout=subprocess.PIPE)
#
# # Chuyển kết quả thành một danh sách các dòng
# lines = result.stdout.decode('utf-8').splitlines()
#
# # Xử lý các dòng để loại bỏ đường dẫn nội bộ
# cleaned_lines = []
# for line in lines:
#     # Thay thế các đường dẫn file bằng regex
#     clean_line = re.sub(r' @ file://.*', '', line)
#     cleaned_lines.append(clean_line)
#
# # Ghi các dòng đã xử lý vào file requirements.txt
# with open("reqs.txt", "w") as f:
#     f.write("\n".join(cleaned_lines))
#
# print("Đã tạo requirements.txt thành công!")

import os

# Lấy giá trị của một biến môi trường cụ thể
print(os.getenv('VARIABLE_NAME'))

# Hoặc liệt kê tất cả các biến môi trường
for key, value in os.environ.items():
    print(f'{key}: {value}')