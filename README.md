# Block Blast Solver

Ứng dụng giải Block Blast puzzle với giao diện đơn giản và hiệu quả.

## Tính năng

- **Giao diện đơn giản**: Layout 2 cột với Current Board + Initial Pieces | 3 Solutions
- **Xử lý ảnh thực tế**: Trích xuất board từ screenshot Block Blast
- **Thuật toán giải**: Sử dụng heuristics để tìm solutions tối ưu
- **Hiển thị trực quan**: Board, pieces và solutions được hiển thị dưới dạng ảnh
- **3 Solutions**: Hiển thị 3 solutions tốt nhất

## Cách sử dụng

### 1. Chạy ứng dụng

```bash
python main.py
```

### 2. Upload ảnh

- Click nút "📁 Choose Image"
- Chọn ảnh screenshot Block Blast (PNG, JPG, JPEG, GIF, BMP)

### 3. Giải puzzle

- Click nút "🚀 SOLVER" để bắt đầu giải
- Ứng dụng sẽ:
  - Xử lý ảnh và trích xuất board
  - Hiển thị current board
  - Hiển thị 3 pieces ban đầu
  - Tìm và hiển thị 3 solutions tốt nhất

### 4. Các nút khác

- **🧪 TEST**: Test với dữ liệu mẫu
- **🗑️ CLEAR**: Xóa tất cả và reset

## Cấu trúc giao diện

```
┌─────────────────────────────────────┐
│ Block Blast Solver                  │
├─────────────────────────────────────┤
│ Upload Section                      │
├─────────────────┬───────────────────┤
│ Current Board   │ Solution 1        │
│ + Initial       │ Solution 2        │
│   Pieces        │ Solution 3        │
├─────────────────┴───────────────────┤
│ 🚀 SOLVER | 🧪 TEST | 🗑️ CLEAR    │
└─────────────────────────────────────┘
```

## Yêu cầu hệ thống

- Python 3.7+
- OpenCV (cv2)
- NumPy
- PIL (Pillow)
- Tkinter (có sẵn với Python)

## Cài đặt dependencies

```bash
pip install -r requirements.txt
```

## Cấu trúc code

- **BlockBlastVision**: Xử lý ảnh và trích xuất board
- **BlockBlastSolver**: Thuật toán giải puzzle
- **BlockBlastGUI**: Giao diện người dùng (Simple UI)
- **Utils**: Các hàm tiện ích

## Lưu ý

- Ứng dụng được tối ưu cho grid 8x8
- Hỗ trợ xử lý ảnh thực tế từ game Block Blast
- Có fallback mechanism khi không xử lý được ảnh
- Hiển thị solutions dưới dạng ảnh trực quan

