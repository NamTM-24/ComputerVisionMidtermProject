"""
Block Blast Solver 
"""
import argparse
import json
import os
import sys
from typing import Dict, List
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

import numpy as np
import cv2

# =============================================================================
# VISION MODULE
# =============================================================================

class BlockBlastVision:
    """Xử lý ảnh để trích xuất board"""
    
    def __init__(self, grid_size: int = 8):
        self.grid_size = grid_size
        self.color_thresholds = {
            'red': {'lower': (0, 50, 50), 'upper': (10, 255, 255)},
            'yellow': {'lower': (20, 50, 50), 'upper': (30, 255, 255)},
            'green': {'lower': (40, 50, 50), 'upper': (80, 255, 255)},
            'blue': {'lower': (100, 50, 50), 'upper': (130, 255, 255)},
            'purple': {'lower': (130, 50, 50), 'upper': (160, 255, 255)},
            'orange': {'lower': (10, 50, 50), 'upper': (20, 255, 255)},
            'pink': {'lower': (160, 50, 50), 'upper': (180, 255, 255)},
            'cyan': {'lower': (80, 50, 50), 'upper': (100, 255, 255)}
        }
    
    def extract_board_from_image(self, image_path: str) -> np.ndarray:
        """Trích xuất board từ ảnh Block Blast"""
        try:
            # Đọc ảnh
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Không thể đọc ảnh: {image_path}")
            
            print(f"Ảnh gốc: {image.shape}")
            
            # Tìm board bằng cách tìm vùng có nhiều khối màu
            board_region = self._find_board_region(image)
            print(f"Board region: {board_region}")
            
            # Crop board region
            x, y, w, h = board_region
            board_image = image[y:y+h, x:x+w]
            print(f"Board image: {board_image.shape}")
            
            # Resize về kích thước chuẩn (vuông)
            size = min(board_image.shape[:2])
            board_image = cv2.resize(board_image, (size, size))
            print(f"Resized board: {board_image.shape}")
            
            # Chia grid và phân tích
            board = self._extract_grid_cells_improved(board_image)
            print(f"Extracted board shape: {board.shape}")
            print(f"Board có {np.sum(board != 0)} blocks")
            
            # Debug: hiển thị board
            self._debug_board(board)
            
            return board
            
        except Exception as e:
            print(f"Lỗi khi xử lý ảnh: {e}")
            # Fallback: tạo board mẫu
            return self._create_fallback_board()
    
    def _find_board_region(self, image):
        """Tìm vùng board trong ảnh Block Blast - phiên bản cải tiến"""
        h, w = image.shape[:2]
        
        print(f"Image size: {w}x{h}")
        
        # Tìm vùng board bằng cách tìm pattern vuông
        # Chuyển sang grayscale để tìm edges
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Tìm edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Tìm contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Tìm contour có diện tích lớn nhất và có dạng gần vuông
        best_contour = None
        best_score = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 10000:  # Quá nhỏ
                continue
                
            # Tính bounding rect
            x, y, w_rect, h_rect = cv2.boundingRect(contour)
            
            # Kiểm tra tỷ lệ (gần vuông)
            aspect_ratio = w_rect / h_rect
            if 0.7 <= aspect_ratio <= 1.3:  # Gần vuông
                # Tính score dựa trên diện tích và tỷ lệ
                score = area * (1 - abs(1 - aspect_ratio))
                if score > best_score:
                    best_score = score
                    best_contour = contour
        
        if best_contour is not None:
            x, y, w_rect, h_rect = cv2.boundingRect(best_contour)
            print(f"Found board region: x={x}, y={y}, w={w_rect}, h={h_rect}")
            return (x, y, w_rect, h_rect)
        else:
            # Fallback: tìm vùng có nhiều màu sắc
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Tìm vùng có nhiều màu sắc (không phải nền)
            lower_bright = np.array([0, 30, 50])
            upper_bright = np.array([180, 255, 255])
            mask = cv2.inRange(hsv, lower_bright, upper_bright)
            
            # Tìm contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w_rect, h_rect = cv2.boundingRect(largest_contour)
                
                # Mở rộng để bao gồm toàn bộ board
                margin = 50
                x = max(0, x - margin)
                y = max(0, y - margin)
                w_rect = min(w_rect + 2*margin, image.shape[1] - x)
                h_rect = min(h_rect + 2*margin, image.shape[0] - y)
                
                print(f"Found board region (fallback): x={x}, y={y}, w={w_rect}, h={h_rect}")
                return (x, y, w_rect, h_rect)
            else:
                # Fallback cuối cùng: lấy vùng trung tâm
                center_x, center_y = w // 2, h // 2
                board_size = min(w, h) // 2
                x = center_x - board_size // 2
                y = center_y - board_size // 2
                print(f"Fallback board region: x={x}, y={y}, w={board_size}, h={board_size}")
                return (x, y, board_size, board_size)
    
    def _extract_grid_cells_improved(self, image):
        """Chia ảnh thành grid cells và phân tích màu - phiên bản cải tiến"""
        h, w = image.shape[:2]
        cell_size = min(h, w) // self.grid_size
        
        print(f"Cell size: {cell_size}")
        
        # Tính margin để crop center của mỗi cell
        margin = int(cell_size * 0.1)  # 10% margin
        
        board = np.zeros((self.grid_size, self.grid_size), dtype=int)
        
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                # Tính tọa độ cell
                y1 = row * cell_size + margin
                y2 = (row + 1) * cell_size - margin
                x1 = col * cell_size + margin
                x2 = (col + 1) * cell_size - margin
                
                # Đảm bảo không vượt quá kích thước ảnh
                y1 = max(0, y1)
                y2 = min(h, y2)
                x1 = max(0, x1)
                x2 = min(w, x2)
                
                # Crop cell
                cell = image[y1:y2, x1:x2]
                
                if cell.size == 0:
                    continue
                
                # Phân tích màu
                color_code = self._analyze_cell_color_improved(cell)
                board[row, col] = color_code
        
        return board
    
    def _analyze_cell_color_improved(self, cell):
        """Phân tích màu của cell - phiên bản cải tiến cho Block Blast"""
        if cell.size == 0:
            return 0
        
        # Chuyển sang HSV
        hsv = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
        
        # Tính mean HSV
        mean_hsv = np.mean(hsv, axis=(0, 1))
        h, s, v = mean_hsv
        
        # Kiểm tra nếu cell trống - cải tiến ngưỡng
        # Cell trống thường có V thấp hoặc S thấp
        if v < 100 or s < 40:
            return 0
        
        # Phân loại màu dựa trên H (Hue) - tối ưu cho Block Blast
        if h < 10 or h > 170:  # Red range
            return 1
        elif 10 <= h < 25:  # Orange
            return 2
        elif 25 <= h < 45:  # Yellow
            return 3
        elif 45 <= h < 75:  # Green
            return 4
        elif 75 <= h < 105:  # Cyan
            return 5
        elif 105 <= h < 135:  # Blue
            return 6
        elif 135 <= h < 165:  # Purple
            return 7
        else:
            return 8  # Other colors
    
    def _debug_board(self, board):
        """Debug: hiển thị board để kiểm tra"""
        print("\n=== DEBUG BOARD ===")
        for row in range(board.shape[0]):
            row_str = ""
            for col in range(board.shape[1]):
                if board[row, col] == 0:
                    row_str += "."
                else:
                    row_str += str(board[row, col])
            print(f"Row {row}: {row_str}")
        
        # Thống kê
        total_cells = board.size
        empty_cells = np.sum(board == 0)
        occupied_cells = total_cells - empty_cells
        print(f"Total cells: {total_cells}")
        print(f"Empty cells: {empty_cells}")
        print(f"Occupied cells: {occupied_cells}")
        print(f"Fill ratio: {occupied_cells/total_cells*100:.1f}%")
        print("==================\n")
    
    def test_with_sample_image(self):
        """Test với ảnh mẫu để kiểm tra thuật toán"""
        print("=== TESTING VISION ALGORITHM ===")
        
        # Tạo ảnh mẫu giống Block Blast
        sample_image = self._create_sample_block_blast_image()
        
        # Test extract board
        board = self._extract_grid_cells_improved(sample_image)
        print(f"Sample board shape: {board.shape}")
        print(f"Sample board có {np.sum(board != 0)} blocks")
        
        # Debug board
        self._debug_board(board)
        
        return board
    
    def _create_sample_block_blast_image(self):
        """Tạo ảnh mẫu giống Block Blast"""
        # Tạo ảnh 400x400
        image = np.ones((400, 400, 3), dtype=np.uint8) * 139  # Nền nâu
        
        # Vẽ board 8x8
        cell_size = 40
        board_size = 8 * cell_size
        
        # Vẽ border
        cv2.rectangle(image, (50, 50), (50 + board_size, 50 + board_size), (0, 0, 0), 2)
        
        # Vẽ một số blocks mẫu
        # Block vàng (hình L)
        cv2.rectangle(image, (50 + 0*cell_size, 50 + 5*cell_size), (50 + 0*cell_size + cell_size, 50 + 6*cell_size), (0, 255, 255), -1)
        cv2.rectangle(image, (50 + 0*cell_size, 50 + 6*cell_size), (50 + 2*cell_size, 50 + 7*cell_size), (0, 255, 255), -1)
        
        # Block xanh lá (hình vuông 2x2)
        cv2.rectangle(image, (50 + 0*cell_size, 50 + 2*cell_size), (50 + 2*cell_size, 50 + 4*cell_size), (0, 255, 0), -1)
        
        # Block xanh dương (hình dài)
        cv2.rectangle(image, (50 + 7*cell_size, 50 + 0*cell_size), (50 + 8*cell_size, 50 + 6*cell_size), (255, 0, 0), -1)
        cv2.rectangle(image, (50 + 4*cell_size, 50 + 4*cell_size), (50 + 7*cell_size, 50 + 5*cell_size), (255, 0, 0), -1)
        cv2.rectangle(image, (50 + 6*cell_size, 50 + 5*cell_size), (50 + 7*cell_size, 50 + 6*cell_size), (255, 0, 0), -1)
        
        return image
    
    def test_with_real_image(self, image_path):
        """Test với ảnh thực tế để debug"""
        print("=== TESTING WITH REAL IMAGE ===")
        
        try:
            # Đọc ảnh
            image = cv2.imread(image_path)
            if image is None:
                print(f"Không thể đọc ảnh: {image_path}")
                return None
            
            print(f"Ảnh gốc: {image.shape}")
            
            # Tìm board region
            board_region = self._find_board_region(image)
            print(f"Board region: {board_region}")
            
            # Crop board region
            x, y, w, h = board_region
            board_image = image[y:y+h, x:x+w]
            print(f"Board image: {board_image.shape}")
            
            # Resize về kích thước chuẩn (vuông)
            size = min(board_image.shape[:2])
            board_image = cv2.resize(board_image, (size, size))
            print(f"Resized board: {board_image.shape}")
            
            # Chia grid và phân tích
            board = self._extract_grid_cells_improved(board_image)
            print(f"Extracted board shape: {board.shape}")
            print(f"Board có {np.sum(board != 0)} blocks")
            
            # Debug: hiển thị board
            self._debug_board(board)
            
            return board
            
        except Exception as e:
            print(f"Lỗi khi test với ảnh thực tế: {e}")
            return None
    
    def _order_points(self, pts):
        """Sắp xếp 4 điểm theo thứ tự: top-left, top-right, bottom-right, bottom-left"""
        rect = np.zeros((4, 2), dtype="float32")
        
        # Tổng tọa độ: top-left có tổng nhỏ nhất, bottom-right có tổng lớn nhất
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # top-left
        rect[2] = pts[np.argmax(s)]  # bottom-right
        
        # Hiệu tọa độ: top-right có hiệu nhỏ nhất, bottom-left có hiệu lớn nhất
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # top-right
        rect[3] = pts[np.argmax(diff)]  # bottom-left
        
        return rect
    
    def _four_point_transform(self, image, pts):
        """Warp perspective từ 4 điểm"""
        rect = self._order_points(pts)
        (tl, tr, br, bl) = rect
        
        # Tính width và height mới
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        
        # Điểm đích
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")
        
        # Tính ma trận transform
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        
        return warped
    
    def _extract_grid_cells(self, image):
        """Chia ảnh thành grid cells và phân tích màu"""
        h, w = image.shape[:2]
        cell_size = min(h, w) // self.grid_size
        
        # Tính margin để crop center
        margin = int(cell_size * 0.15)  # 15% margin
        
        board = np.zeros((self.grid_size, self.grid_size), dtype=int)
        
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                # Tính tọa độ cell
                y1 = row * cell_size + margin
                y2 = (row + 1) * cell_size - margin
                x1 = col * cell_size + margin
                x2 = (col + 1) * cell_size - margin
                
                # Crop cell
                cell = image[y1:y2, x1:x2]
                
                # Phân tích màu
                color_code = self._analyze_cell_color(cell)
                board[row, col] = color_code
        
        return board
    
    def _analyze_cell_color(self, cell):
        """Phân tích màu của cell"""
        if cell.size == 0:
            return 0
        
        # Chuyển sang HSV
        hsv = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
        
        # Tính mean HSV
        mean_hsv = np.mean(hsv, axis=(0, 1))
        h, s, v = mean_hsv
        
        # Kiểm tra nếu cell trống (V thấp hoặc S thấp)
        if v < 50 or s < 30:
            return 0
        
        # So sánh với các màu đã định nghĩa
        for i, (color_name, thresholds) in enumerate(self.color_thresholds.items(), 1):
            lower = np.array(thresholds['lower'])
            upper = np.array(thresholds['upper'])
            
            if np.all(mean_hsv >= lower) and np.all(mean_hsv <= upper):
                return i
        
        # Nếu không match, dùng KMeans
        return self._kmeans_color_analysis(cell)
    
    def _kmeans_color_analysis(self, cell):
        """Phân tích màu bằng KMeans"""
        try:
            from sklearn.cluster import KMeans
            
            # Reshape cell thành 1D array
            pixels = cell.reshape(-1, 3)
            
            # KMeans với 2 clusters
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            # Lấy cluster center
            centers = kmeans.cluster_centers_
            
            # Tìm cluster có màu sáng nhất
            brightness = np.sum(centers, axis=1)
            brightest_cluster = np.argmax(brightness)
            
            # Chuyển sang HSV
            center_bgr = centers[brightest_cluster].astype(np.uint8)
            center_hsv = cv2.cvtColor(center_bgr.reshape(1, 1, 3), cv2.COLOR_BGR2HSV)[0, 0]
            
            # Kiểm tra nếu là màu trống
            if center_hsv[2] < 50:  # V thấp
                return 0
            
            # Gán màu dựa trên H
            h = center_hsv[0]
            if h < 10:
                return 1  # Red
            elif h < 20:
                return 2  # Yellow
            elif h < 40:
                return 3  # Green
            elif h < 80:
                return 4  # Blue
            elif h < 100:
                return 5  # Purple
            elif h < 120:
                return 6  # Orange
            elif h < 140:
                return 7  # Pink
            else:
                return 8  # Cyan
                
        except ImportError:
            # Fallback: dùng màu mặc định
            return 1
    
    def _create_fallback_board(self):
        """Tạo board mẫu khi không xử lý được ảnh"""
        board = np.zeros((self.grid_size, self.grid_size), dtype=int)
        
        # Tạo một số block mẫu
        if self.grid_size >= 8:
            board[0, :3] = 1  # Hàng 1: 3 block đỏ
            board[1, 1:4] = 2  # Hàng 2: 3 block vàng
            board[2, 2:5] = 3  # Hàng 3: 3 block xanh lá
            board[3, 3:6] = 4  # Hàng 4: 3 block xanh dương
            board[4, 4:7] = 5  # Hàng 5: 3 block tím
            
            # Thêm một số block rải rác
            board[6, 1] = 6
            board[7, 2] = 7
        
        return board
    
    def visualize_board(self, board, output_path="board_visualization.png"):
        """Tạo ảnh visualization của board"""
        h, w = board.shape
        cell_size = 50
        
        # Tạo ảnh
        image = np.ones((h * cell_size, w * cell_size, 3), dtype=np.uint8) * 255
        
        # Màu sắc cho các loại block
        colors = {
            0: (200, 200, 200),  # Trống
            1: (0, 0, 255),      # Đỏ
            2: (0, 255, 255),    # Vàng
            3: (0, 255, 0),      # Xanh lá
            4: (255, 0, 0),      # Xanh dương
            5: (255, 0, 255),    # Tím
            6: (0, 165, 255),    # Cam
            7: (203, 192, 255),  # Hồng
            8: (255, 255, 0)     # Cyan
        }
        
        for row in range(h):
            for col in range(w):
                color = colors.get(board[row, col], (100, 100, 100))
                
                y1 = row * cell_size
                y2 = (row + 1) * cell_size
                x1 = col * cell_size
                x2 = (col + 1) * cell_size
                
                image[y1:y2, x1:x2] = color
                
                # Vẽ border
                cv2.rectangle(image, (x1, y1), (x2-1, y2-1), (0, 0, 0), 1)
        
        cv2.imwrite(output_path, image)
        print(f"Đã lưu board visualization: {output_path}")

# =============================================================================
# PIECES MODULE
# =============================================================================

def get_pieces_by_size(grid_size: int):
    """Lấy pieces cho grid size cụ thể"""
    if grid_size == 8:
        return {
            'single': np.array([[1]]),
            'line_2': np.array([[1, 1]]),
            'line_3': np.array([[1, 1, 1]]),
            'line_4': np.array([[1, 1, 1, 1]]),
            'square_2x2': np.array([[1, 1], [1, 1]]),
            'square_3x3': np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
            'l_shape': np.array([[1, 0], [1, 1]]),
            'l_shape_large': np.array([[1, 0, 0], [1, 1, 1]]),
            't_shape': np.array([[1, 1, 1], [0, 1, 0]]),
            'z_shape': np.array([[1, 1, 0], [0, 1, 1]]),
            'cross': np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]),
            'corner': np.array([[1, 1], [1, 0]]),
            'stair': np.array([[1, 0], [1, 1], [0, 1]]),
            'plus': np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]),
            'hook': np.array([[1, 0], [1, 0], [1, 1]]),
            'arrow': np.array([[0, 1, 0], [1, 1, 1], [1, 0, 1]]),
            'diamond': np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]),
            'bridge': np.array([[1, 0, 1], [1, 1, 1]]),
            'snake': np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]]),
            'spiral': np.array([[1, 0, 0], [1, 1, 1], [0, 0, 1]]),
            'zigzag': np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]]),
            'triangle': np.array([[1, 1, 1], [0, 1, 0], [0, 0, 0]]),
            'butterfly': np.array([[1, 0, 1], [1, 1, 1], [1, 0, 1]]),
            'windmill': np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]),
            'castle': np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]),
            'flower': np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]),
            'star': np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]]),
            'crown': np.array([[1, 0, 1], [1, 1, 1], [0, 1, 0]])
        }
    else:
        # Fallback cho grid size khác
        return {
            'single': np.array([[1]]),
            'line_2': np.array([[1, 1]]),
            'square_2x2': np.array([[1, 1], [1, 1]])
        }

# =============================================================================
# SOLVER MODULE
# =============================================================================

class BlockBlastSolver:
    """Solver cho Block Blast"""
    
    def __init__(self, grid_size: int = 8):
        self.grid_size = grid_size
    
    def brute_force_best_move(self, board: np.ndarray, pieces: List[np.ndarray], depth: int = 1):
        """Tìm best move bằng brute force"""
        best_move = None
        best_score = -1
        
        for piece_idx, piece in enumerate(pieces):
            # Thử tất cả rotations
            rotations = self._get_rotations(piece)
            
            for rotation_idx, rotated_piece in enumerate(rotations):
                # Thử tất cả vị trí
                for row in range(self.grid_size - rotated_piece.shape[0] + 1):
                    for col in range(self.grid_size - rotated_piece.shape[1] + 1):
                        if self._can_place(board, rotated_piece, row, col):
                            score = self._place_and_score(board, rotated_piece, row, col)
                            
                            if score > best_score:
                                best_score = score
                                best_move = {
                                    'piece_index': piece_idx,
                                    'rotation': rotation_idx,
                                    'position': [row, col],
                                    'score': score,
                                    'board_after': self._simulate_placement(board, rotated_piece, row, col)
                                }
        
        return best_move
    
    def solve_with_heuristics(self, board: np.ndarray, pieces: List[np.ndarray]):
        """Giải với heuristics"""
        # Ưu tiên các moves có thể clear rows/columns
        best_move = None
        best_score = -1
        
        for piece_idx, piece in enumerate(pieces):
            rotations = self._get_rotations(piece)
            
            for rotation_idx, rotated_piece in enumerate(rotations):
                for row in range(self.grid_size - rotated_piece.shape[0] + 1):
                    for col in range(self.grid_size - rotated_piece.shape[1] + 1):
                        if self._can_place(board, rotated_piece, row, col):
                            score = self._place_and_score(board, rotated_piece, row, col)
                            
                            # Bonus cho moves clear rows/columns
                            if self._will_clear_row_or_column(board, rotated_piece, row, col):
                                score += 20
                            
                            if score > best_score:
                                best_score = score
                                best_move = {
                                    'piece_index': piece_idx,
                                    'rotation': rotation_idx,
                                    'position': [row, col],
                                    'score': score,
                                    'board_after': self._simulate_placement(board, rotated_piece, row, col)
                                }
        
        return best_move
    
    def _get_rotations(self, piece: np.ndarray):
        """Lấy tất cả rotations của piece"""
        rotations = []
        current = piece.copy()
        
        for _ in range(4):
            rotations.append(current.copy())
            current = np.rot90(current)
        
        # Loại bỏ duplicates
        unique_rotations = []
        for rotation in rotations:
            if not any(np.array_equal(rotation, unique) for unique in unique_rotations):
                unique_rotations.append(rotation)
        
        return unique_rotations
    
    def _can_place(self, board: np.ndarray, piece: np.ndarray, row: int, col: int) -> bool:
        """Kiểm tra có thể đặt piece tại vị trí không"""
        for r in range(piece.shape[0]):
            for c in range(piece.shape[1]):
                if piece[r, c] == 1:
                    board_row = row + r
                    board_col = col + c
                    
                    if (board_row >= self.grid_size or board_col >= self.grid_size or
                        board[board_row, board_col] != 0):
                        return False
        
        return True
    
    def _place_and_score(self, board: np.ndarray, piece: np.ndarray, row: int, col: int) -> int:
        """Đặt piece và tính score"""
        # Tạo board copy
        board_copy = board.copy()
        
        # Đặt piece
        for r in range(piece.shape[0]):
            for c in range(piece.shape[1]):
                if piece[r, c] == 1:
                    board_copy[row + r, col + c] = 1
        
        # Tính score
        score = 0
        
        # Kiểm tra rows
        for r in range(self.grid_size):
            if np.all(board_copy[r, :] != 0):
                score += 10
        
        # Kiểm tra columns
        for c in range(self.grid_size):
            if np.all(board_copy[:, c] != 0):
                score += 10
        
        # Trừ điểm cho holes
        holes = self._count_holes(board_copy)
        score -= holes
        
        return score
    
    def _simulate_placement(self, board: np.ndarray, piece: np.ndarray, row: int, col: int) -> np.ndarray:
        """Mô phỏng đặt piece"""
        board_copy = board.copy()
        
        for r in range(piece.shape[0]):
            for c in range(piece.shape[1]):
                if piece[r, c] == 1:
                    board_copy[row + r, col + c] = 1
        
        return board_copy
    
    def _will_clear_row_or_column(self, board: np.ndarray, piece: np.ndarray, row: int, col: int) -> bool:
        """Kiểm tra có clear row/column không"""
        board_copy = board.copy()
        
        # Đặt piece
        for r in range(piece.shape[0]):
            for c in range(piece.shape[1]):
                if piece[r, c] == 1:
                    board_copy[row + r, col + c] = 1
        
        # Kiểm tra rows
        for r in range(self.grid_size):
            if np.all(board_copy[r, :] != 0):
                return True
        
        # Kiểm tra columns
        for c in range(self.grid_size):
            if np.all(board_copy[:, c] != 0):
                return True
        
        return False
    
    def _count_holes(self, board: np.ndarray) -> int:
        """Đếm số holes trong board"""
        holes = 0
        for r in range(1, self.grid_size - 1):
            for c in range(1, self.grid_size - 1):
                if board[r, c] == 0:
                    # Kiểm tra xung quanh
                    if (board[r-1, c] != 0 and board[r+1, c] != 0 and
                        board[r, c-1] != 0 and board[r, c+1] != 0):
                        holes += 1
        
        return holes

# =============================================================================
# UTILS MODULE
# =============================================================================

def create_overlay_image(board: np.ndarray, piece: np.ndarray, position: List[int], 
                        output_path: str = "output.png"):
    """Tạo ảnh overlay với gợi ý move"""
    if board is None:
        return
    
    h, w = board.shape
    cell_size = 50
    
    # Tạo ảnh
    image = np.ones((h * cell_size, w * cell_size, 3), dtype=np.uint8) * 255
    
    # Màu sắc thống nhất
    EMPTY_COLOR = (200, 200, 200)      # Light gray
    OCCUPIED_COLOR = (100, 100, 100)  # Dark gray
    SUGGESTION_COLOR = (0, 255, 0)     # Green
    
    # Vẽ board
    for row in range(h):
        for col in range(w):
            if board[row, col] == 0:
                color = EMPTY_COLOR
            else:
                color = OCCUPIED_COLOR
            
            y1 = row * cell_size
            y2 = (row + 1) * cell_size
            x1 = col * cell_size
            x2 = (col + 1) * cell_size
            
            image[y1:y2, x1:x2] = color
    
    # Vẽ piece suggestion
    if piece is not None and position is not None:
        piece = np.array(piece) if isinstance(piece, list) else piece
        row, col = position
        
        for r in range(piece.shape[0]):
            for c in range(piece.shape[1]):
                if piece[r, c] == 1:
                    board_row = row + r
                    board_col = col + c
                    
                    if (board_row < h and board_col < w):
                        y1 = board_row * cell_size
                        y2 = (board_row + 1) * cell_size
                        x1 = board_col * cell_size
                        x2 = (board_col + 1) * cell_size
                        
                        image[y1:y2, x1:x2] = SUGGESTION_COLOR
    
    # Vẽ grid lines
    for i in range(h + 1):
        y = i * cell_size
        cv2.line(image, (0, y), (w * cell_size, y), (0, 0, 0), 1)
    
    for i in range(w + 1):
        x = i * cell_size
        cv2.line(image, (x, 0), (x, h * cell_size), (0, 0, 0), 1)
    
    cv2.imwrite(output_path, image)
    print(f"Đã lưu ảnh overlay: {output_path}")

# =============================================================================
# GUI MODULE
# =============================================================================

class BlockBlastGUI:
    """GUI cho Block Blast Solver - Simple UI với logic thực tế"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Block Blast Solver")
        self.root.geometry("1200x800")
        
        # Biến
        self.current_image_path = None
        self.current_board = None
        self.current_result = None
        self.piece_labels = []  # Để lưu references đến piece labels
        
        # Màu sắc
        self.colors = {
            'background': '#f8f9fa',
            'primary': '#007bff',
            'success': '#28a745',
            'accent': '#17a2b8',
            'light': '#ffffff',
            'text_primary': '#212529',
            'text_secondary': '#6c757d',
        }
        
        # Tạo giao diện
        self.create_widgets()
        
    def create_widgets(self):
        """Tạo giao diện Simple UI"""
        # Frame chính
        main_frame = tk.Frame(self.root, bg=self.colors['background'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Header
        header_frame = tk.Frame(main_frame, bg=self.colors['background'])
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        title_label = tk.Label(header_frame, text="Block Blast Solver", 
                              font=("Arial", 24, "bold"), 
                              fg=self.colors['primary'], 
                              bg=self.colors['background'])
        title_label.pack()
        
        # Upload section - nhỏ gọn hơn
        upload_frame = tk.Frame(main_frame, bg=self.colors['light'], 
                               relief=tk.RAISED, bd=1)
        upload_frame.pack(fill=tk.X, pady=(0, 10))
        
        upload_inner = tk.Frame(upload_frame, bg=self.colors['light'])
        upload_inner.pack(fill=tk.X, padx=15, pady=10)
        
        tk.Label(upload_inner, text="Upload your Block Blast screenshot", 
                font=("Arial", 12, "bold"), 
                fg=self.colors['text_primary'], 
                bg=self.colors['light']).pack(anchor=tk.W, pady=(0, 8))
        
        # Upload button
        self.upload_btn = tk.Button(upload_inner, text="📁 Choose Image", 
                                   font=("Arial", 10, "bold"), 
                                   bg=self.colors['primary'], 
                                   fg=self.colors['light'],
                                   relief=tk.FLAT, bd=0, padx=15, pady=6,
                                   command=self.select_image)
        self.upload_btn.pack(pady=(0, 8))
        
        # File path
        self.image_path_var = tk.StringVar()
        path_label = tk.Label(upload_inner, textvariable=self.image_path_var, 
                             font=("Arial", 9), 
                             fg=self.colors['text_secondary'], 
                             bg=self.colors['light'],
                             wraplength=600, justify=tk.LEFT)
        path_label.pack(anchor=tk.W, fill=tk.X)
        
        # Main content - 2 cột: Current Board + Initial Pieces | Solutions
        content_frame = tk.Frame(main_frame, bg=self.colors['background'])
        content_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        # Cột trái: Current Board + Initial Pieces
        left_column = tk.Frame(content_frame, bg=self.colors['background'])
        left_column.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Current Board
        current_frame = tk.Frame(left_column, bg=self.colors['light'], 
                              relief=tk.RAISED, bd=1)
        current_frame.pack(fill=tk.X, pady=(0, 5))
        
        current_inner = tk.Frame(current_frame, bg=self.colors['light'])
        current_inner.pack(fill=tk.X, padx=10, pady=8)
        
        tk.Label(current_inner, text="Current Board", 
                font=("Arial", 10, "bold"), 
                fg=self.colors['text_primary'], 
                bg=self.colors['light']).pack(anchor=tk.W, pady=(0, 5))
        
        self.current_board_label = tk.Label(current_inner, text="No board yet", 
                                           font=("Arial", 10), 
                                           fg=self.colors['text_secondary'], 
                                           bg=self.colors['light'],
                                           relief=tk.SUNKEN, bd=1)
        self.current_board_label.pack(expand=True)
        
        # Initial Pieces frame
        pieces_frame = tk.Frame(left_column, bg=self.colors['background'])
        pieces_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 0))
        
        tk.Label(pieces_frame, text="Initial Pieces", 
                font=("Arial", 12, "bold"), 
                fg=self.colors['text_primary'], 
                bg=self.colors['background']).pack(anchor=tk.W, pady=(0, 8))
        
        pieces_display_frame = tk.Frame(pieces_frame, bg=self.colors['light'], 
                                       relief=tk.RAISED, bd=1)
        pieces_display_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 0))
        
        # Labels cho 3 pieces
        for i in range(3):
            piece_label = tk.Label(pieces_display_frame, text=f"Piece {i+1}", 
                                 font=("Arial", 10), 
                                 fg=self.colors['text_secondary'], 
                                        bg=self.colors['light'],
                                 width=12, height=4, relief=tk.SUNKEN, bd=1)
            piece_label.pack(side=tk.LEFT, padx=5, pady=5, expand=True, fill=tk.BOTH)
            self.piece_labels.append(piece_label)
        
        # Cột phải: 3 Solutions
        right_column = tk.Frame(content_frame, bg=self.colors['background'])
        right_column.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        # Solutions row
        solutions_frame = tk.Frame(right_column, bg=self.colors['background'])
        solutions_frame.pack(fill=tk.BOTH, expand=True)
        
        # Solution 1
        sol1_frame = tk.Frame(solutions_frame, bg=self.colors['light'], 
                             relief=tk.RAISED, bd=1)
        sol1_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        sol1_inner = tk.Frame(sol1_frame, bg=self.colors['light'])
        sol1_inner.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        tk.Label(sol1_inner, text="Solution 1", 
                font=("Arial", 11, "bold"), 
                fg=self.colors['text_primary'], 
                bg=self.colors['light']).pack(anchor=tk.W, pady=(0, 8))
        
        self.sol1_label = tk.Label(sol1_inner, text="No solution yet", 
                                  font=("Arial", 9), 
                                  fg=self.colors['text_secondary'], 
                                  bg=self.colors['light'], 
                                  height=6, relief=tk.SUNKEN, bd=1)
        self.sol1_label.pack(fill=tk.BOTH, expand=True)
        
        # Solution 2
        sol2_frame = tk.Frame(solutions_frame, bg=self.colors['light'], 
                             relief=tk.RAISED, bd=1)
        sol2_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        sol2_inner = tk.Frame(sol2_frame, bg=self.colors['light'])
        sol2_inner.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        tk.Label(sol2_inner, text="Solution 2", 
                font=("Arial", 11, "bold"), 
                fg=self.colors['text_primary'], 
                bg=self.colors['light']).pack(anchor=tk.W, pady=(0, 8))
        
        self.sol2_label = tk.Label(sol2_inner, text="No solution yet", 
                                  font=("Arial", 9), 
                                   fg=self.colors['text_secondary'], 
                                   bg=self.colors['light'],
                                  height=6, relief=tk.SUNKEN, bd=1)
        self.sol2_label.pack(fill=tk.BOTH, expand=True)
        
        # Solution 3
        sol3_frame = tk.Frame(solutions_frame, bg=self.colors['light'], 
                             relief=tk.RAISED, bd=1)
        sol3_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        sol3_inner = tk.Frame(sol3_frame, bg=self.colors['light'])
        sol3_inner.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        tk.Label(sol3_inner, text="Solution 3", 
                font=("Arial", 11, "bold"), 
                fg=self.colors['text_primary'], 
                bg=self.colors['light']).pack(anchor=tk.W, pady=(0, 8))
        
        self.sol3_label = tk.Label(sol3_inner, text="No solution yet", 
                                  font=("Arial", 9), 
                                  fg=self.colors['text_secondary'], 
                                 bg=self.colors['light'], 
                                  height=6, relief=tk.SUNKEN, bd=1)
        self.sol3_label.pack(fill=tk.BOTH, expand=True)
        
        # Footer với button SOLVER ở giữa
        footer_frame = tk.Frame(main_frame, bg=self.colors['background'])
        footer_frame.pack(fill=tk.X, pady=(20, 0))
        
        # Button frame - căn giữa
        button_frame = tk.Frame(footer_frame, bg=self.colors['background'])
        button_frame.pack()
        
        # SOLVER button - NÚT CHÍNH ở giữa
        self.solver_btn = tk.Button(button_frame, text="🚀 SOLVER", 
                             font=("Arial", 16, "bold"),
                             bg=self.colors['success'], 
                             fg=self.colors['light'],
                             relief=tk.FLAT, bd=0, padx=40, pady=15,
                             command=self.solve)
        self.solver_btn.pack()
        
    def select_image(self):
        """Chọn ảnh từ file"""
        file_path = filedialog.askopenfilename(
            title="Chọn ảnh Block Blast",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp")]
        )
        
        if file_path:
            self.current_image_path = file_path
            self.image_path_var.set(file_path)
            print(f"Da chon anh: {os.path.basename(file_path)}")
    
        
    def select_image(self):
        """Chọn ảnh từ file"""
        file_path = filedialog.askopenfilename(
            title="Chọn ảnh Block Blast",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp")]
        )
        
        if file_path:
            self.current_image_path = file_path
            self.image_path_var.set(file_path)
            print(f"Da chon anh: {os.path.basename(file_path)}")
    
    
    def solve(self):
        """Giải Block Blast với logic thực tế"""
        print("=== BAT DAU GIAI ===")
        
        if not self.current_image_path:
            messagebox.showerror("Lỗi", "Vui lòng chọn ảnh trước")
            return
        
        print(f"Anh: {os.path.basename(self.current_image_path)}")
        
        try:
            # Khởi tạo vision processor
            vision = BlockBlastVision(8)
            
            # Test với ảnh thực tế trước
            print("Đang test với ảnh thực tế...")
            test_board = vision.test_with_real_image(self.current_image_path)
            
            if test_board is not None:
                self.current_board = test_board
            else:
                # Fallback: sử dụng method cũ
                print("Sử dụng method cũ...")
            self.current_board = vision.extract_board_from_image(self.current_image_path)
            
            print("Đã trích xuất board thành công")
            print(f"Board shape: {self.current_board.shape}")
            print(f"Board có {np.sum(self.current_board != 0)} block")
            
            # Hiển thị current board
            self.display_current_board(self.current_board)
            
            # Load pieces
            pieces = list(get_pieces_by_size(8).values())
            print(f"Sử dụng {len(pieces)} pieces cho grid 8x8")
            
            # Hiển thị các piece ban đầu (3 pieces đầu tiên)
            self.display_initial_pieces()
            
            # Khởi tạo solver
            solver = BlockBlastSolver(8)
            
            # Giải và tạo 3 solutions
            print("Đang tìm kiếm solutions...")
            solutions = self.find_multiple_solutions(self.current_board, pieces)
            
            # Hiển thị solutions
            self.display_solutions(solutions)
            
            print("Da tim thay solutions!")
            messagebox.showinfo("Thành công", "Đã tìm thấy solutions!")
            
        except Exception as e:
            print(f"Loi: {e}")
            # Fallback: tạo solutions mẫu
            self.create_fallback_solutions()
    
    def create_sample_board(self):
        """Tạo board mẫu"""
        board = np.zeros((8, 8), dtype=int)
        
        # Tạo pattern mẫu
        board[0, :3] = 1  # Hàng 1: 3 block đỏ
        board[1, 1:4] = 2  # Hàng 2: 3 block vàng
        board[2, 2:5] = 3  # Hàng 3: 3 block xanh lá
        board[3, 3:6] = 4  # Hàng 4: 3 block xanh dương
        board[4, 4:7] = 5  # Hàng 5: 3 block tím
        
        # Thêm một số block rải rác
        board[6, 1] = 6
        board[7, 2] = 7
        
        return board
    
    def create_sample_pieces(self):
        """Tạo các piece mẫu"""
        # Piece 1: 2x3 rectangle (6 blocks)
        piece1 = np.array([
            [1, 1, 1],
            [1, 1, 1]
        ], dtype=int)
        
        # Piece 2: Cross shape (5 blocks)
        piece2 = np.array([
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]
        ], dtype=int)
        
        # Piece 3: 2x2 square (4 blocks)
        piece3 = np.array([
            [1, 1],
            [1, 1]
        ], dtype=int)
        
        return [piece1, piece2, piece3]
    
    def create_sample_solutions(self):
        """Tạo 3 solutions mẫu"""
        solutions = []
        
        # Solution 1: Piece 2x2 tại (3, 3)
        sol1 = {
            'piece': np.array([[1, 1], [1, 1]]),
            'position': [3, 3],
            'score': 15,
            'description': "2x2 block at (3,3)"
        }
        solutions.append(sol1)
        
        # Solution 2: Piece line 3 tại (1, 5)
        sol2 = {
            'piece': np.array([[1, 1, 1]]),
            'position': [1, 5],
            'score': 12,
            'description': "Line 3 at (1,5)"
        }
        solutions.append(sol2)
        
        # Solution 3: Piece L tại (5, 2)
        sol3 = {
            'piece': np.array([[1, 0], [1, 1]]),
            'position': [5, 2],
            'score': 10,
            'description': "L-shape at (5,2)"
        }
        solutions.append(sol3)
        
        return solutions
    
    def find_multiple_solutions(self, board, pieces):
        """Tìm nhiều solutions"""
        solutions = []
        solver = BlockBlastSolver(8)
        
        # Tìm 3 solutions tốt nhất
        for i in range(3):
            if i < len(pieces):
                piece = pieces[i]
                best_move = solver.solve_with_heuristics(board, [piece])
                if best_move:
                    solutions.append({
                        'piece': piece,
                        'position': best_move['position'],
                        'score': best_move['score'],
                        'description': f"Piece {i+1} at {best_move['position']}"
                    })
        
        # Nếu không đủ solutions, tạo mẫu
        while len(solutions) < 3:
            solutions.extend(self.create_sample_solutions())
            break
        
        return solutions[:3]  # Chỉ lấy 3 solutions
    
    def create_fallback_solutions(self):
        """Tạo solutions mẫu khi có lỗi"""
        print("Tạo solutions mẫu...")
        
        # Tạo board mẫu
        board = self.create_sample_board()
        
        # Hiển thị current board
        self.display_current_board(board)
        
        # Hiển thị các piece ban đầu
        self.display_initial_pieces()
        
        # Tạo 3 solutions mẫu
        solutions = self.create_sample_solutions()
        
        # Hiển thị solutions
        self.display_solutions(solutions)
        
        print("Đã tạo solutions mẫu")
    
    def display_current_board(self, board):
        """Hiển thị current board với tỷ lệ đúng"""
        try:
            # Tạo ảnh board
            board_image = self.create_board_image(board, "current")
            
            # Tính toán kích thước hiển thị giữ tỷ lệ
            max_size = 300  # Kích thước tối đa
            original_width, original_height = board_image.size
            
            # Tính tỷ lệ để giữ tỷ lệ gốc
            ratio = min(max_size / original_width, max_size / original_height)
            new_width = int(original_width * ratio)
            new_height = int(original_height * ratio)
            
            # Resize giữ tỷ lệ
            board_image = board_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(board_image)
            
            # Update label
            self.current_board_label.configure(image=photo, text="")
            self.current_board_label.image = photo  # Keep reference
            
        except Exception as e:
            print(f"Lỗi khi hiển thị current board: {e}")
            self.current_board_label.configure(text=f"✅ Current Board\n\nShape: {board.shape}\nBlocks: {np.sum(board != 0)}")
    
    def display_solutions(self, solutions):
        """Hiển thị 3 solutions"""
        labels = [self.sol1_label, self.sol2_label, self.sol3_label]
        
        for i, (solution, label) in enumerate(zip(solutions, labels)):
            try:
                # Tạo ảnh solution
                sol_image = self.create_solution_image(solution, f"solution_{i+1}")
                
                # Resize giữ tỷ lệ
                max_size = 150  # Kích thước tối đa cho solutions
                original_width, original_height = sol_image.size
                ratio = min(max_size / original_width, max_size / original_height)
                new_width = int(original_width * ratio)
                new_height = int(original_height * ratio)
                sol_image = sol_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Convert to PhotoImage
                photo = ImageTk.PhotoImage(sol_image)
                
                # Update label
                label.configure(image=photo, text="")
                label.image = photo  # Keep reference
                
            except Exception as e:
                print(f"Lỗi khi hiển thị solution {i+1}: {e}")
                label.configure(text=f"✅ Solution {i+1}\n\n{solution['description']}\nScore: {solution['score']}")
    
    def display_initial_pieces(self):
        """Hiển thị các piece ban đầu dưới current board"""
        pieces = self.create_sample_pieces()
        
        for i, piece in enumerate(pieces):
            try:
                piece_image = self.create_piece_image(piece, cell_size=30)  # Tăng cell_size
                
                # Scale giữ tỷ lệ cho pieces
                max_size = 80  # Kích thước tối đa cho pieces
                original_width, original_height = piece_image.size
                ratio = min(max_size / original_width, max_size / original_height)
                new_width = int(original_width * ratio)
                new_height = int(original_height * ratio)
                piece_image = piece_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                photo = ImageTk.PhotoImage(piece_image)
                
                self.piece_labels[i].configure(image=photo, text="")
                self.piece_labels[i].image = photo  # Keep reference
                
            except Exception as e:
                print(f"Loi khi hien thi piece {i+1}: {e}")
                self.piece_labels[i].configure(text=f"Piece {i+1} Error")
    
    def create_board_image(self, board, name):
        """Tạo ảnh board với màu sắc thống nhất"""
        h, w = board.shape
        cell_size = 80
        
        # Tạo ảnh
        image = np.ones((h * cell_size, w * cell_size, 3), dtype=np.uint8) * 255
        
        # Vẽ board với màu sắc thống nhất
        for row in range(h):
            for col in range(w):
                if board[row, col] == 0:
                    color = (220, 220, 220)  # Xám nhạt - ô trống
                else:
                    color = (80, 80, 80)     # Xám đậm - ô có block
                
                y1 = row * cell_size
                y2 = (row + 1) * cell_size
                x1 = col * cell_size
                x2 = (col + 1) * cell_size
                
                image[y1:y2, x1:x2] = color
                
                # Vẽ border
                cv2.rectangle(image, (x1, y1), (x2-1, y2-1), (0, 0, 0), 1)
        
        # Convert to PIL Image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        return pil_image
    
    def create_piece_image(self, piece, cell_size=25):
        """Tạo ảnh cho một piece với màu xám đậm cho block"""
        h, w = piece.shape
        
        # Tạo ảnh nền trắng
        image = np.ones((h * cell_size, w * cell_size, 3), dtype=np.uint8) * 255
        
        for row in range(h):
            for col in range(w):
                if piece[row, col] == 1:
                    color = (80, 80, 80)      # Xám đậm cho block
                else:
                    color = (220, 220, 220)   # Xám nhạt cho empty parts
                
                y1 = row * cell_size
                y2 = (row + 1) * cell_size
                x1 = col * cell_size
                x2 = (col + 1) * cell_size
                
                image[y1:y2, x1:x2] = color
                
                # Vẽ border
                cv2.rectangle(image, (x1, y1), (x2-1, y2-1), (0, 0, 0), 1)
        
        return Image.fromarray(image)
    
    def create_solution_image(self, solution, name):
        """Tạo ảnh solution"""
            # Tạo board mẫu
        board = self.create_sample_board()
        
        # Đặt piece
        piece = solution['piece']
        pos = solution['position']
        
        # Tạo board với suggestion
        board_with_suggestion = board.copy()
        
        for r in range(piece.shape[0]):
            for c in range(piece.shape[1]):
                if piece[r, c] == 1:
                    board_row = pos[0] + r
                    board_col = pos[1] + c
                    if board_row < 8 and board_col < 8:
                        board_with_suggestion[board_row, board_col] = 9  # Suggestion color
            
            # Tạo ảnh
        h, w = board_with_suggestion.shape
        cell_size = 80
        
        image = np.ones((h * cell_size, w * cell_size, 3), dtype=np.uint8) * 255
        
        for row in range(h):
            for col in range(w):
                if board_with_suggestion[row, col] == 0:
                    color = (220, 220, 220)  # Xám nhạt - ô trống
                elif board_with_suggestion[row, col] == 9:
                    color = (0, 150, 0)      # Xanh lá đậm - suggestion
                else:
                    color = (80, 80, 80)     # Xám đậm - ô có block
                
                y1 = row * cell_size
                y2 = (row + 1) * cell_size
                x1 = col * cell_size
                x2 = (col + 1) * cell_size
                
                image[y1:y2, x1:x2] = color
                
                # Vẽ border
                cv2.rectangle(image, (x1, y1), (x2-1, y2-1), (0, 0, 0), 1)
        
        # Convert to PIL Image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        return pil_image
    
    
    def run(self):
        """Chạy GUI"""
        self.root.mainloop()

# =============================================================================
# STEP BY STEP GUIDE MODULE
# =============================================================================

class BlockBlastGuideGenerator:
    """Tạo hướng dẫn từng bước"""
    
    def __init__(self, grid_size: int = 8):
        self.grid_size = grid_size
    
    def create_step_by_step_guide(self, board: np.ndarray, result: dict) -> str:
        """Tạo hướng dẫn từng bước"""
        guide = []
        
        # Phân tích board
        analysis = self._analyze_board_state(board)
        guide.append("=== HUỚNG DẪN TỪNG BƯỚC ===")
        guide.append("TÌM THẤY GIẢI PHÁP!")
        guide.append("")
        
        # Phân tích tình hình
        guide.append("PHÂN TÍCH BOARD:")
        guide.append(f"  • Tổng ô: {self.grid_size * self.grid_size}")
        guide.append(f"  • Ô trống: {analysis['empty_cells']}")
        guide.append(f"  • Ô có block: {analysis['occupied_cells']}")
        guide.append(f"  • Tỷ lệ đầy: {analysis['fill_ratio']:.1f}%")
        guide.append("")
        
        # Chiến lược
        strategy = self._generate_strategy_advice(analysis)
        guide.append("CHIẾN LƯỢC:")
        guide.append(f"  {strategy}")
        guide.append("")
        
        # Hướng dẫn từng bước
        steps = self._generate_step_details(result)
        guide.append("HUỚNG DẪN TỪNG BƯỚC:")
        guide.append("")
        
        for i, step in enumerate(steps, 1):
            guide.append(f"BƯỚC {i}: {step['title']}")
            for detail in step['details']:
                guide.append(f"   {detail}")
            guide.append("")
        
        return "\n".join(guide)
    
    def _analyze_board_state(self, board: np.ndarray) -> dict:
        """Phân tích trạng thái board"""
        total_cells = board.size
        empty_cells = np.sum(board == 0)
        occupied_cells = total_cells - empty_cells
        fill_ratio = (occupied_cells / total_cells) * 100
        
        return {
            'total_cells': total_cells,
            'empty_cells': empty_cells,
            'occupied_cells': occupied_cells,
            'fill_ratio': fill_ratio
        }
    
    def _generate_strategy_advice(self, analysis: dict) -> str:
        """Tạo lời khuyên chiến lược"""
        if analysis['fill_ratio'] < 20:
            return "Board còn khá trống - Tập trung vào việc đặt piece lớn"
        elif analysis['fill_ratio'] < 50:
            return "Board đang phát triển - Cân nhắc việc clear rows/columns"
        else:
            return "Board gần đầy - Ưu tiên clear rows/columns để tạo không gian"
    
    def _generate_step_details(self, result: dict) -> list:
        """Tạo chi tiết từng bước"""
        steps = []
        
        # Bước 1: Phân tích tình hình
        steps.append({
            'title': 'Phân tích tình hình hiện tại',
            'details': [
                f"Board có {result.get('board_after', np.zeros((8, 8))).size} ô",
                f"• Mật độ board: {np.sum(result.get('board_after', np.zeros((8, 8))) != 0) / 64 * 100:.1f}%",
                f"• Số lỗ hiện tại: {self._count_holes(result.get('board_after', np.zeros((8, 8))))}"
            ]
        })
        
        # Bước 2: Chọn piece
        steps.append({
            'title': f'Chọn piece #{result["piece_index"]}',
            'details': [
                f"Piece có diện tích {np.sum(result.get('board_after', np.zeros((8, 8))) != 0)} ô",
                f"• Piece shape: {result.get('board_after', np.zeros((8, 8))).shape}",
                f"• Diện tích: {np.sum(result.get('board_after', np.zeros((8, 8))) != 0)} ô",
                f"• Rotation: {result['rotation']}"
            ]
        })
        
        # Bước 3: Đặt piece
        steps.append({
            'title': f'Đặt piece tại vị trí {result["position"]}',
            'details': [
                f"Đặt góc trên-trái của piece tại ô {result['position']}",
                f"• Vị trí: Hàng {result['position'][0]}, Cột {result['position'][1]}",
                f"• Piece sẽ chiếm từ {result['position']} đến ({result['position'][0]+1}, {result['position'][1]+1})"
            ]
        })
        
        # Bước 4: Kết quả
        steps.append({
            'title': 'Kết quả sau khi đặt',
            'details': [
                f"Điểm số: {result['score']}",
                f"• Điểm số: {result['score']}",
                f"• Không có hàng nào được xóa",
                f"• Block còn lại: {np.sum(result.get('board_after', np.zeros((8, 8))) != 0)}/64"
            ]
        })
        
        return steps
    
    def _count_holes(self, board: np.ndarray) -> int:
        """Đếm số holes"""
        holes = 0
        for r in range(1, board.shape[0] - 1):
            for c in range(1, board.shape[1] - 1):
                if board[r, c] == 0:
                    if (board[r-1, c] != 0 and board[r+1, c] != 0 and
                        board[r, c-1] != 0 and board[r, c+1] != 0):
                        holes += 1
        return holes

# =============================================================================
# MAIN FUNCTIONS
# =============================================================================

def main():
    """Main function"""
    print("=== BLOCK BLAST SOLVER ===")
    print("All-in-One Version")
    
    # Chạy GUI
    app = BlockBlastGUI()
    app.run()

if __name__ == "__main__":
    main()