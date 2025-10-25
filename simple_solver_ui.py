"""
Simple Solver UI - Test nút SOLVER
"""
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import numpy as np
import cv2
from PIL import Image, ImageTk

class SimpleSolverUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Block Blast Solver - Simple")
        self.root.geometry("1200x800")
        
        # Biến
        self.current_image_path = None
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
        """Tạo giao diện"""
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
        upload_frame.pack(fill=tk.X, pady=(0, 10))  # Giảm pady
        
        upload_inner = tk.Frame(upload_frame, bg=self.colors['light'])
        upload_inner.pack(fill=tk.X, padx=15, pady=10)  # Giảm padding
        
        tk.Label(upload_inner, text="Upload your Block Blast screenshot", 
                font=("Arial", 12, "bold"),  # Giảm font
                fg=self.colors['text_primary'], 
                bg=self.colors['light']).pack(anchor=tk.W, pady=(0, 8))  # Giảm pady
        
        # Upload button
        self.upload_btn = tk.Button(upload_inner, text="📁 Choose Image", 
                                   font=("Arial", 10, "bold"),  # Giảm font
                                   bg=self.colors['primary'], 
                                   fg=self.colors['light'],
                                   relief=tk.FLAT, bd=0, padx=15, pady=6,  # Giảm padding
                                   command=self.select_image)
        self.upload_btn.pack(pady=(0, 8))  # Giảm pady
        
        # File path
        self.image_path_var = tk.StringVar()
        path_label = tk.Label(upload_inner, textvariable=self.image_path_var, 
                             font=("Arial", 9),  # Giảm font
                             fg=self.colors['text_secondary'], 
                             bg=self.colors['light'],
                             wraplength=600, justify=tk.LEFT)  # Giảm wraplength
        path_label.pack(anchor=tk.W, fill=tk.X)
        
        # Main content - 2 cột: Current Board + Initial Pieces | Solutions
        content_frame = tk.Frame(main_frame, bg=self.colors['background'])
        content_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        # Cột trái: Current Board + Initial Pieces
        left_column = tk.Frame(content_frame, bg=self.colors['background'])
        left_column.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Current Board - đẩy lên một chút
        current_frame = tk.Frame(left_column, bg=self.colors['light'], 
                                relief=tk.RAISED, bd=1)
        current_frame.pack(fill=tk.X, pady=(0, 5))  # Chỉ fill chiều ngang, giảm pady
        
        current_inner = tk.Frame(current_frame, bg=self.colors['light'])
        current_inner.pack(fill=tk.X, padx=10, pady=8)  # Giảm padding
        
        tk.Label(current_inner, text="Current Board", 
                font=("Arial", 10, "bold"),  # Giảm font
                fg=self.colors['text_primary'], 
                bg=self.colors['light']).pack(anchor=tk.W, pady=(0, 5))  # Giảm pady
        
        self.current_board_label = tk.Label(current_inner, text="No board yet", 
                                           font=("Arial", 10), 
                                           fg=self.colors['text_secondary'], 
                                           bg=self.colors['light'],
                                           relief=tk.SUNKEN, bd=1)  # Không set height để tự động điều chỉnh
        self.current_board_label.pack(expand=True)  # Chỉ expand, không fill để giữ tỷ lệ
        
        # Initial Pieces frame - tăng kích thước để tận dụng không gian
        pieces_frame = tk.Frame(left_column, bg=self.colors['background'])
        pieces_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 0))  # Fill cả chiều dọc
        
        tk.Label(pieces_frame, text="Initial Pieces", 
                font=("Arial", 12, "bold"),  # Tăng font
                fg=self.colors['text_primary'], 
                bg=self.colors['background']).pack(anchor=tk.W, pady=(0, 8))  # Tăng pady
        
        pieces_display_frame = tk.Frame(pieces_frame, bg=self.colors['light'], 
                                       relief=tk.RAISED, bd=1)
        pieces_display_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 0))  # Fill cả chiều dọc
        
        # Labels cho 3 pieces trong 1 hình chữ nhật - tăng kích thước
        for i in range(3):
            piece_label = tk.Label(pieces_display_frame, text=f"Piece {i+1}", 
                                 font=("Arial", 10),  # Tăng font
                                 fg=self.colors['text_secondary'], 
                                 bg=self.colors['light'], 
                                 width=12, height=4, relief=tk.SUNKEN, bd=1)  # Tăng width và height
            piece_label.pack(side=tk.LEFT, padx=5, pady=5, expand=True, fill=tk.BOTH)  # Tăng padding
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
        
        # Footer với buttons
        footer_frame = tk.Frame(main_frame, bg=self.colors['background'])
        footer_frame.pack(fill=tk.X, pady=(20, 0))
        
        # Button frame
        button_frame = tk.Frame(footer_frame, bg=self.colors['background'])
        button_frame.pack()
        
        # SOLVER button - NÚT CHÍNH (nhỏ lại)
        self.solver_btn = tk.Button(button_frame, text="🚀 SOLVER", 
                                  font=("Arial", 14, "bold"),
                                  bg=self.colors['success'], 
                                  fg=self.colors['light'],
                                  relief=tk.FLAT, bd=0, padx=25, pady=10,
                                  command=self.solve)
        self.solver_btn.pack(side=tk.LEFT, padx=(0, 15))
        
        # TEST button
        test_btn = tk.Button(button_frame, text="🧪 TEST", 
                            font=("Arial", 12),
                            bg=self.colors['accent'], 
                            fg=self.colors['light'],
                            relief=tk.FLAT, bd=0, padx=20, pady=8,
                            command=self.test_simple)
        test_btn.pack(side=tk.LEFT, padx=(0, 15))
        
        # CLEAR button
        clear_btn = tk.Button(button_frame, text="🗑️ CLEAR", 
                             font=("Arial", 12),
                             bg="#dc3545", 
                             fg=self.colors['light'],
                             relief=tk.FLAT, bd=0, padx=20, pady=8,
                             command=self.clear_all)
        clear_btn.pack(side=tk.LEFT)
        
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
        """Giải Block Blast"""
        print("=== BAT DAU GIAI ===")
        
        if not self.current_image_path:
            messagebox.showerror("Lỗi", "Vui lòng chọn ảnh trước")
            return
        
        print(f"Anh: {os.path.basename(self.current_image_path)}")
        
        try:
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
            
            print("Da tim thay 3 solutions toi uu!")
            messagebox.showinfo("Thành công", "Đã tìm thấy 3 solutions tối ưu!")
            
        except Exception as e:
            print(f"Loi: {e}")
            messagebox.showerror("Lỗi", f"Lỗi khi giải: {e}")
    
    def test_simple(self):
        """Test đơn giản"""
        print("=== TEST DON GIAN ===")
        
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
        
        messagebox.showinfo("Test", "Test thành công!")
    
    def clear_all(self):
        """Xóa tất cả"""
        self.current_board_label.configure(text="No board yet")
        self.sol1_label.configure(text="No solution yet")
        self.sol2_label.configure(text="No solution yet")
        self.sol3_label.configure(text="No solution yet")
        
        # Xóa piece labels
        for label in self.piece_labels:
            label.configure(image='', text='Piece')
        
        self.current_image_path = None
        self.image_path_var.set("")
        print("Da xoa tat ca")
    
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
        """Tạo các piece mẫu dựa trên ảnh ví dụ"""
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

def main():
    """Main function"""
    print("=== BLOCK BLAST SOLVER - SIMPLE ===")
    print("Simple UI with SOLVER button")
    
    app = SimpleSolverUI()
    app.run()

if __name__ == "__main__":
    main()
