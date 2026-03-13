# Tiny Transformer - Golden Model cho RISC-V SoC Accelerator

Repository này chứa **Golden Model** (mô hình tham chiếu bằng phần mềm) được viết bằng PyTorch cho bộ gia tốc **Tiny Transformer**. Đây là một phần trong dự án thiết kế System-on-Chip (SoC) tích hợp lõi CPU RISC-V cùng một bộ gia tốc AI chuyên dụng. 

Mục đích chính của Golden Model này là cung cấp tính toán chuẩn xác (để verify RTL) và làm cơ sở ánh xạ các phép toán từ mô hình Deep Learning xuống các block phần cứng (Hardware Mapping) như Systolic Array, SRAM, và Vector ALU.

---

## Đặc tả kiến trúc (Architecture Specifications)

Mô hình được thiết kế tinh gọn để phù hợp với tài nguyên phần cứng giới hạn trên SoC. Các cấu hình phần cứng cốt lõi (xem chi tiết tại `config.py`):

- **Embedding Dimension:** 64
- **Attention Heads:** 2 (mỗi head 32 chiều)
- **Encoder Depth:** 3 layers
- **Feed-Forward Multiplier:** 2x
- **Max Sequence Length:** 64 tokens
- **Khả năng đa phương thức (Multi-modal):**
  - **Vision:** Xử lý ảnh đầu vào $64 \times 64 \times 3$ bằng cơ chế chia patch ($8 \times 8$).
  - **Audio:** Xử lý các đặc trưng âm thanh (MFCC/Mel) với 40 features.

---

## Ánh xạ phần cứng (Hardware Mapping)

Mã nguồn được chú thích rõ ràng để định hướng cho việc viết mã RTL. Các ánh xạ phần cứng chính bao gồm:

1. **Multi-Head Attention (`layers.py`):**
   - Các ma trận $W_q, W_k, W_v$ được gộp chung để tối ưu băng thông nạp từ **SRAM**.
   - Phép nhân ma trận $Q \times K^T$ và $Attn \times V$ được thiết kế để đưa vào bộ nhân ma trận phần cứng (**Systolic Array**).
   - Hàm `Softmax` được dự kiến hiện thực bằng xấp xỉ đa thức hoặc Bảng tra cứu (**LUT**) thông qua các phép tính xấp xỉ $e^x$.

2. **Feed-Forward Network / MLP (`layers.py`):**
   - Tái sử dụng lại khối tính toán ma trận (MMU/Systolic Array) của block Attention để tiết kiệm diện tích chip.
   - Hàm kích hoạt `ReLU` được hiện thực đơn giản qua việc so sánh dấu.

3. **Layer Normalization (`layers.py`):**
   - Tính toán trung bình và phương sai được ánh xạ xuống **Vector ALU**.
   - Phép chia căn bậc hai (Inverse Square Root) được thiết kế tối ưu hóa bằng thuật toán Fast InvSqrt hoặc dịch bit (bit-shift) để tránh sử dụng các bộ chia phức tạp.

4. **Embeddings (`embeddings.py`):**
   - **PatchEmbed (Vision):** Sử dụng `Conv2d` có thể tính toán trực tiếp trên hệ thống MAC của bộ gia tốc Transformer hoặc một khối Conv2d phụ trợ.
   - **AudioEmbed (Audio):** Sử dụng các bộ MAC cơ bản để ánh xạ các Linear projection.

---

## Cấu trúc thư mục (File Structure)

- `config.py`: Chứa class `TinyConfig` định nghĩa toàn bộ siêu tham số (hyperparameters) và ràng buộc phần cứng.
- `core.py`: Định nghĩa lõi kiến trúc `Encoder` (với Pre-normalization) và toàn bộ mạng `TinyTransformer` (tích hợp Positional Embedding và Global Average Pooling).
- `embeddings.py`: Các block tiền xử lý dữ liệu đầu vào cho tác vụ Hình ảnh (`PatchEmbed`) và Âm thanh (`AudioEmbed`).
- `layers.py`: Định nghĩa các toán tử nền tảng (`MultiHeadAttention`, `Mlp`, `LayerNorm`) kèm theo các chú thích về cách hiện thực dưới phần cứng.
