# Ollama AIoT – Phân loại sức khoẻ từ cảm biến đeo trên thiết bị biên

Dự án chạy các LLM nhỏ bằng [Ollama](https://ollama.com) để đọc dữ liệu cảm biến đeo theo kiểu PAMAP2: IMU ở tay, ngực, cổ chân và nhịp tim. Với mỗi bản ghi, model trả về đúng một dòng tiếng Việt:

```
HEALTHY: Đang đạp xe, nhịp tim 133 bpm phù hợp với cường độ vận động.
WARNING: Đang ngồi nhưng nhịp tim cao bất thường (128 bpm).
```

| Thành phần | Mục đích |
|---|---|
| [model_files/](model_files/) | Cấu hình (Modelfile) của 3 model |
| [benchmark.py](benchmark.py) | Đo CPU, GPU, RAM, thời gian và FLOPs của từng model |
| [demo.py](demo.py) | Mô phỏng theo dõi sức khoẻ một khách hàng theo từng phút |

## Cài đặt

```bash
pip install psutil requests

# Tạo 3 model từ Modelfile. Ollama tự tải model gốc nếu máy chưa có.
ollama create llama_3.2 -f model_files/llama_3.2.Modelfile
ollama create qwen_2.5  -f model_files/qwen_2.5.Modelfile
ollama create phi_mini  -f model_files/phi_mini.Modelfile
```

Ollama phải đang chạy ở `http://localhost:11434`. Sau mỗi lần sửa Modelfile, chạy lại lệnh `ollama create` tương ứng.

## model_files/

Mỗi file `<tên>.Modelfile` tạo ra một model Ollama tên `<tên>`. Đây cũng là tên dùng cho tham số `--models` và `--model` của hai script.

| Modelfile | Model gốc | Số tham số |
|---|---|---|
| `llama_3.2.Modelfile` | `llama3.2:1b` | 1.24B |
| `qwen_2.5.Modelfile` | `qwen2.5:0.5b` | 0.49B |
| `phi_mini.Modelfile` | `phi3:mini` | 3.82B |

Ba file chỉ khác nhau ở dòng `FROM`. Phần còn lại giống hệt nhau để so sánh công bằng.

**Tham số:**

| Tham số | Giá trị | Lý do |
|---|---|---|
| `temperature` | 0.2 | Câu trả lời ổn định. Ở 0.8, model hay lệch định dạng hoặc trả lời bằng tiếng Trung (Qwen). |
| `num_ctx` | 4096 | Đủ chứa system prompt (khoảng 1700–2700 token) cộng input. |
| `num_predict` | 64 | Giới hạn độ dài câu trả lời. Không có giới hạn này, phi3 từng sinh hơn 5000 token cho một prompt. |
| `stop` | `"\n"`, `"Input:"` | Dừng ngay sau dòng đầu tiên, không cho model tự bịa thêm cặp `Input/Output` mới. |

**System prompt** gồm:
- Mô tả 33 cột input. Cột 1 là tên hoạt động, cột 2 là nhịp tim, các cột 3–32 là cảm biến tay, ngực và cổ chân, cột 33 là `PeopleId` (model bỏ qua).
- Định dạng đầu ra và bảng dịch tên hoạt động sang tiếng Việt.
- Khoảng nhịp tim bình thường theo từng hoạt động.
- Các quy tắc báo `WARNING`: nhịp tim ngoài khoảng bình thường, dưới 45 hoặc trên 190 bpm, có dấu hiệu ngã khi đang nghỉ, nhiệt độ bất thường, hoặc thiếu dữ liệu.
- Quy tắc cho phần `| previous status: ...` mà `demo.py` thêm vào cuối input: đây chỉ là ngữ cảnh của phút trước, không được chép lại.
- Một số ví dụ input/output.

## benchmark.py

Chạy các dòng dữ liệu trong `sample_data_benchmark.csv` qua từng model và đo hiệu năng.

```bash
python3 benchmark.py                               # tất cả model trong model_files/
python3 benchmark.py --models qwen_2.5 phi_mini    # chỉ một số model
```

- Số dòng dữ liệu dùng để đo được đặt bằng `N_SAMPLE_BENCHMARK` ở đầu file (mặc định là 5).
- Mỗi dòng CSV được gửi nguyên dạng giá trị thô. Không đổi sang JSON, vì JSON với 33 tên cột dài khiến model chép lại cả input và chạy chậm khoảng 20 lần.
- Trước khi đo, script gửi một prompt khởi động để nạp model vào bộ nhớ. Prompt này không được tính.

**Các chỉ số đo:**

| Chỉ số | Cách đo |
|---|---|
| CPU % | Lấy mẫu mỗi 0.5 giây bằng `psutil`, rồi tính trung bình |
| GPU % | Lấy mẫu mỗi 0.5 giây. Apple Silicon dùng `ioreg` (không cần sudo), NVIDIA dùng `nvidia-smi` |
| RAM % | `psutil.virtual_memory().percent` |
| Avg s/prompt | Thời gian trung bình của mỗi request |
| TFLOPs/prompt | `2 × N × (số token prompt + số token sinh ra)`, với N là số tham số lấy từ `/api/show` |
| Prefill TFLOPS | `2 × N × số token prompt / thời gian xử lý prompt` |
| Decode TFLOPS, token/s | `2 × N × số token sinh ra / thời gian sinh`, và `số token sinh ra / thời gian sinh` |

Kết quả được in ra màn hình (từng prompt và bảng tổng hợp) và lưu vào `benchmark_results.csv`.

**Lưu ý khi đọc kết quả:**
- **CPU, GPU và RAM % là của cả máy, không riêng Ollama.** Nên tắt các ứng dụng khác trước khi đo.
- **Prefill TFLOPS bị thổi phồng.** Ollama giữ lại phần system prompt đã xử lý giữa các request. Vì vậy `prompt_eval_duration` chỉ tính phần input mới, trong khi `prompt_eval_count` vẫn tính cả system prompt. Trên M3 (GPU 10 nhân), script cho ra khoảng 20–30 TFLOPS, trong khi giá trị thật đo được khi phần đã lưu không dùng lại được là khoảng 2.6 TFLOPS. Các chỉ số TFLOPs/prompt, Decode TFLOPS và token/s vẫn đáng tin.
- **Benchmark chỉ đo hiệu năng, không đo độ chính xác của nhãn `HEALTHY`/`WARNING`.**

## demo.py

Mô phỏng theo dõi liên tục một khách hàng. Mỗi dòng dữ liệu được xem là ảnh chụp 1 phút. Mỗi phút, model nhận dữ liệu phút hiện tại cùng status lần gần nhất của khách hàng, rồi in ra status mới.

```bash
python3 demo.py --people-id 2 --model llama_3.2
python3 demo.py --people-id 2 --model qwen_2.5 --data sample_data_benchmark.csv --interval 5
```

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--people-id` | (bắt buộc) | Mã khách hàng, tức cột `PeopleId` |
| `--model` | (bắt buộc) | Tên model, ví dụ `llama_3.2` |
| `--data` | `dataset2.csv` | File dữ liệu. File được đọc từng dòng nên dùng được với file lớn. |
| `--interval` | 60 | Số giây giữa hai lần xử lý. Đặt nhỏ hơn để demo nhanh. |

Prompt gửi đi có dạng:

```
<dòng CSV hiện tại> | previous status: <status lần trước, hoặc "none" ở phút đầu>
```

Mỗi phút in ra một dòng:

```
[16:46:28] Customer 2 | minute 11 | ascending stairs -> HEALTHY: Đang lên cầu thang, nhịp tim 132 bpm bình thường.
```

- Nếu model trả về câu rỗng, script gọi lại tối đa 3 lần. Nếu vẫn rỗng, nó in `UNKNOWN` và giữ nguyên status cũ.
- `dataset2.csv` là dữ liệu cảm biến thô, khoảng 100 dòng mỗi giây. Một dòng của file này không tương ứng với 1 phút, và nhiều dòng thiếu `heart_rate`. Để chạy đúng kịch bản "mỗi dòng là 1 phút", hãy dùng `--data sample_data_benchmark.csv`, hoặc gộp dữ liệu thô thành từng cửa sổ 1 phút trước khi chạy.

## Dữ liệu

| File | Mô tả |
|---|---|
| `dataset2.csv` | Dữ liệu đầy đủ (khoảng 2.86 triệu dòng, khoảng 800MB), 33 cột theo đúng thứ tự trong system prompt |
| `sample_data_benchmark.csv` | 100 dòng lấy ngẫu nhiên từ `dataset2.csv`, cùng định dạng |
| `sample_data.ipynb` | Notebook tạo `sample_data_benchmark.csv` (`df.sample(n=100, random_state=77)`) |
