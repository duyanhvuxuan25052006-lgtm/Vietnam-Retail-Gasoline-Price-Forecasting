# Dự báo Giá Xăng Bán lẻ Việt Nam

**Vietnam Retail Gasoline Price Forecasting**  
**Tác giả**: Vũ Xuân Duy Anh 
**Lần chạy pipeline gần nhất**: 2026-06-17

Project dự báo giá bán lẻ xăng Việt Nam cho kỳ điều hành kế tiếp, tập trung vào **RON95** và **E5 RON92**. Dữ liệu giá bán lẻ được lấy từ bộ dữ liệu mã nguồn mở [`toanqng/fuel`](https://github.com/toanqng/fuel), bộ dữ liệu này ghi nguồn tham chiếu là Petrolimex; project lưu lại snapshot đã làm sạch trong `data/` để tái lập kết quả. Các biến thị trường quốc tế gồm Brent, WTI, DXY, VIX và USD/VND được tải từ Yahoo Finance.

## Kết quả hiện tại

Weighted MAPE qua walk-forward validation, tính trọng số theo số quan sát trong từng fold:

| Model | RON95 MAPE | E5 RON92 MAPE |
|---|---:|---:|
| Naive Baseline | 3.73% | 3.80% |
| XGBoost | 4.12% | 4.27% |
| MLP Neural Net (Appendix) | 5.54% | 5.94% |
| LightGBM | 6.65% | 6.47% |
| SARIMA | 11.64% | 11.48% |

Forecast snapshot hiện được commit trong `output/forecast_this_thursday.csv`:

| Target date | Market data as of | RON95 forecast | E5 forecast |
|---|---|---:|---:|
| 2026-06-18 | 2026-06-17 | 22,941 VND/L | 21,607 VND/L |

Điểm dự báo dùng **XGBoost** vì đây là ML model tốt nhất trong CV. Khoảng dự báo 2.5%-97.5% dùng **LightGBM Quantile Regression**; đây là prediction interval dựa trên mô hình, chưa được kiểm định calibration coverage. File forecast cũng giữ cột **Naive Baseline** để so sánh với giá kỳ gần nhất.

File forecast có thêm `Generated_On` và `Forecast_Status` để phân biệt snapshot còn hiện hành với snapshot đã stale khi nguồn dữ liệu giá bán lẻ chưa cập nhật kỳ mới.

## Cấu trúc

```text
PVIProject/
├── data/
│   ├── vietnam_retail_prices.csv
│   ├── yfinance_features.csv
│   └── merged_dataset.csv
├── src/
│   ├── data_fetcher.py
│   ├── features.py
│   ├── models.py
│   ├── analysis.py
│   └── create_notebook.py
├── output/
│   ├── forecast_this_thursday.csv
│   ├── model_comparison.csv
│   ├── ron95_cv_results.csv
│   ├── e5_cv_results.csv
│   ├── policy_cycle_comparison.csv
│   ├── sensitivity_matrix.csv
│   ├── policy_analysis.png
│   └── sensitivity_heatmap.png
├── oil_price_forecasting.ipynb
├── pipeline.py
├── requirements.txt
└── README.md
```

## Cài đặt và chạy

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python pipeline.py
```

Pipeline sẽ:

1. Tải và làm sạch dữ liệu giá bán lẻ từ `toanqng/fuel`.
2. Tải biến thị trường từ Yahoo Finance.
3. Ghép dữ liệu theo nguyên tắc chỉ dùng market data có sẵn trước ngày điều hành.
4. Tạo feature, chạy walk-forward CV và lưu bảng so sánh model.
5. Sinh so sánh resampling chu kỳ 7 ngày/10 ngày, sensitivity heatmap, forecast CSV và notebook.

## Kiểm soát rò rỉ dữ liệu (anti-leakage)

| Bước | Cách xử lý |
|---|---|
| Ghép dữ liệu | Market features được dịch +1 ngày trước `merge_asof`, nên tại ngày điều hành $t$ chỉ có giá thị trường đến $t-1$ |
| Missing market | Chỉ `ffill()`, không `bfill()` để tránh kéo giá tương lai về quá khứ |
| Giá bán lẻ | `RON95_lag1`, `RON95_lag2` dùng `.shift()` — không dùng giá kỳ hiện tại làm feature |
| Rolling MA/Std | Tính trên chuỗi thị trường đã căn chỉnh; mỗi hàng chỉ dùng quan sát cùng hàng và các hàng trước |
| Duplicate ngày | Gộp về một bản ghi/ngày trước khi tạo lag |
| Cross-validation | Walk-forward theo năm; winsorizing fit trên train fold, transform test fold |
| Forecast kỳ tới | Lấy snapshot market mới nhất từ `yfinance_features.csv` trước ngày điều hành, không chỉ dùng dòng cuối của `merged_dataset.csv` |

**Lưu ý phương pháp**: Dự báo điểm và sensitivity heatmap huấn luyện trên toàn bộ lịch sử (production inference). Chỉ phần đánh giá model trong `models.py` và so sánh resampling chu kỳ 7 ngày/10 ngày dùng CV có kiểm soát leakage. So sánh chu kỳ này là stress test theo tần suất dữ liệu, không phải kết luận nhân quả về hiệu quả chính sách.

## Ghi chú

Naive Baseline vẫn rất mạnh vì giá bán lẻ xăng dầu Việt Nam có quán tính điều hành lớn. Do đó project giữ baseline này như thước đo nghiêm khắc; ML model chỉ được xem là hữu ích khi vượt hoặc tiến sát baseline trong các giai đoạn biến động.

SARIMA được giữ như benchmark chuỗi thời gian đơn biến (không dùng exogenous variables), nên kết quả kém hơn ML có feature thị trường là điều có thể dự đoán được.
