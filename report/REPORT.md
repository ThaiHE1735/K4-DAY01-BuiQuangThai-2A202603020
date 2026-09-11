# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:** 11/09/2026

**Runtime Colab:** CPU

**Python / PyTorch / Ultralytics:** Python 3.13.15 / PyTorch 2.11.0+cpu / Ultralytics 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`):  
  `{"class_id": 468, "class_name": "cab", "rank": 1, "score": 0.510915, "taxonomy_name": "ImageNet-1K"}`

- Record này mô tả toàn ảnh như thế nào?  
  Đây là prediction cấp ảnh của YOLO11n-cls. `rank=1` là lớp có model score cao nhất đối với toàn bộ ảnh `traffic`.

- Ai định nghĩa class list mà checkpoint có thể dự đoán?  
  Danh sách lớp được xác định bởi taxonomy mà checkpoint đã được huấn luyện với. Trong bài này, `yolo11n-cls.pt` sử dụng taxonomy `ImageNet-1K`.

- Vì sao cần giữ cả ID, tên lớp và tên taxonomy?  
  `class_id` giúp hệ thống xử lý nhãn nhất quán, `class_name` giúp con người hiểu ý nghĩa của lớp, còn `taxonomy_name` xác định hệ phân loại mà ID và tên lớp thuộc về. Giữ đủ các thông tin này giúp tránh nhầm lẫn giữa các taxonomy khác nhau.

- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?  
  Guideline cần quy định rõ tiêu chí chọn lớp đại diện cho toàn ảnh, ví dụ ưu tiên chủ thể chính hoặc đối tượng quan trọng theo mục tiêu của bộ dữ liệu.

- Vì sao model score không phải ground truth?  
  Model score chỉ thể hiện mức độ tự tin của mô hình đối với prediction. Ground truth phải được xác định theo guideline và được annotator hoặc reviewer xác nhận. Prediction của mô hình có thể đúng, sai hoặc bỏ sót.

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`):  
  `{"class_name": "person", "score": 0.912625, "bbox_xyxy": [385.33, 69.24, 498.92, 348.92], "bbox_width": 113.58, "bbox_height": 279.68}`

- Diễn giải vị trí box bằng lời:  
  Bounding box dùng định dạng `xyxy = [x_min, y_min, x_max, y_max]` theo pixel. Gốc tọa độ nằm ở góc trên bên trái ảnh. `(x_min, y_min)` là góc trên-trái và `(x_max, y_max)` là góc dưới-phải của box.

- So sánh số prediction ở hai threshold:  
  Threshold `0.20` có **17** prediction; threshold `0.35` có **11** prediction; threshold `0.60` có **6** prediction.

- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?  
  Khi giảm threshold, mô hình thường trả về nhiều prediction hơn. Điều này có thể tăng độ bao phủ và giảm nguy cơ bỏ sót object, nhưng cũng làm tăng các prediction yếu hoặc false positive nên reviewer phải kiểm tra nhiều hơn. Khi tăng threshold, số prediction giảm nhưng nguy cơ bỏ sót object tăng.

- Đề xuất một quy tắc box chặt:  
  Bounding box phải bao sát toàn bộ phần nhìn thấy của object, không cắt mất phần rõ ràng của vật thể và không chứa quá nhiều background không liên quan.

- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?  
  Guideline cần quy định mức độ nhìn thấy tối thiểu để annotate, cách xử lý object bị cắt ở mép ảnh và box sẽ bao phần nhìn thấy hay phần ước lượng. Nếu không thể quyết định nhất quán thì cần chuyển reviewer hoặc escalation.

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):  
  `{"instance_id": "kitchen-001", "class_name": "person", "score": 0.899318, "polygon_point_count": 348, "polygon_xy_first_8": [[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0], [441.0, 73.0], [439.0, 73.0], [438.0, 74.0]]}`

- Polygon bổ sung chi tiết gì so với box?  
  Bounding box chỉ tạo một hình chữ nhật bao quanh object, còn polygon có thể bám theo đường biên thực tế của vật thể. Vì vậy polygon mô tả chính xác hơn vùng pixel thuộc object và giảm lượng background nằm trong annotation.

- `instance_id` dùng để làm gì và không phải loại ID nào?  
  `instance_id` dùng để phân biệt từng object riêng biệt trong cùng một ảnh. Hai object cùng class vẫn có `instance_id` khác nhau. `instance_id` không phải `class_id`.

- Đề xuất một quy tắc biên mask:  
  Polygon hoặc mask phải bám sát phần biên nhìn thấy của object, không lấy background rõ ràng và không bỏ những vùng rõ ràng thuộc object.

- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?  
  Guideline cần quy định cách xử lý biên không rõ, hai object tiếp xúc nhau và vùng bị che khuất. Nếu annotator không thể xác định nhất quán thì cần chuyển cho reviewer hoặc escalation.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một class cho toàn ảnh | Ảnh có thể có nhiều chủ thể; model có thể dự đoán sai | Chọn class theo guideline và taxonomy | Kiểm tra class và việc áp dụng guideline |
| Phát hiện vật thể | Class + bounding box cho từng object | Model có thể bỏ sót, nhầm class hoặc box quá rộng/hẹp | Gán đủ object và vẽ box sát vật thể | Kiểm tra missing object, class và biên box |
| Instance segmentation | Class + polygon/mask cho từng instance | Biên mờ, che khuất hoặc object tiếp xúc | Tạo polygon riêng cho từng instance | Kiểm tra biên mask, missing object và việc tách instance |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu:  
  Chỉ sử dụng ảnh và dữ liệu nằm trong phạm vi bài thực hành. Không đưa họ tên, MSSV, email, số điện thoại hoặc dữ liệu nhạy cảm vào báo cáo và output.

- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho:  
  Giảng viên, mentor hoặc Lab Coach phụ trách khóa học.

## 6. Danh sách bằng chứng

- [x] `classification_predictions.json`
- [x] `detection_predictions.json`
- [x] `segmentation_predictions.json`
- [x] `IMAGE_ATTRIBUTION.md`
- [x] `visuals/classification_top5.png`
- [x] `visuals/detection_predictions.png`
- [x] `visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`.
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
